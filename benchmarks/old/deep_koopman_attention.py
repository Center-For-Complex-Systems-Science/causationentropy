"""
Per-Pair Deep Koopman with Attention Transition for Causal Discovery on Rössler Oscillators.

Based on deep_koopman_perpair.py, but replaces the fixed linear K matrix
with a single self-attention head. This makes the Koopman transition
input-dependent — different regions of the attractor get different
effective dynamics.

For each candidate pair (j→i):
  - Marginal model: MLP(x_i) → M features, Attention predicts next features
  - Joint model:    MLP([x_i, x_j]) → M features, Attention predicts next features
  Compare prediction quality on held-out test set via bootstrap test.

Methods compared:
  1. Kausal-Attn (per-pair): Per-pair MLPs + Attention transition, bootstrap test
  2. Attn-Koopman + PCMCI:   Shared attention Koopman features + PCMCI
  3. Attn-Koopman + oCSE:    Shared attention Koopman features + oCSE
  4. Linear Granger:         Pairwise DMD in raw state space (no MLP)
  5. PCMCI (raw):            Tigramite PCMCI on derivative basis
  6. oCSE (raw):             oCSE on derivative basis
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import math
import warnings
import sys
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx
from causationentropy.datasets.synthetic import (
    generate_graph_topology,
    simulate_rossler,
    prepare_rossler_data_for_causal_discovery,
)

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# =============================================================================
# ATTENTION TRANSITION (replaces K matrix)
# =============================================================================


class AttentionTransition(nn.Module):
    """Single self-attention head as an input-dependent Koopman transition.

    Treats M latent features as a sequence of M tokens (each dim 1).
    Applies single-head self-attention to compute an input-dependent M→M
    transition, replacing the fixed linear K matrix.

    Args:
        M: number of latent features (sequence length)
        d_head: attention head dimension
    """

    def __init__(self, M, d_head=16):
        super().__init__()
        self.M = M
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)

        self.proj_in = nn.Linear(1, d_head)
        self.W_Q = nn.Linear(d_head, d_head)
        self.W_K = nn.Linear(d_head, d_head)
        self.W_V = nn.Linear(d_head, d_head)
        self.proj_out = nn.Linear(d_head, 1)

    def forward(self, z_t):
        # z_t: (batch, M)
        tokens = self.proj_in(z_t.unsqueeze(-1))  # (batch, M, d_head)
        Q = self.W_Q(tokens)  # (batch, M, d_head)
        K = self.W_K(tokens)  # (batch, M, d_head)
        V = self.W_V(tokens)  # (batch, M, d_head)

        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)  # (batch, M, M)
        out = attn @ V  # (batch, M, d_head)
        z_next = self.proj_out(out).squeeze(-1)  # (batch, M)
        return z_next


class NodeAttentionTransition(nn.Module):
    """Node-level attention transition: each node is a token.

    The attention matrix is (batch, n_nodes, n_nodes) and can be read
    directly as a soft adjacency matrix. attn[i,j] = how much node j
    contributes to node i's next state.

    Args:
        n_nodes: number of nodes (sequence length)
        d_node: features per node (token dimension)
        d_head: attention head dimension
    """

    def __init__(self, n_nodes, d_node, d_head=16):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_node = d_node
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)

        self.W_Q = nn.Linear(d_node, d_head)
        self.W_K = nn.Linear(d_node, d_head)
        self.W_V = nn.Linear(d_node, d_head)
        self.proj_out = nn.Linear(d_head, d_node)

    def forward(self, z_t, return_attn=False):
        # z_t: (batch, n_nodes * d_node)
        tokens = z_t.view(-1, self.n_nodes, self.d_node)  # (batch, n_nodes, d_node)
        Q = self.W_Q(tokens)  # (batch, n_nodes, d_head)
        K = self.W_K(tokens)  # (batch, n_nodes, d_head)
        V = self.W_V(tokens)  # (batch, n_nodes, d_head)

        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)  # (batch, n_nodes, n_nodes)
        out = attn @ V  # (batch, n_nodes, d_head)
        z_next = self.proj_out(out).reshape(-1, self.n_nodes * self.d_node)  # (batch, n_nodes * d_node)

        if return_attn:
            return z_next, attn
        return z_next


# =============================================================================
# SHARED PER-NODE KOOPMAN WITH ATTENTION TRANSITION
# =============================================================================


class NodeEncoder(nn.Module):
    """MLP encoder for a single node: R^3 → R^d_node."""

    def __init__(self, state_dim=3, d_node=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, d_node),
        )

    def forward(self, x):
        return self.net(x)


class PerNodeAttentionKoopman(nn.Module):
    """Per-node encoder/decoder with node-level attention transition.

    Each node gets its own encoder (R^3 → R^d_node) and decoder (R^d_node → R^3).
    The transition uses NodeAttentionTransition where each node is a token,
    producing a (n_nodes, n_nodes) attention matrix interpretable as a
    soft adjacency matrix.
    """

    def __init__(self, n_nodes, d_node, d_head=16, state_per_node=3):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_node = d_node
        self.state_per_node = state_per_node
        self.latent_dim = n_nodes * d_node

        self.encoders = nn.ModuleList([
            NodeEncoder(state_per_node, d_node) for _ in range(n_nodes)
        ])

        self.transition = NodeAttentionTransition(n_nodes, d_node, d_head)

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_node, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, state_per_node),
            )
            for _ in range(n_nodes)
        ])

    def encode(self, x):
        z_parts = []
        for i in range(self.n_nodes):
            xi = x[:, i * self.state_per_node : (i + 1) * self.state_per_node]
            z_parts.append(self.encoders[i](xi))
        return torch.cat(z_parts, dim=1)

    def decode(self, z):
        x_parts = []
        for i in range(self.n_nodes):
            zi = z[:, i * self.d_node : (i + 1) * self.d_node]
            x_parts.append(self.decoders[i](zi))
        return torch.cat(x_parts, dim=1)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def predict(self, x):
        z = self.encode(x)
        z_next = self.transition(z)
        x_next_pred = self.decode(z_next)
        return x_next_pred, z_next

    def encode_per_node(self, data_norm):
        """Encode trajectory and return list of per-node latent arrays."""
        with torch.no_grad():
            x = torch.tensor(data_norm, dtype=torch.float32).to(device)
            per_node = []
            for i in range(self.n_nodes):
                xi = x[:, i * self.state_per_node : (i + 1) * self.state_per_node]
                zi = self.encoders[i](xi)
                per_node.append(zi.cpu().numpy())
            return per_node  # list of (T, d_node) arrays

    def get_attention_adjacency(self, data_norm, threshold_std=1.0):
        """Extract binary adjacency from average attention weights.

        attn[i,j] = how much node j contributes to node i's next state.
        High attn[i,j] → j causes i → adj[j,i] = 1.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(data_norm, dtype=torch.float32).to(device)
            z = self.encode(x)
            _, attn = self.transition(z, return_attn=True)  # (T, n_nodes, n_nodes)
            avg_attn = attn.mean(dim=0).cpu().numpy()  # (n_nodes, n_nodes)

        n = self.n_nodes
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag = avg_attn[off_diag_mask]
        threshold = off_diag.mean() + threshold_std * off_diag.std()

        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j and avg_attn[i, j] > threshold:
                    adj[j, i] = 1  # attn[i,j] high → j causes i
        return adj, avg_attn


# =============================================================================
# PER-PAIR KOOPMAN MODEL WITH ATTENTION
# =============================================================================


class PairKoopmanModel(nn.Module):
    """Koopman model for a single causal pair test, using attention transition.

    Marginal (input_dim=3): encoder sees only node i's state.
    Joint (input_dim=6): encoder sees [node i, node j].
    Decoder always predicts node i's 3D state.
    """

    def __init__(self, input_dim, M=32, target_dim=3, d_head=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, M),
        )
        self.transition = AttentionTransition(M, d_head)
        self.decoder = nn.Sequential(
            nn.Linear(M, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, target_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def predict_next(self, x_t):
        z_t = self.encode(x_t)
        z_next = self.transition(z_t)
        return self.decode(z_next), z_t, z_next


def train_pair_model(
    x_t,
    x_tp1_enc,
    target_t,
    target_tp1,
    input_dim,
    M=32,
    target_dim=3,
    d_head=16,
    epochs=500,
    lr=1e-3,
    batch_size=128,
    lambda_pred=1.0,
    lambda_lin=0.5,
    lambda_recon=0.5,
    weight_decay=1e-4,
):
    """Train one per-pair Koopman model with attention transition.

    Args:
        x_t:         (T, input_dim) encoder input at time t
        x_tp1_enc:   (T, input_dim) encoder input at t+1 (linearity loss)
        target_t:    (T, target_dim) reconstruction target (x_i at t)
        target_tp1:  (T, target_dim) prediction target (x_i at t+1)
    """
    ds = TensorDataset(
        torch.tensor(x_t, dtype=torch.float32),
        torch.tensor(x_tp1_enc, dtype=torch.float32),
        torch.tensor(target_t, dtype=torch.float32),
        torch.tensor(target_tp1, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = PairKoopmanModel(input_dim, M, target_dim, d_head).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        for b_xt, b_xtp1, b_tgt_t, b_tgt_tp1 in loader:
            b_xt = b_xt.to(device)
            b_xtp1 = b_xtp1.to(device)
            b_tgt_t = b_tgt_t.to(device)
            b_tgt_tp1 = b_tgt_tp1.to(device)

            z_t = model.encode(b_xt)
            z_next = model.transition(z_t)

            # Reconstruction: decoder(encoder(x_t)) ≈ x_i(t)
            L_recon = torch.mean((model.decode(z_t) - b_tgt_t) ** 2)

            # Prediction: decoder(attention(encoder(x_t))) ≈ x_i(t+1)
            L_pred = torch.mean((model.decode(z_next) - b_tgt_tp1) ** 2)

            # Linearity: attention(encoder(x_t)) ≈ encoder(x_{t+1})
            z_tp1 = model.encode(b_xtp1)
            L_lin = torch.mean((z_next - z_tp1.detach()) ** 2)

            loss = lambda_recon * L_recon + lambda_pred * L_pred + lambda_lin * L_lin

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def prediction_errors(model, x_t, target_tp1):
    """Per-timestep prediction MSE on test data."""
    model.eval()
    with torch.no_grad():
        pred, _, _ = model.predict_next(
            torch.tensor(x_t, dtype=torch.float32).to(device)
        )
        pred = pred.cpu().numpy()
    return np.sum((target_tp1 - pred) ** 2, axis=1)


# =============================================================================
# SHARED ATTENTION KOOPMAN TRAINING
# =============================================================================


def train_shared_attention_koopman(
    traj, n_nodes, d_node=8, d_head=16, epochs=1000, lr=1e-3, batch_size=256,
    lambda_pred=1.0, lambda_lin=0.5, lambda_sparse=0.01, print_every=200,
):
    """Train a shared per-node Koopman autoencoder with attention transition."""
    T = traj.shape[0]
    state_dim = n_nodes * 3

    data_flat = traj.reshape(T, state_dim)
    mean = data_flat.mean(axis=0)
    std = data_flat.std(axis=0)
    std[std < 1e-8] = 1.0
    data_norm = (data_flat - mean) / std

    x_t = torch.tensor(data_norm[:-1], dtype=torch.float32)
    x_tp1 = torch.tensor(data_norm[1:], dtype=torch.float32)

    dataset = TensorDataset(x_t, x_tp1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = PerNodeAttentionKoopman(n_nodes, d_node, d_head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch_xt, batch_xtp1 in loader:
            batch_xt = batch_xt.to(device)
            batch_xtp1 = batch_xtp1.to(device)

            x_recon, z_t = model(batch_xt)
            x_next_pred, z_next_pred = model.predict(batch_xt)
            _, z_tp1 = model(batch_xtp1)

            L_recon = torch.mean((x_recon - batch_xt) ** 2)
            L_pred = torch.mean((x_next_pred - batch_xtp1) ** 2)
            L_lin = torch.mean((z_next_pred - z_tp1.detach()) ** 2)
            L_sparse = torch.mean(torch.abs(z_t))  # L1 on encoder activations

            loss = L_recon + lambda_pred * L_pred + lambda_lin * L_lin + lambda_sparse * L_sparse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % print_every == 0 or epoch == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"    Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}")

    return model, mean, std


# =============================================================================
# LATENT-SPACE CAUSAL DISCOVERY (PCMCI / oCSE on Koopman features)
# =============================================================================


def latent_pcmci_test(Z_nodes, n_nodes, d_node, alpha=0.05, tau_max=1):
    """Run PCMCI on per-node Koopman latent features, map back to node adjacency."""
    Z_all = np.column_stack(Z_nodes)  # (T, n_nodes * d_node)
    n_vars = n_nodes * d_node
    var_names = [f"z{i}_{k}" for i in range(n_nodes) for k in range(d_node)]

    T = Z_all.shape[0]
    dataframe = pp.DataFrame(
        Z_all, datatime={0: np.arange(T)}, var_names=var_names,
    )
    pcmci_obj = PCMCI(
        dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0,
    )
    pcmci_res = pcmci_obj.run_pcmci(
        tau_min=1, tau_max=tau_max, pc_alpha=alpha,
    )

    # p_matrix[i, j, tau]: p-value for link X_j(t-tau) → X_i(t)
    p_matrix = pcmci_res["p_matrix"]
    adj = np.zeros((n_nodes, n_nodes), dtype=int)

    for dst_var in range(n_vars):
        for src_var in range(n_vars):
            src_node = src_var // d_node
            dst_node = dst_var // d_node
            if src_node == dst_node:
                continue
            for tau in range(1, tau_max + 1):
                if p_matrix[dst_var, src_var, tau] < alpha:
                    adj[src_node, dst_node] = 1

    return adj


def latent_ocse_test(Z_nodes, n_nodes, d_node, alpha=0.05, tau_max=1):
    """Run oCSE on per-node Koopman latent features, map back to node adjacency."""
    Z_all = np.column_stack(Z_nodes)
    var_names = [f"z{i}_{k}" for i in range(n_nodes) for k in range(d_node)]

    Z_df = pd.DataFrame(Z_all, columns=var_names)
    with suppress_stdout():
        oce_network = discover_network(
            data=Z_df, max_lag=tau_max, method="standard",
            information="gaussian", alpha_forward=alpha, alpha_backward=alpha,
        )

    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for edge in oce_network.edges():
        u, v = edge[0], edge[1]
        src_node = int(u.split("_")[0][1:])  # "z0_3" → 0
        dst_node = int(v.split("_")[0][1:])  # "z2_1" → 2
        if src_node != dst_node:
            adj[src_node, dst_node] = 1

    return adj


# =============================================================================
# PER-PAIR KAUSAL TEST (ATTENTION)
# =============================================================================


def kausal_perpair_test(
    traj_nodes_norm,
    n_nodes,
    M=32,
    d_head=16,
    epochs=500,
    lr=1e-3,
    batch_size=128,
    lambda_pred=1.0,
    lambda_lin=0.5,
    lambda_recon=0.5,
    n_bootstrap=200,
    p_crit=0.05,
    train_frac=0.8,
):
    """Per-pair Koopman Granger causality test with attention transition.

    For each pair (j→i):
      1. Marginal: train MLP_i on x_i → predict x_i(t+1)
      2. Joint:    train MLP_ji on [x_i, x_j] → predict x_i(t+1)
      3. Compare held-out prediction errors via bootstrap test.
    """
    T = traj_nodes_norm[0].shape[0]
    split = int((T - 1) * train_frac)

    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    pvals = np.ones((n_nodes, n_nodes))
    delta_means = np.zeros((n_nodes, n_nodes))

    # --- Train marginal models (one per target node) ---
    marg_errors_test = []
    for i in range(n_nodes):
        xi = traj_nodes_norm[i]
        xi_t, xi_tp1 = xi[:-1], xi[1:]

        model = train_pair_model(
            x_t=xi_t[:split],
            x_tp1_enc=xi_tp1[:split],
            target_t=xi_t[:split],
            target_tp1=xi_tp1[:split],
            input_dim=3,
            M=M,
            target_dim=3,
            d_head=d_head,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            lambda_pred=lambda_pred,
            lambda_lin=lambda_lin,
            lambda_recon=lambda_recon,
        )
        errs = prediction_errors(model, xi_t[split:], xi_tp1[split:])
        marg_errors_test.append(errs)

    # --- Train joint models & compare ---
    for i in range(n_nodes):
        xi = traj_nodes_norm[i]
        xi_t, xi_tp1 = xi[:-1], xi[1:]

        for j in range(n_nodes):
            if i == j:
                continue

            xj = traj_nodes_norm[j]
            xj_t, xj_tp1 = xj[:-1], xj[1:]

            joint_t = np.concatenate([xi_t, xj_t], axis=1)
            joint_tp1 = np.concatenate([xi_tp1, xj_tp1], axis=1)

            model = train_pair_model(
                x_t=joint_t[:split],
                x_tp1_enc=joint_tp1[:split],
                target_t=xi_t[:split],
                target_tp1=xi_tp1[:split],
                input_dim=6,
                M=M,
                target_dim=3,
                d_head=d_head,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                lambda_pred=lambda_pred,
                lambda_lin=lambda_lin,
                lambda_recon=lambda_recon,
            )
            errs_joint = prediction_errors(model, joint_t[split:], xi_tp1[split:])
            errs_marg = marg_errors_test[i]

            delta = errs_marg - errs_joint
            T_eff = len(delta)

            rng = np.random.default_rng(42 + i * n_nodes + j)
            boot = np.array(
                [
                    delta[rng.choice(T_eff, size=T_eff, replace=True)].mean()
                    for _ in range(n_bootstrap)
                ]
            )
            p_val = (np.sum(boot <= 0) + 1) / (n_bootstrap + 1)

            delta_means[j, i] = delta.mean()
            pvals[j, i] = p_val
            if p_val < p_crit:
                adj[j, i] = 1

    return adj, pvals, delta_means


# =============================================================================
# LINEAR GRANGER (DMD in raw state space, no MLP)
# =============================================================================


def linear_granger_test(
    traj_nodes_norm,
    n_nodes,
    n_bootstrap=200,
    p_crit=0.05,
    train_frac=0.8,
):
    """Pairwise Granger causality via DMD in raw state space."""
    T = traj_nodes_norm[0].shape[0]
    split = int((T - 1) * train_frac)

    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    pvals = np.ones((n_nodes, n_nodes))

    marg_errors = []
    for i in range(n_nodes):
        xi = traj_nodes_norm[i]
        xi_t, xi_tp1 = xi[:-1], xi[1:]

        X_tr, X_te = xi_t[:split].T, xi_t[split:].T
        Y_tr, Y_te = xi_tp1[:split].T, xi_tp1[split:].T

        K = Y_tr @ np.linalg.pinv(X_tr)
        marg_errors.append(np.sum((Y_te - K @ X_te) ** 2, axis=0))

    for i in range(n_nodes):
        xi = traj_nodes_norm[i]
        xi_t, xi_tp1 = xi[:-1], xi[1:]

        for j in range(n_nodes):
            if i == j:
                continue

            xj = traj_nodes_norm[j]
            xj_t = xj[:-1]

            joint_t = np.concatenate([xi_t, xj_t], axis=1)

            X_tr, X_te = joint_t[:split].T, joint_t[split:].T
            Y_tr, Y_te = xi_tp1[:split].T, xi_tp1[split:].T

            K = Y_tr @ np.linalg.pinv(X_tr)
            errs_joint = np.sum((Y_te - K @ X_te) ** 2, axis=0)

            delta = marg_errors[i] - errs_joint
            T_eff = len(delta)
            rng = np.random.default_rng(42 + i * n_nodes + j)
            boot = np.array(
                [
                    delta[rng.choice(T_eff, size=T_eff, replace=True)].mean()
                    for _ in range(n_bootstrap)
                ]
            )
            p_val = (np.sum(boot <= 0) + 1) / (n_bootstrap + 1)

            pvals[j, i] = p_val
            if p_val < p_crit:
                adj[j, i] = 1

    return adj, pvals


# =============================================================================
# BASELINE HELPERS
# =============================================================================


def extract_node_graph_pcmci(graph_nx, n, basis_map, coupling_on="x"):
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))
    target_start = {"x": 0, "y": n, "z": 2 * n}[coupling_on]
    coupling_start = 3 * n
    for i in range(n):
        target_node_idx = target_start + i
        for idx, (target, source) in enumerate(basis_map):
            coupling_node_idx = coupling_start + idx
            if graph_nx.has_edge(coupling_node_idx, target_node_idx):
                edges = graph_nx[coupling_node_idx][target_node_idx]
                for _, edata in edges.items():
                    if target == i:
                        G.add_edge(source, i)
    return G


def extract_node_graph_oce(graph_nx, n, basis_map, var_names, coupling_on="x"):
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))
    target_vars = [f"d{coupling_on}{i}" for i in range(n)]
    base = 3 * n
    for i in range(n):
        target_var = target_vars[i]
        for idx, (target, source) in enumerate(basis_map):
            coupling_var = var_names[base + idx]
            if graph_nx.has_edge(coupling_var, target_var):
                edges = graph_nx[coupling_var][target_var]
                for _, edata in edges.items():
                    if target == i:
                        G.add_edge(source, i)
    return G


def collapse_to_binary_adjacency(node_graph, n):
    A = np.zeros((n, n), dtype=int)
    for u, v, _ in node_graph.edges(data=True):
        A[u, v] = 1
    return A


# =============================================================================
# METRICS
# =============================================================================


def compute_metrics(predicted_adj, true_adj, time_taken):
    B = (predicted_adj != 0).astype(int)
    A = (true_adj != 0).astype(int)

    tp = np.sum((B == 1) & (A == 1))
    fp = np.sum((B == 1) & (A == 0))
    tn = np.sum((B == 0) & (A == 0))
    fn = np.sum((B == 0) & (A == 1))

    total = tp + fp + tn + fn
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * tpr) / (precision + tpr)
        if (precision + tpr) > 0
        else 0.0
    )
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "F1": f1,
        "Accuracy": accuracy,
        "Time": time_taken,
    }


# =============================================================================
# HELPERS
# =============================================================================


def extract_node_trajectories(traj_sub, n_nodes):
    """(T, n, 3) → list of (T, 3) per-node arrays."""
    return [traj_sub[:, i, :] for i in range(n_nodes)]


def normalize_node_trajectories(traj_nodes):
    """Independently normalize each node's trajectory to zero-mean unit-var."""
    normed = []
    for xi in traj_nodes:
        mu = xi.mean(axis=0)
        sigma = xi.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        normed.append((xi - mu) / sigma)
    return normed


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 5,
    "T": 5000,
    "n_trials": 3,
    "dt": 0.02,
    "subsample": 5,
    "burn_in": 500,
    # Rössler
    "a": 0.2,
    "b": 0.2,
    "c": 5.7,
    "rho_default": 0.4,
    "noise_std": 0.0,
    "init_scale": 1.0,
    "coupling_on": "x",
    "normalize_by_indegree": False,
    # PCMCI / oCSE
    "alpha": 0.05,
    "tau_max": 1,
    # Shared attention Koopman
    "d_node": 5,  # = n_nodes, so attention matrix is (n_nodes, n_nodes)
    "d_head_shared": 16,
    "epochs_shared": 1000,
    "batch_size_shared": 256,
    "lambda_sparse": 0.01,
    # Per-pair Koopman with Attention
    "M": 32,
    "d_head": 16,
    "epochs_perpair": 500,
    "lr": 1e-3,
    "batch_size": 128,
    "lambda_pred": 1.0,
    "lambda_lin": 0.5,
    "lambda_recon": 0.5,
    "train_frac": 0.8,
    # Bootstrap test
    "n_bootstrap": 200,
    "p_crit": 0.05,
}

EXPERIMENTS = {
    "ER_Density": {
        "type": "Erdos-Renyi",
        "vary_param": "p_edge",
        "label": "Edge Probability (p)",
        "values": [0.2, 0.4, 0.6],
        "defaults": {"rho": 0.2},
    },
}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = []

    n_nodes = GLOBAL_PARAMS["n_nodes"]
    M = GLOBAL_PARAMS["M"]
    d_head = GLOBAL_PARAMS["d_head"]
    n_pairs = n_nodes * (n_nodes - 1)

    print("=" * 80)
    print("Per-Pair Koopman (Attention) Causal Discovery — Rössler Oscillators")
    print(f"Device: {device}")
    print(f"{n_nodes} nodes, M={M} features/model, d_head={d_head}, {n_pairs} pair tests per trial")
    print(f"{GLOBAL_PARAMS['n_trials']} trials per condition")
    print("=" * 80)

    total_iters = sum(
        len(cfg["values"]) * GLOBAL_PARAMS["n_trials"]
        for cfg in EXPERIMENTS.values()
    )
    pbar = tqdm(total=total_iters, desc="Total progress")
    os.makedirs("benchmarks/results", exist_ok=True)

    for exp_name, config in EXPERIMENTS.items():
        print(f"\n--- Experiment: {exp_name} ---")

        param_name = config["vary_param"]
        param_values = config["values"]
        defaults = config["defaults"]

        for val in param_values:
            current_params = defaults.copy()
            current_params[param_name] = val

            for trial in range(GLOBAL_PARAMS["n_trials"]):
                seed = 42 + (trial * 100)
                print(f"\n  {config['label']}={val}, Trial {trial}, seed={seed}")

                torch.manual_seed(seed)
                np.random.seed(seed)

                # Generate graph and simulate
                G = generate_graph_topology(
                    config["type"], n_nodes, current_params, seed
                )
                true_adj = nx.to_numpy_array(G).astype(int)

                traj, _ = simulate_rossler(
                    G=G,
                    T=GLOBAL_PARAMS["T"],
                    dt=GLOBAL_PARAMS["dt"],
                    rho=current_params.get("rho", GLOBAL_PARAMS["rho_default"]),
                    seed=seed,
                    a=GLOBAL_PARAMS["a"],
                    b=GLOBAL_PARAMS["b"],
                    c=GLOBAL_PARAMS["c"],
                    init_scale=GLOBAL_PARAMS["init_scale"],
                    noise_std=GLOBAL_PARAMS["noise_std"],
                    burn_in=GLOBAL_PARAMS["burn_in"],
                    normalize_by_indegree=GLOBAL_PARAMS["normalize_by_indegree"],
                    coupling_on=GLOBAL_PARAMS["coupling_on"],
                )

                k = GLOBAL_PARAMS["subsample"]
                traj_sub = traj[::k] if k > 1 else traj
                dt_eff = GLOBAL_PARAMS["dt"] * k

                # Prepare per-node normalized trajectories
                traj_nodes = extract_node_trajectories(traj_sub, n_nodes)
                traj_nodes_norm = normalize_node_trajectories(traj_nodes)

                n_true = int(true_adj.sum())
                n_possible = n_nodes * (n_nodes - 1)
                print(f"  True edges: {n_true}/{n_possible}")

                # ===========================================================
                # METHOD 1: Kausal-Attn (per-pair) — THE MAIN METHOD
                # ===========================================================
                print(f"  Training {n_nodes + n_pairs} per-pair attention models (M={M}, d_head={d_head})...")
                start_time = time.time()

                pred_adj_pp, pvals_pp, deltas_pp = kausal_perpair_test(
                    traj_nodes_norm,
                    n_nodes,
                    M=GLOBAL_PARAMS["M"],
                    d_head=GLOBAL_PARAMS["d_head"],
                    epochs=GLOBAL_PARAMS["epochs_perpair"],
                    lr=GLOBAL_PARAMS["lr"],
                    batch_size=GLOBAL_PARAMS["batch_size"],
                    lambda_pred=GLOBAL_PARAMS["lambda_pred"],
                    lambda_lin=GLOBAL_PARAMS["lambda_lin"],
                    lambda_recon=GLOBAL_PARAMS["lambda_recon"],
                    n_bootstrap=GLOBAL_PARAMS["n_bootstrap"],
                    p_crit=GLOBAL_PARAMS["p_crit"],
                    train_frac=GLOBAL_PARAMS["train_frac"],
                )
                pp_time = time.time() - start_time

                m_pp = compute_metrics(pred_adj_pp, true_adj, pp_time)
                m_pp.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "Kausal-Attn (per-pair)",
                    "Trial": trial,
                })
                results.append(m_pp)
                print(
                    f"  Attn pair  — F1: {m_pp['F1']:.3f}, "
                    f"TP: {m_pp['TP']}, FP: {m_pp['FP']}, "
                    f"TPR: {m_pp['TPR']:.2f}, FPR: {m_pp['FPR']:.2f}, "
                    f"Acc: {m_pp['Accuracy']:.3f}  ({pp_time:.0f}s)"
                )

                # ===========================================================
                # SHARED ATTENTION KOOPMAN (for PCMCI / oCSE on features)
                # ===========================================================
                print(f"  Training shared attention Koopman (d_node={GLOBAL_PARAMS['d_node']}, d_head={GLOBAL_PARAMS['d_head_shared']})...")
                shared_start = time.time()

                shared_model, data_mean, data_std = train_shared_attention_koopman(
                    traj_sub,
                    n_nodes=n_nodes,
                    d_node=GLOBAL_PARAMS["d_node"],
                    d_head=GLOBAL_PARAMS["d_head_shared"],
                    epochs=GLOBAL_PARAMS["epochs_shared"],
                    lr=GLOBAL_PARAMS["lr"],
                    batch_size=GLOBAL_PARAMS["batch_size_shared"],
                    lambda_pred=GLOBAL_PARAMS["lambda_pred"],
                    lambda_lin=GLOBAL_PARAMS["lambda_lin"],
                    lambda_sparse=GLOBAL_PARAMS["lambda_sparse"],
                )
                shared_train_time = time.time() - shared_start

                # Encode to per-node latent representations
                T_sub = traj_sub.shape[0]
                data_flat = traj_sub.reshape(T_sub, n_nodes * 3)
                data_norm = (data_flat - data_mean) / data_std
                Z_nodes = shared_model.encode_per_node(data_norm)
                print(f"  Shared model trained in {shared_train_time:.1f}s")

                # ===========================================================
                # METHOD 2: Attention Adjacency (direct readout)
                # ===========================================================
                print("  Extracting adjacency from attention weights...")
                start_time = time.time()

                pred_adj_attn, avg_attn = shared_model.get_attention_adjacency(
                    data_norm, threshold_std=1.0,
                )
                attn_adj_time = time.time() - start_time + shared_train_time

                m_attn_adj = compute_metrics(pred_adj_attn, true_adj, attn_adj_time)
                m_attn_adj.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "Attn Adjacency",
                    "Trial": trial,
                })
                results.append(m_attn_adj)
                print(
                    f"  Attn Adj   — F1: {m_attn_adj['F1']:.3f}, "
                    f"TP: {m_attn_adj['TP']}, FP: {m_attn_adj['FP']}, "
                    f"TPR: {m_attn_adj['TPR']:.2f}, FPR: {m_attn_adj['FPR']:.2f}, "
                    f"Acc: {m_attn_adj['Accuracy']:.3f}  ({attn_adj_time:.1f}s)"
                )

                # ===========================================================
                # METHOD 3: Attn-Koopman + PCMCI
                # ===========================================================
                print("  Running PCMCI on attention Koopman features...")
                start_time = time.time()

                pred_adj_lpcmci = latent_pcmci_test(
                    Z_nodes, n_nodes, GLOBAL_PARAMS["d_node"],
                    alpha=GLOBAL_PARAMS["alpha"],
                    tau_max=GLOBAL_PARAMS["tau_max"],
                )
                lpcmci_time = time.time() - start_time + shared_train_time

                m_lpcmci = compute_metrics(pred_adj_lpcmci, true_adj, lpcmci_time)
                m_lpcmci.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "Attn-Koopman + PCMCI",
                    "Trial": trial,
                })
                results.append(m_lpcmci)
                print(
                    f"  Attn+PCMCI — F1: {m_lpcmci['F1']:.3f}, "
                    f"TP: {m_lpcmci['TP']}, FP: {m_lpcmci['FP']}, "
                    f"TPR: {m_lpcmci['TPR']:.2f}, FPR: {m_lpcmci['FPR']:.2f}, "
                    f"Acc: {m_lpcmci['Accuracy']:.3f}  ({lpcmci_time:.1f}s)"
                )

                # ===========================================================
                # METHOD 3: Attn-Koopman + oCSE
                # ===========================================================
                print("  Running oCSE on attention Koopman features...")
                start_time = time.time()

                pred_adj_locse = latent_ocse_test(
                    Z_nodes, n_nodes, GLOBAL_PARAMS["d_node"],
                    alpha=GLOBAL_PARAMS["alpha"],
                    tau_max=GLOBAL_PARAMS["tau_max"],
                )
                locse_time = time.time() - start_time + shared_train_time

                m_locse = compute_metrics(pred_adj_locse, true_adj, locse_time)
                m_locse.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "Attn-Koopman + oCSE",
                    "Trial": trial,
                })
                results.append(m_locse)
                print(
                    f"  Attn+oCSE  — F1: {m_locse['F1']:.3f}, "
                    f"TP: {m_locse['TP']}, FP: {m_locse['FP']}, "
                    f"TPR: {m_locse['TPR']:.2f}, FPR: {m_locse['FPR']:.2f}, "
                    f"Acc: {m_locse['Accuracy']:.3f}  ({locse_time:.1f}s)"
                )

                # ===========================================================
                # METHOD 4: Linear Granger (DMD, no MLP)
                # ===========================================================
                print("  Running Linear Granger (DMD)...")
                start_time = time.time()

                pred_adj_lg, pvals_lg = linear_granger_test(
                    traj_nodes_norm,
                    n_nodes,
                    n_bootstrap=GLOBAL_PARAMS["n_bootstrap"],
                    p_crit=GLOBAL_PARAMS["p_crit"],
                    train_frac=GLOBAL_PARAMS["train_frac"],
                )
                lg_time = time.time() - start_time

                m_lg = compute_metrics(pred_adj_lg, true_adj, lg_time)
                m_lg.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "Linear Granger",
                    "Trial": trial,
                })
                results.append(m_lg)
                print(
                    f"  Lin Grang  — F1: {m_lg['F1']:.3f}, "
                    f"TP: {m_lg['TP']}, FP: {m_lg['FP']}, "
                    f"TPR: {m_lg['TPR']:.2f}, FPR: {m_lg['FPR']:.2f}, "
                    f"Acc: {m_lg['Accuracy']:.3f}  ({lg_time:.1f}s)"
                )

                # ===========================================================
                # METHOD 3: Raw PCMCI baseline
                # ===========================================================
                print("  Running PCMCI baseline (raw)...")
                start_time = time.time()

                X_basis, basis_meta, var_names = (
                    prepare_rossler_data_for_causal_discovery(
                        traj_sub,
                        dt=dt_eff,
                        coupling_on=GLOBAL_PARAMS["coupling_on"],
                    )
                )
                T_eff = X_basis.shape[0]
                dataframe = pp.DataFrame(
                    X_basis,
                    datatime={0: np.arange(T_eff)},
                    var_names=var_names,
                )
                pcmci = PCMCI(
                    dataframe=dataframe,
                    cond_ind_test=ParCorr(),
                    verbosity=0,
                )
                pcmci_res = pcmci.run_pcmci(
                    tau_min=1,
                    tau_max=GLOBAL_PARAMS["tau_max"],
                    pc_alpha=GLOBAL_PARAMS["alpha"],
                )
                graph_nx = pcmci_to_networkx(pcmci_res)
                pcmci_node_graph = extract_node_graph_pcmci(
                    graph_nx,
                    basis_meta["n"],
                    basis_meta["basis_map"],
                    coupling_on=GLOBAL_PARAMS["coupling_on"],
                )
                pred_adj_pcmci = collapse_to_binary_adjacency(
                    pcmci_node_graph, n_nodes
                )
                pcmci_time = time.time() - start_time

                m_pcmci = compute_metrics(pred_adj_pcmci, true_adj, pcmci_time)
                m_pcmci.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "PCMCI (raw)",
                    "Trial": trial,
                })
                results.append(m_pcmci)
                print(
                    f"  PCMCI raw  — F1: {m_pcmci['F1']:.3f}, "
                    f"TP: {m_pcmci['TP']}, FP: {m_pcmci['FP']}, "
                    f"TPR: {m_pcmci['TPR']:.2f}, FPR: {m_pcmci['FPR']:.2f}, "
                    f"Acc: {m_pcmci['Accuracy']:.3f}  ({pcmci_time:.1f}s)"
                )

                # ===========================================================
                # METHOD 4: Raw oCSE baseline
                # ===========================================================
                print("  Running oCSE baseline (raw)...")
                start_time = time.time()

                X_df = pd.DataFrame(X_basis, columns=var_names)
                with suppress_stdout():
                    oce_network = discover_network(
                        data=X_df,
                        max_lag=GLOBAL_PARAMS["tau_max"],
                        method="standard",
                        information="gaussian",
                        alpha_forward=GLOBAL_PARAMS["alpha"],
                        alpha_backward=GLOBAL_PARAMS["alpha"],
                    )

                oce_node_graph = extract_node_graph_oce(
                    oce_network,
                    n_nodes,
                    basis_meta["basis_map"],
                    var_names,
                    coupling_on=GLOBAL_PARAMS["coupling_on"],
                )
                pred_adj_oce = collapse_to_binary_adjacency(
                    oce_node_graph, n_nodes
                )
                oce_time = time.time() - start_time

                m_oce = compute_metrics(pred_adj_oce, true_adj, oce_time)
                m_oce.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "oCSE (raw)",
                    "Trial": trial,
                })
                results.append(m_oce)
                print(
                    f"  oCSE raw   — F1: {m_oce['F1']:.3f}, "
                    f"TP: {m_oce['TP']}, FP: {m_oce['FP']}, "
                    f"TPR: {m_oce['TPR']:.2f}, FPR: {m_oce['FPR']:.2f}, "
                    f"Acc: {m_oce['Accuracy']:.3f}  ({oce_time:.1f}s)"
                )

                pbar.update(1)

    pbar.close()

    # =================================================================
    # SAVE
    # =================================================================

    df_results = pd.DataFrame(results)
    csv_path = "benchmarks/results/deep_koopman_attention_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # =================================================================
    # SUMMARY
    # =================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary = (
        df_results.groupby(["Experiment", "Method"])[
            ["F1", "TP", "FP", "FN", "Accuracy", "TPR", "FPR", "Precision", "Time"]
        ]
        .mean()
        .reset_index()
        .sort_values(["Experiment", "F1"], ascending=[True, False])
    )

    for exp_name in EXPERIMENTS:
        print(f"\nExperiment: {exp_name}")
        print("-" * 80)
        exp_summary = summary[summary["Experiment"] == exp_name]
        print(
            exp_summary[
                ["Method", "F1", "TP", "FP", "Accuracy", "TPR", "FPR", "Time"]
            ].to_string(index=False, float_format="%.3f")
        )

    # Reference: per-pair K-matrix Kausal from previous benchmark
    prev_csv = "benchmarks/results/deep_koopman_perpair_results.csv"
    if os.path.exists(prev_csv):
        df_prev = pd.read_csv(prev_csv)
        kausal_pp = df_prev[df_prev["Method"] == "Kausal (per-pair)"]
        if len(kausal_pp) > 0:
            print("\n--- Comparison: K-matrix per-pair Kausal (from previous run) ---")
            print(
                f"  K-matrix per-pair avg: "
                f"F1={kausal_pp['F1'].mean():.3f}, "
                f"TPR={kausal_pp['TPR'].mean():.3f}, "
                f"FPR={kausal_pp['FPR'].mean():.3f}, "
                f"Acc={kausal_pp['Accuracy'].mean():.3f}"
            )

    # =================================================================
    # PLOTTING
    # =================================================================

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for exp_name, config in EXPERIMENTS.items():
        exp_data = df_results[df_results["Experiment"] == exp_name]

        for idx, (metric, label) in enumerate(
            [("F1", "F1 Score"), ("TPR", "True Positive Rate"), ("FPR", "False Positive Rate")]
        ):
            sns.lineplot(
                data=exp_data,
                x="Parameter",
                y=metric,
                hue="Method",
                style="Method",
                markers=True,
                dashes=False,
                ax=axes[idx],
                errorbar=("ci", 68),
            )
            axes[idx].set_title(f"{exp_name}: {label}")
            axes[idx].set_xlabel(config["label"])
            axes[idx].set_ylabel(metric)

    plt.tight_layout()
    plot_path = "benchmarks/results/deep_koopman_attention_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")
