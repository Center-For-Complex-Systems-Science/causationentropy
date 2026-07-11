"""
Two-Stage Causal Discovery: Koopman Forward + Sparse Regression Pruning on Rossler.

Stage 1: Train shared per-node attention Koopman autoencoder.
         Run oCSE on latent features -> over-connected node-level adjacency.
Stage 2: Use LASSO sparse regression on the physics-informed derivative basis
         to prune false positives. The Rossler ODE is:
           dx_i = -y_i - z_i + rho * sum_j A[j,i] * (x_j - x_i)
         So residual r_i = dx_i + y_i + z_i should be sparse in coupling terms.
         LASSO finds the minimal set of coupling terms that explains r_i.

Also includes: oCSE backward pruning (CMI-based), and raw baselines.
"""

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
import sys
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from sklearn.linear_model import LassoCV, Lasso

from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx
from causationentropy.core.discovery import backward, shuffle_test
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
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
# SHARED PER-NODE ATTENTION KOOPMAN (reused from deep_koopman_attention.py)
# =============================================================================


class NodeAttentionTransition(nn.Module):
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
        tokens = z_t.view(-1, self.n_nodes, self.d_node)
        Q = self.W_Q(tokens)
        K = self.W_K(tokens)
        V = self.W_V(tokens)
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out = attn @ V
        z_next = self.proj_out(out).reshape(-1, self.n_nodes * self.d_node)
        if return_attn:
            return z_next, attn
        return z_next


class NodeEncoder(nn.Module):
    def __init__(self, state_dim=3, d_node=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, d_node),
        )

    def forward(self, x):
        return self.net(x)


class PerNodeAttentionKoopman(nn.Module):
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
                nn.Linear(d_node, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.ReLU(),
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
        with torch.no_grad():
            x = torch.tensor(data_norm, dtype=torch.float32).to(device)
            per_node = []
            for i in range(self.n_nodes):
                xi = x[:, i * self.state_per_node : (i + 1) * self.state_per_node]
                zi = self.encoders[i](xi)
                per_node.append(zi.cpu().numpy())
            return per_node


def train_shared_attention_koopman(
    traj, n_nodes, d_node=8, d_head=16, epochs=1000, lr=1e-3, batch_size=256,
    lambda_pred=1.0, lambda_lin=0.5, lambda_sparse=0.01, print_every=200,
):
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
            L_sparse = torch.mean(torch.abs(z_t))

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
# LATENT oCSE (Stage 1: discover over-connected graph)
# =============================================================================


def latent_ocse_node_adjacency(Z_nodes, n_nodes, d_node, alpha=0.05, tau_max=1):
    """Run oCSE on per-node Koopman latent features, return node-level adjacency."""
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
        src_node = int(u.split("_")[0][1:])
        dst_node = int(v.split("_")[0][1:])
        if src_node != dst_node:
            adj[src_node, dst_node] = 1

    return adj


def attention_adjacency(model, data_norm, n_nodes, threshold=0.5):
    """
    Extract the mean attention matrix from the Koopman model and threshold it.

    The attention matrix A[i,j] measures how much node j influences node i's
    next-step prediction. Diagonal (self-attention) is excluded.
    """
    with torch.no_grad():
        x = torch.tensor(data_norm, dtype=torch.float32).to(device)
        z = model.encode(x)
        _, attn = model.transition(z, return_attn=True)
        # attn shape: (T, n_nodes, n_nodes)  — attn[b, i, j] = weight of j on i
        A_mean = attn.mean(dim=0).cpu().numpy()

    # Zero out diagonal (self-attention)
    np.fill_diagonal(A_mean, 0.0)

    # Threshold: A_mean[i,j] > threshold => edge j -> i
    # But attention is row-softmax, so values depend on n_nodes.
    # Use relative threshold: fraction of uniform attention (1/n_nodes)
    uniform = 1.0 / n_nodes
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            if A_mean[i, j] > threshold * uniform:
                adj[j, i] = 1  # j -> i

    return adj, A_mean


# =============================================================================
# STAGE 2: BACKWARD PRUNING ON RAW STATE VARIABLES
# =============================================================================


def twostage_backward_pruning_raw(
    stage1_adj,
    traj_sub,
    n_nodes,
    coupling_on="x",
    alpha_backward=0.05,
    n_shuffles=200,
    information="gaussian",
    max_lag=1,
    k_means=5,
):
    """
    Backward pruning using raw state variables instead of noisy derivatives.

    For each target node i, test whether x_j(t-lag) helps predict x_i(t)
    beyond node i's own lagged state (x_i, y_i, z_i at t-lag).
    """
    rng = np.random.default_rng(42)
    coupling_coord = {"x": 0, "y": 1, "z": 2}[coupling_on]

    T_raw = traj_sub.shape[0]
    # traj_sub shape: (T, n_nodes, 3)
    data_flat = traj_sub.reshape(T_raw, n_nodes * 3)
    # columns: x0,y0,z0, x1,y1,z1, ...

    pruned_adj = np.zeros((n_nodes, n_nodes), dtype=int)

    for i in range(n_nodes):
        target_col = i * 3 + coupling_coord
        Y = data_flat[max_lag:, target_col : target_col + 1]

        # Z_init: own lagged state (all 3 coordinates)
        Z_cols = []
        for coord in range(3):
            col_idx = i * 3 + coord
            for tau in range(1, max_lag + 1):
                Z_cols.append(data_flat[max_lag - tau : T_raw - tau, col_idx])
        Z_init = np.column_stack(Z_cols)  # (T_eff, 3 * max_lag)

        # Collect lagged coupling-coordinate features for each parent
        parent_cols = []
        parent_source = []  # maps index in parent_cols -> source node
        for j in range(n_nodes):
            if j == i:
                continue
            if stage1_adj[j, i] == 0:
                continue
            src_col = j * 3 + coupling_coord
            for tau in range(1, max_lag + 1):
                parent_cols.append(data_flat[max_lag - tau : T_raw - tau, src_col])
                parent_source.append(j)

        if len(parent_cols) == 0:
            continue

        coupling_features = np.column_stack(parent_cols)
        X_combined = np.column_stack([Z_init, coupling_features])

        n_zinit = Z_init.shape[1]
        S_combined = list(range(n_zinit, n_zinit + len(parent_cols)))

        S_survived = backward(
            X_combined,
            Y,
            S_combined,
            rng,
            alpha=alpha_backward,
            n_shuffles=n_shuffles,
            information=information,
            k_means=k_means,
        )

        for s_idx in S_survived:
            src_node = parent_source[s_idx - n_zinit]
            pruned_adj[src_node, i] = 1

    return pruned_adj


# =============================================================================
# STAGE 2: BACKWARD PRUNING ON DERIVATIVE BASIS
# =============================================================================


def twostage_backward_pruning(
    stage1_adj,
    traj_sub,
    dt_eff,
    n_nodes,
    coupling_on="x",
    alpha_backward=0.05,
    n_shuffles=200,
    information="gaussian",
    max_lag=1,
    k_means=5,
    conditioning="target_only",
):
    """
    Given a Stage 1 node-level adjacency matrix (over-connected), run oCSE
    backward elimination on the derivative-basis data to prune false positives.

    conditioning modes:
      - "target_only": condition on lagged target derivative (dx_i) only
      - "target_full": condition on all 3 lagged derivatives of target node i
                       (dx_i, dy_i, dz_i) — richer own-state information
      - "all_self":    condition on ALL 3n lagged self-derivative terms of all
                       nodes — screens out shared dynamical state
    """
    rng = np.random.default_rng(42)

    # Build derivative basis
    X_basis, meta, var_names = prepare_rossler_data_for_causal_discovery(
        traj_sub, dt_eff, coupling_on=coupling_on,
    )

    n = meta["n"]
    basis_map = meta["basis_map"]
    pred_offset = meta["pred_offset"]  # = 3 * n, where coupling terms start

    T_basis, p = X_basis.shape

    # Build lagged predictor matrix (same as discover_network does internally)
    X_lagged = []
    feature_names = []
    for j in range(p):
        for tau in range(1, max_lag + 1):
            col = X_basis[max_lag - tau : T_basis - tau, j]
            X_lagged.append(col)
            feature_names.append((j, tau))

    X_lagged = np.column_stack(X_lagged)  # (T_basis - max_lag, p * max_lag)
    Y_all = X_basis[max_lag:, :]  # aligned targets

    # Map coupling terms: basis_map[k] = (target_node, source_node)
    # coupling term k is at column index (pred_offset + k) in X_basis
    # In X_lagged, that column at lag tau is at index (pred_offset + k) * max_lag + (tau - 1)
    coupling_var_to_lagged_indices = {}
    for k, (tgt, src) in enumerate(basis_map):
        var_idx = pred_offset + k
        for tau in range(1, max_lag + 1):
            lagged_idx = var_idx * max_lag + (tau - 1)
            coupling_var_to_lagged_indices.setdefault((tgt, src), []).append(lagged_idx)

    # For the "standard" method, the initial conditioning set is lagged target values.
    # We also need the lagged derivative indices for the target node's own derivatives.
    # Target dx_i is at column i in X_basis; dy_i at n+i; dz_i at 2n+i.

    pruned_adj = np.zeros((n_nodes, n_nodes), dtype=int)

    # We only need to check the coupling coordinate targets (e.g., dx_i)
    target_var_start = {"x": 0, "y": n, "z": 2 * n}[coupling_on]

    for i in range(n_nodes):
        target_col = target_var_start + i
        Y = Y_all[:, [target_col]]

        # Collect S_init: all coupling term indices for parents of node i
        S_init = []
        parent_source_map = {}  # lagged_idx -> source_node

        for j in range(n_nodes):
            if j == i:
                continue
            if stage1_adj[j, i] == 0:
                continue  # not a Stage 1 parent

            # Find coupling term c_{i<-j}
            lagged_indices = coupling_var_to_lagged_indices.get((i, j), [])
            for lidx in lagged_indices:
                S_init.append(lidx)
                parent_source_map[lidx] = j

        if len(S_init) == 0:
            continue  # no parents to prune

        # Build conditioning columns Z_init based on strategy
        Z_init_cols = []

        if conditioning == "target_only":
            # Original: just lagged target derivative (e.g., dx_i)
            for tau in range(1, max_lag + 1):
                Z_init_cols.append(X_basis[max_lag - tau : T_basis - tau, target_col])

        elif conditioning == "target_full":
            # All 3 lagged derivatives of target node i: dx_i, dy_i, dz_i
            for coord_offset in [0, n, 2 * n]:
                col_idx = coord_offset + i
                for tau in range(1, max_lag + 1):
                    Z_init_cols.append(X_basis[max_lag - tau : T_basis - tau, col_idx])

        elif conditioning == "all_self":
            # All 3n lagged self-derivative terms for every node
            for node_k in range(n):
                for coord_offset in [0, n, 2 * n]:
                    col_idx = coord_offset + node_k
                    for tau in range(1, max_lag + 1):
                        Z_init_cols.append(X_basis[max_lag - tau : T_basis - tau, col_idx])

        else:
            raise ValueError(f"Unknown conditioning mode: {conditioning}")

        Z_init = np.column_stack(Z_init_cols)

        # Build combined predictor matrix: [Z_init | coupling_terms_for_parents]
        # backward() will test each index in S_init, conditioning on all others + Z_init
        coupling_cols = X_lagged[:, S_init]  # (T_eff, len(S_init))
        X_combined = np.column_stack([Z_init, coupling_cols])

        # S_init for backward() now indexes into X_combined
        # Indices 0..max_lag-1 are the Z_init columns (always conditioned on)
        # Indices max_lag..max_lag+len(S_init)-1 are the coupling terms to test
        n_zinit = Z_init.shape[1]
        S_combined = list(range(n_zinit, n_zinit + len(S_init)))

        # Run backward elimination
        S_survived = backward(
            X_combined,
            Y,
            S_combined,
            rng,
            alpha=alpha_backward,
            n_shuffles=n_shuffles,
            information=information,
            k_means=k_means,
        )

        # Map survived indices back to source nodes
        for s_idx in S_survived:
            orig_lagged_idx = S_init[s_idx - n_zinit]
            src_node = parent_source_map[orig_lagged_idx]
            pruned_adj[src_node, i] = 1

    return pruned_adj


# =============================================================================
# STAGE 2: LASSO SPARSE REGRESSION ON DERIVATIVE BASIS
# =============================================================================


# =============================================================================
# PHASE REDUCTION: EXTRACT PHASE FROM ROSSLER AND BUILD KURAMOTO-LIKE BASIS
# =============================================================================


def extract_rossler_phase(traj):
    """
    Extract instantaneous phase from Rossler oscillators.

    Uses arctan2(y, x) on the x-y projection of each oscillator's attractor,
    then unwraps to get continuous phase evolution.
    """
    T, n, d = traj.shape
    x = traj[:, :, 0]
    y = traj[:, :, 1]

    theta = np.arctan2(y, x)  # (T, n)
    theta_unwrapped = np.unwrap(theta, axis=0)  # (T, n)
    return theta_unwrapped


def prepare_rossler_phase_basis(traj, dt):
    """
    Build Kuramoto-like phase basis for Rossler oscillators.

    After extracting phases, builds:
      - Targets: phase velocity omega_i(t+1)
      - Predictors: sin(theta_j(t) - theta_i(t)) for all pairs

    This mirrors prepare_kuramoto_data_for_causal_discovery() exactly.
    """
    T, n, d = traj.shape
    theta = extract_rossler_phase(traj)  # (T, n)

    # Phase velocity via finite differences
    omega = (theta[1:] - theta[:-1]) / dt  # (T-1, n)
    th = theta[:-1]  # (T-1, n)

    # Build coupling basis: sin(theta_j - theta_i)
    S = []
    basis_map = []
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            S.append(np.sin(th[:, j] - th[:, i]))
            basis_map.append((i, j))
    S = np.stack(S, axis=1)  # (T-1, n*(n-1))

    # Shift: predictors at t, targets at t+1
    omega_next = omega[1:]   # (T-2, n)
    S_prev = S[:-1]          # (T-2, n*(n-1))

    X = np.concatenate([omega_next, S_prev], axis=1)

    var_names = [f"w{i}" for i in range(n)] + [
        f"s_{i}<-{j}" for (i, j) in basis_map
    ]
    meta = {
        "n": n, "dt": dt, "pred_offset": n,
        "basis_map": basis_map, "p": X.shape[1],
    }
    return X, meta, var_names


def phase_lasso_pruning(traj_sub, dt_eff, n_nodes, alpha_lasso=None, stage1_adj=None):
    """
    LASSO on the phase-reduced Rossler basis.

    Regresses omega_i on sin(theta_j - theta_i) for candidate parents.
    Non-zero coefficients => edges.
    """
    theta = extract_rossler_phase(traj_sub)  # (T, n)
    omega = (theta[1:] - theta[:-1]) / dt_eff  # (T-1, n)
    th = theta[:-1]

    # Use omega at t+1, sin at t (like prepare_rossler_phase_basis)
    omega_target = omega[1:]  # (T-2, n)
    th_pred = th[:-1]         # (T-2, n)

    adj = np.zeros((n_nodes, n_nodes), dtype=int)

    for i in range(n_nodes):
        y = omega_target[:, i]

        candidates = []
        candidate_nodes = []
        for j in range(n_nodes):
            if j == i:
                continue
            if stage1_adj is not None and stage1_adj[j, i] == 0:
                continue
            candidates.append(np.sin(th_pred[:, j] - th_pred[:, i]))
            candidate_nodes.append(j)

        if len(candidates) == 0:
            continue

        X_coupling = np.column_stack(candidates)

        if alpha_lasso is None:
            model = LassoCV(cv=5, max_iter=10000, n_jobs=-1)
            model.fit(X_coupling, y)
        else:
            model = Lasso(alpha=alpha_lasso, max_iter=10000)
            model.fit(X_coupling, y)

        for k, j in enumerate(candidate_nodes):
            if abs(model.coef_[k]) > 1e-10:
                adj[j, i] = 1

    return adj


def twostage_phase_backward_pruning(
    stage1_adj, traj_sub, dt_eff, n_nodes,
    alpha_backward=0.05, n_shuffles=200,
    information="gaussian", max_lag=1, k_means=5,
):
    """
    Backward pruning using phase-reduced Kuramoto-like basis.

    For each target node i, tests whether sin(theta_j - theta_i)(t-1)
    helps predict omega_i(t) beyond omega_i(t-1), for Stage 1 parents.
    """
    rng = np.random.default_rng(42)

    X_basis, meta, var_names = prepare_rossler_phase_basis(traj_sub, dt_eff)
    n = meta["n"]
    basis_map = meta["basis_map"]
    pred_offset = meta["pred_offset"]  # = n

    T_basis, p = X_basis.shape

    # Build lagged predictor matrix
    X_lagged = []
    for j in range(p):
        for tau in range(1, max_lag + 1):
            col = X_basis[max_lag - tau : T_basis - tau, j]
            X_lagged.append(col)
    X_lagged = np.column_stack(X_lagged)
    Y_all = X_basis[max_lag:, :]

    # Map coupling terms to lagged indices
    coupling_var_to_lagged_indices = {}
    for k, (tgt, src) in enumerate(basis_map):
        var_idx = pred_offset + k
        for tau in range(1, max_lag + 1):
            lagged_idx = var_idx * max_lag + (tau - 1)
            coupling_var_to_lagged_indices.setdefault((tgt, src), []).append(lagged_idx)

    pruned_adj = np.zeros((n_nodes, n_nodes), dtype=int)

    for i in range(n_nodes):
        target_col = i  # omega_i
        Y = Y_all[:, [target_col]]

        S_init = []
        parent_source_map = {}

        for j in range(n_nodes):
            if j == i:
                continue
            if stage1_adj[j, i] == 0:
                continue
            lagged_indices = coupling_var_to_lagged_indices.get((i, j), [])
            for lidx in lagged_indices:
                S_init.append(lidx)
                parent_source_map[lidx] = j

        if len(S_init) == 0:
            continue

        # Conditioning: lagged omega_i
        Z_init_cols = []
        for tau in range(1, max_lag + 1):
            Z_init_cols.append(X_basis[max_lag - tau : T_basis - tau, target_col])
        Z_init = np.column_stack(Z_init_cols)

        # Combine Z_init and coupling terms
        coupling_data = X_lagged[:, S_init]
        X_combined = np.column_stack([Z_init, coupling_data])
        Y_aligned = Y

        n_zinit = Z_init.shape[1]
        S_combined = list(range(n_zinit, n_zinit + len(S_init)))

        S_survived = backward(
            X_combined, Y_aligned, S_combined, rng,
            alpha=alpha_backward, n_shuffles=n_shuffles,
            information=information, k_means=k_means,
        )

        for s_idx in S_survived:
            orig_lagged_idx = S_init[s_idx - n_zinit]
            src_node = parent_source_map[orig_lagged_idx]
            pruned_adj[src_node, i] = 1

    return pruned_adj


def lasso_pruning(
    traj_sub,
    dt_eff,
    n_nodes,
    coupling_on="x",
    alpha_lasso=None,
    a=0.2,
    b=0.2,
    c=5.7,
    stage1_adj=None,
):
    """
    Sparse regression on the physics-informed derivative basis.

    The Rossler ODE for the coupled coordinate (default x) is:
        dx_i = -y_i - z_i + rho * sum_j A[j,i] * (x_j - x_i)

    We compute the residual r_i = dx_i + y_i + z_i, which should equal
    rho * sum_j A[j,i] * (x_j - x_i). Then we regress r_i on the coupling
    terms {x_j - x_i} using LASSO. Non-zero coefficients => edges.

    If stage1_adj is provided, only consider coupling terms for Stage 1 parents.
    If None, consider all possible coupling terms (standalone LASSO).

    If alpha_lasso is None, uses LassoCV to auto-select.
    """
    T, n, d = traj_sub.shape
    assert d == 3

    # Finite differences for derivatives
    dtraj = (traj_sub[1:] - traj_sub[:-1]) / dt_eff  # (T-1, n, 3)

    # Raw state at time t (aligned with derivatives)
    x_state = traj_sub[:-1, :, 0]  # (T-1, n)
    y_state = traj_sub[:-1, :, 1]
    z_state = traj_sub[:-1, :, 2]

    dx = dtraj[:, :, 0]  # (T-1, n)

    coupling_coord = {"x": 0, "y": 1, "z": 2}[coupling_on]
    s = traj_sub[:-1, :, coupling_coord]  # (T-1, n)

    adj = np.zeros((n_nodes, n_nodes), dtype=int)

    for i in range(n_nodes):
        # Compute residual: r_i = dx_i - (-y_i - z_i) = dx_i + y_i + z_i
        r_i = dx[:, i] + y_state[:, i] + z_state[:, i]

        # Build coupling features: (x_j - x_i) for each candidate j
        candidates = []
        candidate_nodes = []
        for j in range(n_nodes):
            if j == i:
                continue
            if stage1_adj is not None and stage1_adj[j, i] == 0:
                continue
            candidates.append(s[:, j] - s[:, i])
            candidate_nodes.append(j)

        if len(candidates) == 0:
            continue

        X_coupling = np.column_stack(candidates)

        # LASSO regression
        if alpha_lasso is None:
            model = LassoCV(cv=5, max_iter=10000, n_jobs=-1)
            model.fit(X_coupling, r_i)
        else:
            model = Lasso(alpha=alpha_lasso, max_iter=10000)
            model.fit(X_coupling, r_i)

        # Non-zero coefficients => edges
        for k, j in enumerate(candidate_nodes):
            if abs(model.coef_[k]) > 1e-10:
                adj[j, i] = 1

    return adj


# =============================================================================
# BASELINE HELPERS (from deep_koopman_attention.py)
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
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "TPR": tpr, "FPR": fpr, "Precision": precision,
        "F1": f1, "Accuracy": accuracy, "Time": time_taken,
    }


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
    # Rossler
    "a": 0.2, "b": 0.2, "c": 5.7,
    "rho_default": 0.4,
    "noise_std": 0.0,
    "init_scale": 1.0,
    "coupling_on": "x",
    "normalize_by_indegree": False,
    # oCSE / PCMCI
    "alpha": 0.05,
    "tau_max": 1,
    # Shared attention Koopman
    "d_node": 5,
    "d_head": 16,
    "epochs": 1000,
    "batch_size": 256,
    "lambda_pred": 1.0,
    "lambda_lin": 0.5,
    "lambda_sparse": 0.01,
    # Backward pruning
    "n_shuffles": 200,
    "alpha_backward_values": [0.05],
    # LASSO
    "lasso_alphas": [None, 0.001, 0.01, 0.1],
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
    d_node = GLOBAL_PARAMS["d_node"]

    print("=" * 80)
    print("Two-Stage Causal Discovery: Koopman Forward + oCSE Backward Pruning")
    print(f"Device: {device}")
    print(f"{n_nodes} nodes, d_node={d_node}")
    print(f"Backward alpha values: {GLOBAL_PARAMS['alpha_backward_values']}")
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
                    config["type"], n_nodes, current_params, seed,
                )
                true_adj = nx.to_numpy_array(G).astype(int)

                traj, _ = simulate_rossler(
                    G=G,
                    T=GLOBAL_PARAMS["T"],
                    dt=GLOBAL_PARAMS["dt"],
                    rho=current_params.get("rho", GLOBAL_PARAMS["rho_default"]),
                    seed=seed,
                    a=GLOBAL_PARAMS["a"], b=GLOBAL_PARAMS["b"], c=GLOBAL_PARAMS["c"],
                    init_scale=GLOBAL_PARAMS["init_scale"],
                    noise_std=GLOBAL_PARAMS["noise_std"],
                    burn_in=GLOBAL_PARAMS["burn_in"],
                    normalize_by_indegree=GLOBAL_PARAMS["normalize_by_indegree"],
                    coupling_on=GLOBAL_PARAMS["coupling_on"],
                )

                k = GLOBAL_PARAMS["subsample"]
                traj_sub = traj[::k] if k > 1 else traj
                dt_eff = GLOBAL_PARAMS["dt"] * k

                n_true = int(true_adj.sum())
                n_possible = n_nodes * (n_nodes - 1)
                print(f"  True edges: {n_true}/{n_possible}")

                # ==========================================================
                # STAGE 1: Train Koopman + oCSE on latent features
                # ==========================================================
                print(f"  Stage 1: Training shared attention Koopman...")
                stage1_start = time.time()

                model, data_mean, data_std = train_shared_attention_koopman(
                    traj_sub,
                    n_nodes=n_nodes,
                    d_node=GLOBAL_PARAMS["d_node"],
                    d_head=GLOBAL_PARAMS["d_head"],
                    epochs=GLOBAL_PARAMS["epochs"],
                    lr=1e-3,
                    batch_size=GLOBAL_PARAMS["batch_size"],
                    lambda_pred=GLOBAL_PARAMS["lambda_pred"],
                    lambda_lin=GLOBAL_PARAMS["lambda_lin"],
                    lambda_sparse=GLOBAL_PARAMS["lambda_sparse"],
                )

                T_sub = traj_sub.shape[0]
                data_flat = traj_sub.reshape(T_sub, n_nodes * 3)
                data_norm = (data_flat - data_mean) / data_std
                Z_nodes = model.encode_per_node(data_norm)

                koopman_time = time.time() - stage1_start

                print(f"  Stage 1: Running oCSE on latent features...")
                ocse_start = time.time()

                stage1_adj = latent_ocse_node_adjacency(
                    Z_nodes, n_nodes, d_node,
                    alpha=GLOBAL_PARAMS["alpha"],
                    tau_max=GLOBAL_PARAMS["tau_max"],
                )

                stage1_time = time.time() - stage1_start

                m_s1 = compute_metrics(stage1_adj, true_adj, stage1_time)
                m_s1.update({
                    "Experiment": exp_name, "Parameter": val,
                    "Method": "Stage 1: Koopman + oCSE",
                    "Trial": trial,
                })
                results.append(m_s1)
                print(
                    f"  Stage 1    -- F1: {m_s1['F1']:.3f}, "
                    f"TP: {m_s1['TP']}, FP: {m_s1['FP']}, "
                    f"TPR: {m_s1['TPR']:.2f}, FPR: {m_s1['FPR']:.2f}"
                )

                # ==========================================================
                # STAGE 2: Backward pruning on derivative basis (best alpha)
                # ==========================================================
                for alpha_bw in GLOBAL_PARAMS["alpha_backward_values"]:
                    print(f"  Stage 2 deriv: Backward (alpha={alpha_bw})...")
                    stage2_start = time.time()

                    pruned_adj = twostage_backward_pruning(
                        stage1_adj,
                        traj_sub,
                        dt_eff,
                        n_nodes,
                        coupling_on=GLOBAL_PARAMS["coupling_on"],
                        alpha_backward=alpha_bw,
                        n_shuffles=GLOBAL_PARAMS["n_shuffles"],
                        information="gaussian",
                        max_lag=GLOBAL_PARAMS["tau_max"],
                    )

                    stage2_time = time.time() - stage2_start
                    total_time = stage1_time + stage2_time

                    m_s2 = compute_metrics(pruned_adj, true_adj, total_time)
                    m_s2.update({
                        "Experiment": exp_name, "Parameter": val,
                        "Method": f"Deriv basis (α={alpha_bw})",
                        "Trial": trial,
                    })
                    results.append(m_s2)
                    print(
                        f"  Deriv      -- F1: {m_s2['F1']:.3f}, "
                        f"TP: {m_s2['TP']}, FP: {m_s2['FP']}, FN: {m_s2['FN']}, "
                        f"TPR: {m_s2['TPR']:.2f}, FPR: {m_s2['FPR']:.2f}  "
                        f"(prune: {stage2_time:.1f}s)"
                    )

                # ==========================================================
                # PHASE-REDUCED METHODS (Kuramoto-like basis on Rossler)
                # ==========================================================

                # Phase LASSO: standalone (all pairs)
                for alpha_l in GLOBAL_PARAMS.get("lasso_alphas", [None, 0.01, 0.1]):
                    label = f"α={alpha_l}" if alpha_l is not None else "CV"
                    print(f"  Phase LASSO standalone ({label})...")
                    start_time = time.time()

                    phase_lasso_adj = phase_lasso_pruning(
                        traj_sub, dt_eff, n_nodes,
                        alpha_lasso=alpha_l, stage1_adj=None,
                    )
                    phase_lasso_time = time.time() - start_time

                    m_pl = compute_metrics(phase_lasso_adj, true_adj, phase_lasso_time)
                    m_pl.update({
                        "Experiment": exp_name, "Parameter": val,
                        "Method": f"Phase LASSO ({label})",
                        "Trial": trial,
                    })
                    results.append(m_pl)
                    print(
                        f"  PhaseLASSO -- F1: {m_pl['F1']:.3f}, "
                        f"TP: {m_pl['TP']}, FP: {m_pl['FP']}, FN: {m_pl['FN']}, "
                        f"TPR: {m_pl['TPR']:.2f}, FPR: {m_pl['FPR']:.2f}  "
                        f"({phase_lasso_time:.1f}s)"
                    )

                # Phase LASSO: two-stage (Stage 1 filtered)
                for alpha_l in GLOBAL_PARAMS.get("lasso_alphas", [None, 0.01, 0.1]):
                    label = f"α={alpha_l}" if alpha_l is not None else "CV"
                    print(f"  Koopman + Phase LASSO ({label})...")
                    start_time = time.time()

                    phase_lasso_adj2 = phase_lasso_pruning(
                        traj_sub, dt_eff, n_nodes,
                        alpha_lasso=alpha_l, stage1_adj=stage1_adj,
                    )
                    phase_lasso_time2 = time.time() - start_time

                    m_pl2 = compute_metrics(phase_lasso_adj2, true_adj, stage1_time + phase_lasso_time2)
                    m_pl2.update({
                        "Experiment": exp_name, "Parameter": val,
                        "Method": f"Koopman+PhaseLASSO ({label})",
                        "Trial": trial,
                    })
                    results.append(m_pl2)
                    print(
                        f"  K+PhLASSO  -- F1: {m_pl2['F1']:.3f}, "
                        f"TP: {m_pl2['TP']}, FP: {m_pl2['FP']}, FN: {m_pl2['FN']}, "
                        f"TPR: {m_pl2['TPR']:.2f}, FPR: {m_pl2['FPR']:.2f}  "
                        f"({phase_lasso_time2:.1f}s)"
                    )

                # Phase backward pruning: two-stage (Stage 1 -> phase backward)
                for alpha_bw in GLOBAL_PARAMS["alpha_backward_values"]:
                    print(f"  Koopman + Phase backward (α={alpha_bw})...")
                    stage2_start = time.time()

                    phase_bkwd_adj = twostage_phase_backward_pruning(
                        stage1_adj, traj_sub, dt_eff, n_nodes,
                        alpha_backward=alpha_bw,
                        n_shuffles=GLOBAL_PARAMS["n_shuffles"],
                        information="gaussian",
                        max_lag=GLOBAL_PARAMS["tau_max"],
                    )
                    stage2_time = time.time() - stage2_start

                    m_pb = compute_metrics(phase_bkwd_adj, true_adj, stage1_time + stage2_time)
                    m_pb.update({
                        "Experiment": exp_name, "Parameter": val,
                        "Method": f"Koopman+PhaseBkwd (α={alpha_bw})",
                        "Trial": trial,
                    })
                    results.append(m_pb)
                    print(
                        f"  K+PhBkwd   -- F1: {m_pb['F1']:.3f}, "
                        f"TP: {m_pb['TP']}, FP: {m_pb['FP']}, FN: {m_pb['FN']}, "
                        f"TPR: {m_pb['TPR']:.2f}, FPR: {m_pb['FPR']:.2f}  "
                        f"(prune: {stage2_time:.1f}s)"
                    )

                # Phase oCSE: standalone (full forward+backward on phase basis)
                print("  Phase oCSE standalone...")
                start_time = time.time()

                X_phase, phase_meta, phase_var_names = prepare_rossler_phase_basis(
                    traj_sub, dt_eff,
                )
                X_phase_df = pd.DataFrame(X_phase, columns=phase_var_names)
                with suppress_stdout():
                    phase_oce_network = discover_network(
                        data=X_phase_df,
                        max_lag=GLOBAL_PARAMS["tau_max"],
                        method="standard",
                        information="gaussian",
                        alpha_forward=GLOBAL_PARAMS["alpha"],
                        alpha_backward=GLOBAL_PARAMS["alpha"],
                    )

                # Manually extract edges from phase oCSE (can't use extract_node_graph_oce
                # because var_names use "w0" instead of "dx0")
                phase_oce_adj = np.zeros((n_nodes, n_nodes), dtype=int)
                for edge in phase_oce_network.edges():
                    u, v = edge[0], edge[1]
                    # coupling vars are s_i<-j, target vars are w_i
                    if u.startswith("s_") and v.startswith("w"):
                        # parse s_i<-j and w_i
                        parts = u.split("<-")
                        tgt_node = int(parts[0].split("_")[1])
                        src_node = int(parts[1])
                        dst_node = int(v[1:])
                        if tgt_node == dst_node:
                            phase_oce_adj[src_node, dst_node] = 1

                phase_oce_time = time.time() - start_time

                m_poce = compute_metrics(phase_oce_adj, true_adj, phase_oce_time)
                m_poce.update({
                    "Experiment": exp_name, "Parameter": val,
                    "Method": "Phase oCSE (raw)",
                    "Trial": trial,
                })
                results.append(m_poce)
                print(
                    f"  PhaseOCSE  -- F1: {m_poce['F1']:.3f}, "
                    f"TP: {m_poce['TP']}, FP: {m_poce['FP']}, "
                    f"TPR: {m_poce['TPR']:.2f}, FPR: {m_poce['FPR']:.2f}  "
                    f"({phase_oce_time:.1f}s)"
                )

                # ==========================================================
                # BASELINE: oCSE (raw) on derivative basis
                # ==========================================================
                print("  Baseline: oCSE (raw) on derivative basis...")
                start_time = time.time()

                X_basis, basis_meta, var_names = (
                    prepare_rossler_data_for_causal_discovery(
                        traj_sub, dt_eff,
                        coupling_on=GLOBAL_PARAMS["coupling_on"],
                    )
                )
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
                    oce_network, n_nodes,
                    basis_meta["basis_map"], var_names,
                    coupling_on=GLOBAL_PARAMS["coupling_on"],
                )
                pred_adj_oce = collapse_to_binary_adjacency(oce_node_graph, n_nodes)
                oce_time = time.time() - start_time

                m_oce = compute_metrics(pred_adj_oce, true_adj, oce_time)
                m_oce.update({
                    "Experiment": exp_name, "Parameter": val,
                    "Method": "oCSE (raw)",
                    "Trial": trial,
                })
                results.append(m_oce)
                print(
                    f"  oCSE raw   -- F1: {m_oce['F1']:.3f}, "
                    f"TP: {m_oce['TP']}, FP: {m_oce['FP']}, "
                    f"TPR: {m_oce['TPR']:.2f}, FPR: {m_oce['FPR']:.2f}  "
                    f"({oce_time:.1f}s)"
                )

                # ==========================================================
                # BASELINE: PCMCI (raw) on derivative basis
                # ==========================================================
                print("  Baseline: PCMCI (raw) on derivative basis...")
                start_time = time.time()

                T_eff = X_basis.shape[0]
                dataframe = pp.DataFrame(
                    X_basis,
                    datatime={0: np.arange(T_eff)},
                    var_names=var_names,
                )
                pcmci_obj = PCMCI(
                    dataframe=dataframe,
                    cond_ind_test=ParCorr(),
                    verbosity=0,
                )
                pcmci_res = pcmci_obj.run_pcmci(
                    tau_min=1,
                    tau_max=GLOBAL_PARAMS["tau_max"],
                    pc_alpha=GLOBAL_PARAMS["alpha"],
                )
                graph_nx = pcmci_to_networkx(pcmci_res)
                pcmci_node_graph = extract_node_graph_pcmci(
                    graph_nx, basis_meta["n"],
                    basis_meta["basis_map"],
                    coupling_on=GLOBAL_PARAMS["coupling_on"],
                )
                pred_adj_pcmci = collapse_to_binary_adjacency(
                    pcmci_node_graph, n_nodes
                )
                pcmci_time = time.time() - start_time

                m_pcmci = compute_metrics(pred_adj_pcmci, true_adj, pcmci_time)
                m_pcmci.update({
                    "Experiment": exp_name, "Parameter": val,
                    "Method": "PCMCI (raw)",
                    "Trial": trial,
                })
                results.append(m_pcmci)
                print(
                    f"  PCMCI raw  -- F1: {m_pcmci['F1']:.3f}, "
                    f"TP: {m_pcmci['TP']}, FP: {m_pcmci['FP']}, "
                    f"TPR: {m_pcmci['TPR']:.2f}, FPR: {m_pcmci['FPR']:.2f}  "
                    f"({pcmci_time:.1f}s)"
                )

                pbar.update(1)

    pbar.close()

    # =================================================================
    # SAVE
    # =================================================================

    df_results = pd.DataFrame(results)
    csv_path = "benchmarks/results/deep_koopman_twostage_results.csv"
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
                ["Method", "F1", "TP", "FP", "FN", "Accuracy", "TPR", "FPR", "Precision", "Time"]
            ].to_string(index=False, float_format="%.3f")
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
    plot_path = "benchmarks/results/deep_koopman_twostage_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")
