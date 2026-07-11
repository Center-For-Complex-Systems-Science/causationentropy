"""
Per-Node Deep Koopman with KAN Autoencoders for Causal Discovery on Rössler Oscillators.

Uses the user's DeepKoopmanCT architecture (continuous-time Koopman with
generator matrix A and expm propagation) with KAN-based per-node encoders/decoders.

Each node gets its own KANEncoder (R^3 → R^d_node) and KANDecoder (R^d_node → R^3).
The generator matrix A is (n_nodes * d_node) × (n_nodes * d_node), and block norms
||A[i_block, j_block]||_F give causal influence from node j to node i.

Methods compared:
  1. KAN-Koopman A-matrix:  Block norm thresholding on generator matrix
  2. KAN-Koopman + PCMCI:   PCMCI on KAN latent features
  3. KAN-Koopman + oCSE:    oCSE on KAN latent features
  4. Linear Granger:        Pairwise DMD in raw state space (no MLP)
  5. PCMCI (raw):           Tigramite PCMCI on derivative basis
  6. oCSE (raw):            oCSE on derivative basis
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
from kan import KANLayer

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
# KAN ENCODER / DECODER (from user's model)
# =============================================================================


class KANEncoder(nn.Module):
    """KAN-based encoder: KANLayer -> ReLU -> Dense."""

    def __init__(
        self,
        input_size,
        hidden_size,
        bottleneck_size,
        grid_size=5,
        spline_order=3,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.kan = KANLayer(
            in_dim=input_size,
            out_dim=hidden_size,
            num=grid_size,
            k=spline_order,
            grid_range=grid_range,
        )
        self.relu = nn.ReLU()
        self.dense = nn.Linear(hidden_size, bottleneck_size)

    def forward(self, x):
        x = self.kan(x)[0]  # KANLayer returns (output, preacts, postacts, postsplines)
        x = self.relu(x)
        x = self.dense(x)
        return x


class KANDecoder(nn.Module):
    """KAN-based decoder: Dense -> ReLU -> KANLayer."""

    def __init__(
        self,
        bottleneck_size,
        hidden_size,
        output_size,
        grid_size=5,
        spline_order=3,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.dense = nn.Linear(bottleneck_size, hidden_size)
        self.relu = nn.ReLU()
        self.kan = KANLayer(
            in_dim=hidden_size,
            out_dim=output_size,
            num=grid_size,
            k=spline_order,
            grid_range=grid_range,
        )

    def forward(self, x):
        x = self.dense(x)
        x = self.relu(x)
        x = self.kan(x)[0]  # KANLayer returns tuple
        return x


class Knet(nn.Module):
    """Bias-free linear layer approximating the Koopman generator matrix."""

    def __init__(self, size):
        super().__init__()
        self.net = nn.Linear(size, size, bias=False)

    def forward(self, X):
        return self.net(X)


# =============================================================================
# PER-NODE KAN KOOPMAN MODEL (continuous-time with expm propagation)
# =============================================================================


class PerNodeKANKoopman(nn.Module):
    """Per-node KAN encoder/decoder with continuous-time Koopman generator.

    Each node gets its own KANEncoder (R^3 -> R^d_node) and KANDecoder.
    The Knet generator matrix A is (n_nodes * d_node) x (n_nodes * d_node).
    Propagation: z_{t+1} = z_t @ expm(A * dt)^T.

    Block norms ||A[i_block, j_block]||_F give causal influence j -> i.
    """

    def __init__(self, n_nodes, d_node, dt, state_per_node=3, hidden_size=32,
                 grid_size=5, spline_order=3):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_node = d_node
        self.state_per_node = state_per_node
        self.latent_dim = n_nodes * d_node
        self.dt = dt

        kan_kwargs = dict(
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=[-3, 3],  # wider for normalized data
        )

        self.encoders = nn.ModuleList([
            KANEncoder(state_per_node, hidden_size, d_node, **kan_kwargs)
            for _ in range(n_nodes)
        ])

        self.K = Knet(self.latent_dim)

        self.decoders = nn.ModuleList([
            KANDecoder(d_node, hidden_size, state_per_node, **kan_kwargs)
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

    def propagate(self, z_t):
        A = self.K.net.weight
        M = torch.matrix_exp(A * self.dt)
        return z_t @ M.T

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def predict(self, x):
        z = self.encode(x)
        z_next = self.propagate(z)
        x_next_pred = self.decode(z_next)
        return x_next_pred, z_next

    def encode_per_node(self, data_norm):
        """Encode trajectory and return list of per-node latent arrays."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(data_norm, dtype=torch.float32).to(device)
            per_node = []
            for i in range(self.n_nodes):
                xi = x[:, i * self.state_per_node : (i + 1) * self.state_per_node]
                zi = self.encoders[i](xi)
                per_node.append(zi.cpu().numpy())
            return per_node  # list of (T, d_node) arrays

    def get_generator_adjacency(self, threshold_std=1.0):
        """Extract binary adjacency from generator matrix A block norms.

        Block A[i_block, j_block] represents how node j's latent features
        affect node i's dynamics. High block norm -> j causes i -> adj[j,i] = 1.
        """
        A = self.K.net.weight.detach().cpu().numpy()  # (latent_dim, latent_dim)
        n = self.n_nodes
        d = self.d_node

        block_norms = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                block = A[i * d : (i + 1) * d, j * d : (j + 1) * d]
                block_norms[i, j] = np.linalg.norm(block, "fro")

        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag = block_norms[off_diag_mask]
        threshold = off_diag.mean() + threshold_std * off_diag.std()

        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j and block_norms[i, j] > threshold:
                    adj[j, i] = 1  # block[i,j] high -> j causes i
        return adj, block_norms


# =============================================================================
# TRAINING
# =============================================================================


def train_kan_koopman(
    traj, n_nodes, d_node=5, dt_eff=0.1, hidden_size=32,
    grid_size=5, spline_order=3,
    epochs=1000, lr=1e-3, batch_size=256,
    lambda_pred=1.0, lambda_lin=0.5, lambda_sparse=0.01,
    print_every=200,
):
    """Train per-node KAN Koopman model with continuous-time propagation."""
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

    model = PerNodeKANKoopman(
        n_nodes, d_node, dt=dt_eff, hidden_size=hidden_size,
        grid_size=grid_size, spline_order=spline_order,
    ).to(device)
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
# LATENT-SPACE CAUSAL DISCOVERY (PCMCI / oCSE on KAN features)
# =============================================================================


def latent_pcmci_test(Z_nodes, n_nodes, d_node, alpha=0.05, tau_max=1):
    """Run PCMCI on per-node KAN latent features, map back to node adjacency."""
    Z_all = np.column_stack(Z_nodes)
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
    """Run oCSE on per-node KAN latent features, map back to node adjacency."""
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


# =============================================================================
# LINEAR GRANGER (DMD baseline)
# =============================================================================


def linear_granger_test(
    traj_nodes_norm, n_nodes, n_bootstrap=200, p_crit=0.05, train_frac=0.8,
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
            boot = np.array([
                delta[rng.choice(T_eff, size=T_eff, replace=True)].mean()
                for _ in range(n_bootstrap)
            ])
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
    return [traj_sub[:, i, :] for i in range(n_nodes)]


def normalize_node_trajectories(traj_nodes):
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
    # KAN Koopman
    "d_node": 5,  # = n_nodes for block-interpretable generator
    "hidden_size": 32,
    "grid_size": 5,
    "spline_order": 3,
    "epochs_kan": 1000,
    "lr": 1e-3,
    "batch_size": 256,
    "lambda_pred": 1.0,
    "lambda_lin": 0.5,
    "lambda_sparse": 0.01,
    # Linear Granger
    "n_bootstrap": 200,
    "p_crit": 0.05,
    "train_frac": 0.8,
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
    dt_eff = GLOBAL_PARAMS["dt"] * GLOBAL_PARAMS["subsample"]

    print("=" * 80)
    print("Per-Node KAN Koopman (Continuous-Time) Causal Discovery — Rössler Oscillators")
    print(f"Device: {device}")
    print(f"{n_nodes} nodes, d_node={d_node}, dt_eff={dt_eff}")
    print(f"KAN: hidden={GLOBAL_PARAMS['hidden_size']}, grid={GLOBAL_PARAMS['grid_size']}, "
          f"spline_order={GLOBAL_PARAMS['spline_order']}")
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

                # Prepare per-node normalized trajectories (for Linear Granger)
                traj_nodes = extract_node_trajectories(traj_sub, n_nodes)
                traj_nodes_norm = normalize_node_trajectories(traj_nodes)

                n_true = int(true_adj.sum())
                n_possible = n_nodes * (n_nodes - 1)
                print(f"  True edges: {n_true}/{n_possible}")

                # ===========================================================
                # TRAIN KAN KOOPMAN MODEL
                # ===========================================================
                print(f"  Training KAN Koopman (d_node={d_node}, hidden={GLOBAL_PARAMS['hidden_size']}, "
                      f"grid={GLOBAL_PARAMS['grid_size']})...")
                kan_start = time.time()

                kan_model, data_mean, data_std = train_kan_koopman(
                    traj_sub,
                    n_nodes=n_nodes,
                    d_node=d_node,
                    dt_eff=dt_eff,
                    hidden_size=GLOBAL_PARAMS["hidden_size"],
                    grid_size=GLOBAL_PARAMS["grid_size"],
                    spline_order=GLOBAL_PARAMS["spline_order"],
                    epochs=GLOBAL_PARAMS["epochs_kan"],
                    lr=GLOBAL_PARAMS["lr"],
                    batch_size=GLOBAL_PARAMS["batch_size"],
                    lambda_pred=GLOBAL_PARAMS["lambda_pred"],
                    lambda_lin=GLOBAL_PARAMS["lambda_lin"],
                    lambda_sparse=GLOBAL_PARAMS["lambda_sparse"],
                )
                kan_train_time = time.time() - kan_start
                print(f"  KAN model trained in {kan_train_time:.1f}s")

                # Normalize data for inference
                T_sub = traj_sub.shape[0]
                data_flat = traj_sub.reshape(T_sub, n_nodes * 3)
                data_norm = (data_flat - data_mean) / data_std

                # ===========================================================
                # METHOD 1: KAN-Koopman A-matrix (block norm thresholding)
                # ===========================================================
                print("  Extracting adjacency from generator matrix A...")
                start_time = time.time()

                pred_adj_gen, block_norms = kan_model.get_generator_adjacency(
                    threshold_std=1.0,
                )
                gen_time = time.time() - start_time + kan_train_time

                m_gen = compute_metrics(pred_adj_gen, true_adj, gen_time)
                m_gen.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "KAN-Koopman A-matrix",
                    "Trial": trial,
                })
                results.append(m_gen)
                print(
                    f"  KAN A-mat  — F1: {m_gen['F1']:.3f}, "
                    f"TP: {m_gen['TP']}, FP: {m_gen['FP']}, "
                    f"TPR: {m_gen['TPR']:.2f}, FPR: {m_gen['FPR']:.2f}, "
                    f"Acc: {m_gen['Accuracy']:.3f}  ({gen_time:.1f}s)"
                )

                # ===========================================================
                # METHOD 2: KAN-Koopman + PCMCI
                # ===========================================================
                print("  Running PCMCI on KAN latent features...")
                start_time = time.time()

                Z_nodes = kan_model.encode_per_node(data_norm)
                pred_adj_lpcmci = latent_pcmci_test(
                    Z_nodes, n_nodes, d_node,
                    alpha=GLOBAL_PARAMS["alpha"],
                    tau_max=GLOBAL_PARAMS["tau_max"],
                )
                lpcmci_time = time.time() - start_time + kan_train_time

                m_lpcmci = compute_metrics(pred_adj_lpcmci, true_adj, lpcmci_time)
                m_lpcmci.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "KAN-Koopman + PCMCI",
                    "Trial": trial,
                })
                results.append(m_lpcmci)
                print(
                    f"  KAN+PCMCI  — F1: {m_lpcmci['F1']:.3f}, "
                    f"TP: {m_lpcmci['TP']}, FP: {m_lpcmci['FP']}, "
                    f"TPR: {m_lpcmci['TPR']:.2f}, FPR: {m_lpcmci['FPR']:.2f}, "
                    f"Acc: {m_lpcmci['Accuracy']:.3f}  ({lpcmci_time:.1f}s)"
                )

                # ===========================================================
                # METHOD 3: KAN-Koopman + oCSE
                # ===========================================================
                print("  Running oCSE on KAN latent features...")
                start_time = time.time()

                pred_adj_locse = latent_ocse_test(
                    Z_nodes, n_nodes, d_node,
                    alpha=GLOBAL_PARAMS["alpha"],
                    tau_max=GLOBAL_PARAMS["tau_max"],
                )
                locse_time = time.time() - start_time + kan_train_time

                m_locse = compute_metrics(pred_adj_locse, true_adj, locse_time)
                m_locse.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "KAN-Koopman + oCSE",
                    "Trial": trial,
                })
                results.append(m_locse)
                print(
                    f"  KAN+oCSE   — F1: {m_locse['F1']:.3f}, "
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
                # METHOD 5: Raw PCMCI baseline
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
                # METHOD 6: Raw oCSE baseline
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
    csv_path = "benchmarks/results/deep_koopman_kan_results.csv"
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

    # Compare with attention results if available
    attn_csv = "benchmarks/results/deep_koopman_attention_results.csv"
    if os.path.exists(attn_csv):
        df_attn = pd.read_csv(attn_csv)
        print("\n--- Comparison: Attention-based results (from previous run) ---")
        attn_summary = (
            df_attn.groupby("Method")[["F1", "TPR", "FPR", "Accuracy"]]
            .mean()
            .sort_values("F1", ascending=False)
        )
        print(attn_summary.to_string(float_format="%.3f"))

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
    plot_path = "benchmarks/results/deep_koopman_kan_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")
