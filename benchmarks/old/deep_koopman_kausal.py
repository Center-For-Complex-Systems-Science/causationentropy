"""
Kausal-style Deep Koopman for Causal Discovery on Rössler Oscillators.

Implements the core idea from:
  "Deep Koopman operators for causal discovery" (Nat. Commun. Phys., 2025)

For each candidate pair (j→i), compare:
  - Marginal model: predict node i using only node i's observables
  - Joint model: predict node i using node i + node j's observables
If joint predicts significantly better (bootstrap test), j causes i.

This avoids the latent-pair collapse problem entirely — each causal test
is a direct node-to-node comparison.
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
from torch.utils.data import TensorDataset, DataLoader

# Tigramite imports (for baselines)
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# User Library Imports
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
# PER-NODE ENCODER (observable estimator)
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


class PerNodeKoopmanAutoencoder(nn.Module):
    """Per-node encoder/decoder with shared K matrix."""

    def __init__(self, n_nodes, d_node, state_per_node=3):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_node = d_node
        self.state_per_node = state_per_node
        self.latent_dim = n_nodes * d_node

        self.encoders = nn.ModuleList([
            NodeEncoder(state_per_node, d_node) for _ in range(n_nodes)
        ])

        self.K = nn.Parameter(torch.randn(self.latent_dim, self.latent_dim) * 0.01)

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
        z_next = z @ self.K.T
        x_next_pred = self.decode(z_next)
        return x_next_pred, z_next

    def block_sparsity_loss(self):
        penalty = torch.tensor(0.0, device=self.K.device)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    block = self.K[
                        i * self.d_node : (i + 1) * self.d_node,
                        j * self.d_node : (j + 1) * self.d_node,
                    ]
                    penalty = penalty + torch.norm(block, p="fro")
        return penalty

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


# =============================================================================
# TRAINING
# =============================================================================


def train_koopman(
    traj, n_nodes, d_node=8, epochs=1000, lr=1e-3, batch_size=256,
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

    model = PerNodeKoopmanAutoencoder(n_nodes, d_node).to(device)
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

            loss = L_recon + lambda_pred * L_pred + lambda_lin * L_lin

            if lambda_sparse > 0:
                loss = loss + lambda_sparse * model.block_sparsity_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % print_every == 0 or epoch == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.6f}")

    return model, mean, std


# =============================================================================
# KAUSAL-STYLE PAIRWISE GRANGER TEST IN KOOPMAN SPACE
# =============================================================================


def dmd_predict(Z_predictors, Z_target_next):
    """
    Fit Koopman operator via DMD (pseudoinverse) and return prediction error.

    K = Z_target_next @ pinv(Z_predictors)
    error = ||Z_target_next - K @ Z_predictors||² per timestep
    """
    # Z_predictors: (T, d_pred), Z_target_next: (T, d_target)
    # DMD: K = Y @ X^+ where X = Z_predictors.T, Y = Z_target_next.T
    X = Z_predictors.T  # (d_pred, T)
    Y = Z_target_next.T  # (d_target, T)

    K = Y @ np.linalg.pinv(X)  # (d_target, d_pred)
    Y_pred = K @ X  # (d_target, T)
    errors = np.sum((Y - Y_pred) ** 2, axis=0)  # (T,)
    return errors


def kausal_pairwise_test(
    Z_nodes, n_nodes, n_bootstrap=200, p_crit=0.05,
):
    """
    Kausal-style pairwise Granger causality in Koopman latent space.

    For each pair (j→i):
      - Marginal: predict Z_i(t+1) from Z_i(t) alone
      - Joint: predict Z_i(t+1) from [Z_i(t), Z_j(t)]
      - Delta = error_marginal - error_joint (per timestep)
      - Bootstrap test: is Delta significantly > 0?

    Parameters
    ----------
    Z_nodes : list of np.ndarray
        Per-node latent trajectories, each shape (T, d_node).
    n_nodes : int
    n_bootstrap : int
        Number of bootstrap samples for significance testing.
    p_crit : float
        P-value threshold.

    Returns
    -------
    adj : np.ndarray (n_nodes, n_nodes)
        Binary adjacency matrix.
    pvals : np.ndarray (n_nodes, n_nodes)
        P-values for each pair.
    deltas : np.ndarray (n_nodes, n_nodes)
        Mean causal effect (error_marg - error_joint) for each pair.
    """
    T = Z_nodes[0].shape[0]

    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    pvals = np.ones((n_nodes, n_nodes))
    deltas = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        Zi = Z_nodes[i]  # (T, d_node)
        Zi_t = Zi[:-1]   # predictors at time t
        Zi_tp1 = Zi[1:]  # target at time t+1

        # Marginal errors (predict i from i alone)
        errors_marg = dmd_predict(Zi_t, Zi_tp1)

        for j in range(n_nodes):
            if i == j:
                continue

            Zj = Z_nodes[j]
            Zj_t = Zj[:-1]

            # Joint predictors: [Z_i(t), Z_j(t)]
            Z_joint_t = np.concatenate([Zi_t, Zj_t], axis=1)
            errors_joint = dmd_predict(Z_joint_t, Zi_tp1)

            # Causal effect per timestep
            delta = errors_marg - errors_joint  # positive = j helps predict i

            # Bootstrap test: is mean(delta) > 0?
            T_eff = len(delta)
            bootstrap_means = np.zeros(n_bootstrap)
            rng = np.random.default_rng(42 + i * n_nodes + j)
            for b in range(n_bootstrap):
                idx = rng.choice(T_eff, size=T_eff, replace=True)
                bootstrap_means[b] = delta[idx].mean()

            # One-sided p-value: fraction of bootstrap means <= 0
            p_val = (np.sum(bootstrap_means <= 0) + 1) / (n_bootstrap + 1)

            deltas[j, i] = delta.mean()
            pvals[j, i] = p_val

            if p_val < p_crit:
                adj[j, i] = 1

    return adj, pvals, deltas


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


def compute_metrics(predicted_adj, true_adj, time_taken, prefix=""):
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
        f"{prefix}TP": int(tp),
        f"{prefix}FP": int(fp),
        f"{prefix}FN": int(fn),
        f"{prefix}TPR": tpr,
        f"{prefix}FPR": fpr,
        f"{prefix}Precision": precision,
        f"{prefix}F1": f1,
        f"{prefix}Accuracy": accuracy,
        f"{prefix}Time": time_taken,
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
    # Koopman
    "d_node": 8,
    "epochs": 1000,
    "lr": 1e-3,
    "batch_size": 256,
    "lambda_pred": 1.0,
    "lambda_lin": 0.5,
    "lambda_sparse": 0.01,
    # Kausal test
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
    d_node = GLOBAL_PARAMS["d_node"]

    print("=" * 80)
    print("Kausal-style Pairwise Koopman Causal Discovery — Rössler Oscillators")
    print(f"Device: {device}")
    print(f"{n_nodes} nodes, d_node={d_node}")
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

                # -----------------------------------------------------------
                # TRAIN PER-NODE KOOPMAN AUTOENCODER
                # -----------------------------------------------------------
                print("  Training Per-Node Koopman...")
                train_start = time.time()

                model, data_mean, data_std = train_koopman(
                    traj_sub,
                    n_nodes=n_nodes,
                    d_node=d_node,
                    epochs=GLOBAL_PARAMS["epochs"],
                    lr=GLOBAL_PARAMS["lr"],
                    batch_size=GLOBAL_PARAMS["batch_size"],
                    lambda_pred=GLOBAL_PARAMS["lambda_pred"],
                    lambda_lin=GLOBAL_PARAMS["lambda_lin"],
                    lambda_sparse=GLOBAL_PARAMS["lambda_sparse"],
                )

                train_time = time.time() - train_start

                # Encode to per-node latent representations
                T_sub = traj_sub.shape[0]
                data_flat = traj_sub.reshape(T_sub, n_nodes * 3)
                data_norm = (data_flat - data_mean) / data_std
                Z_nodes = model.encode_per_node(data_norm)

                print(f"  Trained in {train_time:.1f}s")

                # -----------------------------------------------------------
                # METHOD 1: Kausal-style pairwise Granger in Koopman space
                # -----------------------------------------------------------
                print("  Running Kausal pairwise test...")
                start_time = time.time()

                pred_adj_kausal, pvals, deltas = kausal_pairwise_test(
                    Z_nodes, n_nodes,
                    n_bootstrap=GLOBAL_PARAMS["n_bootstrap"],
                    p_crit=GLOBAL_PARAMS["p_crit"],
                )
                kausal_time = time.time() - start_time + train_time

                metrics_kausal = compute_metrics(
                    pred_adj_kausal, true_adj, kausal_time
                )
                metrics_kausal.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "Koopman + Kausal",
                    "Trial": trial,
                })
                results.append(metrics_kausal)
                print(
                    f"  Kausal     — F1: {metrics_kausal['F1']:.3f}, "
                    f"TP: {metrics_kausal['TP']}, FP: {metrics_kausal['FP']}, "
                    f"Acc: {metrics_kausal['Accuracy']:.3f}"
                )

                # -----------------------------------------------------------
                # METHOD 2: K-matrix thresholding
                # -----------------------------------------------------------
                pred_adj_K = model.extract_adjacency() if hasattr(model, 'extract_adjacency') else np.zeros((n_nodes, n_nodes))
                # extract_adjacency not on this model, compute manually
                K_np = model.K.detach().cpu().numpy()
                block_norms = np.zeros((n_nodes, n_nodes))
                for ii in range(n_nodes):
                    for jj in range(n_nodes):
                        block = K_np[
                            ii * d_node : (ii + 1) * d_node,
                            jj * d_node : (jj + 1) * d_node,
                        ]
                        block_norms[ii, jj] = np.linalg.norm(block, "fro")
                off_diag = block_norms[~np.eye(n_nodes, dtype=bool)]
                threshold = off_diag.mean() + 1.0 * off_diag.std()
                pred_adj_K = np.zeros((n_nodes, n_nodes), dtype=int)
                for ii in range(n_nodes):
                    for jj in range(n_nodes):
                        if ii != jj and block_norms[ii, jj] > threshold:
                            pred_adj_K[jj, ii] = 1

                metrics_K = compute_metrics(pred_adj_K, true_adj, train_time)
                metrics_K.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "Koopman (K-matrix)",
                    "Trial": trial,
                })
                results.append(metrics_K)
                print(
                    f"  K-matrix   — F1: {metrics_K['F1']:.3f}, "
                    f"TP: {metrics_K['TP']}, FP: {metrics_K['FP']}, "
                    f"Acc: {metrics_K['Accuracy']:.3f}"
                )

                # -----------------------------------------------------------
                # METHOD 3: Raw PCMCI baseline (no Koopman)
                # -----------------------------------------------------------
                print("  Running PCMCI baseline (raw)...")
                start_time = time.time()

                X_basis, basis_meta, var_names = (
                    prepare_rossler_data_for_causal_discovery(
                        traj_sub, dt=dt_eff,
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
                    graph_nx, basis_meta["n"], basis_meta["basis_map"],
                    coupling_on=GLOBAL_PARAMS["coupling_on"],
                )
                pred_adj_pcmci = collapse_to_binary_adjacency(
                    pcmci_node_graph, n_nodes
                )
                pcmci_time = time.time() - start_time

                metrics_pcmci = compute_metrics(
                    pred_adj_pcmci, true_adj, pcmci_time
                )
                metrics_pcmci.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "PCMCI (raw)",
                    "Trial": trial,
                })
                results.append(metrics_pcmci)
                print(
                    f"  PCMCI raw  — F1: {metrics_pcmci['F1']:.3f}, "
                    f"TP: {metrics_pcmci['TP']}, FP: {metrics_pcmci['FP']}, "
                    f"Acc: {metrics_pcmci['Accuracy']:.3f}"
                )

                # -----------------------------------------------------------
                # METHOD 4: Raw oCSE baseline (no Koopman)
                # -----------------------------------------------------------
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
                    oce_network, n_nodes, basis_meta["basis_map"],
                    var_names, coupling_on=GLOBAL_PARAMS["coupling_on"],
                )
                pred_adj_oce = collapse_to_binary_adjacency(
                    oce_node_graph, n_nodes
                )
                oce_time = time.time() - start_time

                metrics_oce = compute_metrics(
                    pred_adj_oce, true_adj, oce_time
                )
                metrics_oce.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "oCSE (raw)",
                    "Trial": trial,
                })
                results.append(metrics_oce)
                print(
                    f"  oCSE raw   — F1: {metrics_oce['F1']:.3f}, "
                    f"TP: {metrics_oce['TP']}, FP: {metrics_oce['FP']}, "
                    f"Acc: {metrics_oce['Accuracy']:.3f}"
                )

                pbar.update(1)

    pbar.close()

    # =================================================================
    # SAVE
    # =================================================================

    df_results = pd.DataFrame(results)
    df_results.to_csv(
        "benchmarks/results/deep_koopman_kausal_results.csv", index=False
    )
    print("\nResults saved to benchmarks/results/deep_koopman_kausal_results.csv")

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

    # =================================================================
    # PLOTTING
    # =================================================================

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for exp_name, config in EXPERIMENTS.items():
        exp_data = df_results[df_results["Experiment"] == exp_name]

        sns.lineplot(
            data=exp_data, x="Parameter", y="F1",
            hue="Method", style="Method", markers=True, dashes=False,
            ax=axes[0], errorbar=("ci", 68),
        )
        axes[0].set_title(f"{exp_name}: F1 Score")
        axes[0].set_xlabel(config["label"])
        axes[0].set_ylabel("F1")

        sns.lineplot(
            data=exp_data, x="Parameter", y="TPR",
            hue="Method", style="Method", markers=True, dashes=False,
            ax=axes[1], errorbar=("ci", 68),
        )
        axes[1].set_title(f"{exp_name}: True Positive Rate")
        axes[1].set_xlabel(config["label"])
        axes[1].set_ylabel("TPR")

        sns.lineplot(
            data=exp_data, x="Parameter", y="FPR",
            hue="Method", style="Method", markers=True, dashes=False,
            ax=axes[2], errorbar=("ci", 68),
        )
        axes[2].set_title(f"{exp_name}: False Positive Rate")
        axes[2].set_xlabel(config["label"])
        axes[2].set_ylabel("FPR")

    plt.tight_layout()
    plt.savefig(
        "benchmarks/results/deep_koopman_kausal_analysis.png",
        dpi=300, bbox_inches="tight",
    )
    print("\nPlot saved to benchmarks/results/deep_koopman_kausal_analysis.png")

    # Sources:
    # - https://www.nature.com/articles/s42005-025-02426-1
    # - https://github.com/juannat7/kausal
    # - https://arxiv.org/abs/2505.14828
