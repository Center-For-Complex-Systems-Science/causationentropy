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

# Tigramite imports
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# User Library Imports
from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx
from causationentropy.datasets.synthetic import (
    generate_graph_topology,
    simulate_rossler,
)

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# STDOUT SUPPRESSION
# =============================================================================


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# =============================================================================
# DEEP KOOPMAN AUTOENCODER
# =============================================================================


class KoopmanAutoencoder(nn.Module):
    """
    Deep Koopman autoencoder for causal discovery.

    Encoder φ: R^(n*3) → R^(n*d_node)      (lift to Koopman space)
    K matrix:  R^(n*d_node) → R^(n*d_node)  (linear dynamics)
    Decoder ψ: R^(n*d_node) → R^(n*3)       (reconstruct)
    """

    def __init__(self, state_dim, latent_dim):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        self.K = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        )

    def forward(self, x):
        """Encode and reconstruct."""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def predict(self, x):
        """Encode, apply K, decode → predicted next state."""
        z = self.encoder(x)
        z_next = z @ self.K.T
        x_next_pred = self.decoder(z_next)
        return x_next_pred, z_next

    def block_sparsity_loss(self, n, d_node):
        """
        Group-LASSO penalty on off-diagonal blocks of K.

        Penalizes sum of Frobenius norms of off-diagonal (i,j) blocks,
        encouraging K to be block-diagonal (no spurious cross-node coupling).
        """
        penalty = torch.tensor(0.0, device=self.K.device)
        for i in range(n):
            for j in range(n):
                if i != j:
                    block = self.K[
                        i * d_node : (i + 1) * d_node,
                        j * d_node : (j + 1) * d_node,
                    ]
                    penalty = penalty + torch.norm(block, p="fro")
        return penalty

    def encode_trajectory(self, traj_flat_norm):
        """Encode a full normalized trajectory to latent space."""
        with torch.no_grad():
            x = torch.tensor(traj_flat_norm, dtype=torch.float32).to(device)
            z = self.encoder(x)
            return z.cpu().numpy()

    def extract_adjacency(self, n, d_node, threshold_std=1.0):
        """
        Extract binary adjacency from K's block structure.

        K_block[i,j] governs how node j's latent dims influence node i's next latent dims.
        A[j, i] = 1 if ||K_block[i,j]||_F > threshold (j influences i).
        """
        K_np = self.K.detach().cpu().numpy()
        block_norms = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                block = K_np[
                    i * d_node : (i + 1) * d_node,
                    j * d_node : (j + 1) * d_node,
                ]
                block_norms[i, j] = np.linalg.norm(block, "fro")

        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag_norms = block_norms[off_diag_mask]
        threshold = off_diag_norms.mean() + threshold_std * off_diag_norms.std()

        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j and block_norms[i, j] > threshold:
                    adj[j, i] = 1

        return adj


# =============================================================================
# TRAINING
# =============================================================================


def train_koopman(
    traj,
    n_nodes,
    d_node=8,
    epochs=1000,
    lr=1e-3,
    batch_size=256,
    lambda_pred=1.0,
    lambda_lin=0.5,
    lambda_sparse=0.0,
    print_every=200,
):
    """
    Train a Deep Koopman autoencoder on Rössler trajectory data.

    Returns the trained model along with normalization stats (mean, std)
    needed to encode new data.
    """
    T = traj.shape[0]
    state_dim = n_nodes * 3
    latent_dim = n_nodes * d_node

    data_flat = traj.reshape(T, state_dim)

    mean = data_flat.mean(axis=0)
    std = data_flat.std(axis=0)
    std[std < 1e-8] = 1.0
    data_norm = (data_flat - mean) / std

    x_t = torch.tensor(data_norm[:-1], dtype=torch.float32)
    x_tp1 = torch.tensor(data_norm[1:], dtype=torch.float32)

    dataset = TensorDataset(x_t, x_tp1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = KoopmanAutoencoder(state_dim, latent_dim).to(device)
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
                L_sparse = model.block_sparsity_loss(n_nodes, d_node)
                loss = loss + lambda_sparse * L_sparse

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
# LATENT-SPACE CAUSAL DISCOVERY HELPERS
# =============================================================================


def latent_var_names(n_nodes, d_node):
    """Generate variable names for the latent space: z0_0, z0_1, ..., z1_0, ..."""
    names = []
    for i in range(n_nodes):
        for k in range(d_node):
            names.append(f"z{i}_{k}")
    return names


def latent_graph_to_node_adjacency(graph_nx, n_nodes, d_node):
    """
    Collapse a graph over latent variables back to node-level adjacency.

    If any latent dim of node j has an edge to any latent dim of node i (j≠i),
    then A[j, i] = 1 in the node-level adjacency.
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=int)

    for u, v, _ in graph_nx.edges(data=True):
        # Determine which node each latent variable belongs to
        if isinstance(u, str):
            # oCSE returns string node names like "z0_3"
            node_u = int(u.split("_")[0][1:])  # "z0_3" → 0
            node_v = int(v.split("_")[0][1:])
        else:
            # PCMCI returns integer indices
            node_u = u // d_node
            node_v = v // d_node

        if node_u != node_v:
            adj[node_u, node_v] = 1

    return adj


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
    f1 = (
        2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    )
    accuracy = (tp + tn) / total if total > 0 else 0.0
    shd = fp + fn

    return {
        f"{prefix}TP": int(tp),
        f"{prefix}FP": int(fp),
        f"{prefix}FN": int(fn),
        f"{prefix}TPR": tpr,
        f"{prefix}FPR": fpr,
        f"{prefix}Precision": precision,
        f"{prefix}F1": f1,
        f"{prefix}Accuracy": accuracy,
        f"{prefix}SHD": shd,
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
    # Rössler params
    "a": 0.2,
    "b": 0.2,
    "c": 5.7,
    "rho_default": 0.4,
    "noise_std": 0.0,
    "init_scale": 1.0,
    "coupling_on": "x",
    "normalize_by_indegree": False,
    # PCMCI / oCSE params
    "alpha": 0.05,
    "tau_max": 1,
    # Koopman params
    "d_node": 8,
    "epochs": 1000,
    "lr": 1e-3,
    "batch_size": 256,
    "lambda_pred": 1.0,
    "lambda_lin": 0.5,
    "lambda_sparse": 0.01,
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
# MAIN EXPERIMENT LOOP
# =============================================================================

if __name__ == "__main__":
    results = []

    n_nodes = GLOBAL_PARAMS["n_nodes"]
    d_node = GLOBAL_PARAMS["d_node"]
    latent_dim = n_nodes * d_node
    var_names_latent = latent_var_names(n_nodes, d_node)

    print("=" * 80)
    print("Deep Koopman + oCSE/PCMCI on Lifted Space — Rössler Oscillators")
    print(f"Device: {device}")
    print(f"{n_nodes} nodes, d_node={d_node}, latent_dim={latent_dim}")
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
                print(
                    f"\n  {config['label']}={val}, Trial {trial}, seed={seed}"
                )

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

                # Subsample trajectory
                k = GLOBAL_PARAMS["subsample"]
                traj_sub = traj[::k] if k > 1 else traj
                dt_eff = GLOBAL_PARAMS["dt"] * k

                # -----------------------------------------------------------
                # TRAIN DEEP KOOPMAN
                # -----------------------------------------------------------
                print("  Training Deep Koopman...")
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

                # Encode full trajectory to latent space
                T_sub = traj_sub.shape[0]
                data_flat = traj_sub.reshape(T_sub, n_nodes * 3)
                data_norm = (data_flat - data_mean) / data_std
                Z = model.encode_trajectory(data_norm)  # (T_sub, latent_dim)

                print(f"  Koopman trained in {train_time:.1f}s, "
                      f"latent shape: {Z.shape}")

                # -----------------------------------------------------------
                # METHOD 1: K-matrix thresholding (direct)
                # -----------------------------------------------------------
                pred_adj_K = model.extract_adjacency(n_nodes, d_node)
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
                # METHOD 2: oCSE on lifted latent space
                # -----------------------------------------------------------
                print("  Running oCSE on latent space...")
                start_time = time.time()

                Z_df = pd.DataFrame(Z, columns=var_names_latent)
                max_lag = GLOBAL_PARAMS["tau_max"]

                with suppress_stdout():
                    oce_graph = discover_network(
                        data=Z_df,
                        max_lag=max_lag,
                        method="standard",
                        information="gaussian",
                        alpha_forward=GLOBAL_PARAMS["alpha"],
                        alpha_backward=GLOBAL_PARAMS["alpha"],
                    )

                pred_adj_oce = latent_graph_to_node_adjacency(
                    oce_graph, n_nodes, d_node
                )
                oce_time = time.time() - start_time + train_time

                metrics_oce = compute_metrics(pred_adj_oce, true_adj, oce_time)
                metrics_oce.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "Koopman + oCSE",
                    "Trial": trial,
                })
                results.append(metrics_oce)
                print(
                    f"  Koop+oCSE  — F1: {metrics_oce['F1']:.3f}, "
                    f"TP: {metrics_oce['TP']}, FP: {metrics_oce['FP']}, "
                    f"Acc: {metrics_oce['Accuracy']:.3f}, "
                    f"Time: {oce_time:.1f}s"
                )

                # -----------------------------------------------------------
                # METHOD 3: PCMCI on lifted latent space
                # -----------------------------------------------------------
                print("  Running PCMCI on latent space...")
                start_time = time.time()

                dataframe = pp.DataFrame(
                    Z,
                    datatime={0: np.arange(Z.shape[0])},
                    var_names=var_names_latent,
                )
                pcmci = PCMCI(
                    dataframe=dataframe,
                    cond_ind_test=ParCorr(),
                    verbosity=0,
                )
                pcmci_res = pcmci.run_pcmci(
                    tau_min=1,
                    tau_max=max_lag,
                    pc_alpha=GLOBAL_PARAMS["alpha"],
                )
                pcmci_graph = pcmci_to_networkx(pcmci_res)

                pred_adj_pcmci = latent_graph_to_node_adjacency(
                    pcmci_graph, n_nodes, d_node
                )
                pcmci_time = time.time() - start_time + train_time

                metrics_pcmci = compute_metrics(
                    pred_adj_pcmci, true_adj, pcmci_time
                )
                metrics_pcmci.update({
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "Koopman + PCMCI",
                    "Trial": trial,
                })
                results.append(metrics_pcmci)
                print(
                    f"  Koop+PCMCI — F1: {metrics_pcmci['F1']:.3f}, "
                    f"TP: {metrics_pcmci['TP']}, FP: {metrics_pcmci['FP']}, "
                    f"Acc: {metrics_pcmci['Accuracy']:.3f}, "
                    f"Time: {pcmci_time:.1f}s"
                )

                pbar.update(1)

    pbar.close()

    # =================================================================
    # SAVE RESULTS
    # =================================================================

    df_results = pd.DataFrame(results)
    df_results.to_csv("benchmarks/results/deep_koopman_results.csv", index=False)
    print(f"\nResults saved to benchmarks/results/deep_koopman_results.csv")

    # =================================================================
    # SUMMARY
    # =================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary = (
        df_results.groupby(["Experiment", "Method"])[
            ["F1", "TP", "FP", "FN", "Accuracy", "TPR", "FPR", "Precision", "SHD", "Time"]
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
        "benchmarks/results/deep_koopman_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("\nPlot saved to benchmarks/results/deep_koopman_analysis.png")
