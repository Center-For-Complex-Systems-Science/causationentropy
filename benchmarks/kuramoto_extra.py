"""
Additional algorithms on Kuramoto data for completeness.
Adds: LASSO (CV + fixed alphas), Linear Granger, Koopman + oCSE.
Uses same experimental setup as kuramoto.py.
"""
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings
import sys
import os
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LassoCV, Lasso

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx
from causationentropy.datasets.synthetic import (
    generate_graph_topology,
    simulate_kuramoto,
    prepare_kuramoto_data_for_causal_discovery,
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


# =========================================================================
# METRICS
# =========================================================================

def compute_metrics(predicted_adj, true_adj, time_taken):
    B = (predicted_adj != 0).astype(int)
    A = (true_adj != 0).astype(int)
    tp = np.sum((B == 1) & (A == 1))
    fp = np.sum((B == 1) & (A == 0))
    tn = np.sum((B == 0) & (A == 0))
    fn = np.sum((B == 0) & (A == 1))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "TPR": tpr, "FPR": fpr, "Precision": precision,
        "F1": f1, "Time": time_taken,
    }


# =========================================================================
# LASSO ON KURAMOTO BASIS
# =========================================================================

def lasso_kuramoto(X_basis, meta, var_names, n_nodes, alpha=None):
    """
    LASSO regression on the Kuramoto sin-basis.
    For each target v_i, regress on sin(theta_j - theta_i) terms.
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    basis_map = meta["basis_map"]
    pred_offset = meta["pred_offset"]  # = n_nodes

    for i in range(n_nodes):
        Y = X_basis[:, i]  # velocity v_i

        # Coupling columns for this target
        coupling_cols = []
        coupling_pairs = []
        for idx, (target, source) in enumerate(basis_map):
            if target == i:
                coupling_cols.append(pred_offset + idx)
                coupling_pairs.append((target, source))

        if not coupling_cols:
            continue

        X_coup = X_basis[:, coupling_cols]

        if alpha is None:
            model = LassoCV(cv=5, max_iter=10000)
        else:
            model = Lasso(alpha=alpha, max_iter=10000)

        model.fit(X_coup, Y)

        for k, (target, source) in enumerate(coupling_pairs):
            if abs(model.coef_[k]) > 1e-8:
                adj[source, i] = 1

    return adj


# =========================================================================
# LINEAR GRANGER ON RAW THETA
# =========================================================================

def linear_granger_kuramoto(theta, n_nodes, alpha=0.05):
    """
    Simple linear Granger causality on raw theta time series.
    For each target i, test if lagged theta_j improves prediction of theta_i.
    """
    from sklearn.linear_model import LinearRegression
    from scipy import stats

    T = theta.shape[0]
    adj = np.zeros((n_nodes, n_nodes), dtype=int)

    Y_all = theta[1:]  # (T-1, n)
    X_all = theta[:-1]  # (T-1, n)

    for i in range(n_nodes):
        y = Y_all[:, i]

        # Restricted model: just own lag
        X_restricted = X_all[:, [i]]
        reg_r = LinearRegression().fit(X_restricted, y)
        rss_r = np.sum((y - reg_r.predict(X_restricted))**2)

        for j in range(n_nodes):
            if j == i:
                continue

            # Unrestricted: own lag + j's lag
            X_unrest = X_all[:, [i, j]]
            reg_u = LinearRegression().fit(X_unrest, y)
            rss_u = np.sum((y - reg_u.predict(X_unrest))**2)

            # F-test
            p_added = 1
            n_obs = len(y)
            k_u = 2
            k_r = 1
            f_stat = ((rss_r - rss_u) / p_added) / (rss_u / (n_obs - k_u))
            p_value = 1 - stats.f.cdf(f_stat, p_added, n_obs - k_u)

            if p_value < alpha:
                adj[j, i] = 1

    return adj


# =========================================================================
# KOOPMAN + oCSE FOR KURAMOTO
# =========================================================================

class KuramotoKoopman(nn.Module):
    """Simple shared-encoder Koopman for 1D-per-node Kuramoto data."""

    def __init__(self, n_nodes, d_node, d_hidden=64):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_node = d_node
        # Shared encoder: 1D -> d_node per node
        self.encoder = nn.Sequential(
            nn.Linear(1, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_node),
        )
        # Shared decoder: d_node -> 1D per node
        self.decoder = nn.Sequential(
            nn.Linear(d_node, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )
        # Linear transition in latent space
        total_latent = n_nodes * d_node
        self.K = nn.Linear(total_latent, total_latent, bias=False)

    def encode_per_node(self, x_flat):
        """x_flat: (T, n_nodes). Returns list of (T, d_node) arrays."""
        x_tensor = torch.tensor(x_flat, dtype=torch.float32).to(
            next(self.parameters()).device
        )
        z_list = []
        with torch.no_grad():
            for i in range(self.n_nodes):
                xi = x_tensor[:, i:i+1]  # (T, 1)
                zi = self.encoder(xi)     # (T, d_node)
                z_list.append(zi.cpu().numpy())
        return z_list

    def forward(self, x):
        """x: (batch, n_nodes)"""
        batch = x.shape[0]
        # Encode each node
        z_parts = []
        for i in range(self.n_nodes):
            xi = x[:, i:i+1]  # (batch, 1)
            zi = self.encoder(xi)  # (batch, d_node)
            z_parts.append(zi)
        z = torch.cat(z_parts, dim=1)  # (batch, n_nodes * d_node)

        # Linear transition
        z_next = self.K(z)

        # Decode
        x_recon_parts = []
        for i in range(self.n_nodes):
            zi_next = z_next[:, i*self.d_node:(i+1)*self.d_node]
            xi_recon = self.decoder(zi_next)  # (batch, 1)
            x_recon_parts.append(xi_recon)
        x_recon = torch.cat(x_recon_parts, dim=1)  # (batch, n_nodes)

        return x_recon, z, z_next


def train_kuramoto_koopman(theta, n_nodes, d_node=5, epochs=500, lr=1e-3,
                           batch_size=256, print_every=250):
    T = theta.shape[0]
    data = theta.astype(np.float32)
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0) + 1e-8
    data_norm = (data - data_mean) / data_std

    X = data_norm[:-1]
    Y = data_norm[1:]

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = KuramotoKoopman(n_nodes, d_node).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred, z, z_next = model(xb)

            loss_pred = nn.functional.mse_loss(y_pred, yb)
            # Linearity loss
            z_next_true = []
            with torch.no_grad():
                for i in range(n_nodes):
                    zi = model.encoder(yb[:, i:i+1])
                    z_next_true.append(zi)
            z_next_true = torch.cat(z_next_true, dim=1)
            loss_lin = nn.functional.mse_loss(z_next, z_next_true)

            loss = loss_pred + 0.5 * loss_lin
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % print_every == 0 or epoch == 1:
            print(f"    Epoch {epoch}/{epochs}  Loss: {total_loss/len(loader):.6f}")

    model.eval()
    return model, data_mean, data_std


def latent_ocse_kuramoto(model, theta, n_nodes, d_node, data_mean, data_std,
                         alpha=0.05, tau_max=1):
    """Run oCSE on Koopman latent features for Kuramoto."""
    data_norm = (theta.astype(np.float32) - data_mean) / data_std
    Z_nodes = model.encode_per_node(data_norm)

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


# =========================================================================
# KURAMOTO-SPECIFIC EXTRACTION (from kuramoto.py)
# =========================================================================

def extract_node_adjacency_from_basis_oce(graph_nx, n, basis_map, var_names):
    A_node = np.zeros((n, n), dtype=int)
    velocity_vars = [f"v{i}" for i in range(n)]
    for i in range(n):
        velocity_var = velocity_vars[i]
        for idx, (target, source) in enumerate(basis_map):
            coupling_var = var_names[n + idx]
            if graph_nx.has_edge(coupling_var, velocity_var):
                edges = graph_nx[coupling_var][velocity_var]
                for key, edge_data in edges.items():
                    if edge_data["lag"] <= 1:
                        if target == i:
                            A_node[source, i] = 1
                        elif source == i:
                            A_node[target, i] = 1
                        break
    return A_node


def extract_node_adjacency_from_basis_pcmci(graph_nx, n, basis_map):
    A_node = np.zeros((n, n), dtype=int)
    for i in range(n):
        velocity_node_idx = i
        for idx, (target, source) in enumerate(basis_map):
            coupling_node_idx = n + idx
            if graph_nx.has_edge(coupling_node_idx, velocity_node_idx):
                if target == i:
                    A_node[source, i] = 1
                elif source == i:
                    A_node[target, i] = 1
    return A_node


# =========================================================================
# CONFIG (matches kuramoto.py)
# =========================================================================

GLOBAL_PARAMS = {
    "n_nodes": 10,
    "T": 5000,
    "n_trials": 3,  # Fewer trials since Koopman+oCSE is slow
    "alpha": 0.05,
    "tau_max": 1,
    "dt": 0.05,
    "omega_std": 1.0,
    "phase_noise_std": 0.02,
    "burn_in": 50,
    "normalize_by_indegree": False,
    "d_node": 3,  # Keep latent dim small (30 total) for oCSE speed
}

EXPERIMENTS = {
    "ER_Density": {
        "type": "Erdos-Renyi",
        "vary_param": "p_edge",
        "label": "Edge Probability (p)",
        "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "defaults": {"rho": 0.7},
    },
    "Coupling_Strength": {
        "type": "Erdos-Renyi",
        "vary_param": "rho",
        "label": "Coupling Strength (rho)",
        "values": [0.1, 0.3, 0.5, 0.7, 0.9],
        "defaults": {"p_edge": 0.3},
    },
}


# =========================================================================
# MAIN
# =========================================================================

results = []
n_nodes = GLOBAL_PARAMS["n_nodes"]

total_iters = sum(
    len(cfg["values"]) * GLOBAL_PARAMS["n_trials"]
    for cfg in EXPERIMENTS.values()
)

print("=" * 80)
print("Kuramoto Extra Benchmarks: LASSO, Linear Granger, Koopman+oCSE, oCSE, PCMCI")
print(f"Device: {device}")
print(f"{n_nodes} nodes, {GLOBAL_PARAMS['n_trials']} trials per condition")
print(f"Total iterations: {total_iters}")
print("=" * 80)

pbar = tqdm(total=total_iters, desc="Total progress")

for exp_name, config in EXPERIMENTS.items():
    print(f"\n--- {exp_name} ---")

    param_name = config["vary_param"]
    param_values = config["values"]
    defaults = config["defaults"]

    for val in param_values:
        current_params = defaults.copy()
        current_params[param_name] = val

        for trial in range(GLOBAL_PARAMS["n_trials"]):
            seed = 42 + (trial * 100)
            np.random.seed(seed)
            torch.manual_seed(seed)

            G = generate_graph_topology(
                config["type"], n_nodes, current_params, seed
            )
            true_adj = nx.to_numpy_array(G).astype(int)

            theta, _ = simulate_kuramoto(
                G=G, T=GLOBAL_PARAMS["T"], dt=GLOBAL_PARAMS["dt"],
                rho=current_params.get("rho", 0.7), seed=seed,
                omega_mean=0.0, omega_std=GLOBAL_PARAMS["omega_std"],
                phase_noise_std=GLOBAL_PARAMS["phase_noise_std"],
                burn_in=GLOBAL_PARAMS["burn_in"],
                normalize_by_indegree=GLOBAL_PARAMS["normalize_by_indegree"],
            )

            X_basis, basis_meta, var_names = prepare_kuramoto_data_for_causal_discovery(
                theta, dt=GLOBAL_PARAMS["dt"]
            )
            T_eff = X_basis.shape[0]

            n_true = int(true_adj.sum())
            n_possible = n_nodes * (n_nodes - 1)

            common = {"Experiment": exp_name, "Parameter": val, "Trial": trial}

            # ----- LASSO (CV) on Kuramoto basis -----
            start = time.time()
            adj_lasso_cv = lasso_kuramoto(X_basis, basis_meta, var_names, n_nodes, alpha=None)
            m = compute_metrics(adj_lasso_cv, true_adj, time.time() - start)
            m.update(common); m["Method"] = "LASSO (CV)"
            results.append(m)

            # ----- LASSO (fixed alphas) -----
            for alpha_l in [0.001, 0.01, 0.1]:
                start = time.time()
                adj_lasso = lasso_kuramoto(X_basis, basis_meta, var_names, n_nodes, alpha=alpha_l)
                m = compute_metrics(adj_lasso, true_adj, time.time() - start)
                m.update(common); m["Method"] = f"LASSO (α={alpha_l})"
                results.append(m)

            # ----- Linear Granger on raw theta -----
            start = time.time()
            adj_granger = linear_granger_kuramoto(theta, n_nodes, alpha=0.05)
            m = compute_metrics(adj_granger, true_adj, time.time() - start)
            m.update(common); m["Method"] = "Linear Granger"
            results.append(m)

            # ----- Koopman + oCSE -----
            d_node = GLOBAL_PARAMS["d_node"]
            start = time.time()
            model, data_mean, data_std = train_kuramoto_koopman(
                theta, n_nodes, d_node=d_node, epochs=500, lr=1e-3,
                batch_size=256, print_every=500,
            )
            adj_koopman = latent_ocse_kuramoto(
                model, theta, n_nodes, d_node=d_node,
                data_mean=data_mean, data_std=data_std,
                alpha=GLOBAL_PARAMS["alpha"], tau_max=GLOBAL_PARAMS["tau_max"],
            )
            m = compute_metrics(adj_koopman, true_adj, time.time() - start)
            m.update(common); m["Method"] = "Koopman + oCSE"
            results.append(m)

            # Print summary for this trial
            trial_results = [r for r in results if r["Trial"] == trial
                           and r["Parameter"] == val and r["Experiment"] == exp_name]
            print(f"  {config['label']}={val}, Trial {trial} "
                  f"(true edges: {n_true}/{n_possible})")
            for r in trial_results:
                print(f"    {r['Method']:25s}  TP={r['TP']:2d} FP={r['FP']:2d} "
                      f"FN={r['FN']:2d} F1={r['F1']:.3f}")

            pbar.update(1)

pbar.close()

# =========================================================================
# SAVE
# =========================================================================

df = pd.DataFrame(results)
df.to_csv("benchmarks/results/kuramoto_extra_results.csv", index=False)
print(f"\nResults saved to benchmarks/results/kuramoto_extra_results.csv")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: Mean across all conditions and trials")
print("=" * 80)
summary = df.groupby("Method")[["TPR", "FPR", "F1"]].agg(["mean", "std"]).reset_index()
summary.columns = ["Method", "TPR_mean", "TPR_std", "FPR_mean", "FPR_std", "F1_mean", "F1_std"]
summary = summary.sort_values("F1_mean", ascending=False)
print(summary.to_string(index=False, float_format="%.3f"))
