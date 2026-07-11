"""
Kuramoto ROC Curve Benchmark
=============================
Runs oCSE, PCMCI, and Linear Granger at multiple significance levels (alpha)
on coupled Kuramoto oscillators with physics-informed basis to generate
ROC curves. Each alpha is a true operating point — the algorithm is re-run
at each level, not just thresholded post-hoc.

oCSE uses a targeted approach: forward/backward selection on velocity
targets only (10 instead of 100 variables), which is where the causal
edges live.

Produces:
  - benchmarks/results/kuramoto_roc_results.csv
  - benchmarks/results/kuramoto_roc.png
"""
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
import time
import warnings
import os

from sklearn.linear_model import LinearRegression
from scipy import stats

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from causationentropy.core.discovery import standard_optimal_causation_entropy
from causationentropy.graph import pcmci_to_networkx
from causationentropy.datasets.synthetic import (
    generate_graph_topology,
    simulate_kuramoto,
    prepare_kuramoto_data_for_causal_discovery,
)

warnings.filterwarnings("ignore")


# =============================================================================
# METHOD WRAPPERS
# =============================================================================

def run_targeted_oce(X_basis, basis_meta, alpha, n_shuffles=100):
    """
    Run oCSE forward+backward on velocity targets only.
    Returns node-level adjacency matrix.
    """
    n = basis_meta["n"]
    basis_map = basis_meta["basis_map"]
    X_lagged = X_basis[:-1, :]
    Y_all = X_basis[1:, :]
    rng = np.random.default_rng(42)

    A_node = np.zeros((n, n), dtype=int)

    for i in range(n):
        Y = Y_all[:, [i]]
        Z_init = X_lagged[:, [i]]

        S = standard_optimal_causation_entropy(
            X_lagged, Y, Z_init, rng,
            alpha1=alpha, alpha2=alpha,
            n_shuffles=n_shuffles,
            information="gaussian",
        )

        for s in S:
            if s < n:
                continue
            coupling_idx = s - n
            target, source = basis_map[coupling_idx]
            if target == i:
                A_node[source, i] = 1
            elif source == i:
                A_node[target, i] = 1

    return A_node


def extract_pcmci_adjacency(graph_nx, n, basis_map):
    """Extract node-level adjacency from PCMCI graph on expanded basis."""
    A_node = np.zeros((n, n), dtype=int)
    for i in range(n):
        for idx, (target, source) in enumerate(basis_map):
            coupling_node_idx = n + idx
            if graph_nx.has_edge(coupling_node_idx, i):
                if target == i:
                    A_node[source, i] = 1
                elif source == i:
                    A_node[target, i] = 1
    return A_node


def linear_granger(theta, n_nodes, alpha=0.05):
    """Pairwise linear Granger causality."""
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    Y_all = theta[1:]
    X_all = theta[:-1]
    for i in range(n_nodes):
        y = Y_all[:, i]
        X_r = X_all[:, [i]]
        reg_r = LinearRegression().fit(X_r, y)
        rss_r = np.sum((y - reg_r.predict(X_r)) ** 2)
        for j in range(n_nodes):
            if j == i:
                continue
            X_u = X_all[:, [i, j]]
            reg_u = LinearRegression().fit(X_u, y)
            rss_u = np.sum((y - reg_u.predict(X_u)) ** 2)
            n_obs = len(y)
            f_stat = ((rss_r - rss_u) / 1) / (rss_u / (n_obs - 2))
            p_value = 1 - stats.f.cdf(f_stat, 1, n_obs - 2)
            if p_value < alpha:
                adj[j, i] = 1
    return adj


def compute_metrics(predicted_adj, true_adj):
    B = (predicted_adj != 0).astype(int)
    A = (true_adj != 0).astype(int)
    tp = int(np.sum((B == 1) & (A == 1)))
    fp = int(np.sum((B == 1) & (A == 0)))
    tn = int(np.sum((B == 0) & (A == 0)))
    fn = int(np.sum((B == 0) & (A == 1)))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * tpr / (precision + tpr)
        if (precision + tpr) > 0 else 0.0
    )
    return {"TP": tp, "FP": fp, "FN": fn,
            "TPR": tpr, "FPR": fpr, "Precision": precision, "F1": f1}


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 10,
    "T": 5000,
    "n_trials": 5,
    "dt": 0.05,
    "omega_std": 1.0,
    "phase_noise_std": 0.02,
    "burn_in": 50,
    "normalize_by_indegree": False,
    "n_shuffles": 100,
}

TOPOLOGIES = {
    "Hub": {
        "type": "Scale-Free",
        "params": {"m_attachment": 2, "rho": 0.7},
    },
    "Small-World": {
        "type": "Small-World",
        "params": {"k_neighbors": 4, "p_rewire": 0.3, "rho": 0.7},
    },
    "Erdos-Renyi": {
        "type": "Erdos-Renyi",
        "params": {"p_edge": 0.3, "rho": 0.7},
    },
}

ALPHA_VALUES = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs("benchmarks/results", exist_ok=True)
    n_nodes = GLOBAL_PARAMS["n_nodes"]
    n_trials = GLOBAL_PARAMS["n_trials"]
    n_alphas = len(ALPHA_VALUES)
    n_topos = len(TOPOLOGIES)

    total_iters = n_topos * n_trials * n_alphas
    print("=" * 70)
    print("Kuramoto ROC Curve Benchmark")
    print(f"Methods: oCSE (targeted), PCMCI, Linear Granger")
    print(f"Topologies: {list(TOPOLOGIES.keys())}")
    print(f"Alpha values: {ALPHA_VALUES}")
    print(f"{n_nodes} nodes, {n_trials} trials, {n_alphas} alpha points")
    print(f"Total method runs: {total_iters}")
    print("=" * 70)

    results = []
    pbar = tqdm(total=total_iters, desc="Progress")

    for topo_name, topo_cfg in TOPOLOGIES.items():
        print(f"\n--- {topo_name} ({topo_cfg['type']}) ---")

        for trial in range(n_trials):
            seed = 42 + trial * 100
            np.random.seed(seed)

            # Generate data ONCE per (topology, trial)
            G = generate_graph_topology(
                topo_cfg["type"], n_nodes, topo_cfg["params"], seed
            )
            true_adj = nx.to_numpy_array(G).astype(int)
            n_edges = int(true_adj.sum())

            theta, _ = simulate_kuramoto(
                G=G, T=GLOBAL_PARAMS["T"], dt=GLOBAL_PARAMS["dt"],
                rho=topo_cfg["params"]["rho"], seed=seed,
                omega_mean=0.0, omega_std=GLOBAL_PARAMS["omega_std"],
                phase_noise_std=GLOBAL_PARAMS["phase_noise_std"],
                burn_in=GLOBAL_PARAMS["burn_in"],
                normalize_by_indegree=GLOBAL_PARAMS["normalize_by_indegree"],
            )

            X_basis, basis_meta, var_names = (
                prepare_kuramoto_data_for_causal_discovery(
                    theta, dt=GLOBAL_PARAMS["dt"]
                )
            )
            T_eff = X_basis.shape[0]

            # Sweep alpha — re-run each method at each alpha
            for alpha in ALPHA_VALUES:
                common = {
                    "Topology": topo_name,
                    "Alpha": alpha,
                    "Trial": trial,
                    "N_true_edges": n_edges,
                }

                # --- oCSE (targeted: velocity targets only) ---
                t0 = time.time()
                adj_oce = run_targeted_oce(
                    X_basis, basis_meta, alpha=alpha,
                    n_shuffles=GLOBAL_PARAMS["n_shuffles"],
                )
                m = compute_metrics(adj_oce, true_adj)
                m.update(common)
                m["Method"] = "oCSE"
                m["Time"] = time.time() - t0
                results.append(m)

                # --- PCMCI ---
                t0 = time.time()
                dataframe = pp.DataFrame(
                    X_basis, datatime={0: np.arange(T_eff)},
                    var_names=var_names,
                )
                pcmci = PCMCI(
                    dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0
                )
                pcmci_res = pcmci.run_pcmci(
                    tau_min=1, tau_max=1, pc_alpha=alpha,
                )
                graph_nx = pcmci_to_networkx(pcmci_res)
                adj_pcmci = extract_pcmci_adjacency(
                    graph_nx, basis_meta["n"], basis_meta["basis_map"]
                )
                m = compute_metrics(adj_pcmci, true_adj)
                m.update(common)
                m["Method"] = "PCMCI"
                m["Time"] = time.time() - t0
                results.append(m)

                # --- Linear Granger ---
                t0 = time.time()
                adj_granger = linear_granger(theta, n_nodes, alpha=alpha)
                m = compute_metrics(adj_granger, true_adj)
                m.update(common)
                m["Method"] = "Linear Granger"
                m["Time"] = time.time() - t0
                results.append(m)

                pbar.update(1)

            # Print summary for this trial at alpha=0.05
            r05 = [r for r in results
                   if r["Topology"] == topo_name
                   and r["Trial"] == trial
                   and np.isclose(r["Alpha"], 0.05)]
            line = f"  Trial {trial}: {n_edges} edges |"
            for r in r05:
                line += f"  {r['Method']}: F1={r['F1']:.3f}"
            print(line)

    pbar.close()

    # Save
    df = pd.DataFrame(results)
    df.to_csv("benchmarks/results/kuramoto_roc_results.csv", index=False)
    print(f"\nResults saved ({len(df)} rows)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (alpha=0.05)")
    print("=" * 70)
    df05 = df[np.isclose(df["Alpha"], 0.05)]
    summary = (
        df05.groupby(["Topology", "Method"])[["TPR", "FPR", "F1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "Topology", "Method",
        "TPR_mean", "TPR_std", "FPR_mean", "FPR_std", "F1_mean", "F1_std",
    ]
    print(summary.to_string(index=False, float_format="%.3f"))

    # =========================================================================
    # ROC PLOT
    # =========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "sans-serif",
    })

    methods = ["oCSE", "PCMCI", "Linear Granger"]
    colors = {"oCSE": "#e74c3c", "PCMCI": "#3498db", "Linear Granger": "#9b59b6"}
    markers = {"oCSE": "o", "PCMCI": "s", "Linear Granger": "^"}

    fig, axes = plt.subplots(1, n_topos, figsize=(6 * n_topos, 5.5), squeeze=False)

    for col, topo_name in enumerate(TOPOLOGIES.keys()):
        ax = axes[0, col]
        topo_data = df[df["Topology"] == topo_name]

        for method in methods:
            mdata = topo_data[topo_data["Method"] == method]
            grouped = mdata.groupby("Alpha")[["FPR", "TPR"]].agg(["mean", "std"])

            fpr_mean = grouped[("FPR", "mean")].values
            fpr_std = grouped[("FPR", "std")].values
            tpr_mean = grouped[("TPR", "mean")].values
            tpr_std = grouped[("TPR", "std")].values

            # Sort by FPR
            order = np.argsort(fpr_mean)
            fpr_mean, fpr_std = fpr_mean[order], fpr_std[order]
            tpr_mean, tpr_std = tpr_mean[order], tpr_std[order]

            ax.errorbar(
                fpr_mean, tpr_mean,
                xerr=fpr_std, yerr=tpr_std,
                label=method,
                color=colors[method], marker=markers[method],
                capsize=3, linewidth=2, markersize=7,
            )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(topo_name)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="lower right")

    fig.suptitle(
        r"Kuramoto Oscillators: ROC ($\alpha$ sweep)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        "benchmarks/results/kuramoto_roc.png", dpi=300, bbox_inches="tight"
    )
    print("ROC figure saved to benchmarks/results/kuramoto_roc.png")
