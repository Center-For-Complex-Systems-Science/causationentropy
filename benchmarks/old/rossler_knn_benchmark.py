"""
Rossler benchmark with KNN estimators - small graph, varying coupling strength.

This benchmark uses a small graph (n=3 nodes) to make KNN estimation tractable,
and varies coupling strength to test detection sensitivity.
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

# Tigramite imports
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.cmiknn import CMIknn

# User Library Imports
from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx
from causationentropy.datasets.synthetic import (
    simulate_rossler,
    prepare_rossler_data_for_causal_discovery,
)

warnings.filterwarnings("ignore")


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


def extract_node_adjacency_from_basis_oce_rossler(
    graph_nx, n, basis_map, var_names, coupling_on="x"
):
    """Extract node-level adjacency from OCE graph on expanded basis."""
    A_node = np.zeros((n, n), dtype=int)

    if coupling_on == "x":
        target_vars = [f"dx{i}" for i in range(n)]
    elif coupling_on == "y":
        target_vars = [f"dy{i}" for i in range(n)]
    else:
        target_vars = [f"dz{i}" for i in range(n)]

    base = 3 * n
    for i in range(n):
        target_var = target_vars[i]
        for idx, (target, source) in enumerate(basis_map):
            coupling_var = var_names[base + idx]
            if graph_nx.has_edge(coupling_var, target_var):
                edges = graph_nx[coupling_var][target_var]
                for _, edata in edges.items():
                    if edata.get("lag", 0) <= 1:
                        if target == i:
                            A_node[source, i] = 1
                        break
    return A_node


def extract_node_adjacency_from_basis_pcmci_rossler(
    graph_nx, n, basis_map, coupling_on="x"
):
    """Extract node-level adjacency from PCMCI graph on expanded basis."""
    A_node = np.zeros((n, n), dtype=int)

    if coupling_on == "x":
        target_start = 0
    elif coupling_on == "y":
        target_start = n
    else:
        target_start = 2 * n

    coupling_start = 3 * n

    for i in range(n):
        target_node_idx = target_start + i
        for idx, (target, source) in enumerate(basis_map):
            coupling_node_idx = coupling_start + idx
            if graph_nx.has_edge(coupling_node_idx, target_node_idx):
                if target == i:
                    A_node[source, i] = 1
    return A_node


def compute_metrics(predicted_adj, true_adj, time_taken):
    """Compute classification metrics."""
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
    shd = fp + fn

    return {
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "F1": f1,
        "SHD": shd,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Time": time_taken,
    }


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 3,  # Small graph for tractable KNN
    "T": 500,  # Moderate length
    "n_trials": 3,  # Fewer trials since we're testing k values
    "alpha": 0.01,  # Stricter significance threshold
    "tau_max": 1,
    "dt": 0.02,
    "burn_in": 500,
    "normalize_by_indegree": False,
    # Rossler parameters
    "a": 0.2,
    "b": 0.2,
    "c": 5.7,
    "noise_std": 0.0,
    "init_scale": 1.0,
    "coupling_on": "x",
    # KNN parameters
    "n_shuffles": 200,  # More shuffles for better p-value resolution
}

# Test different k values for KNN
K_VALUES = [5, 10, 20, 50]

# Fixed coupling strength (use moderate value where signal should be detectable)
COUPLING_VALUES = [0.3]

# Fixed graph structure for consistency
def create_fixed_graph(n, seed=42):
    """Create a fixed small graph for testing."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    # Create a simple chain with some additional edges
    # 0 -> 1 -> 2, plus 2 -> 0 for a cycle
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)
    return G


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = []

    # Create fixed graph
    G = create_fixed_graph(GLOBAL_PARAMS["n_nodes"])
    true_adj = nx.to_numpy_array(G).astype(int)

    print("=" * 60)
    print("Rossler KNN Benchmark - Varying k Parameter")
    print("=" * 60)
    print(f"Nodes: {GLOBAL_PARAMS['n_nodes']}")
    print(f"Time series length: {GLOBAL_PARAMS['T']}")
    print(f"Trials per k value: {GLOBAL_PARAMS['n_trials']}")
    print(f"k values: {K_VALUES}")
    print(f"Coupling strength: {COUPLING_VALUES[0]}")
    print(f"Alpha: {GLOBAL_PARAMS['alpha']}")
    print(f"\nTrue adjacency matrix:")
    print(true_adj)
    print(f"True edges: {list(G.edges())}")
    print("=" * 60)

    total_iters = len(K_VALUES) * GLOBAL_PARAMS["n_trials"]
    pbar = tqdm(total=total_iters, desc="Total progress")

    rho = COUPLING_VALUES[0]  # Fixed coupling

    for k_knn in K_VALUES:
        for trial in range(GLOBAL_PARAMS["n_trials"]):
            seed = 42 + trial * 100

            # Simulate Rossler
            traj, _ = simulate_rossler(
                G=G,
                T=GLOBAL_PARAMS["T"],
                dt=GLOBAL_PARAMS["dt"],
                rho=rho,
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

            X, meta, var_names = prepare_rossler_data_for_causal_discovery(
                traj, dt=GLOBAL_PARAMS["dt"], coupling_on=GLOBAL_PARAMS["coupling_on"]
            )
            T_eff = X.shape[0]

            dataframe = pp.DataFrame(
                X, datatime={0: np.arange(T_eff)}, var_names=var_names
            )

            # -----------------------------------------------------------------
            # METHOD 1: OCE with KNN
            # -----------------------------------------------------------------
            start_time = time.time()
            X_df = pd.DataFrame(X, columns=var_names)

            with suppress_stdout():
                network = discover_network(
                    data=X_df,
                    max_lag=GLOBAL_PARAMS["tau_max"],
                    method="standard",
                    information="knn",
                    k_means=k_knn,
                    n_shuffles=GLOBAL_PARAMS["n_shuffles"],
                    alpha_forward=GLOBAL_PARAMS["alpha"],
                    alpha_backward=GLOBAL_PARAMS["alpha"],
                )

            pred_adj_oce = extract_node_adjacency_from_basis_oce_rossler(
                network,
                meta["n"],
                meta["basis_map"],
                var_names,
                coupling_on=GLOBAL_PARAMS["coupling_on"],
            )

            metrics_oce = compute_metrics(
                pred_adj_oce, true_adj, time.time() - start_time
            )
            metrics_oce.update(
                {
                    "k": k_knn,
                    "Method": "OCE (knn)",
                    "Trial": trial,
                }
            )
            results.append(metrics_oce)

            # -----------------------------------------------------------------
            # METHOD 2: PCMCI with CMIknn
            # -----------------------------------------------------------------
            start_time = time.time()
            cmi_knn = CMIknn(
                knn=k_knn,
                shuffle_neighbors=k_knn,
                significance="shuffle_test",
                sig_samples=GLOBAL_PARAMS["n_shuffles"],
            )
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cmi_knn, verbosity=0)
            pcmci_res = pcmci.run_pcmci(
                tau_min=1,
                tau_max=GLOBAL_PARAMS["tau_max"],
                pc_alpha=GLOBAL_PARAMS["alpha"],
            )
            graph_nx = pcmci_to_networkx(pcmci_res)

            pred_adj_pcmci = extract_node_adjacency_from_basis_pcmci_rossler(
                graph_nx,
                meta["n"],
                meta["basis_map"],
                coupling_on=GLOBAL_PARAMS["coupling_on"],
            )

            metrics_pcmci = compute_metrics(
                pred_adj_pcmci, true_adj, time.time() - start_time
            )
            metrics_pcmci.update(
                {
                    "k": k_knn,
                    "Method": "PCMCI (CMIknn)",
                    "Trial": trial,
                }
            )
            results.append(metrics_pcmci)

            pbar.update(1)

    pbar.close()

    # =========================================================================
    # SAVE + ANALYSIS
    # =========================================================================

    df_results = pd.DataFrame(results)
    df_results.to_csv("rossler_knn_results.csv", index=False)
    print("\nResults saved to rossler_knn_results.csv")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY BY K VALUE")
    print("=" * 60)

    summary = (
        df_results.groupby(["k", "Method"])[["F1", "TPR", "FPR", "TP", "FP", "FN", "Time"]]
        .mean()
        .reset_index()
    )

    for k in K_VALUES:
        print(f"\nk = {k}")
        print("-" * 50)
        k_data = summary[summary["k"] == k]
        print(
            k_data[["Method", "F1", "TPR", "FPR", "TP", "FP", "FN", "Time"]].to_string(
                index=False, float_format="%.3f"
            )
        )

    # Plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # F1 Score
    sns.lineplot(
        data=df_results,
        x="k",
        y="F1",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=axes[0],
        errorbar=("ci", 68),
    )
    axes[0].set_title("F1 Score vs k (KNN neighbors)")
    axes[0].set_xlabel("k (number of neighbors)")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_ylim(0, 1)

    # TPR
    sns.lineplot(
        data=df_results,
        x="k",
        y="TPR",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=axes[1],
        errorbar=("ci", 68),
    )
    axes[1].set_title("True Positive Rate vs k")
    axes[1].set_xlabel("k (number of neighbors)")
    axes[1].set_ylabel("TPR")
    axes[1].set_ylim(0, 1)
    axes[1].get_legend().remove()

    # FPR
    sns.lineplot(
        data=df_results,
        x="k",
        y="FPR",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=axes[2],
        errorbar=("ci", 68),
    )
    axes[2].set_title("False Positive Rate vs k")
    axes[2].set_xlabel("k (number of neighbors)")
    axes[2].set_ylabel("FPR")
    axes[2].set_ylim(0, 1)
    axes[2].get_legend().remove()

    plt.tight_layout()
    plt.savefig("rossler_knn_analysis.png", dpi=300, bbox_inches="tight")
    print("\nPlot saved to rossler_knn_analysis.png")
