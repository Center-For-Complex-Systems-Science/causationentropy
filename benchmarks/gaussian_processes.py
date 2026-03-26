"""
Coupled Gaussian Process Causal Discovery Benchmark
====================================================
Compares four methods on linear stochastic Gaussian processes
across multiple graph topologies:
  1. oCSE (Optimal Causation Entropy) — Gaussian estimator
  2. PCMCI (ParCorr) — partial correlation
  3. VARLiNGAM — Vector Autoregressive LiNGAM
  4. Linear Granger — F-test on lagged regressors

Sweeps: ER density, coupling strength, scale-free hubs, small-world rewiring.
"""
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
import time
import warnings

import lingam
from sklearn.linear_model import LinearRegression
from scipy import stats

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx
from causationentropy.datasets.synthetic import (
    generate_graph_topology,
    linear_stochastic_gaussian_process,
)

warnings.filterwarnings("ignore")


# =============================================================================
# LINEAR GRANGER CAUSALITY
# =============================================================================

def linear_granger(data, n_nodes, alpha=0.05):
    """
    Pairwise linear Granger causality.
    For each target i, F-test whether lagged X_j improves prediction.
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    Y_all = data[1:]
    X_all = data[:-1]

    for i in range(n_nodes):
        y = Y_all[:, i]
        X_restricted = X_all[:, [i]]
        reg_r = LinearRegression().fit(X_restricted, y)
        rss_r = np.sum((y - reg_r.predict(X_restricted)) ** 2)

        for j in range(n_nodes):
            if j == i:
                continue
            X_unrest = X_all[:, [i, j]]
            reg_u = LinearRegression().fit(X_unrest, y)
            rss_u = np.sum((y - reg_u.predict(X_unrest)) ** 2)

            n_obs = len(y)
            f_stat = ((rss_r - rss_u) / 1) / (rss_u / (n_obs - 2))
            p_value = 1 - stats.f.cdf(f_stat, 1, n_obs - 2)
            if p_value < alpha:
                adj[j, i] = 1

    return adj


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
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * (precision * tpr) / (precision + tpr)
        if (precision + tpr) > 0
        else 0.0
    )
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "TPR": tpr, "FPR": fpr, "Precision": precision,
        "F1": f1, "Time": time_taken,
    }


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 10,
    "T": 300,
    "n_trials": 5,
    "max_lag": 1,
    "alpha": 0.05,
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
    "ScaleFree_Hubs": {
        "type": "Scale-Free",
        "vary_param": "m_attachment",
        "label": "Attachment Edges (m)",
        "values": [1, 2, 3, 4],
        "defaults": {"rho": 0.7},
    },
    "SmallWorld_Rewiring": {
        "type": "Small-World",
        "vary_param": "p_rewire",
        "label": "Rewiring Probability (p)",
        "values": [0.0, 0.1, 0.3, 0.5, 0.8, 1.0],
        "defaults": {"rho": 0.7, "k_neighbors": 2},
    },
}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    os.makedirs("benchmarks/results", exist_ok=True)

    results = []
    n_nodes = GLOBAL_PARAMS["n_nodes"]

    total_iters = sum(
        len(cfg["values"]) * GLOBAL_PARAMS["n_trials"]
        for cfg in EXPERIMENTS.values()
    )

    print("=" * 70)
    print("Coupled Gaussian Process Causal Discovery Benchmark")
    print(f"Methods: oCSE, PCMCI, VARLiNGAM, Linear Granger")
    print(f"{n_nodes} nodes, {GLOBAL_PARAMS['n_trials']} trials per condition")
    print(f"Total iterations: {total_iters}")
    print("=" * 70)

    pbar = tqdm(total=total_iters, desc="Progress")

    for exp_name, config in EXPERIMENTS.items():
        print(f"\n--- {exp_name} ---")
        param_name = config["vary_param"]
        defaults = config["defaults"]

        for val in config["values"]:
            current_params = defaults.copy()
            current_params[param_name] = val

            for trial in range(GLOBAL_PARAMS["n_trials"]):
                seed = 42 + (trial * 100)
                np.random.seed(seed)

                # Generate graph
                G = generate_graph_topology(
                    config["type"], n_nodes, current_params, seed
                )
                true_adj = nx.to_numpy_array(G)
                sim_p = nx.density(G)

                # Simulate coupled Gaussian process
                data, _ = linear_stochastic_gaussian_process(
                    rho=current_params.get("rho", 0.7),
                    n=n_nodes,
                    T=GLOBAL_PARAMS["T"],
                    p=sim_p,
                    seed=seed,
                    G=G,
                )

                var_names = [f"X{i}" for i in range(n_nodes)]
                n_true = int((true_adj != 0).sum())
                common = {
                    "Experiment": exp_name, "Parameter": val, "Trial": trial,
                }

                # ----- oCSE (Gaussian estimator) -----
                start = time.time()
                network = discover_network(
                    data=data,
                    max_lag=GLOBAL_PARAMS["max_lag"],
                    information="gaussian",
                )
                pred_adj = nx.to_numpy_array(network)
                m = compute_metrics(pred_adj, true_adj, time.time() - start)
                m.update(common)
                m["Method"] = "oCSE"
                results.append(m)

                # ----- PCMCI (ParCorr) -----
                start = time.time()
                dataframe = pp.DataFrame(
                    data,
                    datatime={0: np.arange(GLOBAL_PARAMS["T"])},
                    var_names=var_names,
                )
                pcmci = PCMCI(
                    dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0
                )
                pcmci_res = pcmci.run_pcmci(
                    tau_max=GLOBAL_PARAMS["max_lag"],
                    pc_alpha=GLOBAL_PARAMS["alpha"],
                )
                graph_nx = pcmci_to_networkx(pcmci_res)
                pred_adj = nx.to_numpy_array(graph_nx)
                m = compute_metrics(pred_adj, true_adj, time.time() - start)
                m.update(common)
                m["Method"] = "PCMCI"
                results.append(m)

                # ----- VARLiNGAM -----
                start = time.time()
                model = lingam.VARLiNGAM(lags=1)
                model.fit(data)
                adj_var = np.zeros((n_nodes, n_nodes), dtype=int)
                for lag_idx, B_lag in enumerate(model.adjacency_matrices_):
                    if lag_idx == 0:
                        continue
                    adj_var = adj_var | (np.abs(B_lag) > 1e-8).astype(int)
                np.fill_diagonal(adj_var, 0)
                m = compute_metrics(adj_var, true_adj, time.time() - start)
                m.update(common)
                m["Method"] = "VARLiNGAM"
                results.append(m)

                # ----- Linear Granger -----
                start = time.time()
                adj_granger = linear_granger(data, n_nodes, alpha=0.05)
                m = compute_metrics(adj_granger, true_adj, time.time() - start)
                m.update(common)
                m["Method"] = "Linear Granger"
                results.append(m)

                print(
                    f"  {config['label']}={val}, Trial {trial}  "
                    f"true={n_true}  "
                    f"oCSE F1={results[-4]['F1']:.3f}  "
                    f"PCMCI F1={results[-3]['F1']:.3f}  "
                    f"VARLiNGAM F1={results[-2]['F1']:.3f}  "
                    f"Granger F1={results[-1]['F1']:.3f}"
                )

                pbar.update(1)

    pbar.close()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("benchmarks/results/gaussian_process_results.csv", index=False)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = (
        df.groupby("Method")[["TPR", "FPR", "F1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "Method", "TPR_mean", "TPR_std",
        "FPR_mean", "FPR_std", "F1_mean", "F1_std",
    ]
    summary = summary.sort_values("F1_mean", ascending=False)
    print(summary.to_string(index=False, float_format="%.3f"))

    # Per-experiment summary
    for exp_name in EXPERIMENTS:
        exp_data = df[df["Experiment"] == exp_name]
        print(f"\n--- {exp_name} ---")
        exp_summary = (
            exp_data.groupby("Method")[["F1", "TPR", "FPR"]]
            .mean()
            .sort_values("F1", ascending=False)
        )
        print(exp_summary.to_string(float_format="%.3f"))

    print(f"\nResults saved to benchmarks/results/gaussian_process_results.csv")

    # =========================================================================
    # PLOTS
    # =========================================================================
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.size"] = 11

    methods = ["oCSE", "PCMCI", "VARLiNGAM", "Linear Granger"]
    colors = {"oCSE": "#e74c3c", "PCMCI": "#3498db",
              "VARLiNGAM": "#2ecc71", "Linear Granger": "#9b59b6"}
    markers = {"oCSE": "o", "PCMCI": "s", "VARLiNGAM": "D", "Linear Granger": "^"}

    n_exp = len(EXPERIMENTS)
    fig, axes = plt.subplots(n_exp, 3, figsize=(18, 5 * n_exp), squeeze=False)

    for row, (exp_name, config) in enumerate(EXPERIMENTS.items()):
        exp_data = df[df["Experiment"] == exp_name]

        for col, metric in enumerate(["F1", "TPR", "FPR"]):
            ax = axes[row, col]
            for method in methods:
                mdata = exp_data[exp_data["Method"] == method]
                grouped = mdata.groupby("Parameter")[metric]
                mean = grouped.mean()
                std = grouped.std()
                ax.errorbar(
                    mean.index, mean.values, yerr=std.values,
                    label=method, color=colors[method],
                    marker=markers[method], capsize=3, linewidth=2,
                )
            ax.set_xlabel(config["label"])
            ax.set_ylabel(metric)
            ax.set_title(f"{exp_name}: {metric}")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(
        "benchmarks/results/gaussian_process_plots.png", dpi=300, bbox_inches="tight"
    )
    print("Plots saved to benchmarks/results/gaussian_process_plots.png")
