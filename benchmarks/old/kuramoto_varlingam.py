"""
Run VARLiNGAM on Kuramoto data, then compile all Kuramoto results.
"""
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
import time
import warnings

import lingam

from causationentropy.datasets.synthetic import (
    generate_graph_topology,
    simulate_kuramoto,
)

warnings.filterwarnings("ignore")


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
# CONFIG (matches kuramoto.py)
# =========================================================================

GLOBAL_PARAMS = {
    "n_nodes": 10,
    "T": 5000,
    "n_trials": 3,
    "dt": 0.05,
    "omega_std": 1.0,
    "phase_noise_std": 0.02,
    "burn_in": 50,
    "normalize_by_indegree": False,
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

print("=" * 70)
print("Kuramoto VARLiNGAM Benchmark")
print(f"{n_nodes} nodes, {GLOBAL_PARAMS['n_trials']} trials per condition")
print(f"Total iterations: {total_iters}")
print("=" * 70)

pbar = tqdm(total=total_iters, desc="Progress")

for exp_name, config in EXPERIMENTS.items():
    param_name = config["vary_param"]
    defaults = config["defaults"]

    for val in config["values"]:
        current_params = defaults.copy()
        current_params[param_name] = val

        for trial in range(GLOBAL_PARAMS["n_trials"]):
            seed = 42 + (trial * 100)
            np.random.seed(seed)

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

            n_true = int(true_adj.sum())
            common = {"Experiment": exp_name, "Parameter": val, "Trial": trial}

            # ----- VARLiNGAM -----
            start = time.time()
            model = lingam.VARLiNGAM(lags=1)
            model.fit(theta)

            # Extract adjacency from lag-1 matrices
            # model.adjacency_matrices_ is list of (n, n) for lag 0, 1, ...
            adj_var = np.zeros((n_nodes, n_nodes), dtype=int)
            for lag_idx, B_lag in enumerate(model.adjacency_matrices_):
                if lag_idx == 0:
                    continue  # skip contemporaneous
                # B_lag[i, j] != 0 means j -> i at this lag
                adj_var = adj_var | (np.abs(B_lag) > 1e-8).astype(int)
            # Remove diagonal
            np.fill_diagonal(adj_var, 0)

            m = compute_metrics(adj_var, true_adj, time.time() - start)
            m.update(common)
            m["Method"] = "VARLiNGAM"
            results.append(m)

            print(f"  {config['label']}={val}, Trial {trial}  "
                  f"true={n_true}  TP={m['TP']} FP={m['FP']} FN={m['FN']} "
                  f"F1={m['F1']:.3f}")

            pbar.update(1)

pbar.close()

df = pd.DataFrame(results)
df.to_csv("benchmarks/results/kuramoto_varlingam_results.csv", index=False)

print("\n" + "=" * 70)
print("VARLiNGAM SUMMARY")
print("=" * 70)
summary = df.groupby("Method")[["TPR", "FPR", "F1"]].agg(["mean", "std"]).reset_index()
summary.columns = ["Method", "TPR_mean", "TPR_std", "FPR_mean", "FPR_std", "F1_mean", "F1_std"]
print(summary.to_string(index=False, float_format="%.3f"))

# =========================================================================
# COMPILE ALL KURAMOTO RESULTS
# =========================================================================

print("\n\n" + "=" * 70)
print("ALL KURAMOTO METHODS COMPILED")
print("=" * 70)

all_dfs = []

# Original oCSE + PCMCI (10 trials)
try:
    df_orig = pd.read_csv("benchmarks/results/kuramoto_simplified_results.csv")
    # Rename to match
    df_orig = df_orig.rename(columns={"Method": "Method"})
    all_dfs.append(df_orig[["TPR", "FPR", "F1", "Method", "Experiment", "Parameter", "Trial"]])
except Exception as e:
    print(f"  Skipping original: {e}")

# Extra methods (LASSO, Granger, Koopman)
try:
    df_extra = pd.read_csv("benchmarks/results/kuramoto_extra_results.csv")
    all_dfs.append(df_extra[["TPR", "FPR", "F1", "Method", "Experiment", "Parameter", "Trial"]])
except Exception as e:
    print(f"  Skipping extra: {e}")

# VARLiNGAM
all_dfs.append(df[["TPR", "FPR", "F1", "Method", "Experiment", "Parameter", "Trial"]])

df_all = pd.concat(all_dfs, ignore_index=True)

# Rename for consistency
df_all["Method"] = df_all["Method"].replace({
    "OCE (gaussian)": "oCSE (basis)",
    "PCMCI (ParCorr)": "PCMCI (basis)",
})

combined = df_all.groupby("Method")[["TPR", "FPR", "F1"]].agg(["mean", "std"]).reset_index()
combined.columns = ["Method", "TPR_mean", "TPR_std", "FPR_mean", "FPR_std", "F1_mean", "F1_std"]
combined = combined.sort_values("F1_mean", ascending=False)

print(combined.to_string(index=False, float_format="%.3f"))

df_all.to_csv("benchmarks/results/kuramoto_all_results.csv", index=False)
print(f"\nAll results saved to benchmarks/results/kuramoto_all_results.csv")
