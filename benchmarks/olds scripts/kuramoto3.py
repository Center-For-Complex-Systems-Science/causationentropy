import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings

# Tigramite imports
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# User Library Imports
from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx

# Import our enhanced recovery functions
import sys

sys.path.insert(0, "/home/claude")
from enhanced_recovery import (
    recover_node_adj_simple,
    recover_node_adj_with_companion,
    analyze_lag_structure,
)

warnings.filterwarnings("ignore")


# =============================================================================
# KURAMOTO SIMULATION (same as before)
# =============================================================================


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def simulate_kuramoto(
    G,
    T,
    dt=0.05,
    rho=0.7,
    seed=0,
    omega_mean=0.0,
    omega_std=1.0,
    phase_noise_std=0.02,
    burn_in=50,
    normalize_by_indegree=True,
):
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes).astype(float)

    omega = rng.normal(loc=omega_mean, scale=omega_std, size=n)
    theta = rng.uniform(low=-np.pi, high=np.pi, size=n)

    if normalize_by_indegree:
        indeg = A.sum(axis=0)
        indeg_safe = np.where(indeg > 0, indeg, 1.0)
    else:
        indeg_safe = np.ones(n)

    total_steps = burn_in + T
    out = np.zeros((T, n), dtype=float)

    for step in range(total_steps):
        coupling = (A.T * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)
        dtheta = omega + (rho * coupling / indeg_safe)
        noise = rng.normal(loc=0.0, scale=phase_noise_std, size=n)
        theta = theta + dt * dtheta + noise
        theta = wrap_to_pi(theta)

        if step >= burn_in:
            out[step - burn_in] = theta

    return out, {"omega": omega, "dt": dt, "rho": rho}


def phase_velocity(theta, dt):
    dtheta = wrap_to_pi(theta[1:] - theta[:-1])
    return dtheta / dt


def build_kuramoto_basis(theta, dt):
    T, n = theta.shape
    v = phase_velocity(theta, dt)
    th = theta[:-1]
    S = np.zeros((T - 1, n * (n - 1)), dtype=float)

    basis_map = []
    names = [f"v{i}" for i in range(n)]

    col = 0
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            S[:, col] = np.sin(th[:, j] - th[:, i])
            basis_map.append((i, j))
            names.append(f"s_{i}<-{j}")
            col += 1

    X = np.concatenate([v, S], axis=1)
    meta = {
        "n": n,
        "dt": dt,
        "pred_offset": n,
        "basis_map": basis_map,
        "p": X.shape[1],
    }
    return X, meta, names


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 10,
    "T": 600,
    "n_trials": 5,
    "alpha": 0.05,
    "tau_max": 2,  # Increased to 2 so we can use companion structure!
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
        # "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "values": [0.1, 0.3, 0.5, 0.7],
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


# =============================================================================
# HELPERS
# =============================================================================


def generate_graph_topology(topo_type, n_nodes, params, seed):
    if topo_type == "Erdos-Renyi":
        p = params.get("p_edge", 0.2)
        return nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    if topo_type == "Scale-Free":
        m = params.get("m_attachment", 1)
        G_und = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        return nx.DiGraph(G_und)
    if topo_type == "Small-World":
        k = params.get("k_neighbors", 2)
        p = params.get("p_rewire", 0.1)
        G_und = nx.connected_watts_strogatz_graph(n_nodes, k, p, tries=100, seed=seed)
        return nx.DiGraph(G_und)
    raise ValueError(f"Unknown topology: {topo_type}")


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
    shd = fp + fn

    return {
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "F1": f1,
        "SHD": shd,
        "Time": time_taken,
    }


# =============================================================================
# MAIN
# =============================================================================

results = []

print(f"Starting Kuramoto-basis Experiments with companion matrix validation")
print(
    f"{GLOBAL_PARAMS['n_nodes']} nodes, tau_max={GLOBAL_PARAMS['tau_max']}, "
    f"{GLOBAL_PARAMS['n_trials']} trials..."
)

total_iters = sum(
    len(cfg["values"]) * GLOBAL_PARAMS["n_trials"] for cfg in EXPERIMENTS.values()
)
pbar = tqdm(total=total_iters, desc="Total progress")

# Flag to print detailed analysis for first trial
first_trial_analyzed = False

for exp_name, config in EXPERIMENTS.items():
    print(f"\n--- Running Experiment: {exp_name} ---")

    param_name = config["vary_param"]
    param_values = config["values"]
    defaults = config["defaults"]

    for val in param_values:
        current_params = defaults.copy()
        current_params[param_name] = val

        for trial in range(GLOBAL_PARAMS["n_trials"]):
            seed = 42 + (trial * 100)

            # Generate graph and simulate
            G = generate_graph_topology(
                config["type"], GLOBAL_PARAMS["n_nodes"], current_params, seed
            )
            true_adj = nx.to_numpy_array(G).astype(int)

            theta, _ = simulate_kuramoto(
                G=G,
                T=GLOBAL_PARAMS["T"],
                dt=GLOBAL_PARAMS["dt"],
                rho=current_params.get("rho", 0.7),
                seed=seed,
                omega_mean=0.0,
                omega_std=GLOBAL_PARAMS["omega_std"],
                phase_noise_std=GLOBAL_PARAMS["phase_noise_std"],
                burn_in=GLOBAL_PARAMS["burn_in"],
                normalize_by_indegree=GLOBAL_PARAMS["normalize_by_indegree"],
            )

            X, basis_meta, var_names = build_kuramoto_basis(
                theta, dt=GLOBAL_PARAMS["dt"]
            )
            T_eff = X.shape[0]

            dataframe = pp.DataFrame(
                X, datatime={0: np.arange(T_eff)}, var_names=var_names
            )

            # -------------------------------------------------------
            # METHOD 1: OCE with simple recovery
            # -------------------------------------------------------
            start_time = time.time()
            network = discover_network(
                data=X, max_lag=GLOBAL_PARAMS["tau_max"], information="gaussian"
            )

            pred_adj_simple = recover_node_adj_simple(network, basis_meta)
            metrics = compute_metrics(
                pred_adj_simple, true_adj, time.time() - start_time
            )
            metrics.update(
                {
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "OCE (Simple)",
                    "Trial": trial,
                }
            )
            results.append(metrics)

            # -------------------------------------------------------
            # METHOD 2: OCE with companion-aware recovery
            # -------------------------------------------------------
            if GLOBAL_PARAMS["tau_max"] > 1:
                start_time = time.time()
                pred_adj_companion = recover_node_adj_with_companion(
                    network,
                    basis_meta,
                    tau_max=GLOBAL_PARAMS["tau_max"],
                    use_lag_consistency=True,
                    consistency_weight=0.3,
                    threshold=0.01,
                )
                metrics = compute_metrics(
                    pred_adj_companion, true_adj, time.time() - start_time
                )
                metrics.update(
                    {
                        "Experiment": exp_name,
                        "Parameter": val,
                        "Method": "OCE (Companion)",
                        "Trial": trial,
                    }
                )
                results.append(metrics)

            # -------------------------------------------------------
            # METHOD 3: PCMCI with simple recovery
            # -------------------------------------------------------
            start_time = time.time()
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
            pcmci_res = pcmci.run_pcmci(
                tau_max=GLOBAL_PARAMS["tau_max"], pc_alpha=GLOBAL_PARAMS["alpha"]
            )
            graph_nx = pcmci_to_networkx(pcmci_res)

            pred_adj_simple = recover_node_adj_simple(graph_nx, basis_meta)
            metrics = compute_metrics(
                pred_adj_simple, true_adj, time.time() - start_time
            )
            metrics.update(
                {
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "PCMCI (Simple)",
                    "Trial": trial,
                }
            )
            results.append(metrics)

            # -------------------------------------------------------
            # METHOD 4: PCMCI with companion-aware recovery
            # -------------------------------------------------------
            if GLOBAL_PARAMS["tau_max"] > 1:
                start_time = time.time()
                pred_adj_companion = recover_node_adj_with_companion(
                    graph_nx,
                    basis_meta,
                    tau_max=GLOBAL_PARAMS["tau_max"],
                    use_lag_consistency=True,
                    consistency_weight=0.3,
                    threshold=0.01,
                )
                metrics = compute_metrics(
                    pred_adj_companion, true_adj, time.time() - start_time
                )
                metrics.update(
                    {
                        "Experiment": exp_name,
                        "Parameter": val,
                        "Method": "PCMCI (Companion)",
                        "Trial": trial,
                    }
                )
                results.append(metrics)

            # Diagnostic analysis for first trial
            if not first_trial_analyzed and GLOBAL_PARAMS["tau_max"] > 1:
                print("\n" + "=" * 80)
                print("DIAGNOSTIC ANALYSIS (First Trial)")
                print("=" * 80)
                print("\nOCE Results:")
                analyze_lag_structure(network, basis_meta, GLOBAL_PARAMS["tau_max"])
                print("\nPCMCI Results:")
                analyze_lag_structure(graph_nx, basis_meta, GLOBAL_PARAMS["tau_max"])
                first_trial_analyzed = True

            pbar.update(1)

pbar.close()

# =============================================================================
# SAVE + PLOTS
# =============================================================================

df_results = pd.DataFrame(results)
df_results.to_csv("kuramoto_companion_results.csv", index=False)
print("\nResults saved to kuramoto_companion_results.csv")

# Create comparison plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
experiments_list = list(EXPERIMENTS.keys())

fig, axes = plt.subplots(
    len(experiments_list), 2, figsize=(14, 5 * len(experiments_list))
)
if len(experiments_list) == 1:
    axes = np.array([axes])

for idx, exp_name in enumerate(experiments_list):
    exp_data = df_results[df_results["Experiment"] == exp_name]
    config = EXPERIMENTS[exp_name]

    ax_f1 = axes[idx, 0]
    sns.lineplot(
        data=exp_data,
        x="Parameter",
        y="F1",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=ax_f1,
        ci=68,
    )
    ax_f1.set_title(f"{exp_name}: F1 Score")
    ax_f1.set_xlabel(config["label"])
    ax_f1.set_ylabel("F1")

    ax_tpr = axes[idx, 1]
    sns.lineplot(
        data=exp_data,
        x="Parameter",
        y="TPR",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=ax_tpr,
        ci=68,
    )
    ax_tpr.set_title(f"{exp_name}: True Positive Rate")
    ax_tpr.set_xlabel(config["label"])
    ax_tpr.set_ylabel("TPR")

    if idx == 0:
        ax_f1.legend(bbox_to_anchor=(1.05, 1.2), loc="upper left", ncol=2)
        ax_tpr.get_legend().remove()
    else:
        ax_f1.get_legend().remove()
        ax_tpr.get_legend().remove()

plt.tight_layout()
plt.savefig("kuramoto_companion_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Summary statistics
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON: Simple vs Companion Recovery")
print("=" * 80)

summary = df_results.groupby(["Experiment", "Method"])[
    ["F1", "TPR", "FPR", "Time"]
].mean()
summary = summary.reset_index().sort_values(
    ["Experiment", "F1"], ascending=[True, False]
)

for exp_name in experiments_list:
    print(f"\nExperiment: {exp_name}")
    print("-" * 80)
    exp_summary = summary[summary["Experiment"] == exp_name]
    print(
        exp_summary[["Method", "F1", "TPR", "FPR", "Time"]].to_string(
            index=False, float_format="%.3f"
        )
    )

    # Compare simple vs companion for each base method
    if GLOBAL_PARAMS["tau_max"] > 1:
        for base_method in ["OCE", "PCMCI"]:
            simple = exp_summary[exp_summary["Method"] == f"{base_method} (Simple)"]
            companion = exp_summary[
                exp_summary["Method"] == f"{base_method} (Companion)"
            ]
            if len(simple) > 0 and len(companion) > 0:
                f1_simple = simple["F1"].values[0]
                f1_companion = companion["F1"].values[0]
                improvement = (
                    ((f1_companion - f1_simple) / f1_simple * 100)
                    if f1_simple > 0
                    else 0
                )
                print(f"  → {base_method} F1 improvement: {improvement:+.1f}%")

print("\n" + "=" * 80)
print("Key Insights:")
print("- 'Simple' recovery uses only direct lag-1 edges")
print("- 'Companion' recovery validates using lag-2 consistency (A_2 ≈ A_1²)")
print("- Positive improvement suggests companion structure helps filter spurious edges")
print("=" * 80)
