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
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn

# User Library Imports
from causationentropy import discover_network
from causationentropy.datasets.synthetic import linear_stochastic_gaussian_process
from causationentropy.core.stats import Compute_TPR_FPR
from causationentropy.graph import pcmci_to_networkx

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore")


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
    """
    Simulate a (directed) Kuramoto network on graph G.

    Returns:
        data: np.ndarray of shape (T, n_nodes), where each column is theta_i(t)
              (wrapped to [-pi, pi]).
        meta: dict with omega, dt, etc.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))  # adjacency matrix (n x n)

    # Natural frequencies
    omega = rng.normal(loc=omega_mean, scale=omega_std, size=n)

    # Initial phases
    theta = rng.uniform(low=-np.pi, high=np.pi, size=n)

    # Precompute indegree normalization if desired
    if normalize_by_indegree:
        indeg = A.sum(
            axis=0
        )  # column-sum: incoming edges to each node i if A_{ji} means j->i
        # To be consistent with your current A_ij usage in the rest of the script,
        # we interpret A_ij as i->j (row i to col j). For Kuramoto, we want influence into i:
        # sum over j: A_{ji} * sin(theta_j - theta_i).
        # Thus indegree for i is sum_j A_{ji} = column sum.
        indeg_safe = np.where(indeg > 0, indeg, 1.0)
    else:
        indeg_safe = np.ones(n)

    # Total steps includes burn-in
    total_steps = burn_in + T
    out = np.zeros((T, n), dtype=float)

    for step in range(total_steps):
        # Coupling term: sum_j A_{ji} sin(theta_j - theta_i)
        # A_T = A.T has entries A_{ji} in position (i, j)
        coupling = (A.T * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)

        # Normalize and scale
        dtheta = omega + (rho * coupling / indeg_safe)

        # Euler-Maruyama style phase noise
        noise = rng.normal(loc=0.0, scale=phase_noise_std, size=n)

        theta = theta + dt * dtheta + noise

        # Wrap to [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        if step >= burn_in:
            out[step - burn_in] = theta

    meta = {"omega": omega, "dt": dt, "rho": rho}
    return out, meta


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 10,
    "T": 300,
    "n_trials": 5,  # Number of random seeds per parameter value (for error bars)
    "max_lag": 1,
    "alpha": 0.05,
}

# Define the "Sweeps" - What variable are we changing?
# Format: 'experiment_name': {'param_name': 'x_label', 'values': [...], 'fixed_params': {...}}
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
        "values": [1, 2, 3, 4],  # m must be < n_nodes
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
# HELPER FUNCTIONS
# =============================================================================


def generate_graph_topology(topo_type, n_nodes, params, seed):
    """Factory function for graph generation based on type and params."""
    if topo_type == "Erdos-Renyi":
        p = params.get("p_edge", 0.2)
        return nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)

    elif topo_type == "Scale-Free":
        m = params.get("m_attachment", 1)
        # Barabasi-Albert is undirected by default in NX, convert to directed
        G_und = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        return nx.DiGraph(G_und)

    elif topo_type == "Small-World":
        k = params.get("k_neighbors", 2)
        p = params.get("p_rewire", 0.1)
        # Watts-Strogatz is undirected, convert to directed
        G_und = nx.connected_watts_strogatz_graph(n_nodes, k, p, tries=100, seed=seed)
        return nx.DiGraph(G_und)

    else:
        raise ValueError(f"Unknown topology: {topo_type}")


def compute_metrics(predicted_adj, true_adj, time_taken):
    """Computes publication metrics."""
    # Ensure binary matrices
    B = (predicted_adj != 0).astype(int)
    A = (true_adj != 0).astype(int)

    # Basic counts
    tp = np.sum((B == 1) & (A == 1))
    fp = np.sum((B == 1) & (A == 0))
    tn = np.sum((B == 0) & (A == 0))
    fn = np.sum((B == 0) & (A == 1))

    # Rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    # Structural Hamming Distance (simplified for adjacency)
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
# MAIN EXECUTION ENGINE
# =============================================================================

results = []

print(
    f"Starting Experiments with {GLOBAL_PARAMS['n_nodes']} nodes over {GLOBAL_PARAMS['n_trials']} trials..."
)

for exp_name, config in EXPERIMENTS.items():
    print(f"\n--- Running Experiment: {exp_name} ---")

    param_name = config["vary_param"]
    param_values = config["values"]
    defaults = config["defaults"]

    # Progress bar for the parameter sweep
    pbar = tqdm(total=len(param_values) * GLOBAL_PARAMS["n_trials"])

    for val in param_values:
        # Construct current parameters
        current_params = defaults.copy()
        current_params[param_name] = val

        for trial in range(GLOBAL_PARAMS["n_trials"]):
            seed = 42 + (trial * 100)
            np.random.seed(seed)

            # 1. Generate Graph
            G = generate_graph_topology(
                config["type"], GLOBAL_PARAMS["n_nodes"], current_params, seed
            )
            true_adj = nx.to_numpy_array(G)

            # 2. Simulate Data
            # Note: Edge prob calculation for simulation physics (approximate for non-ER graphs)
            sim_p = nx.density(G)

            # 2. Simulate Data (Kuramoto)
            data, _ = simulate_kuramoto(
                G=G,
                T=GLOBAL_PARAMS["T"],
                dt=0.5,
                rho=current_params.get("rho", 0.7),
                seed=seed,
                omega_mean=0.0,
                omega_std=1.0,
                phase_noise_std=0.02,
                burn_in=50,
                normalize_by_indegree=False,
            )

            # Tigramite DataFrame format
            dataframe = pp.DataFrame(
                data,
                datatime={0: np.arange(GLOBAL_PARAMS["T"])},
                var_names=[f"X{i}" for i in range(GLOBAL_PARAMS["n_nodes"])],
            )

            # -------------------------------------------------------
            # METHOD 1: Optimal Causation Entropy (Your Method)
            # -------------------------------------------------------
            oce_estimators = ["gaussian", "knn", "kde"]  # , 'geometric_knn']

            for est in oce_estimators:
                start_time = time.time()
                # Passing the estimator parameter as requested
                network = discover_network(
                    data=data, max_lag=GLOBAL_PARAMS["max_lag"], information=est
                )

                pred_adj = nx.to_numpy_array(network)
                metrics = compute_metrics(pred_adj, true_adj, time.time() - start_time)

                metrics.update(
                    {
                        "Experiment": exp_name,
                        "Parameter": val,
                        "Method": f"OCE ({est})",
                        "Trial": trial,
                    }
                )
                results.append(metrics)

            # -------------------------------------------------------
            # METHOD 2: Tigramite Variants
            # -------------------------------------------------------
            tigramite_configs = [
                ("PCMCI (ParCorr)", ParCorr()),
                # ('PCMCI (GPDC)', GPDC()),
                # ('PCMCI (CMIknn)', CMIknn())
            ]

            for method_label, ind_test in tigramite_configs:
                start_time = time.time()

                pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ind_test, verbosity=0)
                pcmci_res = pcmci.run_pcmci(
                    tau_max=GLOBAL_PARAMS["max_lag"], pc_alpha=GLOBAL_PARAMS["alpha"]
                )

                # Helper from your code to convert
                graph_nx = pcmci_to_networkx(pcmci_res)
                pred_adj = nx.to_numpy_array(graph_nx)

                metrics = compute_metrics(pred_adj, true_adj, time.time() - start_time)

                metrics.update(
                    {
                        "Experiment": exp_name,
                        "Parameter": val,
                        "Method": method_label,
                        "Trial": trial,
                    }
                )
                results.append(metrics)

            pbar.update(1)
    pbar.close()

# Convert to DataFrame for Analysis
df_results = pd.DataFrame(results)

# Save raw results for publication data
df_results.to_csv("causal_discovery_simulation_results.csv", index=False)
print("\nSimulation complete. Results saved to CSV.")


# Set style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
experiments_list = list(EXPERIMENTS.keys())

# Create a figure with rows = experiments, cols = 2 (F1 Score, TPR)
fig, axes = plt.subplots(
    len(experiments_list), 2, figsize=(14, 5 * len(experiments_list))
)

if len(experiments_list) == 1:
    axes = np.array([axes])  # Handle single experiment case

for idx, exp_name in enumerate(experiments_list):
    exp_data = df_results[df_results["Experiment"] == exp_name]
    config = EXPERIMENTS[exp_name]

    # Left Column: F1 Score
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
        ci=68,  # Standard Error
    )
    ax_f1.set_title(f"{exp_name}: F1 Score Performance")
    ax_f1.set_xlabel(config["label"])
    ax_f1.set_ylabel("F1 Score")

    # Right Column: TPR (Power) or Precision
    ax_metric = axes[idx, 1]
    sns.lineplot(
        data=exp_data,
        x="Parameter",
        y="TPR",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=ax_metric,
        ci=68,
    )
    ax_metric.set_title(f"{exp_name}: True Positive Rate (Sensitivity)")
    ax_metric.set_xlabel(config["label"])
    ax_metric.set_ylabel("TPR")

    # Handle Legend: Only show on the first plot to avoid clutter, or put outside
    if idx == 0:
        ax_f1.legend(bbox_to_anchor=(1.05, 1.2), loc="upper left", ncol=4)
        ax_metric.get_legend().remove()
    else:
        ax_f1.get_legend().remove()
        ax_metric.get_legend().remove()

plt.tight_layout()
plt.savefig("comprehensive_causal_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 80)
print("AGGREGATED PERFORMANCE BY EXPERIMENT")
print("=" * 80)

# Group by Experiment and Method, calculate mean F1 and Runtime
summary = (
    df_results.groupby(["Experiment", "Method"])[["F1", "TPR", "FPR", "Time"]]
    .mean()
    .reset_index()
)
summary.to_csv("kuramoto_results.csv")
for exp_name in experiments_list:
    print(f"\nExperiment: {exp_name}")
    print("-" * 80)
    exp_summary = summary[summary["Experiment"] == exp_name].sort_values(
        "F1", ascending=False
    )
    print(
        exp_summary[["Method", "F1", "TPR", "FPR", "Time"]].to_string(
            index=False, float_format="%.3f"
        )
    )
