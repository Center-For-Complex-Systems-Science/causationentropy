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
from tigramite.independence_tests.parcorr import ParCorr

# User Library Imports
from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx

warnings.filterwarnings("ignore")


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
# KURAMOTO SIMULATION + CORRECT-BASIS TRANSFORM
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
    """
    Build expanded data matrix with velocity and coupling terms.

    Returns:
        X: (T-1, n + n*(n-1)) data matrix
        meta: metadata dict with 'n', 'basis_map', etc.
        var_names: list of variable names
    """
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


def build_kuramoto_basis_lag1(theta, dt):
    T, n = theta.shape
    v = phase_velocity(theta, dt)  # shape (T-1, n)
    th = theta[:-1]  # shape (T-1, n)

    # build S aligned with v(t)
    S = []
    basis_map = []
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            S.append(np.sin(th[:, j] - th[:, i]))
            basis_map.append((i, j))
    S = np.stack(S, axis=1)  # (T-1, n*(n-1))

    # SHIFT: predictors at t, targets at t+1
    v_next = v[1:]  # (T-2, n)
    S_prev = S[:-1]  # (T-2, n*(n-1))

    X = np.concatenate([v_next, S_prev], axis=1)

    var_names = [f"v{i}" for i in range(n)] + [f"s_{i}<-{j}" for (i, j) in basis_map]
    meta = {"n": n, "dt": dt, "pred_offset": n, "basis_map": basis_map, "p": X.shape[1]}
    return X, meta, var_names


def extract_node_adjacency_from_basis_oce(graph_nx, n, basis_map, var_names):
    """
    Extract node-level adjacency from OCE expanded basis graph.

    The discover_network function returns a MultiDiGraph where:
    - Nodes are variable names (strings): "v0", "v1", ..., "s_0<-1", "s_0<-2", ...
    - Edges have a 'lag' attribute indicating the time lag
    - We want edges: s_{i<-j} (at any lag) -> v_i (predicting velocity i)
    - Such an edge indicates j -> i in the original Kuramoto network

    Parameters
    ----------
    graph_nx : networkx.MultiDiGraph
        Graph from discover_network() on expanded variables
    n : int
        Number of original nodes
    basis_map : list of (i, j) tuples
        Maps coupling variable to (target, source) pair
    var_names : list of str
        Variable names in order

    Returns
    -------
    A : (n, n) np.ndarray
        Binary adjacency where A[j, i] = 1 means j -> i
    """
    A_node = np.zeros((n, n), dtype=int)

    # Get velocity variable names
    velocity_vars = [f"v{i}" for i in range(n)]

    # For each coupling term s_{i<-j}, check if it has an edge to v_i
    for idx, (i, j) in enumerate(basis_map):
        coupling_var_name = var_names[n + idx]  # e.g., "s_0<-1"
        velocity_var_name = velocity_vars[i]  # e.g., "v0"

        # Check if there's any edge from coupling term to velocity
        # (MultiDiGraph allows multiple edges, so we check if any exist)
        if graph_nx.has_edge(coupling_var_name, velocity_var_name):
            # Edge exists: s_{i<-j} -> v_i, so j causes i
            A_node[j, i] = 1

    return A_node


def extract_node_adjacency_from_basis_pcmci(graph_nx, n, basis_map):
    """
    Extract node-level adjacency from PCMCI expanded basis graph.

    The pcmci_to_networkx function returns a MultiDiGraph where:
    - Nodes are integers: 0, 1, ..., p-1 where p = n + n*(n-1)
    - First n nodes (0 to n-1) are velocities v0, v1, ..., v_{n-1}
    - Next n*(n-1) nodes are coupling terms in basis_map order
    - Edges have a 'lag' attribute
    - We want edges: coupling_node -> velocity_node to infer causality

    Parameters
    ----------
    graph_nx : networkx.MultiDiGraph
        Graph from pcmci_to_networkx()
    n : int
        Number of original nodes
    basis_map : list of (i, j) tuples
        Maps coupling variable index to (target, source) pair

    Returns
    -------
    A : (n, n) np.ndarray
        Binary adjacency where A[j, i] = 1 means j -> i
    """
    A_node = np.zeros((n, n), dtype=int)

    # For each coupling term, check if it has an edge to the corresponding velocity
    for idx, (i, j) in enumerate(basis_map):
        coupling_node_idx = n + idx  # Node index for s_{i<-j}
        velocity_node_idx = i  # Node index for v_i

        # Check if there's any edge from coupling to velocity (at any lag)
        if graph_nx.has_edge(coupling_node_idx, velocity_node_idx):
            # Edge exists: coupling term predicts velocity, so j causes i
            A_node[j, i] = 1

    return A_node


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 3,
    "T": 100,
    "n_trials": 1,
    "alpha": 0.05,
    "tau_max": 1,
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

print(f"Starting Kuramoto-basis Experiments (Simplified)")
print(f"{GLOBAL_PARAMS['n_nodes']} nodes, {GLOBAL_PARAMS['n_trials']} trials...")

total_iters = sum(
    len(cfg["values"]) * GLOBAL_PARAMS["n_trials"] for cfg in EXPERIMENTS.values()
)
pbar = tqdm(total=total_iters, desc="Total progress")

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

            X, basis_meta, var_names = build_kuramoto_basis_lag1(
                theta, dt=GLOBAL_PARAMS["dt"]
            )
            T_eff = X.shape[0]

            dataframe = pp.DataFrame(
                X, datatime={0: np.arange(T_eff)}, var_names=var_names
            )

            # -------------------------------------------------------
            # METHOD 1: OCE (causationentropy)
            # -------------------------------------------------------
            start_time = time.time()

            # Convert to DataFrame with proper column names for discover_network
            X_df = pd.DataFrame(X, columns=var_names)

            with suppress_stdout():
                network = discover_network(
                    data=X_df,  # Pass DataFrame instead of array
                    max_lag=GLOBAL_PARAMS["tau_max"],
                    information="gaussian",
                )

            # Extract node-level adjacency using corrected function
            pred_adj = extract_node_adjacency_from_basis_oce(
                network, basis_meta["n"], basis_meta["basis_map"], var_names
            )

            metrics = compute_metrics(pred_adj, true_adj, time.time() - start_time)
            metrics.update(
                {
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "OCE (gaussian)",
                    "Trial": trial,
                }
            )
            results.append(metrics)

            # -------------------------------------------------------
            # METHOD 2: PCMCI (Tigramite)
            # -------------------------------------------------------
            start_time = time.time()
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
            pcmci_res = pcmci.run_pcmci(
                tau_max=GLOBAL_PARAMS["tau_max"], pc_alpha=GLOBAL_PARAMS["alpha"]
            )
            graph_nx = pcmci_to_networkx(pcmci_res)

            # Extract node-level adjacency using corrected function
            pred_adj_pcmci = extract_node_adjacency_from_basis_pcmci(
                graph_nx, basis_meta["n"], basis_meta["basis_map"]
            )

            metrics = compute_metrics(
                pred_adj_pcmci, true_adj, time.time() - start_time
            )
            metrics.update(
                {
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "PCMCI (ParCorr)",
                    "Trial": trial,
                }
            )
            results.append(metrics)

            pbar.update(1)

pbar.close()

# =============================================================================
# SAVE + PLOTS
# =============================================================================

df_results = pd.DataFrame(results)
df_results.to_csv("kuramoto_simplified_results.csv", index=False)
print("\nResults saved to kuramoto_simplified_results.csv")

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
        ax_f1.legend(bbox_to_anchor=(1.05, 1.2), loc="upper left")
        ax_tpr.get_legend().remove()
    else:
        ax_f1.get_legend().remove()
        ax_tpr.get_legend().remove()

plt.tight_layout()
plt.savefig("kuramoto_simplified_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n" + "=" * 80)
print("SUMMARY")
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
