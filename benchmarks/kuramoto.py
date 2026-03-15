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
import pkg_resources
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, accuracy_score

# Tigramite imports
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# User Library Imports
from causationentropy import discover_network
from causationentropy.graph import pcmci_to_networkx
from causationentropy.core.plotting import plot_causal_network
from causationentropy.datasets.synthetic import (
    generate_graph_topology,
    simulate_kuramoto,
    prepare_kuramoto_data_for_causal_discovery,
    wrap_to_pi,
)

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
# KURAMOTO-SPECIFIC EXTRACTION FUNCTIONS
# =============================================================================


def extract_node_scores_from_basis_oce(graph_nx, n, basis_map, var_names):
    """
    Extracts continuous causal scores from the OCE graph on the expanded basis.

    After running causal discovery on the transformed data, this function
    translates the edge weights (CMI values) from the high-dimensional graph
    back into a continuous adjacency matrix (scores) for the original n-node
    Kuramoto network.

    An edge `s_{i<-j} -> v_i` in the basis graph implies a causal link `j -> i`
    in the original network. This function finds such edges and uses their
    CMI values as the score for the `j -> i` interaction. It also correctly
    handles the symmetry `sin(theta_j - theta_i) = -sin(theta_i - theta_j)`.

    Parameters
    ----------
    graph_nx : nx.MultiDiGraph
        The causal graph discovered by OCE on the transformed data.
    n : int
        The number of nodes in the original Kuramoto network.
    basis_map : list of tuple
        A map from sine-term index to the pair (target, source) of oscillators.
    var_names : list of str
        A list of variable names in the transformed data.

    Returns
    -------
    scores : np.ndarray
        An (n, n) matrix where `scores[j, i]` is the maximum CMI score for the
        causal link `j -> i`.
    """
    scores = np.zeros((n, n), dtype=float)
    velocity_vars = [f"v{i}" for i in range(n)]

    for i in range(n):
        velocity_var = velocity_vars[i]

        for idx, (target, source) in enumerate(basis_map):
            coupling_var = var_names[n + idx]

            if graph_nx.has_edge(coupling_var, velocity_var):
                edges = graph_nx[coupling_var][velocity_var]
                max_cmi = 0.0
                for key, edge_data in edges.items():
                    if edge_data["lag"] <= 1:
                        cmi = edge_data.get("cmi", 0.0)
                        max_cmi = max(max_cmi, abs(cmi))

                        if target == i:
                            scores[source, i] = max(scores[source, i], max_cmi)
                        elif source == i:
                            scores[target, i] = max(scores[target, i], max_cmi)

    return scores


def extract_node_adjacency_from_basis_oce(graph_nx, n, basis_map, var_names):
    """
    Extracts binary adjacency from the OCE graph on the expanded basis.

    After running causal discovery on the transformed data, this function
    translates the causal links from the high-dimensional graph back into a
    binary adjacency matrix for the original n-node Kuramoto network.

    An edge `s_{i<-j} -> v_i` in the basis graph implies a causal link `j -> i`
    in the original network. This function detects such edges (with lag 0 or 1)
    and sets the corresponding entry in the adjacency matrix to 1.

    It correctly handles the symmetry where a link from `sin(theta_j - theta_i)`
    or `sin(theta_i - theta_j)` to `v_i` can both imply `j -> i` or `i -> j`.

    Parameters
    ----------
    graph_nx : nx.MultiDiGraph
        The causal graph discovered by OCE on the transformed data.
    n : int
        The number of nodes in the original Kuramoto network.
    basis_map : list of tuple
        A map from sine-term index to the pair (target, source) of oscillators.
    var_names : list of str
        A list of variable names in the transformed data.

    Returns
    -------
    A_node : np.ndarray
        An (n, n) binary adjacency matrix where `A_node[j, i] = 1` indicates
        a discovered causal link `j -> i`.
    """
    A_node = np.zeros((n, n), dtype=int)

    # Get velocity variable names
    velocity_vars = [f"v{i}" for i in range(n)]

    # For each velocity v_i, check which coupling terms predict it
    # Due to symmetry sin(θ_j - θ_i) = -sin(θ_i - θ_j), we need to check both:
    # - s_{i<-j} -> v_i (direct)
    # - s_{j<-i} -> v_i (due to negative correlation)
    for i in range(n):
        velocity_var = velocity_vars[i]

        # Check all coupling terms that might predict v_i
        for idx, (target, source) in enumerate(basis_map):
            coupling_var = var_names[n + idx]

            # Check if this coupling term predicts v_i
            if graph_nx.has_edge(coupling_var, velocity_var):
                edges = graph_nx[coupling_var][velocity_var]
                # Accept lag-0 or lag-1 edges (lag-1 due to discretization)
                for key, edge_data in edges.items():
                    if edge_data["lag"] <= 1:
                        # s_{target<-source} -> v_i detected
                        # This means either source->i or target->i due to symmetry
                        if target == i:
                            # s_{i<-source} -> v_i: source causes i
                            A_node[source, i] = 1
                        elif source == i:
                            # s_{target<-i} -> v_i: target causes i (via symmetry)
                            A_node[target, i] = 1
                        break

    return A_node


def extract_node_scores_from_basis_pcmci(pcmci_result, n, basis_map):
    """
    Extract node-level continuous scores from PCMCI results.

    Parameters
    ----------
    pcmci_result : dict
        Results from pcmci.run_pcmci()
    n : int
        Number of original nodes
    basis_map : list of (i, j) tuples
        Maps coupling variable to (target, source) pair

    Returns
    -------
    scores : (n, n) np.ndarray
        Continuous scores where scores[j, i] is the strength of j -> i
        Uses 1 - p_value as the score (higher = more confident)
    """
    scores = np.zeros((n, n), dtype=float)
    p_matrix = pcmci_result["p_matrix"]  # shape: (n_vars, n_vars, tau_max+1)

    # For each velocity, check which coupling terms predict it
    for i in range(n):
        velocity_node_idx = i

        # Check all coupling terms
        for idx, (target, source) in enumerate(basis_map):
            coupling_node_idx = n + idx

            # Check for edges from coupling to velocity
            # Look at lag 1 (tau_max is typically 1)
            if p_matrix.shape[2] > 1:
                p_val = p_matrix[i, coupling_node_idx, 1]  # target, source, lag
                score = 1.0 - p_val  # Convert p-value to confidence score

                if target == i:
                    scores[source, i] = max(scores[source, i], score)
                elif source == i:
                    scores[target, i] = max(scores[target, i], score)

    return scores


def extract_node_adjacency_from_basis_pcmci(graph_nx, n, basis_map):
    """
    Extract node-level adjacency from PCMCI expanded basis graph.

    The pcmci_to_networkx function returns a MultiDiGraph where:
    - Nodes are integers: 0, 1, ..., p-1 where p = n + n*(n-1)
    - First n nodes (0 to n-1) are velocities v0, v1, ..., v_{n-1}
    - Next n*(n-1) nodes are coupling terms in basis_map order
    - Edges have a 'lag' attribute
    - Due to Kuramoto symmetry, we check both directions

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

    # For each velocity, check which coupling terms predict it
    # Handle symmetry: sin(θ_j - θ_i) = -sin(θ_i - θ_j)
    for i in range(n):
        velocity_node_idx = i

        # Check all coupling terms
        for idx, (target, source) in enumerate(basis_map):
            coupling_node_idx = n + idx

            # Check if coupling predicts this velocity
            if graph_nx.has_edge(coupling_node_idx, velocity_node_idx):
                # s_{target<-source} -> v_i detected
                if target == i:
                    # s_{i<-source} -> v_i: source causes i
                    A_node[source, i] = 1
                elif source == i:
                    # s_{target<-i} -> v_i: target causes i (via symmetry)
                    A_node[target, i] = 1

    return A_node


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 10,
    "T": 5000,  # Much longer for 9-dimensional discovery
    "n_trials": 10,  # Full trials for robust statistics
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


def compute_metrics(predicted_adj, true_adj, time_taken, predicted_scores=None):
    """
    Compute classification metrics for causal discovery.

    Parameters
    ----------
    predicted_adj : np.ndarray
        Binary predicted adjacency matrix
    true_adj : np.ndarray
        Binary true adjacency matrix
    time_taken : float
        Time taken for discovery
    predicted_scores : np.ndarray, optional
        Continuous scores (e.g., CMI values, 1-p_values) for AUC computation

    Returns
    -------
    dict : Dictionary of metrics
    """
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
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # Compute AUC if continuous scores are provided
    auc = np.nan
    if predicted_scores is not None:
        # Flatten matrices for sklearn
        y_true = A.flatten()
        y_score = predicted_scores.flatten()
        # Only compute AUC if we have both positive and negative classes
        if len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, y_score)
            except ValueError:
                auc = np.nan

    return {
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "F1": f1,
        "SHD": shd,
        "Accuracy": accuracy,
        "AUC": auc,
        "Time": time_taken,
    }


def adjacency_to_graph(adj_matrix, var_names=None):
    """
    Convert adjacency matrix to networkx MultiDiGraph for visualization.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Binary adjacency matrix where adj[i,j]=1 means i->j
    var_names : list, optional
        Variable names for nodes

    Returns
    -------
    G : nx.MultiDiGraph
        Graph with lag=1 edges (for Kuramoto, these are contemporaneous in reality)
    """
    n = adj_matrix.shape[0]
    if var_names is None:
        var_names = [f"Node_{i}" for i in range(n)]

    G = nx.MultiDiGraph()
    G.add_nodes_from(var_names)

    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] != 0:
                # Add edge with lag=0 for contemporaneous Kuramoto coupling
                G.add_edge(var_names[i], var_names[j], lag=0, cmi=1.0, p_value=0.0)

    return G


def plot_trial_summary(
    theta,
    dt,
    true_adj,
    pred_adj_oce,
    pred_adj_pcmci,
    metrics_oce,
    metrics_pcmci,
    node_names,
    save_path,
    title,
):
    """
    Create a comprehensive 2-row plot: networks on top, time series on bottom.

    Parameters
    ----------
    theta : np.ndarray
        Phase trajectories, shape (T, n)
    dt : float
        Time step
    true_adj : np.ndarray
        True adjacency matrix
    pred_adj_oce : np.ndarray
        OCE predicted adjacency matrix
    pred_adj_pcmci : np.ndarray
        PCMCI predicted adjacency matrix
    metrics_oce : dict
        Dictionary of metrics for the OCE method.
    metrics_pcmci : dict
        Dictionary of metrics for the PCMCI method.
    node_names : list
        Node names for visualization
    save_path : str
        Path to save figure
    title : str
        Overall title
    """
    # Convert adjacency matrices to graphs
    true_graph = adjacency_to_graph(true_adj, node_names)
    oce_graph = adjacency_to_graph(pred_adj_oce, node_names)
    pcmci_graph = adjacency_to_graph(pred_adj_pcmci, node_names)

    # Extract metrics
    f1_oce = metrics_oce["F1"]
    tp_oce = int(metrics_oce["TPR"] * np.sum(true_adj == 1))
    fp_oce = int(metrics_oce["FPR"] * np.sum(true_adj == 0))
    fn_oce = np.sum((pred_adj_oce == 0) & (true_adj == 1))

    f1_pcmci = metrics_pcmci["F1"]
    tp_pcmci = int(metrics_pcmci["TPR"] * np.sum(true_adj == 1))
    fp_pcmci = int(metrics_pcmci["FPR"] * np.sum(true_adj == 0))
    fn_pcmci = np.sum((pred_adj_pcmci == 0) & (true_adj == 1))

    # Create figure with 2 rows
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # ========== ROW 1: NETWORKS ==========
    graphs = [true_graph, oce_graph, pcmci_graph]
    network_titles = [
        f"TRUE NETWORK\n{true_graph.number_of_edges()} edges",
        f"OCE (Optimal Causation Entropy)\n{oce_graph.number_of_edges()} edges | F1={f1_oce:.2f} | TP={tp_oce} FP={fp_oce} FN={fn_oce}",
        f"PCMCI (Partial Correlation)\n{pcmci_graph.number_of_edges()} edges | F1={f1_pcmci:.2f} | TP={tp_pcmci} FP={fp_pcmci} FN={fn_pcmci}",
    ]

    # Optimize layout once for all plots
    from causationentropy.core.plotting import (
        optimize_circular_order,
        _circular_positions,
    )

    order = optimize_circular_order(true_graph, rng=42)
    pos = _circular_positions(order, radius=1.0)

    for col, (graph, subtitle) in enumerate(zip(graphs, network_titles)):
        ax = fig.add_subplot(gs[0, col])

        if graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Empty Graph", ha="center", va="center", fontsize=14)
            ax.set_title(subtitle, fontsize=11, fontweight="bold", pad=10)
            ax.axis("off")
            continue

        # Draw nodes
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=2000,
            node_color="white",
            edgecolors="black",
            linewidths=2.5,
            ax=ax,
        )

        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=11, font_weight="bold", ax=ax)

        # Draw edges with colors based on lag
        edge_colors = []
        edge_widths = []
        for u, v, data in graph.edges(data=True):
            lag = data.get("lag", 0)
            cmi = data.get("cmi", 1.0)

            if lag == 0:
                edge_colors.append("red")
                edge_widths.append(2.5 + cmi * 2)
            elif lag == 1:
                edge_colors.append("blue")
                edge_widths.append(1.5 + cmi * 1.5)
            else:
                edge_colors.append("gray")
                edge_widths.append(1 + cmi)

        if len(edge_colors) > 0:
            nx.draw_networkx_edges(
                graph,
                pos,
                edge_color=edge_colors,
                width=edge_widths,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=18,
                node_size=2000,
                ax=ax,
            )

        ax.set_title(subtitle, fontsize=11, fontweight="bold", pad=10)
        ax.axis("off")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)

    # ========== ROW 2: TIME SERIES ==========
    T, n = theta.shape
    time = np.arange(T) * dt

    # Phase trajectories
    ax_phase = fig.add_subplot(gs[1, :2])
    for i in range(n):
        ax_phase.plot(time, theta[:, i], label=f"Node {i}", linewidth=2)
    ax_phase.set_xlabel("Time", fontsize=11, fontweight="bold")
    ax_phase.set_ylabel("Phase θ (rad)", fontsize=11, fontweight="bold")
    ax_phase.set_title("Kuramoto Oscillator Phases", fontsize=12, fontweight="bold")
    ax_phase.legend(fontsize=9)
    ax_phase.grid(True, alpha=0.3)
    ax_phase.set_ylim(-np.pi, np.pi)

    # Phase differences
    ax_diff = fig.add_subplot(gs[1, 2])
    for i in range(1, n):
        diff = wrap_to_pi(theta[:, i] - theta[:, 0])
        ax_diff.plot(time, diff, label=f"Node {i} - Node 0", linewidth=2)
    ax_diff.set_xlabel("Time", fontsize=11, fontweight="bold")
    ax_diff.set_ylabel("Phase Diff (rad)", fontsize=11, fontweight="bold")
    ax_diff.set_title("Phase Synchronization", fontsize=12, fontweight="bold")
    ax_diff.legend(fontsize=9)
    ax_diff.grid(True, alpha=0.3)
    ax_diff.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax_diff.set_ylim(-np.pi, np.pi)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def save_environment_info():
    """Saves the versions of key libraries to a file."""
    key_libraries = [
        "numpy",
        "pandas",
        "networkx",
        "scikit-learn",
        "tigramite",
        "causationentropy",
    ]
    with open("environment_info.txt", "w") as f:
        f.write("Python Package Versions:\n")
        for lib in key_libraries:
            try:
                version = pkg_resources.get_distribution(lib).version
                f.write(f"- {lib}: {version}\n")
            except pkg_resources.DistributionNotFound:
                f.write(f"- {lib}: Not Found\n")


# =============================================================================
# MAIN
# =============================================================================

# Create output directory for figures
os.makedirs("figs", exist_ok=True)

results = []

print(f"Starting Kuramoto-basis Experiments (Simplified)")
print(f"{GLOBAL_PARAMS['n_nodes']} nodes, {GLOBAL_PARAMS['n_trials']} trials...")
print(f"Sample size: T={GLOBAL_PARAMS['T']}")

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

            # Node names for visualization
            node_names = [f"Node_{i}" for i in range(GLOBAL_PARAMS["n_nodes"])]

            X, basis_meta, var_names = prepare_kuramoto_data_for_causal_discovery(
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
                    method="standard",  # Condition on lagged targets
                    information="gaussian",
                )

            # Extract node-level adjacency and scores
            pred_adj = extract_node_adjacency_from_basis_oce(
                network, basis_meta["n"], basis_meta["basis_map"], var_names
            )
            pred_scores = extract_node_scores_from_basis_oce(
                network, basis_meta["n"], basis_meta["basis_map"], var_names
            )

            metrics_oce = compute_metrics(
                pred_adj, true_adj, time.time() - start_time, pred_scores
            )
            metrics_oce.update(
                {
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "OCE (gaussian)",
                    "Trial": trial,
                }
            )
            results.append(metrics_oce)

            # -------------------------------------------------------
            # METHOD 2: PCMCI (Tigramite)
            # -------------------------------------------------------
            start_time = time.time()
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
            pcmci_res = pcmci.run_pcmci(
                tau_min=1,
                tau_max=GLOBAL_PARAMS["tau_max"],
                pc_alpha=GLOBAL_PARAMS["alpha"],
            )
            graph_nx = pcmci_to_networkx(pcmci_res)

            # Extract node-level adjacency and scores
            pred_adj_pcmci = extract_node_adjacency_from_basis_pcmci(
                graph_nx, basis_meta["n"], basis_meta["basis_map"]
            )
            pred_scores_pcmci = extract_node_scores_from_basis_pcmci(
                pcmci_res, basis_meta["n"], basis_meta["basis_map"]
            )

            metrics_pcmci = compute_metrics(
                pred_adj_pcmci, true_adj, time.time() - start_time, pred_scores_pcmci
            )
            metrics_pcmci.update(
                {
                    "Experiment": exp_name,
                    "Parameter": val,
                    "Method": "PCMCI (ParCorr)",
                    "Trial": trial,
                }
            )
            results.append(metrics_pcmci)

            # -------------------------------------------------------
            # VISUALIZATION: Single Comprehensive Plot
            # -------------------------------------------------------
            plot_filename = f"figs/{exp_name}_param{val:.2f}_trial{trial}.png"
            plot_title = f"{exp_name} | {config['label']}={val:.2f} | Trial {trial}"
            plot_trial_summary(
                theta=theta,
                dt=GLOBAL_PARAMS["dt"],
                true_adj=true_adj,
                pred_adj_oce=pred_adj,
                pred_adj_pcmci=pred_adj_pcmci,
                metrics_oce=metrics_oce,
                metrics_pcmci=metrics_pcmci,
                node_names=node_names,
                save_path=plot_filename,
                title=plot_title,
            )

            pbar.update(1)

pbar.close()

# =============================================================================
# SAVE + PLOTS
# =============================================================================

df_results = pd.DataFrame(results)
df_results.to_csv("kuramoto_simplified_results.csv", index=False)
print("\n" + "=" * 80)
print("RESULTS SAVED")
print("=" * 80)
print("\nResults saved to kuramoto_simplified_results.csv")
print("Metrics included: TPR, FPR, Precision, F1, SHD, Accuracy, AUC, Time")
print(f"\nVisualizations saved to figs/ directory:")
print(f"  - {total_iters} comprehensive plots showing:")
print(f"    • Top row: True, OCE, and PCMCI networks side-by-side")
print(f"    • Bottom row: Oscillator time series")
print(f"  Total: {total_iters} figures")
print(f"\nFile naming: {list(EXPERIMENTS.keys())[0]}_param<value>_trial<n>.png")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
experiments_list = list(EXPERIMENTS.keys())

# Create comprehensive metrics plot (2 rows x 3 cols per experiment)
fig, axes = plt.subplots(
    len(experiments_list), 3, figsize=(18, 5 * len(experiments_list))
)
if len(experiments_list) == 1:
    axes = np.array([axes])

for idx, exp_name in enumerate(experiments_list):
    exp_data = df_results[df_results["Experiment"] == exp_name]
    config = EXPERIMENTS[exp_name]

    # F1 Score
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

    # Accuracy
    ax_acc = axes[idx, 1]
    sns.lineplot(
        data=exp_data,
        x="Parameter",
        y="Accuracy",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=ax_acc,
        ci=68,
    )
    ax_acc.set_title(f"{exp_name}: Accuracy")
    ax_acc.set_xlabel(config["label"])
    ax_acc.set_ylabel("Accuracy")

    # AUC
    ax_auc = axes[idx, 2]
    sns.lineplot(
        data=exp_data,
        x="Parameter",
        y="AUC",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=ax_auc,
        ci=68,
    )
    ax_auc.set_title(f"{exp_name}: AUC")
    ax_auc.set_xlabel(config["label"])
    ax_auc.set_ylabel("AUC")

    # Legend management
    if idx == 0:
        ax_f1.legend(bbox_to_anchor=(1.05, 1.2), loc="upper left")
        ax_acc.get_legend().remove()
        ax_auc.get_legend().remove()
    else:
        ax_f1.get_legend().remove()
        ax_acc.get_legend().remove()
        ax_auc.get_legend().remove()

plt.tight_layout()
plt.savefig("kuramoto_simplified_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary = df_results.groupby(["Experiment", "Method"])[
    ["F1", "TPR", "FPR", "Accuracy", "AUC", "Time"]
].mean()
summary = summary.reset_index().sort_values(
    ["Experiment", "F1"], ascending=[True, False]
)

for exp_name in experiments_list:
    print(f"\nExperiment: {exp_name}")
    print("-" * 80)
    exp_summary = summary[summary["Experiment"] == exp_name]
    print(
        exp_summary[
            ["Method", "F1", "TPR", "FPR", "Accuracy", "AUC", "Time"]
        ].to_string(index=False, float_format="%.3f")
    )

save_environment_info()
print("\nEnvironment info saved to environment_info.txt")
