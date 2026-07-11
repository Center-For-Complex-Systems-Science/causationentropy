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
from causationentropy.core.linalg import companion_matrix
from causationentropy.datasets.synthetic import (
    generate_graph_topology,
    simulate_rossler,
    prepare_rossler_data_for_causal_discovery,
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
# RÖSSLER-SPECIFIC EXTRACTION FUNCTIONS
# =============================================================================


def extract_node_graph_from_basis_oce_rossler(
    graph_nx, n, basis_map, var_names, coupling_on="x"
):
    """
    For OCE graph (string nodes):
    Return nx.MultiDiGraph with integer nodes [0..n-1].
    Edges carry lag, cmi, p_value attributes.
    """
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))

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
                    if target == i:
                        G.add_edge(
                            source,
                            i,
                            lag=edata.get("lag", 0),
                            cmi=edata.get("cmi", 0.0),
                            p_value=edata.get("p_value", 1.0),
                        )
    return G


def extract_node_graph_from_basis_pcmci_rossler(
    graph_nx, n, basis_map, coupling_on="x"
):
    """
    For PCMCI graph (integer nodes):
    Return nx.MultiDiGraph with integer nodes [0..n-1].
    Edges carry lag, cmi, p_value attributes.
    """
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(n))

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
                edges = graph_nx[coupling_node_idx][target_node_idx]
                for _, edata in edges.items():
                    if target == i:
                        G.add_edge(
                            source,
                            i,
                            lag=edata.get("lag", 0),
                            cmi=edata.get("val", edata.get("cmi", 0.0)),
                            p_value=edata.get("p_value", 1.0),
                        )
    return G


def collapse_to_binary_adjacency(node_graph, n):
    """Collapse MultiDiGraph to binary n x n adjacency matrix."""
    A = np.zeros((n, n), dtype=int)
    for u, v, _ in node_graph.edges(data=True):
        A[u, v] = 1
    return A


# =============================================================================
# COMPANION MATRIX HELPERS
# =============================================================================


def build_true_companion_top_rows(true_adj, n, max_lag):
    """Build (n, n*max_lag) true companion top rows. Lag-1 block = true_adj, rest zeros."""
    top = np.zeros((n, n * max_lag))
    top[:, :n] = (true_adj != 0).astype(int)
    return top


def build_pred_companion_top_rows(node_graph, n, max_lag):
    """Build (n, n*max_lag) predicted companion top rows via companion_matrix()."""
    C = companion_matrix(node_graph)
    if C.size == 0:
        return np.zeros((n, n * max_lag))
    top = C[:n, :]
    # Pad or truncate to standard width n*max_lag
    width = n * max_lag
    if top.shape[1] >= width:
        return top[:, :width]
    padded = np.zeros((n, width))
    padded[:, : top.shape[1]] = top
    return padded


# =============================================================================
# METRICS + GRAPH HELPERS
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
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    shd = fp + fn

    return {
        f"{prefix}TPR": tpr,
        f"{prefix}FPR": fpr,
        f"{prefix}Precision": precision,
        f"{prefix}F1": f1,
        f"{prefix}Accuracy": accuracy,
        f"{prefix}SHD": shd,
        f"{prefix}Time": time_taken,
    }


def adjacency_to_graph(adj_matrix, var_names=None):
    n = adj_matrix.shape[0]
    if var_names is None:
        var_names = [f"Node_{i}" for i in range(n)]

    G = nx.MultiDiGraph()
    G.add_nodes_from(var_names)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] != 0:
                G.add_edge(var_names[i], var_names[j], lag=0, cmi=1.0, p_value=0.0)
    return G


def plot_trial_summary_rossler(
    traj,
    dt,
    true_adj,
    pred_adj_oce,
    pred_adj_pcmci,
    node_names,
    save_path,
    title,
    coupling_on="x",
):
    """
    Similar layout to your Kuramoto plot:
      Row 1: true / OCE / PCMCI graphs
      Row 2: a few time series (x for all nodes + optionally phase-like projection)
    """
    true_graph = adjacency_to_graph(true_adj, node_names)
    oce_graph = adjacency_to_graph(pred_adj_oce, node_names)
    pcmci_graph = adjacency_to_graph(pred_adj_pcmci, node_names)

    tp_oce = np.sum((pred_adj_oce == 1) & (true_adj == 1))
    fp_oce = np.sum((pred_adj_oce == 1) & (true_adj == 0))
    fn_oce = np.sum((pred_adj_oce == 0) & (true_adj == 1))
    f1_oce = (
        2 * tp_oce / (2 * tp_oce + fp_oce + fn_oce)
        if (2 * tp_oce + fp_oce + fn_oce) > 0
        else 0
    )

    tp_pcmci = np.sum((pred_adj_pcmci == 1) & (true_adj == 1))
    fp_pcmci = np.sum((pred_adj_pcmci == 1) & (true_adj == 0))
    fn_pcmci = np.sum((pred_adj_pcmci == 0) & (true_adj == 1))
    f1_pcmci = (
        2 * tp_pcmci / (2 * tp_pcmci + fp_pcmci + fn_pcmci)
        if (2 * tp_pcmci + fp_pcmci + fn_pcmci) > 0
        else 0
    )

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    graphs = [true_graph, oce_graph, pcmci_graph]
    network_titles = [
        f"TRUE NETWORK\n{true_graph.number_of_edges()} edges",
        f"OCE (gaussian)\n{oce_graph.number_of_edges()} edges | F1={f1_oce:.2f} | TP={tp_oce} FP={fp_oce} FN={fn_oce}",
        f"PCMCI (ParCorr)\n{pcmci_graph.number_of_edges()} edges | F1={f1_pcmci:.2f} | TP={tp_pcmci} FP={fp_pcmci} FN={fn_pcmci}",
    ]

    # stable circular layout
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

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_size=2000,
            node_color="white",
            edgecolors="black",
            linewidths=2.5,
            ax=ax,
        )
        nx.draw_networkx_labels(graph, pos, font_size=11, font_weight="bold", ax=ax)

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

    # Row 2: time series for x (and maybe y,z of node 0)
    T, n, _ = traj.shape
    t = np.arange(T) * dt

    ax_x = fig.add_subplot(gs[1, :2])
    for i in range(n):
        ax_x.plot(t, traj[:, i, 0], label=f"Node {i} x", linewidth=2)
    ax_x.set_xlabel("Time", fontsize=11, fontweight="bold")
    ax_x.set_ylabel("x", fontsize=11, fontweight="bold")
    ax_x.set_title(
        f"Rössler x(t) per node (coupling on {coupling_on})",
        fontsize=12,
        fontweight="bold",
    )
    ax_x.legend(fontsize=9)
    ax_x.grid(True, alpha=0.3)

    ax0 = fig.add_subplot(gs[1, 2])
    ax0.plot(t, traj[:, 0, 0], label="x0", linewidth=2)
    ax0.plot(t, traj[:, 0, 1], label="y0", linewidth=2)
    ax0.plot(t, traj[:, 0, 2], label="z0", linewidth=2)
    ax0.set_xlabel("Time", fontsize=11, fontweight="bold")
    ax0.set_ylabel("State", fontsize=11, fontweight="bold")
    ax0.set_title("Node 0 state (x,y,z)", fontsize=12, fontweight="bold")
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # plt.close()

    return fig


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_PARAMS = {
    "n_nodes": 5,
    "T": 5000,  # chaotic needs decent length; adjust as needed
    "n_trials": 2,
    "alpha": 0.05,
    "tau_max": 1,
    "dt": 0.02,
    "burn_in": 500,
    "normalize_by_indegree": False,
    # Rössler-specific
    "a": 0.2,
    "b": 0.2,
    "c": 5.7,
    "rho_default": 0.4,
    "noise_std": 0.0,
    "init_scale": 1.0,
    "coupling_on": "x",
}

EXPERIMENTS = {
    "Method_Compare": {
        "type": "Erdos-Renyi",
        "vary_param": "p_edge",
        "label": "Edge Probability (p)",
        "values": [0.2, 0.4, 0.6],
        "defaults": {"rho": 0.2, "subsample": 5},
    },
}


# =============================================================================
# MAIN
# =============================================================================

results = []

print("Starting Rössler-basis Experiments (Simplified)")
print(f"{GLOBAL_PARAMS['n_nodes']} nodes, {GLOBAL_PARAMS['n_trials']} trials...")

total_iters = sum(
    len(cfg["values"]) * GLOBAL_PARAMS["n_trials"] for cfg in EXPERIMENTS.values()
)
pbar = tqdm(total=total_iters, desc="Total progress")

os.makedirs("figs", exist_ok=True)

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

            node_names = [f"Node_{i}" for i in range(GLOBAL_PARAMS["n_nodes"])]

            # Subsample trajectory to increase effective dt
            k = int(current_params.get("subsample", 1))
            traj_sub = traj[::k] if k > 1 else traj
            dt_eff = GLOBAL_PARAMS["dt"] * k

            X, basis_meta, var_names = prepare_rossler_data_for_causal_discovery(
                traj_sub, dt=dt_eff, coupling_on=GLOBAL_PARAMS["coupling_on"]
            )
            T_eff = X.shape[0]

            dataframe = pp.DataFrame(
                X, datatime={0: np.arange(T_eff)}, var_names=var_names
            )

            n = basis_meta["n"]
            max_lag = int(current_params.get("tau_max_oce", GLOBAL_PARAMS["tau_max"]))
            X_df = pd.DataFrame(X, columns=var_names)
            true_comp = build_true_companion_top_rows(true_adj, n, max_lag)

            # -------------------------------------------------------
            # OCE METHODS
            # -------------------------------------------------------
            OCE_METHODS = [
                ("standard", "gaussian", "OCE (standard)"),
                ("alternative", "gaussian", "OCE (alternative)"),
                ("lasso", "gaussian", "OCE (lasso)"),
            ]

            # Store the last OCE prediction for the visualization
            pred_adj_oce = None

            alpha_oce = current_params.get("alpha_oce", GLOBAL_PARAMS["alpha"])

            for oce_method, oce_info, oce_label in OCE_METHODS:
                start_time = time.time()
                with suppress_stdout():
                    network = discover_network(
                        data=X_df,
                        max_lag=max_lag,
                        method=oce_method,
                        information=oce_info,
                        alpha_forward=alpha_oce,
                        alpha_backward=alpha_oce,
                    )

                oce_node_graph = extract_node_graph_from_basis_oce_rossler(
                    network,
                    n,
                    basis_meta["basis_map"],
                    var_names,
                    coupling_on=GLOBAL_PARAMS["coupling_on"],
                )
                pred_adj = collapse_to_binary_adjacency(oce_node_graph, n)
                oce_time = time.time() - start_time

                # Binary metrics
                metrics = compute_metrics(pred_adj, true_adj, oce_time)
                # Companion metrics
                pred_comp = build_pred_companion_top_rows(oce_node_graph, n, max_lag)
                comp_metrics = compute_metrics(pred_comp, true_comp, oce_time, prefix="comp_")
                metrics.update(comp_metrics)
                metrics.update(
                    {
                        "Experiment": exp_name,
                        "Parameter": val,
                        "Method": oce_label,
                        "Trial": trial,
                    }
                )
                results.append(metrics)
                pred_adj_oce = pred_adj  # keep last for plot

            # -------------------------------------------------------
            # PCMCI (Tigramite)
            # -------------------------------------------------------
            start_time = time.time()
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
            pcmci_res = pcmci.run_pcmci(
                tau_min=1,
                tau_max=max_lag,
                pc_alpha=GLOBAL_PARAMS["alpha"],
            )
            graph_nx = pcmci_to_networkx(pcmci_res)

            pcmci_node_graph = extract_node_graph_from_basis_pcmci_rossler(
                graph_nx,
                n,
                basis_meta["basis_map"],
                coupling_on=GLOBAL_PARAMS["coupling_on"],
            )
            pred_adj_pcmci = collapse_to_binary_adjacency(pcmci_node_graph, n)
            pcmci_time = time.time() - start_time

            # -------------------------------------------------------
            # VISUALIZATION
            # -------------------------------------------------------
            plot_filename = f"figs/{exp_name}_param{val:.3f}_trial{trial}.png"
            plot_title = f"{exp_name} | {config['label']}={val:.3f} | Trial {trial}"
            plot_trial_summary_rossler(
                traj=traj_sub,
                dt=dt_eff,
                true_adj=true_adj,
                pred_adj_oce=pred_adj_oce,
                pred_adj_pcmci=pred_adj_pcmci,
                node_names=node_names,
                save_path=plot_filename,
                title=plot_title,
                coupling_on=GLOBAL_PARAMS["coupling_on"],
            )

            # Binary metrics
            metrics = compute_metrics(pred_adj_pcmci, true_adj, pcmci_time)
            # Companion metrics
            pred_comp_pcmci = build_pred_companion_top_rows(pcmci_node_graph, n, max_lag)
            comp_metrics = compute_metrics(pred_comp_pcmci, true_comp, pcmci_time, prefix="comp_")
            metrics.update(comp_metrics)
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
df_results.to_csv("rossler.csv", index=False)
print("\nResults saved to rossler.csv")
print("\nVisualizations saved to figs/ directory.")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
experiments_list = list(EXPERIMENTS.keys())

fig, axes = plt.subplots(
    len(experiments_list), 3, figsize=(20, 5 * len(experiments_list))
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
    ax_f1.set_title(f"{exp_name}: F1 Score (Binary)")
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

    ax_comp = axes[idx, 2]
    sns.lineplot(
        data=exp_data,
        x="Parameter",
        y="comp_F1",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=ax_comp,
        ci=68,
    )
    ax_comp.set_title(f"{exp_name}: F1 Score (Companion)")
    ax_comp.set_xlabel(config["label"])
    ax_comp.set_ylabel("comp_F1")

    if idx == 0:
        ax_f1.legend(bbox_to_anchor=(1.05, 1.2), loc="upper left")
        ax_tpr.get_legend().remove()
        ax_comp.get_legend().remove()
    else:
        ax_f1.get_legend().remove()
        ax_tpr.get_legend().remove()
        ax_comp.get_legend().remove()

plt.tight_layout()
plt.savefig("rossler.png", dpi=300, bbox_inches="tight")
# plt.show()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary = df_results.groupby(["Experiment", "Method"])[
    ["F1", "Accuracy", "TPR", "FPR", "Time", "comp_F1", "comp_Accuracy", "comp_TPR"]
].mean()
summary = summary.reset_index().sort_values(
    ["Experiment", "F1"], ascending=[True, False]
)

for exp_name in experiments_list:
    print(f"\nExperiment: {exp_name}")
    print("-" * 80)
    exp_summary = summary[summary["Experiment"] == exp_name]
    print(
        exp_summary[["Method", "F1", "Accuracy", "TPR", "FPR", "comp_F1", "comp_Accuracy", "comp_TPR", "Time"]].to_string(
            index=False, float_format="%.3f"
        )
    )
