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


def extract_node_adjacency_from_basis_oce_rossler(
    graph_nx, n, basis_map, var_names, coupling_on="x"
):
    """
    For OCE graph (string nodes):
      nodes are var_names strings.
    We infer edge j -> i if coupling variable c_{i<-j} predicts d(s_i)/dt variable.

    If coupling_on == "x": look at edges c_{i<-j} -> dx_i
    If coupling_on == "y": look at edges c_{i<-j} -> dy_i
    If coupling_on == "z": look at edges c_{i<-j} -> dz_i

    Accept lag<=1 (like your Kuramoto code).
    """
    A_node = np.zeros((n, n), dtype=int)

    if coupling_on == "x":
        target_vars = [f"dx{i}" for i in range(n)]
        target_offset = 0
    elif coupling_on == "y":
        target_vars = [f"dy{i}" for i in range(n)]
        target_offset = n
    else:
        target_vars = [f"dz{i}" for i in range(n)]
        target_offset = 2 * n

    # coupling vars begin after 3n in var_names
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
    """
    For PCMCI graph (integer nodes):
      first 3n nodes are [dx0..dx{n-1}, dy0.., dz0..]
      coupling nodes start at index 3n

    We infer j -> i if c_{i<-j} -> d(s_i)/dt is present.
    """
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


# =============================================================================
# METRICS + GRAPH HELPERS
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
    "n_nodes": 10,
    "T": 1000,  # chaotic needs decent length; adjust as needed
    "n_trials": 5,
    "alpha": 0.05,
    "tau_max": 1,
    "dt": 0.02,
    "burn_in": 500,
    "normalize_by_indegree": False,
    # Rössler-specific
    "a": 0.2,
    "b": 0.2,
    "c": 5.7,
    "rho_default": 0.1,
    "noise_std": 0.0,
    "init_scale": 1.0,
    "coupling_on": "x",
}

EXPERIMENTS = {
    "ER_Density": {
        "type": "Erdos-Renyi",
        "vary_param": "p_edge",
        "label": "Edge Probability (p)",
        "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "defaults": {"rho": 0.1},
    },
    "Coupling_Strength": {
        "type": "Erdos-Renyi",
        "vary_param": "rho",
        "label": "Coupling Strength (rho)",
        "values": [0.02, 0.05, 0.1, 0.2, 0.3],
        "defaults": {"p_edge": 0.3},
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

            X, basis_meta, var_names = prepare_rossler_data_for_causal_discovery(
                traj, dt=GLOBAL_PARAMS["dt"], coupling_on=GLOBAL_PARAMS["coupling_on"]
            )
            T_eff = X.shape[0]

            dataframe = pp.DataFrame(
                X, datatime={0: np.arange(T_eff)}, var_names=var_names
            )

            # -------------------------------------------------------
            # METHOD 1: OCE (causationentropy)
            # -------------------------------------------------------
            start_time = time.time()
            X_df = pd.DataFrame(X, columns=var_names)

            with suppress_stdout():
                network = discover_network(
                    data=X_df,
                    max_lag=GLOBAL_PARAMS["tau_max"],
                    method="standard",
                    information="gaussian",
                )

            pred_adj_oce = extract_node_adjacency_from_basis_oce_rossler(
                network,
                basis_meta["n"],
                basis_meta["basis_map"],
                var_names,
                coupling_on=GLOBAL_PARAMS["coupling_on"],
            )

            metrics = compute_metrics(pred_adj_oce, true_adj, time.time() - start_time)
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
                tau_min=1,
                tau_max=GLOBAL_PARAMS["tau_max"],
                pc_alpha=GLOBAL_PARAMS["alpha"],
            )
            graph_nx = pcmci_to_networkx(pcmci_res)

            pred_adj_pcmci = extract_node_adjacency_from_basis_pcmci_rossler(
                graph_nx,
                basis_meta["n"],
                basis_meta["basis_map"],
                coupling_on=GLOBAL_PARAMS["coupling_on"],
            )

            # -------------------------------------------------------
            # VISUALIZATION
            # -------------------------------------------------------
            plot_filename = f"figs/{exp_name}_param{val:.3f}_trial{trial}.png"
            plot_title = f"{exp_name} | {config['label']}={val:.3f} | Trial {trial}"
            plot_trial_summary_rossler(
                traj=traj,
                dt=GLOBAL_PARAMS["dt"],
                true_adj=true_adj,
                pred_adj_oce=pred_adj_oce,
                pred_adj_pcmci=pred_adj_pcmci,
                node_names=node_names,
                save_path=plot_filename,
                title=plot_title,
                coupling_on=GLOBAL_PARAMS["coupling_on"],
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
df_results.to_csv("rossler_simplified_results.csv", index=False)
print("\nResults saved to rossler_simplified_results.csv")
print("\nVisualizations saved to figs/ directory.")

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
plt.savefig("rossler_analysis.png", dpi=300, bbox_inches="tight")
# plt.show()

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
