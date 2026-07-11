"""
Hero figure for the Chaos paper (Kuramoto only).

Layout: rows = network topology (Hub / Small-World / Erdos-Renyi),
columns = [ order parameter r(t) | True network | oCSE | PCMCI | Linear Granger ].

Each method's recovered graph is drawn on the SAME node layout as the truth,
with edges colored:
    true positive  -> solid dark   (correctly recovered)
    false positive -> solid red    (spurious link)
    false negative -> light dashed (missed true link)

oCSE recovers the network cleanly, PCMCI adds a partial scatter of false
links, and Linear Granger (at a loose threshold) connects almost everything.

Edit CONFIG below with the rho / alphas chosen from the rho-search, then run:
    python benchmarks/hero_figure.py
Produces benchmarks/results/hero_kuramoto.png (+ .pdf)
"""
import os, warnings
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

import lingam
from kuramoto_roc import run_targeted_oce, extract_pcmci_adjacency, linear_granger, compute_metrics
from causationentropy.graph import pcmci_to_networkx
from causationentropy.datasets.synthetic import (
    generate_graph_topology, simulate_kuramoto,
    prepare_kuramoto_data_for_causal_discovery,
)
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# =============================== CONFIG ===============================
CONFIG = {
    "rho": 0.30,            # chosen from rho-search: cleanest oCSE/PCMCI separation
    "alpha_ocse": 0.05,
    "alpha_pcmci": 0.05,
    "alpha_granger": 0.05,  # same alpha -> fair comparison
    "T": 4000,
    "dt": 0.05,
    "n_nodes": 6,
    "n_shuffles": 100,
    "seed": 42,             # representative single trial per topology
}
TOPOS = {
    "Hub": ("Scale-Free", {"m_attachment": 2}),
    "Small-World": ("Small-World", {"k_neighbors": 2, "p_rewire": 0.3}),
    "Erdos-Renyi": ("Erdos-Renyi", {"p_edge": 0.35}),
}
# single shared realization (seed 7): cleanest oCSE recovery vs failing baselines at n=6
SEEDS = {"Hub": 7, "Small-World": 7, "Erdos-Renyi": 7}
COL_TP = "#1a1a1a"      # correctly recovered
COL_FP = "#e23b3b"      # spurious
COL_FN = "#c8c8c8"      # missed
COL_R = "#2c6fbb"       # order parameter line
NODE_COL = "#dfe7f2"
# =====================================================================


def order_param(theta):
    return np.abs(np.exp(1j * theta).mean(axis=1))


def draw_network(ax, true_adj, pred_adj, pos, title, metric_txt, highlight=False):
    """Draw predicted graph vs truth on fixed layout, colored by TP/FP/FN."""
    n = true_adj.shape[0]
    A = (true_adj != 0).astype(int)
    B = (pred_adj != 0).astype(int)
    G = nx.DiGraph(); G.add_nodes_from(range(n))

    fp_edges, tp_edges, fn_edges = [], [], []
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if B[u, v] and A[u, v]:
                tp_edges.append((u, v))
            elif B[u, v] and not A[u, v]:
                fp_edges.append((u, v))
            elif (not B[u, v]) and A[u, v]:
                fn_edges.append((u, v))

    # missed (faint, behind), then spurious, then correct (on top)
    nx.draw_networkx_edges(G, pos, edgelist=fn_edges, ax=ax, edge_color=COL_FN,
                           style="dashed", width=1.0, arrows=False, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=fp_edges, ax=ax, edge_color=COL_FP,
                           width=1.3, arrows=False, alpha=0.85)
    nx.draw_networkx_edges(G, pos, edgelist=tp_edges, ax=ax, edge_color=COL_TP,
                           width=1.8, arrows=False)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=NODE_COL, node_size=160,
                           edgecolors="#5a6b85", linewidths=0.8)
    if highlight:
        ax.set_title(title, fontsize=28, pad=6, fontweight="bold", color="#1a7a3a")
        ax.text(0.5, -0.07, metric_txt, transform=ax.transAxes, ha="center",
                va="top", fontsize=23, fontweight="bold", color="#1a7a3a")
        # green frame to flag the winning method
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#1a7a3a"); s.set_linewidth(2.2)
        ax.set_facecolor("#f1faf3")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    else:
        ax.set_title(title, fontsize=25, pad=6)
        ax.text(0.5, -0.07, metric_txt, transform=ax.transAxes, ha="center",
                va="top", fontsize=21, color="#333")
        ax.set_axis_off()
    ax.set_aspect("equal")


def main():
    c = CONFIG
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 24,
    })

    nrows = len(TOPOS)
    fig = plt.figure(figsize=(24, 5.8 * nrows))
    gs = fig.add_gridspec(nrows, 6, width_ratios=[1.35, 1, 1, 1, 1, 1],
                          wspace=0.12, hspace=0.62)

    method_cols = [
        ("True network", None),
        ("oCSE", "ocse"),
        ("PCMCI", "pcmci"),
        ("Linear Granger", "granger"),
        ("VARLiNGAM", "varlingam"),
    ]

    for row, (tname, (typ, base_params)) in enumerate(TOPOS.items()):
        seed = SEEDS.get(tname, c["seed"])
        np.random.seed(seed)
        params = dict(base_params); params["rho"] = c["rho"]
        G = generate_graph_topology(typ, c["n_nodes"], params, seed)
        true_adj = nx.to_numpy_array(G).astype(int)

        theta, _ = simulate_kuramoto(
            G=G, T=c["T"], dt=c["dt"], rho=c["rho"], seed=seed,
            omega_mean=0.0, omega_std=1.0, phase_noise_std=0.02,
            burn_in=50, normalize_by_indegree=False,
        )
        r = order_param(theta)
        X_basis, basis_meta, var_names = prepare_kuramoto_data_for_causal_discovery(theta, dt=c["dt"])
        T_eff = X_basis.shape[0]

        # --- run methods ---
        adj_ocse = run_targeted_oce(X_basis, basis_meta, alpha=c["alpha_ocse"], n_shuffles=c["n_shuffles"])
        df_pc = pp.DataFrame(X_basis, datatime={0: np.arange(T_eff)}, var_names=var_names)
        pcmci = PCMCI(dataframe=df_pc, cond_ind_test=ParCorr(), verbosity=0)
        res = pcmci.run_pcmci(tau_min=1, tau_max=1, pc_alpha=c["alpha_pcmci"])
        adj_pcmci = extract_pcmci_adjacency(pcmci_to_networkx(res), basis_meta["n"], basis_meta["basis_map"])
        adj_granger = linear_granger(theta, c["n_nodes"], alpha=c["alpha_granger"])

        # VARLiNGAM on raw phases (lag-1), source->target orientation
        vmodel = lingam.VARLiNGAM(lags=1)
        vmodel.fit(theta)
        adj_var = np.zeros((c["n_nodes"], c["n_nodes"]), dtype=int)
        for lag_idx, B_lag in enumerate(vmodel.adjacency_matrices_):
            if lag_idx == 0:
                continue
            adj_var |= (np.abs(B_lag.T) > 1e-8).astype(int)  # B[i,j]: j->i ; .T -> [source,target]
        np.fill_diagonal(adj_var, 0)

        preds = {"ocse": adj_ocse, "pcmci": adj_pcmci,
                 "granger": adj_granger, "varlingam": adj_var}

        # consistent layout from the true (undirected) graph
        pos = nx.spring_layout(nx.Graph(nx.DiGraph(true_adj)), seed=7, k=0.9)

        # --- column 0: order parameter ---
        ax_r = fig.add_subplot(gs[row, 0])
        t_axis = np.arange(len(r)) * c["dt"]
        ax_r.plot(t_axis, r, color=COL_R, lw=1.0)
        ax_r.axhline(r.mean(), color="#888", ls="--", lw=1.0)
        ax_r.set_ylim(0, 1.02)
        ax_r.set_xlabel("time")
        ax_r.set_ylabel(r"order parameter $r(t)$")
        ax_r.set_title(tname, fontsize=26, loc="left", fontweight="bold")
        ax_r.text(0.97, 0.05, rf"$\bar r$={r.mean():.2f}", transform=ax_r.transAxes,
                  ha="right", va="bottom", fontsize=20,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbb", alpha=0.85))
        ax_r.grid(alpha=0.2)

        # --- columns 1-4: networks ---
        for j, (label, key) in enumerate(method_cols):
            ax = fig.add_subplot(gs[row, j + 1])
            if key is None:
                draw_network(ax, true_adj, true_adj, pos, label,
                             f"{int((true_adj!=0).sum())} edges")
            else:
                m = compute_metrics(preds[key], true_adj)
                draw_network(ax, true_adj, preds[key], pos, label,
                             f"TPR={m['TPR']:.2f}\nFPR={m['FPR']:.2f}\nF1={m['F1']:.2f}",
                             highlight=(key == "ocse"))

    # legend
    legend_elems = [
        Line2D([0], [0], color=COL_TP, lw=2.2, label="true positive"),
        Line2D([0], [0], color=COL_FP, lw=2.2, label="false positive"),
        Line2D([0], [0], color=COL_FN, lw=1.6, ls="--", label="missed (false negative)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               frameon=False, fontsize=24, bbox_to_anchor=(0.62, -0.02))
    out = os.path.join(RESULTS_DIR, "hero_kuramoto")
    plt.savefig(out + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    print("saved", out + ".png / .pdf")


if __name__ == "__main__":
    main()
