"""
Paper Figure Generation
========================
Generates the two main benchmark figures:

  Figure 1: ROC curves (from kuramoto_roc_results.csv)
  Figure 2: F1 / accuracy vs parameter sweep (from kuramoto_results.csv
            or gaussian_process_results.csv)

Run after: kuramoto_roc.py and kuramoto.py (or gaussian_processes.py)
"""
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "sans-serif",
})

RESULTS_DIR = "benchmarks/results"

COLORS = {
    "oCSE": "#e74c3c",
    "PCMCI": "#3498db",
    "VARLiNGAM": "#2ecc71",
    "Linear Granger": "#9b59b6",
}
MARKERS = {"oCSE": "o", "PCMCI": "s", "VARLiNGAM": "D", "Linear Granger": "^"}


# =============================================================================
# FIGURE 1: ROC CURVES
# =============================================================================

def plot_roc(df, outpath):
    """Multi-panel ROC from alpha-sweep data (kuramoto_roc_results.csv)."""
    topologies = df["Topology"].unique()
    methods = [m for m in ["oCSE", "PCMCI", "Linear Granger"]
               if m in df["Method"].unique()]
    n_panels = len(topologies)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), squeeze=False)

    for col, topo in enumerate(topologies):
        ax = axes[0, col]
        topo_data = df[df["Topology"] == topo]

        for method in methods:
            mdata = topo_data[topo_data["Method"] == method]
            grouped = mdata.groupby("Alpha")[["FPR", "TPR"]].agg(["mean", "std"])

            fpr_mean = grouped[("FPR", "mean")].values
            fpr_std = grouped[("FPR", "std")].values
            tpr_mean = grouped[("TPR", "mean")].values
            tpr_std = grouped[("TPR", "std")].values

            order = np.argsort(fpr_mean)
            fpr_mean, fpr_std = fpr_mean[order], fpr_std[order]
            tpr_mean, tpr_std = tpr_mean[order], tpr_std[order]

            _, unique_idx = np.unique(fpr_mean, return_index=True)
            fpr_mean = fpr_mean[unique_idx]
            fpr_std = fpr_std[unique_idx]
            tpr_mean = tpr_mean[unique_idx]
            tpr_std = tpr_std[unique_idx]

            fpr_full = np.concatenate([[0], fpr_mean, [1]])
            tpr_full = np.concatenate([[0], tpr_mean, [1]])
            auc = np.trapz(tpr_full, fpr_full)

            ax.errorbar(
                fpr_mean, tpr_mean, xerr=fpr_std, yerr=tpr_std,
                label=f"{method} (AUC={auc:.2f})",
                color=COLORS[method], marker=MARKERS[method],
                capsize=3, linewidth=2, markersize=6,
            )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(topo)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="lower right")

    fig.suptitle(
        "Kuramoto Oscillators: ROC Curves (significance level sweep)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Figure 1 saved to {outpath}")
    plt.close()


# =============================================================================
# FIGURE 2: ACCURACY (F1 / TPR / FPR) vs PARAMETER
# =============================================================================

EXPERIMENT_LABELS = {
    "ER_Density": "Edge Probability (p)",
    "Coupling_Strength": "Coupling Strength (rho)",
    "ScaleFree_Hubs": "Attachment Edges (m)",
    "SmallWorld_Rewiring": "Rewiring Probability (p)",
}

def plot_accuracy(df, outpath, experiments=None):
    """
    Multi-row accuracy figure: one row per experiment, columns for F1, TPR, FPR.
    """
    if experiments is None:
        experiments = df["Experiment"].unique()

    methods = [m for m in ["oCSE", "PCMCI", "VARLiNGAM", "Linear Granger"]
               if m in df["Method"].unique()]

    has_trials = "Trial" in df.columns
    n_exp = len(experiments)
    fig, axes = plt.subplots(n_exp, 3, figsize=(18, 4.5 * n_exp), squeeze=False)

    for row, exp_name in enumerate(experiments):
        exp_data = df[df["Experiment"] == exp_name]
        xlabel = EXPERIMENT_LABELS.get(exp_name, exp_name)

        for col, metric in enumerate(["F1", "TPR", "FPR"]):
            ax = axes[row, col]
            for method in methods:
                mdata = exp_data[exp_data["Method"] == method]
                if has_trials:
                    grouped = mdata.groupby("Parameter")[metric]
                    mean = grouped.mean()
                    std = grouped.std()
                    ax.errorbar(
                        mean.index, mean.values, yerr=std.values,
                        label=method, color=COLORS[method],
                        marker=MARKERS[method], capsize=3, linewidth=2,
                    )
                else:
                    ax.plot(
                        mdata["Parameter"], mdata[metric],
                        label=method, color=COLORS[method],
                        marker=MARKERS[method], linewidth=2,
                    )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric)
            ax.set_title(f"{exp_name}: {metric}")
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Figure 2 saved to {outpath}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Figure 1: ROC ---
    roc_path = os.path.join(RESULTS_DIR, "kuramoto_roc_results.csv")
    if os.path.exists(roc_path):
        df_roc = pd.read_csv(roc_path)
        plot_roc(df_roc, os.path.join(RESULTS_DIR, "figure1_roc.png"))
    else:
        print(f"ROC data not found at {roc_path}")
        print("  Run: python -m benchmarks.kuramoto_roc")

    # --- Figure 2: Accuracy ---
    kur_path = os.path.join(RESULTS_DIR, "kuramoto_results.csv")
    gp_path = os.path.join(RESULTS_DIR, "gaussian_process_results.csv")

    if os.path.exists(kur_path):
        df_acc = pd.read_csv(kur_path)
        if "Trial" in df_acc.columns and len(df_acc) > 20:
            print(f"Using Kuramoto per-trial data ({len(df_acc)} rows)")
            plot_accuracy(df_acc, os.path.join(RESULTS_DIR, "figure2_accuracy.png"))
        else:
            print(f"Kuramoto data has only {len(df_acc)} rows (summary only).")
            print("  Re-run: python -m benchmarks.kuramoto  to get per-trial data.")
            if os.path.exists(gp_path):
                print("  Using Gaussian process data as fallback.")
                df_acc = pd.read_csv(gp_path)
                plot_accuracy(
                    df_acc, os.path.join(RESULTS_DIR, "figure2_accuracy_gp.png"),
                )
    elif os.path.exists(gp_path):
        df_acc = pd.read_csv(gp_path)
        print(f"Using Gaussian process data ({len(df_acc)} rows)")
        plot_accuracy(df_acc, os.path.join(RESULTS_DIR, "figure2_accuracy_gp.png"))
    else:
        print("No accuracy data found. Run kuramoto.py or gaussian_processes.py first.")
