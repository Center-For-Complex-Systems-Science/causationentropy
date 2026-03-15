"""
Compile all Rossler benchmark results into ROC-style plots.
Each method is plotted as a point (FPR, TPR) averaged across all conditions and trials.
Methods with multiple operating points (different alpha/threshold) are connected as curves.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11

# =========================================================================
# LOAD ALL RESULTS
# =========================================================================

results_dir = "benchmarks/results"

# All Rossler experiment CSVs with compatible columns
files_and_prefixes = [
    ("deep_koopman_twostage_results.csv", ""),
    ("deep_koopman_attention_results.csv", ""),
    ("deep_koopman_kan_results.csv", ""),
    ("deep_koopman_results.csv", ""),
    ("deep_koopman_pernode_results.csv", ""),
    ("deep_koopman_perpair_results.csv", ""),
    ("deep_koopman_kausal_results.csv", ""),
]

all_dfs = []
for fname, prefix in files_and_prefixes:
    try:
        df = pd.read_csv(f"{results_dir}/{fname}")
        # Ensure common columns
        for col in ["TP", "FP", "FN", "TPR", "FPR", "F1", "Method", "Experiment", "Parameter", "Trial"]:
            if col not in df.columns:
                continue
        if prefix:
            df["Method"] = prefix + df["Method"]
        all_dfs.append(df[["TPR", "FPR", "F1", "Method", "Experiment", "Parameter", "Trial"]])
    except Exception as e:
        print(f"  Skipping {fname}: {e}")

df_all = pd.concat(all_dfs, ignore_index=True)

# =========================================================================
# COMPUTE MEAN TPR, FPR, F1 PER METHOD (across all densities and trials)
# =========================================================================

summary = (
    df_all.groupby("Method")[["TPR", "FPR", "F1"]]
    .agg(["mean", "std", "count"])
    .reset_index()
)
summary.columns = ["Method", "TPR_mean", "TPR_std", "TPR_n",
                    "FPR_mean", "FPR_std", "FPR_n",
                    "F1_mean", "F1_std", "F1_n"]

# Sort by F1 descending
summary = summary.sort_values("F1_mean", ascending=False)

print("=" * 90)
print("ALL METHODS — Mean across densities {0.2, 0.4, 0.6} and trials")
print("=" * 90)
print(summary[["Method", "F1_mean", "TPR_mean", "FPR_mean", "TPR_n"]].to_string(index=False, float_format="%.3f"))

# =========================================================================
# DEFINE METHOD CATEGORIES FOR COLORING
# =========================================================================

def categorize(method):
    m = method.lower()
    if "stage 1" in m:
        return "Stage 1 (Koopman+oCSE)"
    if "deriv" in m:
        return "Two-stage: Deriv backward"
    if "phase" in m:
        return "Phase reduction"
    if "lasso" in m and "phase" not in m:
        return "LASSO (derivative)"
    if "attn" in m:
        return "Attention"
    if "ocse" in m.replace(" ", "").replace("∩", "") and "phase" not in m:
        return "Baseline: oCSE"
    if "pcmci" in m:
        return "Baseline: PCMCI"
    if "granger" in m or "linear" in m:
        return "Baseline: Linear Granger"
    if "kan" in m:
        return "KAN-Koopman"
    if "kausal" in m and "per-pair" not in m:
        return "Koopman + Kausal"
    if "per-pair" in m:
        return "Per-pair"
    if "pernode" in m or "per-node" in m or "k-matrix" in m or "koopman (k" in m:
        return "Koopman (K-matrix)"
    return "Other"

summary["Category"] = summary["Method"].apply(categorize)

# =========================================================================
# PLOT 1: ROC-STYLE SCATTER (ALL METHODS)
# =========================================================================

category_colors = {
    "Stage 1 (Koopman+oCSE)": "#e74c3c",
    "Two-stage: Deriv backward": "#2ecc71",
    "Phase reduction": "#9b59b6",
    "LASSO (derivative)": "#e67e22",
    "Attention": "#95a5a6",
    "Baseline: oCSE": "#3498db",
    "Baseline: PCMCI": "#1abc9c",
    "Baseline: Linear Granger": "#f1c40f",
    "KAN-Koopman": "#e91e63",
    "Koopman + Kausal": "#ff5722",
    "Per-pair": "#795548",
    "Koopman (K-matrix)": "#607d8b",
    "Other": "#bdc3c7",
}

category_markers = {
    "Stage 1 (Koopman+oCSE)": "D",
    "Two-stage: Deriv backward": "s",
    "Phase reduction": "^",
    "LASSO (derivative)": "v",
    "Attention": "x",
    "Baseline: oCSE": "o",
    "Baseline: PCMCI": "o",
    "Baseline: Linear Granger": "o",
    "KAN-Koopman": "P",
    "Koopman + Kausal": "*",
    "Per-pair": "h",
    "Koopman (K-matrix)": "p",
    "Other": ".",
}

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- Left: ROC scatter ---
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

plotted_categories = set()
for _, row in summary.iterrows():
    cat = row["Category"]
    label = cat if cat not in plotted_categories else None
    plotted_categories.add(cat)
    color = category_colors.get(cat, "#bdc3c7")
    marker = category_markers.get(cat, "o")
    ax.scatter(
        row["FPR_mean"], row["TPR_mean"],
        c=color, marker=marker, s=80, zorder=5,
        label=label, edgecolors="black", linewidths=0.5,
    )

ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.set_title("ROC Space: All Rossler Benchmark Methods")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8, loc="lower right")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# --- Right: F1 bar chart (top methods) ---
ax2 = axes[1]
top = summary.head(20).copy()
top = top.sort_values("F1_mean", ascending=True)  # for horizontal bars
colors = [category_colors.get(c, "#bdc3c7") for c in top["Category"]]
bars = ax2.barh(range(len(top)), top["F1_mean"], xerr=top["F1_std"],
                color=colors, edgecolor="black", linewidth=0.5, capsize=3)
ax2.set_yticks(range(len(top)))
ax2.set_yticklabels(top["Method"], fontsize=8)
ax2.set_xlabel("F1 Score (mean ± std)")
ax2.set_title("Top 20 Methods by F1 Score")
ax2.set_xlim(0, 1)
ax2.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("benchmarks/results/rossler_roc_all_methods.png", dpi=300, bbox_inches="tight")
print(f"\nPlot saved to benchmarks/results/rossler_roc_all_methods.png")

# =========================================================================
# PLOT 2: ROC BY DENSITY (separate panels for p=0.2, 0.4, 0.6)
# =========================================================================

# Only use key methods for clarity
key_methods = [
    "Stage 1: Koopman + oCSE",
    "Deriv basis (α=0.05)",
    "PCMCI (raw)",
    "oCSE (raw)",
    "Linear Granger",
    "Phase LASSO (CV)",
    "Phase LASSO (α=0.001)",
    "Koopman + Kausal",
    "Koopman (K-matrix)",
    "PerNode K-matrix",
    "Kausal (per-pair)",
    "Attn Adjacency",
    "Attn-Koopman + PCMCI",
    "Attn-Koopman + oCSE",
    "KAN-Koopman A-matrix",
    "KAN-Koopman + PCMCI",
    "KAN-Koopman + oCSE",
    "Koopman+PhaseLASSO (CV)",
    "Koopman+PhaseBkwd (α=0.05)",
    "Phase oCSE (raw)",
]

densities = [0.2, 0.4, 0.6]
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))

for idx, p_val in enumerate(densities):
    ax = axes2[idx]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

    df_p = df_all[df_all["Parameter"] == p_val]
    summary_p = df_p.groupby("Method")[["TPR", "FPR", "F1"]].mean().reset_index()

    plotted = set()
    for _, row in summary_p.iterrows():
        cat = categorize(row["Method"])
        label = cat if cat not in plotted else None
        plotted.add(cat)
        color = category_colors.get(cat, "#bdc3c7")
        marker = category_markers.get(cat, "o")

        ax.scatter(
            row["FPR"], row["TPR"],
            c=color, marker=marker, s=70, zorder=5,
            label=label, edgecolors="black", linewidths=0.5,
        )

        # Annotate key methods
        if row["Method"] in [
            "Stage 1: Koopman + oCSE", "Deriv basis (α=0.05)",
            "PCMCI (raw)", "oCSE (raw)",
        ]:
            ax.annotate(
                row["Method"].replace("Stage 1: Koopman + oCSE", "Stage 1")
                             .replace("Deriv basis (α=0.05)", "Deriv Bkwd")
                             .replace("PCMCI (raw)", "PCMCI")
                             .replace("oCSE (raw)", "oCSE"),
                (row["FPR"], row["TPR"]),
                textcoords="offset points", xytext=(8, 5),
                fontsize=7, alpha=0.8,
            )

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ER Density p={p_val}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    if idx == 2:
        ax.legend(fontsize=7, loc="lower right")

plt.suptitle("ROC Space by Edge Density — Coupled Rossler Oscillators", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("benchmarks/results/rossler_roc_by_density.png", dpi=300, bbox_inches="tight")
print(f"Plot saved to benchmarks/results/rossler_roc_by_density.png")
