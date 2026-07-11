#!/usr/bin/env python3
"""
Simple figure + stats script.

Input:  causal_discovery_simulation_results.csv
Output: saves figures into ./figs and prints statistical tests at the end.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon

CSV = "causal_discovery_simulation_results.csv"
OUTDIR = "figs"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------
# Load + aggregate (avg over trials first)
# -----------------------
df = pd.read_csv(CSV)

avg = df.groupby(["Experiment", "Parameter", "Method"], as_index=False).agg(
    F1=("F1", "mean"), SHD=("SHD", "mean"), Time=("Time", "mean")
)

# Paired matrix across identical (Experiment, Parameter) conditions
pivot_f1 = avg.pivot_table(
    index=["Experiment", "Parameter"], columns="Method", values="F1"
).dropna()

# Order methods by mean F1 (simple + readable)
method_order = pivot_f1.mean(axis=0).sort_values(ascending=False).index.tolist()

# -----------------------
# FIG 1: Boxplot of F1 by method (paired conditions; trials averaged)
# -----------------------
plt.figure(figsize=(7.2, 3.6))
plt.boxplot(
    [pivot_f1[m].to_numpy() for m in method_order],
    labels=method_order,
    showfliers=False,
)
plt.ylabel("F1 score")
plt.xlabel("Method")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "boxplot_f1_by_method.pdf"), bbox_inches="tight")
plt.savefig(
    os.path.join(OUTDIR, "boxplot_f1_by_method.png"), dpi=600, bbox_inches="tight"
)
plt.close()

# -----------------------
# FIG 2+: Mean F1 vs Parameter, one figure per Experiment
# -----------------------
for exp in sorted(avg["Experiment"].unique()):
    sub = avg[avg["Experiment"] == exp].copy()
    # ensure sorted lines
    sub = sub.sort_values(["Method", "Parameter"])

    plt.figure(figsize=(6.2, 3.6))
    for m in sorted(sub["Method"].unique()):
        mdf = sub[sub["Method"] == m]
        plt.plot(
            mdf["Parameter"],
            mdf["F1"],
            marker="o",
            linewidth=1.2,
            markersize=3,
            label=m,
        )

    plt.xlabel("Parameter")
    plt.ylabel("Mean F1 score")
    plt.legend(frameon=False, ncols=2)
    plt.tight_layout()
    stem = f"line_f1_vs_param_{exp}"
    plt.savefig(os.path.join(OUTDIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUTDIR, f"{stem}.png"), dpi=600, bbox_inches="tight")
    plt.close()

# -----------------------
# FIG 3: Boxplot of SHD by method (lower is better)
# -----------------------
pivot_shd = avg.pivot_table(
    index=["Experiment", "Parameter"], columns="Method", values="SHD"
).dropna()
order_shd = pivot_shd.mean(axis=0).sort_values(ascending=True).index.tolist()

plt.figure(figsize=(7.2, 3.6))
plt.boxplot(
    [pivot_shd[m].to_numpy() for m in order_shd], labels=order_shd, showfliers=False
)
plt.ylabel("SHD (lower is better)")
plt.xlabel("Method")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "boxplot_shd_by_method.pdf"), bbox_inches="tight")
plt.savefig(
    os.path.join(OUTDIR, "boxplot_shd_by_method.png"), dpi=600, bbox_inches="tight"
)
plt.close()

# -----------------------
# FIG 4: Boxplot of runtime by method (log scale)
# -----------------------
pivot_time = avg.pivot_table(
    index=["Experiment", "Parameter"], columns="Method", values="Time"
).dropna()
order_time = pivot_time.mean(axis=0).sort_values(ascending=True).index.tolist()

plt.figure(figsize=(7.2, 3.6))
plt.boxplot(
    [pivot_time[m].to_numpy() for m in order_time], labels=order_time, showfliers=False
)
plt.yscale("log")
plt.ylabel("Runtime (s, log scale)")
plt.xlabel("Method")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "boxplot_time_by_method.pdf"), bbox_inches="tight")
plt.savefig(
    os.path.join(OUTDIR, "boxplot_time_by_method.png"), dpi=600, bbox_inches="tight"
)
plt.close()

# -----------------------
# STATS: Friedman + pairwise Wilcoxon on F1 (paired), with Holm correction
# -----------------------
methods = pivot_f1.columns.tolist()
fried_stat, fried_p = friedmanchisquare(*[pivot_f1[m].to_numpy() for m in methods])

# Pairwise Wilcoxon: compare all pairs (simple + standard)
pairs = []
pvals = []
for i in range(len(methods)):
    for j in range(i + 1, len(methods)):
        a, b = methods[i], methods[j]
        stat, p = wilcoxon(pivot_f1[a], pivot_f1[b])
        pairs.append((a, b, stat))
        pvals.append(p)

pvals = np.array(pvals)

# Holm-Bonferroni adjustment
order = np.argsort(pvals)
holm_adj = np.empty_like(pvals)
m = len(pvals)
for k, idx in enumerate(order):
    holm_adj[idx] = min(1.0, (m - k) * pvals[idx])
# enforce monotonicity (optional but nice)
# (make adjusted p-values non-decreasing in sorted order)
sorted_adj = holm_adj[order].copy()
for k in range(1, m):
    sorted_adj[k] = max(sorted_adj[k], sorted_adj[k - 1])
holm_adj[order] = sorted_adj

print(
    "\n=== Statistical tests on F1 (paired over Experiment×Parameter; trials averaged) ==="
)
print(f"Paired conditions (n) = {pivot_f1.shape[0]}")
print(f"Friedman chi-square = {fried_stat:.6g}, p = {fried_p:.6g}")

print("\nPairwise Wilcoxon signed-rank tests (two-sided), Holm-corrected:")
# Pretty print in ascending raw p
for rank, idx in enumerate(order, start=1):
    a, b, stat = pairs[idx]
    print(
        f"{rank:2d}. {a} vs {b}: "
        f"W={stat:.6g}, p={pvals[idx]:.6g}, p_holm={holm_adj[idx]:.6g}"
    )

print(f"\nSaved figures to: {os.path.abspath(OUTDIR)}")
