"""Consolidated figure for causal discovery on emergent regimes (all 4 regimes,
merging the main benchmark CSV with the dedicated chimera CSV)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
rd = os.path.join(here, "results")
df = pd.concat([
    pd.read_csv(os.path.join(rd, "emergent_causal_results.csv")),
    pd.read_csv(os.path.join(rd, "emergent_causal_chimera.csv")),
], ignore_index=True)

regimes = ["cyclops", "metastable", "traveling_wave", "chimera"]
methods = ["oCSE", "oCSE+OP", "PCMCI", "Granger", "VARLiNGAM"]
colors = {"oCSE": "#1a7a3a", "oCSE+OP": "#27c463", "PCMCI": "#3498db",
          "Granger": "#9b59b6", "VARLiNGAM": "#e67e22"}

fig, axes = plt.subplots(2, len(regimes), figsize=(4.4 * len(regimes), 8.5), squeeze=False)
for col, reg in enumerate(regimes):
    sub = df[df["Regime"] == reg]
    for ri, metric in enumerate(["FPR", "TPR"]):
        ax = axes[ri][col]
        x = np.arange(len(methods))
        means = [sub[sub["Method"] == m][metric].mean() for m in methods]
        stds = [sub[sub["Method"] == m][metric].std() for m in methods]
        means = [0 if (isinstance(v, float) and np.isnan(v)) else v for v in means]
        ax.bar(x, means, yerr=stds, color=[colors[m] for m in methods], capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.02)
        if ri == 0:
            ax.set_title(reg, fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel(metric, fontsize=12)
        if reg == "cyclops" and metric == "FPR":
            ax.text(0.5, 0.5, "FPR undefined\n(complete graph)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#888")

fig.suptitle("Causal discovery on emergent regimes: oCSE keeps low FPR; "
             "order-parameter conditioning lowers it further",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(rd, "emergent_causal_summary.png")
plt.savefig(out, dpi=180, bbox_inches="tight")
print("saved", out)
