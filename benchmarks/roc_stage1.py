"""
ROC curve for Stage 1 (Koopman + oCSE) across many edge densities.
Each density is one operating point in (FPR, TPR) space.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import warnings
import sys
import os

import torch

from causationentropy.datasets.synthetic import (
    generate_graph_topology,
    simulate_rossler,
)

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.rcParams['font.size'] = 12

sys.path.insert(0, "benchmarks")
from deep_koopman_twostage import (
    train_shared_attention_koopman,
    latent_ocse_node_adjacency,
    compute_metrics,
)

# =========================================================================
# CONFIG
# =========================================================================

N_NODES = 5
D_NODE = 5
T = 5000
DT = 0.02
SUBSAMPLE = 5
BURN_IN = 500
EPOCHS = 1000
N_TRIALS = 3
ALPHA = 0.05

DENSITIES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]

# =========================================================================
# MAIN
# =========================================================================

results = []

print("=" * 70)
print("ROC Curve: Stage 1 (Koopman + oCSE) — Density Sweep")
print(f"Device: {device}")
print(f"Densities: {DENSITIES}, {N_TRIALS} trials each")
print("=" * 70)

for p_edge in DENSITIES:
    for trial in range(N_TRIALS):
        seed = 42 + trial * 100
        print(f"\n  p={p_edge}, Trial {trial}, seed={seed}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        G = generate_graph_topology("Erdos-Renyi", N_NODES, {"p_edge": p_edge}, seed)
        true_adj = nx.to_numpy_array(G).astype(int)

        traj, _ = simulate_rossler(
            G=G, T=T, dt=DT, rho=0.2, seed=seed,
            a=0.2, b=0.2, c=5.7,
            init_scale=1.0, noise_std=0.0, burn_in=BURN_IN,
            normalize_by_indegree=False, coupling_on="x",
        )
        traj_sub = traj[::SUBSAMPLE]

        n_true = int(true_adj.sum())
        n_possible = N_NODES * (N_NODES - 1)
        print(f"    True edges: {n_true}/{n_possible}")

        print(f"    Training Koopman...")
        model, data_mean, data_std = train_shared_attention_koopman(
            traj_sub, n_nodes=N_NODES, d_node=D_NODE, d_head=16,
            epochs=EPOCHS, lr=1e-3, batch_size=256,
            lambda_pred=1.0, lambda_lin=0.5, lambda_sparse=0.01,
            print_every=500,
        )

        T_sub = traj_sub.shape[0]
        data_flat = traj_sub.reshape(T_sub, N_NODES * 3)
        data_norm = (data_flat - data_mean) / data_std
        Z_nodes = model.encode_per_node(data_norm)

        adj = latent_ocse_node_adjacency(
            Z_nodes, N_NODES, D_NODE, alpha=ALPHA, tau_max=1,
        )

        m = compute_metrics(adj, true_adj, 0)
        m.update({"Density": p_edge, "Trial": trial})
        results.append(m)
        print(f"    TP={m['TP']:2d}  FP={m['FP']:2d}  FN={m['FN']:2d}  "
              f"TPR={m['TPR']:.2f}  FPR={m['FPR']:.2f}  F1={m['F1']:.3f}")

df = pd.DataFrame(results)
df.to_csv("benchmarks/results/stage1_roc_density_sweep.csv", index=False)

# =========================================================================
# PLOT (can also run standalone from CSV)
# =========================================================================

try:
    df
except NameError:
    df = pd.read_csv("benchmarks/results/stage1_roc_density_sweep.csv")

sweep = df.groupby("Density")[["TPR", "FPR", "F1"]].mean().reset_index()
sweep = sweep.sort_values("FPR")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random", linewidth=1)

ax.plot(
    sweep["FPR"], sweep["TPR"],
    "D-", color="#e74c3c", markersize=8, linewidth=2.5,
    label="Stage 1 (Koopman+oCSE)", zorder=5,
    markeredgecolor="black", markeredgewidth=0.5,
)

# Annotate only a few key points to avoid overlap
from adjustText import adjust_text
texts = []
for _, row in sweep.iterrows():
    texts.append(ax.text(
        row["FPR"], row["TPR"],
        f"p={row['Density']:.2f}",
        fontsize=8, alpha=0.8,
    ))
adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))

ax.set_xlabel("False Positive Rate (FPR)", fontsize=13)
ax.set_ylabel("True Positive Rate (TPR)", fontsize=13)
ax.set_title("ROC: Koopman + oCSE (Stage 1) — Varying Edge Density", fontsize=14)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc="lower right")

plt.tight_layout()
plt.savefig("benchmarks/results/stage1_roc_curve.png", dpi=300, bbox_inches="tight")
print(f"\nPlot saved to benchmarks/results/stage1_roc_curve.png")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(sweep.to_string(index=False, float_format="%.3f"))
