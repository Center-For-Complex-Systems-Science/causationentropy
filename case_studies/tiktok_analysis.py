"""
TikTok Causal Discovery Analysis
=================================
Discovers causal relationships between TikTok political content,
media coverage, and gun background checks using oCSE.
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pingouin as pg
from collections import defaultdict
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from causationentropy.core import discover_network


# ── 1. Load & merge data ─────────────────────────────────────────────

df = pd.read_csv("data/all_dailies_reg.csv")
proquest_df = pd.read_csv("data/proquest.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.floor("D")

proquest_df["PubDate"] = pd.to_datetime(proquest_df["PubDate"])
proquest_df["date"] = proquest_df["PubDate"].dt.floor("D")

articles_per_day = (
    proquest_df.groupby("date").size().reset_index(name="Media Articles")
)

merged = (
    df.merge(articles_per_day, on="date", how="left")
      .drop(columns=["PubDate"], errors="ignore")
      .rename(columns={
          "bg_checks": "Background Checks",
          "anti_reg": "Anti-Reg",
          "pro_reg": "Pro-Reg",
      })
)
merged["Media Articles"] = merged["Media Articles"].fillna(0)
merged = merged.set_index("date").sort_index().asfreq("D")


# ── 2. Deseasonalize background checks ───────────────────────────────

decomp = seasonal_decompose(merged["Background Checks"], model="additive", period=365)
merged["Deseasonalized Background Checks"] = decomp.observed - decomp.seasonal


# ── 3. Plot raw time series ──────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Time Series Analysis", fontsize=16, fontweight="bold", y=0.995)

colors = ["#2E86AB", "#A23B72", "#F18F01", "#06A77D"]
variables = ["Background Checks", "Anti-Reg", "Pro-Reg", "Media Articles"]

for ax, var, color in zip(axes, variables, colors):
    ax.plot(merged.index, merged[var], color=color, linewidth=1.5, alpha=0.8)
    ax.set_ylabel(var, fontsize=11, fontweight="600")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Date", fontsize=11, fontweight="600")
plt.tight_layout()
plt.savefig("results/timeseries.pdf", dpi=300)
plt.close()


# ── 4. Causal discovery ──────────────────────────────────────────────

analysis_cols = ["Deseasonalized Background Checks", "Anti-Reg", "Pro-Reg", "Media Articles"]
data = merged[analysis_cols].copy()

scaler = StandardScaler()
data = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns,
    index=data.index,
)

network = discover_network(
    data,
    information="gaussian",
    max_lag=7,
    alpha_forward=0.005,
    alpha_backward=0.005,
    n_shuffles=200,
)


# ── 5. Plot full causal network ──────────────────────────────────────

PLOT_CONFIG = {
    "node_size": 58000,
    "arrowsize": 100,
    "label_fontsize": 25,
    "figsize": (14, 14),
    "edge_width_range": (1.0, 6.0),
    "node_linewidth": 2.0,
    "colormaps": ["Blues", "Greens", "Oranges", "Purples", "Reds", "Greys", "YlOrBr"],
}


def plot_causal_network(G, pos, title, save_path, config=PLOT_CONFIG):
    edge_data = defaultdict(list)
    for u, v, k, d in G.edges(keys=True, data=True):
        lag = d.get("lag", 0)
        cmi = max(0.0, float(d.get("cmi", 0.0)))
        edge_data[lag].append((u, v, cmi))

    fig, ax = plt.subplots(figsize=config["figsize"])

    nx.draw_networkx_nodes(
        G, pos,
        node_size=config["node_size"],
        node_color="white",
        edgecolors="black",
        linewidths=config["node_linewidth"],
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=config["label_fontsize"], ax=ax)

    for i, (lag, edges) in enumerate(sorted(edge_data.items())):
        if not edges:
            continue

        cmis = np.array([e[2] for e in edges])
        max_cmi = cmis.max() if cmis.max() > 0 else 1.0

        w_lo, w_hi = config["edge_width_range"]
        widths = w_lo + (w_hi - w_lo) * (cmis / max_cmi)

        cmap = plt.cm.get_cmap(config["colormaps"][i % len(config["colormaps"])])
        color = cmap(0.7)
        colors = [color] * len(edges)

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(e[0], e[1]) for e in edges],
            edge_color=colors,
            width=widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=config["arrowsize"],
            connectionstyle=f"arc3,rad={0.15 * (i - len(edge_data) / 2)}",
            node_size=config["node_size"],
            ax=ax,
        )

    legend_elements = []
    for i, lag in enumerate(sorted(edge_data.keys())):
        cmap = plt.cm.get_cmap(config["colormaps"][i % len(config["colormaps"])])
        legend_elements.append(Patch(facecolor=cmap(0.7), edgecolor="black", label=f"Lag {lag}"))

    ax.legend(
        handles=legend_elements, loc="upper right", fontsize=18,
        title="Lag Groups", title_fontsize=20, framealpha=0.9,
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=28, pad=20)
    plt.margins(0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


pos = nx.circular_layout(network)
plot_causal_network(network, pos, "TikTok Causal Network", "results/network.pdf")


# ── 6. Prune to key causal pathway ───────────────────────────────────

PRO = "Pro-Reg"
MEDIA = "Media Articles"
BC = "Deseasonalized Background Checks"

allowed_pairs = {(PRO, MEDIA), (MEDIA, BC), (PRO, BC)}

pruned = nx.MultiDiGraph()
pruned.add_nodes_from([PRO, MEDIA, BC])

for u, v, k, d in network.edges(keys=True, data=True):
    if (u, v) in allowed_pairs:
        pruned.add_edge(u, v, key=k, **d)

pos_pruned = nx.circular_layout(pruned)
plot_causal_network(pruned, pos_pruned, "Pruned Causal Network", "results/pruned.pdf")


# ── 7. Build lagged dataframe for validation ─────────────────────────

df_lagged = pd.DataFrame(index=merged.index)

for node in pruned.nodes:
    if node in merged.columns:
        df_lagged[node] = merged[node]

for u, v, k, d in network.edges(keys=True, data=True):
    lag = int(d["lag"])
    col_name = f"{u}_lag{lag}"
    df_lagged[col_name] = merged[u].shift(lag)

df_lagged = df_lagged.dropna()


# ── 8. Partial correlation validation ────────────────────────────────

Y = "Deseasonalized Background Checks"

parents = []
for u, v, key, d in network.in_edges(Y, keys=True, data=True):
    parents.append((u, d["lag"]))

print(f"\nParents of '{Y}': {parents}\n")

x_lag = 7
X_col = f"Pro-Reg_lag{x_lag}"

covars = [
    f"{var}_lag{lag}"
    for (var, lag) in parents
    if not (var == "Pro-Reg" and lag == x_lag)
]

cols_needed = [Y, X_col] + covars
df_test = df_lagged[cols_needed].dropna()

result = pg.partial_corr(data=df_test, x=X_col, y=Y, covar=covars)
print("Partial correlation: Pro-Reg (lag 7) -> Background Checks")
print(result)
