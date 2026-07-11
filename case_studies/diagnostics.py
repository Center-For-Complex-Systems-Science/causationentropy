"""
Diagnostic Plots: Pro-Reg -> Background Checks (lag 7)
======================================================
Visualize the nature of the dependency that oCSE detected:
  1. Scatter plot (raw relationship)
  2. Shuffle null distribution vs observed CMI
  3. Gaussian vs k-NN CMI comparison
  4. Spearman rank correlation (nonlinear alternative to Pearson)
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from causationentropy.core.discovery import shuffle_test


# ── Load & prepare data (same pipeline as tiktok_analysis.py) ────────

df = pd.read_csv("data/all_dailies_reg.csv")
proquest_df = pd.read_csv("data/proquest.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.floor("D")
proquest_df["PubDate"] = pd.to_datetime(proquest_df["PubDate"])
proquest_df["date"] = proquest_df["PubDate"].dt.floor("D")

articles_per_day = proquest_df.groupby("date").size().reset_index(name="Media Articles")

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

decomp = seasonal_decompose(merged["Background Checks"], model="additive", period=365)
merged["Deseasonalized Background Checks"] = decomp.observed - decomp.seasonal

# Standardize (same as discovery)
analysis_cols = ["Deseasonalized Background Checks", "Anti-Reg", "Pro-Reg", "Media Articles"]
data = merged[analysis_cols].copy().dropna()

scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(data), columns=data.columns, index=data.index,
)

# Build aligned arrays: X = Pro-Reg_{t-7}, Y = Background Checks_t
lag = 7
X_series = data_scaled["Pro-Reg"].shift(lag).dropna()
Y_series = data_scaled["Deseasonalized Background Checks"].loc[X_series.index]

# Also get all the parents of D (the conditioning set from discovery)
# Parents: Media Articles lag 7, Pro-Reg lag 7
# For the P->D edge, we condition on everything EXCEPT P_lag7, i.e. just M_lag7
Z_series = data_scaled["Media Articles"].shift(lag).loc[X_series.index]

X = X_series.values.reshape(-1, 1)
Y = Y_series.values.reshape(-1, 1)
Z = Z_series.values.reshape(-1, 1)

n = len(X)
print(f"Aligned samples: {n}")


# ── 1. Scatter plot ──────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Raw scatter
axes[0].scatter(X, Y, alpha=0.15, s=8, color="#2E86AB")
axes[0].set_xlabel("Pro-Reg (lag 7, standardized)")
axes[0].set_ylabel("Background Checks (standardized)")
axes[0].set_title("Raw Scatter")

# Hexbin density
hb = axes[1].hexbin(X.ravel(), Y.ravel(), gridsize=30, cmap="Blues", mincnt=1)
axes[1].set_xlabel("Pro-Reg (lag 7, standardized)")
axes[1].set_ylabel("Background Checks (standardized)")
axes[1].set_title("Density (hexbin)")
plt.colorbar(hb, ax=axes[1], label="Count")

# Binned means (nonlinearity check)
n_bins = 20
x_flat = X.ravel()
y_flat = Y.ravel()
bin_edges = np.percentile(x_flat, np.linspace(0, 100, n_bins + 1))
bin_centers = []
bin_means = []
bin_sems = []
for i in range(n_bins):
    mask = (x_flat >= bin_edges[i]) & (x_flat < bin_edges[i + 1])
    if i == n_bins - 1:
        mask = (x_flat >= bin_edges[i]) & (x_flat <= bin_edges[i + 1])
    if mask.sum() > 0:
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        bin_means.append(y_flat[mask].mean())
        bin_sems.append(y_flat[mask].std() / np.sqrt(mask.sum()))

axes[2].errorbar(bin_centers, bin_means, yerr=bin_sems, fmt="o-", color="#A23B72",
                 capsize=3, markersize=5)
axes[2].axhline(0, color="gray", linestyle="--", alpha=0.5)
axes[2].set_xlabel("Pro-Reg (lag 7, standardized)")
axes[2].set_ylabel("Mean Background Checks")
axes[2].set_title("Binned Means (nonlinearity check)")

plt.tight_layout()
plt.savefig("results/diagnostic_scatter.pdf", dpi=300)
plt.close()
print("Saved: results/diagnostic_scatter.pdf")


# ── 2. CMI: Gaussian vs k-NN vs KDE ─────────────────────────────────

print("\n--- CMI Estimates (P_lag7 -> D, no conditioning) ---")
cmi_gauss = conditional_mutual_information(X, Y, method="gaussian")
cmi_knn = conditional_mutual_information(X, Y, method="knn", k=6)
cmi_kde = conditional_mutual_information(X, Y, method="kde")

print(f"  Gaussian:  {cmi_gauss:.6f}")
print(f"  k-NN:      {cmi_knn:.6f}")
print(f"  KDE:       {cmi_kde:.6f}")

print("\n--- CMI Estimates (P_lag7 -> D | M_lag7) ---")
cmi_gauss_cond = conditional_mutual_information(X, Y, Z, method="gaussian")
cmi_knn_cond = conditional_mutual_information(X, Y, Z, method="knn", k=6)
cmi_kde_cond = conditional_mutual_information(X, Y, Z, method="kde")

print(f"  Gaussian:  {cmi_gauss_cond:.6f}")
print(f"  k-NN:      {cmi_knn_cond:.6f}")
print(f"  KDE:       {cmi_kde_cond:.6f}")


# ── 3. Shuffle null distribution ─────────────────────────────────────

n_shuffles = 1000

print(f"\n--- Running shuffle test ({n_shuffles} permutations) ---")

# Unconditional: I(P_lag7; D)
result_uncond = shuffle_test(
    X, Y, Z=None, observed_cmi=cmi_gauss,
    alpha=0.05, n_shuffles=n_shuffles, rng=42, information="gaussian",
)

# Conditional: I(P_lag7; D | M_lag7)
result_cond = shuffle_test(
    X, Y, Z=Z, observed_cmi=cmi_gauss_cond,
    alpha=0.05, n_shuffles=n_shuffles, rng=42, information="gaussian",
)

# Also generate the null distributions explicitly for plotting
rng = np.random.default_rng(42)
null_uncond = np.empty(n_shuffles)
null_cond = np.empty(n_shuffles)
for i in range(n_shuffles):
    perm = rng.permutation(n)
    X_perm = X[perm]
    null_uncond[i] = conditional_mutual_information(X_perm, Y, method="gaussian")
    null_cond[i] = conditional_mutual_information(X_perm, Y, Z, method="gaussian")

print(f"\n  Unconditional I(P_lag7; D):")
print(f"    Observed: {cmi_gauss:.6f}")
print(f"    Threshold (95th): {result_uncond['Threshold']:.6f}")
print(f"    p-value: {result_uncond['P_value']:.4f}")
print(f"    Significant: {result_uncond['Pass']}")

print(f"\n  Conditional I(P_lag7; D | M_lag7):")
print(f"    Observed: {cmi_gauss_cond:.6f}")
print(f"    Threshold (95th): {result_cond['Threshold']:.6f}")
print(f"    p-value: {result_cond['P_value']:.4f}")
print(f"    Significant: {result_cond['Pass']}")


# Plot null distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(null_uncond, bins=50, color="#2E86AB", alpha=0.7, edgecolor="white")
axes[0].axvline(cmi_gauss, color="red", linewidth=2, label=f"Observed = {cmi_gauss:.5f}")
axes[0].axvline(result_uncond["Threshold"], color="orange", linewidth=2, linestyle="--",
                label=f"95th pct = {result_uncond['Threshold']:.5f}")
axes[0].set_xlabel("CMI (Gaussian)")
axes[0].set_ylabel("Count")
axes[0].set_title(f"I(Pro-Reg_lag7 ; BG Checks)\np = {result_uncond['P_value']:.4f}")
axes[0].legend(fontsize=9)

axes[1].hist(null_cond, bins=50, color="#A23B72", alpha=0.7, edgecolor="white")
axes[1].axvline(cmi_gauss_cond, color="red", linewidth=2, label=f"Observed = {cmi_gauss_cond:.5f}")
axes[1].axvline(result_cond["Threshold"], color="orange", linewidth=2, linestyle="--",
                label=f"95th pct = {result_cond['Threshold']:.5f}")
axes[1].set_xlabel("CMI (Gaussian)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"I(Pro-Reg_lag7 ; BG Checks | Media_lag7)\np = {result_cond['P_value']:.4f}")
axes[1].legend(fontsize=9)

plt.suptitle("Shuffle Null Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/diagnostic_shuffle.pdf", dpi=300)
plt.close()
print("\nSaved: results/diagnostic_shuffle.pdf")


# ── 4. Correlation comparison ────────────────────────────────────────

from scipy.stats import pearsonr

r_pearson, p_pearson = pearsonr(X.ravel(), Y.ravel())
r_spearman, p_spearman = spearmanr(X.ravel(), Y.ravel())

print(f"\n--- Correlation Comparison (P_lag7 vs D) ---")
print(f"  Pearson:   r = {r_pearson:+.4f},  p = {p_pearson:.6f}")
print(f"  Spearman:  r = {r_spearman:+.4f},  p = {p_spearman:.6f}")
print(f"  Gaussian CMI:  {cmi_gauss:.6f}")
print(f"  k-NN CMI:      {cmi_knn:.6f}")

# If k-NN CMI >> Gaussian CMI, the dependency is nonlinear
ratio = cmi_knn / cmi_gauss if cmi_gauss > 0 else float("inf")
print(f"\n  k-NN / Gaussian CMI ratio: {ratio:.2f}")
if ratio > 2:
    print("  => Substantial nonlinear component detected")
elif ratio > 1.3:
    print("  => Mild nonlinear component")
else:
    print("  => Relationship is mostly linear (or absent)")
