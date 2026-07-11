"""
Causal discovery on emergent regimes  (Stage 2)
===============================================

Runs causal-network reconstruction on the emergent regimes generated in
`emergent_regimes.py` (chimera, traveling wave, metastability, cyclops), scoring
each method against the KNOWN ground-truth coupling graph (binary support of W).

Two arms:

  ARM A -- standard comparison, every method on the sine basis:
      oCSE, PCMCI, Linear Granger, VARLiNGAM.

  ARM B -- order-parameter conditioning (the new idea):
      oCSE+OP feeds the global mean-field order parameter Z_m(t) = <exp(i m theta)>
      into oCSE's conditioning set. In synchronized / emergent regimes the mean
      field is the common driver that manufactures spurious correlations between
      non-neighbours. Because oCSE conditions on its whole set, conditioning on
      Z(t) should block that common-cause path and remove false-positive links.

Adjacency convention: A[source, target]; ground truth GT[j, i] = 1 iff W[i, j] > 0.

NOTE on cyclops: its coupling is all-to-all, so the ground-truth graph is complete
-> there are no true negatives -> FPR is undefined (reported as NaN). Only
recall / precision are meaningful there.

Produces:
    benchmarks/results/emergent_causal_results.csv
    benchmarks/results/emergent_causal.png
"""
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from causationentropy.core.discovery import standard_optimal_causation_entropy
from causationentropy.graph import pcmci_to_networkx
from causationentropy.datasets.synthetic import prepare_kuramoto_data_for_causal_discovery

from kuramoto_roc import extract_pcmci_adjacency, compute_metrics
from basis_all_methods import build_basis_series, granger_basis, varlingam_lag1_adj
import emergent_regimes as er


# =============================================================================
# ORDER-PARAMETER CONDITIONING SET
# =============================================================================

def order_parameter_features(theta_rows, harmonics=(1, 2)):
    """[Re Z_m, Im Z_m] for each harmonic m, per time row.
    Z_m(t) = (1/n) sum_j exp(i m theta_j(t))."""
    cols = []
    for m in harmonics:
        Z = np.exp(1j * m * theta_rows).mean(axis=1)
        cols.append(Z.real)
        cols.append(Z.imag)
    return np.column_stack(cols)


def run_targeted_oce(X_basis, basis_meta, alpha, n_shuffles, cond_basis=None):
    """oCSE forward/backward on velocity targets only (node-level adjacency).

    cond_basis : optional (rows == X_basis rows) array of EXTRA conditioning
    variables (e.g. the order parameter) added to every causation-entropy test
    but never selectable as an edge. This is ARM B when supplied.
    """
    n = basis_meta["n"]
    basis_map = basis_meta["basis_map"]
    X_lagged = X_basis[:-1, :]
    Y_all = X_basis[1:, :]
    cond_lagged = None if cond_basis is None else cond_basis[:-1, :]
    rng = np.random.default_rng(42)

    A_node = np.zeros((n, n), dtype=int)
    for i in range(n):
        Y = Y_all[:, [i]]
        Z_init = X_lagged[:, [i]]                      # own past velocity
        if cond_lagged is not None:
            Z_init = np.hstack([Z_init, cond_lagged])  # + order parameter
        S = standard_optimal_causation_entropy(
            X_lagged, Y, Z_init, rng,
            alpha1=alpha, alpha2=alpha,
            n_shuffles=n_shuffles, information="gaussian",
        )
        for s in S:
            if s < n:                                  # skip self-velocity cols
                continue
            target, source = basis_map[s - n]
            if target == i:
                A_node[source, i] = 1
            elif source == i:
                A_node[target, i] = 1
    return A_node


# =============================================================================
# METRICS  (handle the degenerate complete-graph / no-negatives case)
# =============================================================================

def regime_metrics(pred, true_adj):
    m = compute_metrics(pred, true_adj)
    A = (true_adj != 0).astype(int)
    n = A.shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    n_neg = int((A[offdiag] == 0).sum())
    if n_neg == 0:                                     # complete graph -> no TN
        m["FPR"] = float("nan")
    return m


def ground_truth_from_W(W):
    """GT[source, target] = 1 iff source drives target, i.e. W[target, source] > 0."""
    GT = (W.T > 0).astype(int)
    np.fill_diagonal(GT, 0)
    return GT


# =============================================================================
# CONFIG
# =============================================================================

REGIME_ORDER = ["cyclops", "metastable", "traveling_wave", "chimera"]  # small N first
REGIME_N = {"chimera": 36, "traveling_wave": 24}  # shrink the expensive regimes
SEEDS = [0, 1]
ALPHA = 0.05
N_SHUFFLES = 20             # tractable-run setting (raise for final numbers)
T_CAP = 2000                # cap time samples to bound oCSE CMI cost
PCMCI_MAX_BASIS = 1300      # skip PCMCI above this basis dimension (intractable)
OP_HARMONICS = (1, 2)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "emergent_causal_results.csv")

    print("=" * 74)
    print("Causal discovery on emergent regimes  (Arm A: all methods; Arm B: oCSE+OP)")
    print("=" * 74)

    rows = []
    t_start = time.time()
    for regime in REGIME_ORDER:
        for seed in SEEDS:
            bkw = {"seed": seed}
            if regime in REGIME_N:
                bkw["n"] = REGIME_N[regime]
            cfg = getattr(er, f"regime_{regime}")(**bkw)
            cfg.pop("name")
            cfg.pop("comm", None)
            W = cfg["W"]
            n = W.shape[0]
            true_adj = ground_truth_from_W(W)
            n_edges = int(true_adj.sum())

            theta, _ = er.simulate_phase_oscillators(**cfg)
            if T_CAP:
                theta = theta[:T_CAP]
            dt = cfg["dt"]

            X_basis, basis_meta, var_names = prepare_kuramoto_data_for_causal_discovery(theta, dt=dt)
            basis_dim = X_basis.shape[1]
            T_eff = X_basis.shape[0]
            _, _, _, M, basis_map = build_basis_series(theta, dt)
            cond_basis = order_parameter_features(theta[:T_eff], harmonics=OP_HARMONICS)

            common = {"Regime": regime, "Seed": seed, "N": n,
                      "N_true_edges": n_edges, "basis_dim": basis_dim}

            def record(method, adj, t0, note=""):
                if adj is None:
                    m = {"TP": np.nan, "FP": np.nan, "FN": np.nan, "TPR": np.nan,
                         "FPR": np.nan, "Precision": np.nan, "F1": np.nan}
                else:
                    m = regime_metrics(adj, true_adj)
                m.update(common)
                m["Method"] = method
                m["Time"] = time.time() - t0
                m["Note"] = note
                rows.append(m)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                return m

            # ---- ARM A: oCSE ----
            t0 = time.time()
            adj = run_targeted_oce(X_basis, basis_meta, ALPHA, N_SHUFFLES)
            mA = record("oCSE", adj, t0)

            # ---- ARM B: oCSE + order parameter ----
            t0 = time.time()
            adj = run_targeted_oce(X_basis, basis_meta, ALPHA, N_SHUFFLES, cond_basis=cond_basis)
            mB = record("oCSE+OP", adj, t0)

            # ---- PCMCI (basis) ----
            t0 = time.time()
            if basis_dim <= PCMCI_MAX_BASIS:
                try:
                    dframe = pp.DataFrame(X_basis, datatime={0: np.arange(T_eff)}, var_names=var_names)
                    pcmci = PCMCI(dataframe=dframe, cond_ind_test=ParCorr(), verbosity=0)
                    res = pcmci.run_pcmci(tau_min=1, tau_max=1, pc_alpha=ALPHA)
                    adj = extract_pcmci_adjacency(pcmci_to_networkx(res), basis_meta["n"], basis_meta["basis_map"])
                    record("PCMCI", adj, t0)
                except Exception as e:
                    record("PCMCI", None, t0, note=f"failed:{type(e).__name__}")
            else:
                record("PCMCI", None, t0, note="skipped:basis_too_large")

            # ---- Granger (basis) ----
            t0 = time.time()
            try:
                record("Granger", granger_basis(theta, dt, n, alpha=ALPHA), t0)
            except Exception as e:
                record("Granger", None, t0, note=f"failed:{type(e).__name__}")

            # ---- VARLiNGAM (basis) ----
            t0 = time.time()
            try:
                record("VARLiNGAM", varlingam_lag1_adj(M, n, basis_map=basis_map), t0)
            except Exception as e:
                record("VARLiNGAM", None, t0, note=f"failed:{type(e).__name__}")

            el = time.time() - t_start
            fpr_a = "nan" if np.isnan(mA["FPR"]) else f"{mA['FPR']:.2f}"
            fpr_b = "nan" if np.isnan(mB["FPR"]) else f"{mB['FPR']:.2f}"
            print(f"[{el:6.0f}s] {regime:14s} seed{seed} N={n:>2} edges={n_edges:>3} | "
                  f"oCSE F1={mA['F1']:.2f} FPR={fpr_a} -> "
                  f"oCSE+OP F1={mB['F1']:.2f} FPR={fpr_b}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path} ({len(df)} rows)")

    # ---- summary ----
    print("\n" + "=" * 74)
    print("SUMMARY  (mean over seeds)")
    print("=" * 74)
    summ = df.groupby(["Regime", "Method"])[["TPR", "FPR", "F1"]].mean()
    print(summ.round(3).to_string())

    _plot(df, os.path.join(results_dir, "emergent_causal.png"))


def _plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["oCSE", "oCSE+OP", "PCMCI", "Granger", "VARLiNGAM"]
    colors = {"oCSE": "#1a7a3a", "oCSE+OP": "#27c463", "PCMCI": "#3498db",
              "Granger": "#9b59b6", "VARLiNGAM": "#e67e22"}
    regimes = [r for r in REGIME_ORDER if r in df["Regime"].unique()]

    fig, axes = plt.subplots(2, len(regimes), figsize=(4.2 * len(regimes), 9), squeeze=False)
    for col, regime in enumerate(regimes):
        sub = df[df["Regime"] == regime]
        for ri, metric in enumerate(["F1", "FPR"]):
            ax = axes[ri][col]
            means = [sub[sub["Method"] == m][metric].mean() for m in methods]
            stds = [sub[sub["Method"] == m][metric].std() for m in methods]
            x = np.arange(len(methods))
            ax.bar(x, means, yerr=stds, color=[colors[m] for m in methods], capsize=3)
            ax.set_xticks(x); ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            if ri == 0:
                ax.set_title(regime)
            if col == 0:
                ax.set_ylabel(metric)
            if metric == "F1":
                ax.set_ylim(0, 1.05)
    fig.suptitle("Causal discovery on emergent regimes (Arm A vs oCSE+OP)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print("Saved", path)


if __name__ == "__main__":
    main()
