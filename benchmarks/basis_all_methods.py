"""
Does the sine basis, or the method, drive oCSE's advantage?
=========================================================

oCSE "linearizes" coupled-Kuramoto causal discovery by regressing onto the
physics-informed basis  s_{i<-j}(t) = sin(theta_j(t) - theta_i(t))  instead of
the raw phases.  oCSE and PCMCI already use this basis; Linear Granger and
VARLiNGAM, as benchmarked in the hero figure, use the raw phase series.

This script applies the SAME sine basis to every method and reports node-edge
recovery, so the basis effect can be separated from the method:

    oCSE      (basis)        - conditional causation-entropy selection
    PCMCI     (basis)        - conditional independence on the lagged basis
    Granger   (raw vs basis) - pairwise F-test
    VARLiNGAM (raw vs basis) - lag-1 LiNGAM structural model

Design matrix (per node i), one-step prediction:
    target   y_i      = v_i(t+1)                 (phase velocity, next step)
    own past p_i      = v_i(t)
    feature  s_{i<-j} = sin(theta_j(t) - theta_i(t))   -> node edge  j -> i

Adjacency convention (matches kuramoto_roc): A[source, target] = A[j, i] = 1.

Produces:
    benchmarks/results/basis_all_methods_results.csv
    benchmarks/results/basis_all_methods.png
"""
import os
import time
import warnings

import numpy as np
import networkx as nx
import pandas as pd
import lingam
from sklearn.linear_model import LinearRegression
from scipy import stats

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from kuramoto_roc import (
    run_targeted_oce,
    extract_pcmci_adjacency,
    linear_granger,
    compute_metrics,
)
from causationentropy.graph import pcmci_to_networkx
from causationentropy.datasets.synthetic import (
    generate_graph_topology,
    simulate_kuramoto,
    prepare_kuramoto_data_for_causal_discovery,
    phase_velocity,
)

warnings.filterwarnings("ignore")


# =============================================================================
# BASIS BUILDERS
# =============================================================================

def build_basis_series(theta, dt):
    """
    Time-aligned design pieces for the sine basis.

    Returns
    -------
    y_next : (T-2, n)            target velocities v_i(t+1)
    p_own  : (T-2, n)            own past v_i(t)
    s_prev : (T-2, n*(n-1))      features s_{i<-j}(t) = sin(theta_j - theta_i)
    M      : (T-1, n + n*(n-1))  [v(t), s(t)] aligned for VARLiNGAM lag-1
    basis_map : list[(i, j)]     column idx -> (target i, source j)
    """
    T, n = theta.shape
    v = phase_velocity(theta, dt)          # (T-1, n), v aligned with th below
    th = theta[:-1]                        # (T-1, n)

    cols, basis_map = [], []
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            cols.append(np.sin(th[:, j] - th[:, i]))   # s_{i<-j} aligned with v
            basis_map.append((i, j))
    S = np.stack(cols, axis=1)             # (T-1, n*(n-1))

    y_next = v[1:]                         # v(t+1)   (T-2, n)
    p_own = v[:-1]                         # v(t)     (T-2, n)
    s_prev = S[:-1]                        # s(t)     (T-2, n*(n-1))
    M = np.concatenate([v, S], axis=1)     # [v(t), s(t)]  (T-1, .)
    return y_next, p_own, s_prev, M, basis_map


# =============================================================================
# BASIS-AWARE METHODS
# =============================================================================

def granger_basis(theta, dt, n, alpha=0.05):
    """
    Pairwise Granger on the sine basis. For each ordered pair (i, j) test whether
    the coupling feature s_{i<-j}(t) improves prediction of v_i(t+1) beyond v_i's
    own past v_i(t). Significant -> edge j -> i.  Parallels the raw linear_granger.
    """
    y_next, p_own, s_prev, _, basis_map = build_basis_series(theta, dt)
    adj = np.zeros((n, n), dtype=int)
    n_obs = y_next.shape[0]
    for idx, (i, j) in enumerate(basis_map):
        y = y_next[:, i]
        X_r = p_own[:, [i]]
        reg_r = LinearRegression().fit(X_r, y)
        rss_r = np.sum((y - reg_r.predict(X_r)) ** 2)

        X_u = np.column_stack([p_own[:, i], s_prev[:, idx]])
        reg_u = LinearRegression().fit(X_u, y)
        rss_u = np.sum((y - reg_u.predict(X_u)) ** 2)

        df_u = n_obs - 2
        if rss_u <= 0 or df_u <= 0:
            continue
        f_stat = ((rss_r - rss_u) / 1) / (rss_u / df_u)
        p_value = 1 - stats.f.cdf(f_stat, 1, df_u)
        if p_value < alpha:
            adj[j, i] = 1   # source j -> target i
    return adj


def varlingam_lag1_adj(series, n, basis_map=None, thr=1e-8):
    """
    Fit VARLiNGAM(lags=1) on `series` and return a node-level adjacency.

    If basis_map is None, `series` is the raw phases (n columns): lag-1 effect of
    theta_j(t-1) on theta_i(t) -> edge j -> i.

    If basis_map is given, `series` is M = [v(t), s(t)] (n + n*(n-1) columns):
    lag-1 effect of feature s_{i<-j}(t-1) on velocity v_i(t) -> edge j -> i.

    criterion=None pins the model to lags=1 and skips VAR order selection, which
    is ill-conditioned on the sign-antisymmetric sine basis (s_{i<-j} = -s_{j<-i}).
    """
    model = lingam.VARLiNGAM(lags=1, criterion=None)
    try:
        model.fit(series)
    except Exception as e:                  # singular / non-PD covariance, etc.
        print("    [VARLiNGAM fit failed: %s]" % type(e).__name__, flush=True)
        return None
    B1 = model.adjacency_matrices_[1]      # B1[a, b]: effect of var b(t-1) on a(t)
    adj = np.zeros((n, n), dtype=int)
    if basis_map is None:
        for i in range(n):
            for j in range(n):
                if i != j and abs(B1[i, j]) > thr:
                    adj[j, i] = 1
    else:
        for idx, (i, j) in enumerate(basis_map):
            if abs(B1[i, n + idx]) > thr:   # s_{i<-j}(t-1) -> v_i(t)
                adj[j, i] = 1
    return adj


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL = {
    "n_nodes": 6,
    "rho": 0.3,
    "T": 4000,
    "dt": 0.05,
    "omega_std": 1.0,
    "phase_noise_std": 0.02,
    "burn_in": 50,
    "alpha": 0.05,
    "n_shuffles": 100,
}
TOPOS = {
    "Hub": ("Scale-Free", {"m_attachment": 2}),
    "Small-World": ("Small-World", {"k_neighbors": 2, "p_rewire": 0.3}),
    "Erdos-Renyi": ("Erdos-Renyi", {"p_edge": 0.35}),
}
SEEDS = [7, 42, 142, 242, 342]


def main():
    g = GLOBAL
    n = g["n_nodes"]
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 72)
    print("Sine basis applied to ALL methods (Kuramoto, n=%d, rho=%.2f)" % (n, g["rho"]))
    print("=" * 72)

    rows = []
    for tname, (typ, base_params) in TOPOS.items():
        for seed in SEEDS:
            np.random.seed(seed)
            params = dict(base_params); params["rho"] = g["rho"]
            G = generate_graph_topology(typ, n, params, seed)
            true_adj = nx.to_numpy_array(G).astype(int)
            n_edges = int(true_adj.sum())
            if n_edges == 0:
                continue

            theta, _ = simulate_kuramoto(
                G=G, T=g["T"], dt=g["dt"], rho=g["rho"], seed=seed,
                omega_mean=0.0, omega_std=g["omega_std"],
                phase_noise_std=g["phase_noise_std"], burn_in=g["burn_in"],
                normalize_by_indegree=False,
            )
            X_basis, basis_meta, var_names = prepare_kuramoto_data_for_causal_discovery(theta, dt=g["dt"])
            T_eff = X_basis.shape[0]
            _, _, _, M, basis_map = build_basis_series(theta, g["dt"])

            common = {"Topology": tname, "Seed": seed, "N_true_edges": n_edges}

            def record(method, basis_label, adj, t0):
                if adj is None:
                    return None
                m = compute_metrics(adj, true_adj)
                m.update(common)
                m["Method"] = method
                m["Input"] = basis_label
                m["Time"] = time.time() - t0
                rows.append(m)
                return m

            # ---- oCSE (basis) ----
            t0 = time.time()
            adj = run_targeted_oce(X_basis, basis_meta, alpha=g["alpha"], n_shuffles=g["n_shuffles"])
            record("oCSE", "basis", adj, t0)

            # ---- PCMCI (basis) ----
            t0 = time.time()
            dframe = pp.DataFrame(X_basis, datatime={0: np.arange(T_eff)}, var_names=var_names)
            pcmci = PCMCI(dataframe=dframe, cond_ind_test=ParCorr(), verbosity=0)
            res = pcmci.run_pcmci(tau_min=1, tau_max=1, pc_alpha=g["alpha"])
            adj = extract_pcmci_adjacency(pcmci_to_networkx(res), basis_meta["n"], basis_meta["basis_map"])
            record("PCMCI", "basis", adj, t0)

            # ---- Granger (raw) ----
            t0 = time.time()
            record("Granger", "raw", linear_granger(theta, n, alpha=g["alpha"]), t0)
            # ---- Granger (basis) ----
            t0 = time.time()
            record("Granger", "basis", granger_basis(theta, g["dt"], n, alpha=g["alpha"]), t0)

            # ---- VARLiNGAM (raw) ----
            t0 = time.time()
            record("VARLiNGAM", "raw", varlingam_lag1_adj(theta, n, basis_map=None), t0)
            # ---- VARLiNGAM (basis) ----
            t0 = time.time()
            record("VARLiNGAM", "basis", varlingam_lag1_adj(M, n, basis_map=basis_map), t0)

            o = [r for r in rows if r["Seed"] == seed and r["Topology"] == tname]
            tag = {(r["Method"], r["Input"]): r["F1"] for r in o}

            def gv(method, inp):
                v = tag.get((method, inp))
                return "  na" if v is None else f"{v:.2f}"
            print(f"  {tname:12s} seed{seed:>3}: "
                  f"oCSE={gv('oCSE','basis')} | "
                  f"Granger raw={gv('Granger','raw')}->basis={gv('Granger','basis')} | "
                  f"VARLiNGAM raw={gv('VARLiNGAM','raw')}->basis={gv('VARLiNGAM','basis')}",
                  flush=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "basis_all_methods_results.csv")
    df.to_csv(csv_path, index=False)
    print("\nSaved", csv_path, f"({len(df)} rows)")

    # ---- summary ----
    df["Label"] = df["Method"] + " (" + df["Input"] + ")"
    summ = df.groupby("Label")[["TPR", "FPR", "F1"]].agg(["mean", "std"])
    print("\n" + "=" * 72)
    print("SUMMARY (mean over topologies x seeds)")
    print("=" * 72)
    print(summ.round(3).to_string())

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["oCSE (basis)", "PCMCI (basis)",
             "Granger (raw)", "Granger (basis)",
             "VARLiNGAM (raw)", "VARLiNGAM (basis)"]
    order = [l for l in order if l in summ.index]
    f1m = [summ.loc[l, ("F1", "mean")] for l in order]
    f1s = [summ.loc[l, ("F1", "std")] for l in order]
    fprm = [summ.loc[l, ("FPR", "mean")] for l in order]
    fprs = [summ.loc[l, ("FPR", "std")] for l in order]

    colors = ["#1a7a3a" if "oCSE" in l else
              ("#3498db" if "PCMCI" in l else
               ("#9b59b6" if "Granger" in l else "#e67e22")) for l in order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(order))
    axes[0].bar(x, f1m, yerr=f1s, color=colors, capsize=4)
    axes[0].set_ylabel("F1"); axes[0].set_title("F1 (higher better)")
    axes[1].bar(x, fprm, yerr=fprs, color=colors, capsize=4)
    axes[1].set_ylabel("FPR"); axes[1].set_title("False positive rate (lower better)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Sine basis applied to all methods (Kuramoto, n=6)", fontweight="bold")
    plt.tight_layout()
    png = os.path.join(results_dir, "basis_all_methods.png")
    plt.savefig(png, dpi=200, bbox_inches="tight")
    print("Saved", png)


if __name__ == "__main__":
    main()
