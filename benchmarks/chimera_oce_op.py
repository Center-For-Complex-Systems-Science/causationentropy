"""
Decisive chimera test: does conditioning oCSE on the order parameter cut its
false positives in the strong-mean-field regime?

Runs ONLY the fast methods (oCSE, oCSE+OP, Granger) on the chimera regime --
PCMCI and VARLiNGAM are skipped because they take dozens of hours at N=36.
Lighter settings (T_CAP, fewer shuffles) for a directional result in ~1.5-2h.

Writes benchmarks/results/emergent_causal_chimera.csv (separate from the main
benchmark CSV so existing cyclops/metastable/traveling results are preserved).
"""
import os
import time
import numpy as np
import pandas as pd

import emergent_regimes as er
import emergent_causal_benchmark as b
from basis_all_methods import granger_basis
from causationentropy.datasets.synthetic import prepare_kuramoto_data_for_causal_discovery

CHIMERA_N = 24           # small N -> 552-feature candidate pool -> oCSE ~25 min/run
CHIMERA_RADIUS = 3       # sparse (~26% density, clear split) -> meaningful FPR
SEEDS = [0, 1]
ALPHA = 0.05
N_SHUFFLES = 20          # >=20 so the shuffle test can reach p<0.05 (1/21=0.048);
                         # 10 was a bug -> nothing could pass -> artifactual TPR~0
T_CAP = 2000             # match the other regimes for comparable FPR
OP_HARMONICS = (1, 2)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "emergent_causal_chimera.csv")

    print("=" * 70)
    print(f"Chimera oCSE vs oCSE+OP  (N={CHIMERA_N}, shuffles={N_SHUFFLES}, T_cap={T_CAP})")
    print("=" * 70)

    rows = []
    t_start = time.time()
    for seed in SEEDS:
        cfg = er.regime_chimera(seed=seed, n=CHIMERA_N, radius=CHIMERA_RADIUS)
        cfg.pop("name")
        W = cfg["W"]
        n = W.shape[0]
        true_adj = b.ground_truth_from_W(W)
        n_edges = int(true_adj.sum())

        theta, _ = er.simulate_phase_oscillators(**cfg)
        if T_CAP:
            theta = theta[:T_CAP]
        dt = cfg["dt"]

        X_basis, basis_meta, var_names = prepare_kuramoto_data_for_causal_discovery(theta, dt=dt)
        T_eff = X_basis.shape[0]
        cond_basis = b.order_parameter_features(theta[:T_eff], harmonics=OP_HARMONICS)

        common = {"Regime": "chimera", "Seed": seed, "N": n, "N_true_edges": n_edges}

        def record(method, adj, t0):
            m = b.regime_metrics(adj, true_adj)
            m.update(common)
            m["Method"] = method
            m["Time"] = time.time() - t0
            rows.append(m)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            return m

        t0 = time.time()
        mA = record("oCSE", b.run_targeted_oce(X_basis, basis_meta, ALPHA, N_SHUFFLES), t0)
        print(f"[{time.time()-t_start:6.0f}s] seed{seed} oCSE     "
              f"F1={mA['F1']:.2f} TPR={mA['TPR']:.2f} FPR={mA['FPR']:.3f} "
              f"({mA['Time']:.0f}s)", flush=True)

        t0 = time.time()
        mB = record("oCSE+OP", b.run_targeted_oce(X_basis, basis_meta, ALPHA, N_SHUFFLES, cond_basis=cond_basis), t0)
        print(f"[{time.time()-t_start:6.0f}s] seed{seed} oCSE+OP  "
              f"F1={mB['F1']:.2f} TPR={mB['TPR']:.2f} FPR={mB['FPR']:.3f} "
              f"({mB['Time']:.0f}s)   <-- FPR {mA['FPR']:.3f} -> {mB['FPR']:.3f}", flush=True)

        t0 = time.time()
        mG = record("Granger", granger_basis(theta, dt, n, alpha=ALPHA), t0)
        print(f"[{time.time()-t_start:6.0f}s] seed{seed} Granger  "
              f"F1={mG['F1']:.2f} TPR={mG['TPR']:.2f} FPR={mG['FPR']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print("\nSUMMARY (mean over seeds):")
    print(df.groupby("Method")[["TPR", "FPR", "F1"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
