import timeit
import numpy as np
from causationentropy.core.information.conditional_mutual_information import (
    gaussian_conditional_mutual_information
)

def baseline_4pass_cmi(X, Y, Z):
    def _detcorr(A):
        C = np.corrcoef(A.T)
        return 0.0 if np.ndim(C) == 0 else np.linalg.slogdet(C)[1]
    sz = _detcorr(Z)
    sxz = _detcorr(np.hstack((X, Z)))
    syz = _detcorr(np.hstack((Y, Z)))
    sxyz = _detcorr(np.hstack((X, Y, Z)))
    return 0.5 * (sxz + syz - sz - sxyz)

print("=" * 65)
print("Gaussian CMI Performance Benchmark (Issue #7)")
print("=" * 65)
configurations = [
    {"n": 500,  "kx": 1, "ky": 1, "kz": 1, "runs": 2000},
    {"n": 1000, "kx": 2, "ky": 1, "kz": 2, "runs": 2000},
    {"n": 1000, "kx": 5, "ky": 1, "kz": 5, "runs": 1000},
    {"n": 2000, "kx": 3, "ky": 1, "kz": 4, "runs": 1000},
]

for cfg in configurations:
    n, kx, ky, kz, runs = cfg["n"], cfg["kx"], cfg["ky"], cfg["kz"], cfg["runs"]
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, kx))
    Y = rng.standard_normal((n, ky))
    Z = rng.standard_normal((n, kz))

    val_base = baseline_4pass_cmi(X, Y, Z)
    val_opt = gaussian_conditional_mutual_information(X, Y, Z)
    diff = abs(val_base - val_opt)
    assert diff < 1e-12, f"Discrepancy: {diff}"

    t_base = timeit.timeit(lambda: baseline_4pass_cmi(X, Y, Z), number=runs)
    t_opt = timeit.timeit(lambda: gaussian_conditional_mutual_information(X, Y, Z), number=runs)
    speedup = t_base / t_opt

    print(f"N={n:4d}, kX={kx}, kY={ky}, kZ={kz} ({runs} runs):")
    print(f"  Baseline (4-pass):  {t_base*1000/runs:.3f} ms/call")
    print(f"  Optimized (1-pass): {t_opt*1000/runs:.3f} ms/call")
    print(f"  Speedup:             {speedup:.2f}x (max abs diff: {diff:.1e})")
    print("")
