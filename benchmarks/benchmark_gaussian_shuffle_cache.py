"""Compare Gaussian surrogate calculations on identical index sequences."""

from time import perf_counter

import numpy as np

from causationentropy.core.information.conditional_mutual_information import (
    gaussian_conditional_mutual_information,
)
from causationentropy.core.information.gaussian_shuffle_cache import (
    GaussianShuffleCache,
)


def run(n, kx, kz, draws, mode):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, kx))
    Y = 0.2 * X[:, :1] + rng.normal(size=(n, 1))
    Z = rng.normal(size=(n, kz))
    cache = GaussianShuffleCache.build(X, Y, Z)
    assert cache is not None
    indices = [
        rng.permutation(n) if mode == "global" else rng.integers(0, n, size=n)
        for _ in range(draws)
    ]
    t0 = perf_counter()
    original = [
        gaussian_conditional_mutual_information(X[idx], Y, Z) for idx in indices
    ]
    t1 = perf_counter()
    cached = [cache.evaluate(idx, is_permutation=(mode == "global")) for idx in indices]
    t2 = perf_counter()
    np.testing.assert_allclose(cached, original, atol=1e-10, rtol=1e-10)
    baseline = t1 - t0
    optimized = t2 - t1
    max_difference = np.max(np.abs(np.array(original) - np.array(cached)))
    print(
        f"mode={mode:10s} N={n:4d} kx={kx} kz={kz} draws={draws}: "
        f"baseline={baseline:.4f}s cached={optimized:.4f}s "
        f"speedup={baseline / optimized:.2f}x "
        f"max_diff={max_difference:.2e}"
    )


if __name__ == "__main__":
    for mode in ("global", "reassigned"):
        for n, kx, kz, draws in ((200, 1, 2, 100), (500, 2, 4, 100)):
            run(n, kx, kz, draws, mode)
