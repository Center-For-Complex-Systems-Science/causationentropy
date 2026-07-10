import time

import numpy as np
import pytest

from causationentropy.core.discovery import discover_network
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)
from causationentropy.core.information.mutual_information import (
    geometric_knn_mutual_information,
    knn_mutual_information,
)


class TestKDTreeCorrectness:
    """kd_tree=True and kd_tree=False must produce numerically identical results."""

    def test_knn_mutual_information_correctness(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((150, 2))
        Y = rng.standard_normal((150, 1))
        mi_bf  = knn_mutual_information(X, Y, k=3, kd_tree=False)
        mi_kdt = knn_mutual_information(X, Y, k=3, kd_tree=True)
        assert abs(mi_bf - mi_kdt) < 1e-9, (
            f"knn MI mismatch: brute={mi_bf:.8f}, kd_tree={mi_kdt:.8f}"
        )

    def test_geometric_knn_mutual_information_correctness(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 2))
        Y = rng.standard_normal((100, 1))
        mi_bf  = geometric_knn_mutual_information(X, Y, k=2, kd_tree=False)
        mi_kdt = geometric_knn_mutual_information(X, Y, k=2, kd_tree=True)
        assert abs(mi_bf - mi_kdt) < 1e-9

    def test_knn_cmi_correctness(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((120, 1))
        Y = rng.standard_normal((120, 1))
        Z = rng.standard_normal((120, 2))
        cmi_bf  = conditional_mutual_information(X, Y, Z, method="knn", k=3, kd_tree=False)
        cmi_kdt = conditional_mutual_information(X, Y, Z, method="knn", k=3, kd_tree=True)
        assert abs(cmi_bf - cmi_kdt) < 1e-9

    def test_geometric_knn_cmi_correctness(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((80, 1))
        Y = rng.standard_normal((80, 1))
        Z = rng.standard_normal((80, 1))
        cmi_bf  = conditional_mutual_information(X, Y, Z, method="geometric_knn", k=2, kd_tree=False)
        cmi_kdt = conditional_mutual_information(X, Y, Z, method="geometric_knn", k=2, kd_tree=True)
        assert abs(cmi_bf - cmi_kdt) < 1e-9

    def test_discover_network_kd_tree_flag(self):
        """discover_network runs with kd_tree=True and produces a valid graph."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((60, 3))
        G = discover_network(data, information="knn", max_lag=1, n_shuffles=10, kd_tree=True)
        import networkx as nx
        assert isinstance(G, nx.MultiDiGraph)
        assert G.number_of_nodes() == 3

    def test_default_behavior_unchanged(self):
        """kd_tree=False (default) still works exactly as before."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal((60, 3))
        G = discover_network(data, information="knn", max_lag=1, n_shuffles=10)
        import networkx as nx
        assert isinstance(G, nx.MultiDiGraph)


class TestKDTreeBenchmark:
    """Benchmark to show speedup. Not a strict assertion — print results."""

    @pytest.mark.slow
    def test_runtime_scaling(self):
        for N in [300, 800]:
            rng = np.random.default_rng(0)
            X = rng.standard_normal((N, 2))
            Y = rng.standard_normal((N, 1))
            t0 = time.perf_counter()
            knn_mutual_information(X, Y, k=5, kd_tree=False)
            t_bf = time.perf_counter() - t0

            t0 = time.perf_counter()
            knn_mutual_information(X, Y, k=5, kd_tree=True)
            t_kdt = time.perf_counter() - t0

            speedup = t_bf / t_kdt if t_kdt > 0 else float("inf")
            print(f"\nN={N}: brute={t_bf:.3f}s  kd_tree={t_kdt:.3f}s  speedup={speedup:.1f}x")
            # At N=800 in ≥2D the KD-Tree should be meaningfully faster
            if N >= 800:
                assert speedup > 1.5, f"Expected >1.5x speedup at N={N}, got {speedup:.2f}x"
