import numpy as np
import pytest

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
    geometric_knn_conditional_mutual_information,
    knn_conditional_mutual_information,
)
from causationentropy.core.information.mutual_information import (
    geometric_knn_mutual_information,
    knn_mutual_information,
)
from causationentropy.core.information.neighbors import (
    kth_neighbor_distance,
    knn_indices,
    radius_neighbor_counts,
)


class TestKthNeighborDistance:
    def test_matches_brute_force_euclidean(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))

        kd = kth_neighbor_distance(X, k=4, metric="euclidean", kd_tree=True)
        brute = kth_neighbor_distance(X, k=4, metric="euclidean", kd_tree=False)

        np.testing.assert_allclose(kd, brute, rtol=1e-10)

    def test_matches_brute_force_chebyshev(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 2))

        kd = kth_neighbor_distance(X, k=2, metric="chebyshev", kd_tree=True)
        brute = kth_neighbor_distance(X, k=2, metric="chebyshev", kd_tree=False)

        np.testing.assert_allclose(kd, brute, rtol=1e-10)

    def test_unsupported_metric_falls_back(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(20, 2))

        kd = kth_neighbor_distance(X, k=1, metric="correlation", kd_tree=True)
        brute = kth_neighbor_distance(X, k=1, metric="correlation", kd_tree=False)

        np.testing.assert_allclose(kd, brute, rtol=1e-10)


class TestRadiusNeighborCounts:
    def test_matches_brute_force(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(50, 2))
        radii = kth_neighbor_distance(X, k=3, metric="euclidean")

        kd = radius_neighbor_counts(X, radii, metric="euclidean", kd_tree=True)
        brute = radius_neighbor_counts(X, radii, metric="euclidean", kd_tree=False)

        np.testing.assert_array_equal(kd, brute)


class TestKnnIndices:
    def test_matches_brute_force(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(30, 2))

        kd = knn_indices(X, k=3, metric="euclidean", kd_tree=True)
        brute = knn_indices(X, k=3, metric="euclidean", kd_tree=False)

        for i in range(X.shape[0]):
            assert set(kd[i]) == set(brute[i])


class TestKdTreeEquivalence:
    def test_knn_mutual_information_equivalence(self):
        rng = np.random.default_rng(5)
        n = 80
        X = rng.normal(size=(n, 1))
        Y = 0.5 * X + rng.normal(scale=0.3, size=(n, 1))

        mi_tree = knn_mutual_information(X, Y, k=4, kd_tree=True)
        mi_brute = knn_mutual_information(X, Y, k=4, kd_tree=False)

        assert np.isclose(mi_tree, mi_brute, rtol=1e-10)

    def test_knn_conditional_mutual_information_equivalence(self):
        rng = np.random.default_rng(6)
        n = 80
        Z = rng.normal(size=(n, 1))
        X = Z + rng.normal(scale=0.2, size=(n, 1))
        Y = Z + rng.normal(scale=0.2, size=(n, 1))

        cmi_tree = knn_conditional_mutual_information(X, Y, Z, k=4, kd_tree=True)
        cmi_brute = knn_conditional_mutual_information(X, Y, Z, k=4, kd_tree=False)

        assert np.isclose(cmi_tree, cmi_brute, rtol=1e-10)

    def test_geometric_knn_mutual_information_equivalence(self):
        rng = np.random.default_rng(7)
        n = 60
        X = rng.normal(size=(n, 2))
        Y = rng.normal(size=(n, 1))

        mi_tree = geometric_knn_mutual_information(X, Y, k=3, kd_tree=True)
        mi_brute = geometric_knn_mutual_information(X, Y, k=3, kd_tree=False)

        assert np.isclose(mi_tree, mi_brute, rtol=1e-8)

    def test_geometric_knn_conditional_mutual_information_equivalence(self):
        rng = np.random.default_rng(8)
        n = 60
        Z = rng.normal(size=(n, 1))
        X = rng.normal(size=(n, 1))
        Y = rng.normal(size=(n, 1))

        cmi_tree = geometric_knn_conditional_mutual_information(
            X, Y, Z, k=3, kd_tree=True
        )
        cmi_brute = geometric_knn_conditional_mutual_information(
            X, Y, Z, k=3, kd_tree=False
        )

        assert np.isclose(cmi_tree, cmi_brute, rtol=1e-8)

    @pytest.mark.parametrize("method", ["knn", "geometric_knn"])
    def test_conditional_mutual_information_dispatch_equivalence(self, method):
        rng = np.random.default_rng(9)
        n = 70
        Z = rng.normal(size=(n, 1))
        X = rng.normal(size=(n, 1))
        Y = rng.normal(size=(n, 1))

        cmi_tree = conditional_mutual_information(
            X, Y, Z, method=method, k=3, kd_tree=True
        )
        cmi_brute = conditional_mutual_information(
            X, Y, Z, method=method, k=3, kd_tree=False
        )

        assert np.isclose(cmi_tree, cmi_brute, rtol=1e-8)
