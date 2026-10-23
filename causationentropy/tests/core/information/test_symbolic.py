"""Tests for the symbolic (partition-based) information estimators (issue #59)."""

import networkx as nx
import numpy as np
import pytest

from causationentropy.core.discovery import discover_network
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
    symbolic_conditional_mutual_information,
)
from causationentropy.core.information.entropy import symbolic_entropy, symbolize
from causationentropy.core.information.mutual_information import (
    symbolic_mutual_information,
)


def _cmi_by_definition(x, y, z):
    """I(X;Y|Z) from Eq. (15) of Cafaro et al. (2015), summed term by term (nats)."""
    n = len(x)
    total = 0.0
    for xv, yv, zv in set(zip(x, y, z)):
        p_xyz = np.sum((x == xv) & (y == yv) & (z == zv)) / n
        p_z = np.sum(z == zv) / n
        p_xz = np.sum((x == xv) & (z == zv)) / n
        p_yz = np.sum((y == yv) & (z == zv)) / n
        total += p_xyz * np.log((p_xyz / p_z) / ((p_xz / p_z) * (p_yz / p_z)))
    return total


class TestSymbolize:
    def test_median_split(self):
        x = np.array([0.1, 0.9, 0.4, 0.6])
        assert symbolize(x, n_symbols=2).ravel().tolist() == [0, 1, 0, 1]

    def test_quantile_cells_are_equal_frequency(self):
        x = np.random.default_rng(0).normal(size=1000)
        counts = np.bincount(symbolize(x, n_symbols=4).ravel())
        assert counts.tolist() == [250, 250, 250, 250]

    def test_uniform_cells(self):
        x = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        assert symbolize(x, n_symbols=2, partition="uniform").ravel().tolist() == [
            0,
            0,
            1,
            1,
            1,
        ]

    def test_edge_value_goes_to_upper_cell(self):
        x = np.array([0.0, 0.5, 1.0])
        assert symbolize(x, n_symbols=2, partition="uniform").ravel().tolist() == [
            0,
            1,
            1,
        ]

    def test_columns_are_symbolized_separately(self):
        X = np.column_stack([[1.0, 2.0, 3.0, 4.0], [400.0, 300.0, 200.0, 100.0]])
        S = symbolize(X, n_symbols=2)
        assert S.shape == (4, 2)
        assert S[:, 0].tolist() == [0, 0, 1, 1]
        assert S[:, 1].tolist() == [1, 1, 0, 0]

    def test_precomputed_returns_integers(self):
        S = symbolize(np.array([2.0, 0.0, 1.0]), partition="precomputed")
        assert S.dtype == np.int64
        assert S.ravel().tolist() == [2, 0, 1]

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            symbolize(np.arange(5.0), n_symbols=1)
        with pytest.raises(ValueError):
            symbolize(np.arange(5.0), partition="unknown")
        with pytest.raises(ValueError):
            symbolize(np.array([0.5, 1.0]), partition="precomputed")

    @pytest.mark.parametrize("partition", ["quantile", "uniform", "precomputed"])
    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_values_are_rejected(self, partition, bad):
        with pytest.raises(ValueError, match="finite"):
            symbolize(np.array([1.0, bad, 2.0, 3.0]), partition=partition)


class TestSymbolicEntropy:
    def test_uniform_symbols(self):
        x = np.array([0, 1, 2, 3])
        assert np.isclose(symbolic_entropy(x, partition="precomputed"), np.log(4))

    def test_constant_has_zero_entropy(self):
        assert symbolic_entropy(np.zeros(10), partition="precomputed") == 0.0

    def test_joint_entropy_of_two_columns(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        assert np.isclose(symbolic_entropy(X, partition="precomputed"), np.log(4))


class TestSymbolicMutualInformation:
    def test_self_information_equals_entropy(self):
        x = np.random.default_rng(1).normal(size=500)
        assert np.isclose(
            symbolic_mutual_information(x, x, n_symbols=3),
            symbolic_entropy(x, n_symbols=3),
        )

    def test_independent_variables_near_zero(self):
        rng = np.random.default_rng(2)
        mi = symbolic_mutual_information(rng.normal(size=20000), rng.normal(size=20000))
        assert 0.0 <= mi < 1e-3

    def test_invariant_to_monotone_transform(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=1000)
        y = x + rng.normal(size=1000)
        assert np.isclose(
            symbolic_mutual_information(x, y, n_symbols=3),
            symbolic_mutual_information(np.exp(x), y**3, n_symbols=3),
        )


class TestSymbolicConditionalMutualInformation:
    def test_matches_definition(self):
        rng = np.random.default_rng(4)
        z = rng.integers(0, 3, 400)
        x = (z + rng.integers(0, 2, 400)) % 3
        y = (x + rng.integers(0, 2, 400)) % 2
        got = symbolic_conditional_mutual_information(x, y, z, partition="precomputed")
        assert np.isclose(got, _cmi_by_definition(x, y, z))

    def test_chain_is_screened_off(self):
        # X -> Z -> Y with Y a function of Z: I(X;Y) > 0 but I(X;Y|Z) = 0.
        rng = np.random.default_rng(5)
        x = rng.integers(0, 2, 2000)
        z = np.where(rng.random(2000) < 0.9, x, 1 - x)
        y = z.copy()
        assert symbolic_mutual_information(x, y, partition="precomputed") > 0.2
        assert np.isclose(
            symbolic_conditional_mutual_information(x, y, z, partition="precomputed"),
            0.0,
        )

    def test_none_z_returns_mutual_information(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=300)
        y = x + rng.normal(size=300)
        assert symbolic_conditional_mutual_information(x, y, None) == pytest.approx(
            symbolic_mutual_information(x, y)
        )

    def test_logistic_map_generating_partition_is_memoryless(self):
        # The logistic map at r = 4 is conjugate to the tent map; the partition at
        # 1/2 is generating and its symbol sequence is a fair coin (Cafaro et al.,
        # 2015, Sec. IV), so one past symbol carries no information about the next.
        x = np.empty(20001)
        x[0] = 0.1234
        for t in range(20000):
            x[t + 1] = 4.0 * x[t] * (1.0 - x[t])
        s = symbolize(x, n_symbols=2, partition="uniform").ravel()
        assert symbolic_entropy(s, partition="precomputed") == pytest.approx(
            np.log(2), abs=1e-3
        )
        assert (
            symbolic_mutual_information(s[:-1], s[1:], partition="precomputed") < 1e-3
        )

    def test_dispatcher(self):
        rng = np.random.default_rng(7)
        x, y, z = rng.normal(size=(3, 500))
        direct = symbolic_conditional_mutual_information(x, y, z, n_symbols=3)
        via = conditional_mutual_information(x, y, z, method="symbolic", n_symbols=3)
        assert via == pytest.approx(max(0.0, direct))

    def test_dispatcher_lists_symbolic_on_error(self):
        with pytest.raises(ValueError, match="symbolic"):
            conditional_mutual_information(
                np.zeros(5), np.zeros(5), None, method="nonexistent"
            )


class TestSymbolicDiscovery:
    def test_recovers_lagged_edge(self):
        rng = np.random.default_rng(42)
        T = 600
        x0 = rng.normal(size=T)
        x1 = np.zeros(T)
        x1[1:] = 0.9 * x0[:-1] + 0.3 * rng.normal(size=T - 1)
        G = discover_network(
            np.column_stack([x0, x1]),
            information="symbolic",
            max_lag=2,
            n_shuffles=100,
            n_jobs=1,
        )
        assert isinstance(G, nx.MultiDiGraph)
        edges = {(u, v, a.get("lag")) for u, v, a in G.edges(data=True)}
        assert edges == {("X0", "X1", 1)}


class TestSymbolizeTies:
    def test_heavily_tied_binary_data_keeps_both_symbols(self):
        # 70% zeros: a median edge at 0 would put every point in one cell.
        x = np.array([0.0] * 70 + [1.0] * 30)
        assert symbolize(x, n_symbols=2).ravel().tolist() == [0] * 70 + [1] * 30
        assert symbolic_entropy(x) == pytest.approx(
            -(0.7 * np.log(0.7) + 0.3 * np.log(0.3))
        )

    def test_tied_values_share_a_cell(self):
        x = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=float)
        s = symbolize(x, n_symbols=3).ravel()
        for v in np.unique(x):
            assert len(set(s[x == v])) == 1

    def test_constant_column(self):
        assert symbolize(np.ones(10), n_symbols=3).ravel().tolist() == [0] * 10
        assert symbolic_entropy(np.ones(10)) == 0.0

    def test_count_data_dependence_is_detected(self):
        rng = np.random.default_rng(8)
        x = rng.poisson(0.5, 3000).astype(float)
        y = x + rng.poisson(0.2, 3000)
        assert symbolic_mutual_information(x, y, n_symbols=3) > 0.1
