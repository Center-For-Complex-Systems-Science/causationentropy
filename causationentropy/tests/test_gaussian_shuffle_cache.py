"""Tests for cached Gaussian conditional MI surrogate calculations."""

import numpy as np
import pytest

from causationentropy.core.discovery import shuffle_test
from causationentropy.core.information.conditional_mutual_information import (
    gaussian_conditional_mutual_information,
)
from causationentropy.core.information.gaussian_shuffle_cache import (
    GaussianShuffleCache,
)


@pytest.mark.parametrize("seed,kx,kz", [(1, 1, 1), (2, 2, 3), (3, 3, 2)])
def test_cached_cmi_matches_original_permutations(seed, kx, kz):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(180, kx))
    Y = 0.2 * X[:, :1] + rng.normal(size=(180, 1))
    Z = rng.normal(size=(180, kz))
    cache = GaussianShuffleCache.build(X, Y, Z)
    assert cache is not None
    for _ in range(40):
        idx = rng.permutation(len(X))
        old = gaussian_conditional_mutual_information(X[idx], Y, Z)
        new = cache.evaluate(idx)
        np.testing.assert_allclose(new, old, atol=1e-10, rtol=1e-10)


def test_local_donor_reassignment_recomputes_x_covariance():
    rng = np.random.default_rng(10)
    X, Y, Z = [rng.normal(size=(150, d)) for d in (2, 1, 3)]
    cache = GaussianShuffleCache.build(X, Y, Z)
    assert cache is not None
    for _ in range(30):
        donors = rng.integers(0, len(X), size=len(X))
        old = gaussian_conditional_mutual_information(X[donors], Y, Z)
        new = cache.evaluate(donors, is_permutation=False)
        np.testing.assert_allclose(new, old, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("early_stop", [False, True])
@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_shuffle_test_matches_reference(monkeypatch, early_stop, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(130, 2))
    Y = rng.normal(size=(130, 1))
    Z = rng.normal(size=(130, 2))
    observed = gaussian_conditional_mutual_information(X, Y, Z)
    ref = shuffle_test(
        X,
        Y,
        Z,
        observed,
        n_shuffles=50,
        alpha=0.1,
        rng=seed,
        information="gaussian",
        early_stop=early_stop,
    )
    with monkeypatch.context() as m:
        m.setattr(GaussianShuffleCache, "build", lambda *args: None)
        old = shuffle_test(
            X,
            Y,
            Z,
            observed,
            n_shuffles=50,
            alpha=0.1,
            rng=seed,
            information="gaussian",
            early_stop=early_stop,
        )
    assert ref["N_Completed"] == old["N_Completed"]
    assert ref["Early_Stopped"] == old["Early_Stopped"]
    assert ref["Pass"] == old["Pass"]
    assert ref["P_value"] == old["P_value"]
    np.testing.assert_allclose(ref["Threshold"], old["Threshold"], atol=1e-10)


def test_singular_fixed_block_falls_back():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(100, 1))
    Y = rng.normal(size=(100, 1))
    Z = np.column_stack((Y[:, 0], Y[:, 0]))
    assert GaussianShuffleCache.build(X, Y, Z) is None


def test_no_conditioning_uses_existing_mi_path():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(100, 1))
    Y = rng.normal(size=(100, 1))
    assert GaussianShuffleCache.build(X, Y, None) is None
    observed = gaussian_conditional_mutual_information(X, Y, None)
    result = shuffle_test(X, Y, None, observed, rng=1, n_shuffles=20)
    assert result["N_Completed"] == 20


def test_cache_preserves_rng_consumption(monkeypatch):
    rng = np.random.default_rng(32)
    X = rng.normal(size=(110, 2))
    Y = rng.normal(size=(110, 1))
    Z = rng.normal(size=(110, 2))
    observed = gaussian_conditional_mutual_information(X, Y, Z)
    fast_rng = np.random.default_rng(15)
    baseline_rng = np.random.default_rng(15)

    result = shuffle_test(X, Y, Z, observed, rng=fast_rng, n_shuffles=30)
    with monkeypatch.context() as patcher:
        patcher.setattr(GaussianShuffleCache, "build", lambda *args: None)
        baseline = shuffle_test(X, Y, Z, observed, rng=baseline_rng, n_shuffles=30)

    assert result["N_Completed"] == baseline["N_Completed"]
    assert result["P_value"] == baseline["P_value"]
    assert fast_rng.integers(0, 2**31) == baseline_rng.integers(0, 2**31)


def test_constant_predictor_uses_reference():
    rng = np.random.default_rng(34)
    X = np.ones((80, 1))
    Y = rng.normal(size=(80, 1))
    Z = rng.normal(size=(80, 1))
    assert GaussianShuffleCache.build(X, Y, Z) is None


@pytest.mark.parametrize("seed", [7, 19])
def test_discover_network_gaussian_cache_preserves_graph(monkeypatch, seed):
    from causationentropy.core.discovery import discover_network

    rng = np.random.default_rng(seed)
    data = rng.normal(size=(180, 4))

    for t in range(1, len(data)):
        data[t, 1] += 0.7 * data[t - 1, 0]
        data[t, 2] += 0.5 * data[t - 1, 1]

    settings = dict(
        method="standard",
        information="gaussian",
        max_lag=2,
        n_shuffles=35,
        n_jobs=1,
        random_state=101,
        alpha_forward=0.1,
        alpha_backward=0.1,
    )

    with monkeypatch.context() as patcher:
        patcher.setattr(GaussianShuffleCache, "build", lambda *args: None)
        baseline = discover_network(data, **settings)

    optimized = discover_network(data, **settings)

    def edge_data(graph):
        return {
            (u, v, d["lag"]): (d["cmi"], d["p_value"])
            for u, v, d in graph.edges(data=True)
        }

    expected = edge_data(baseline)
    actual = edge_data(optimized)

    assert expected
    assert expected.keys() == actual.keys()

    for edge in expected:
        np.testing.assert_allclose(
            actual[edge],
            expected[edge],
            atol=1e-10,
            rtol=1e-10,
            equal_nan=True,
        )
