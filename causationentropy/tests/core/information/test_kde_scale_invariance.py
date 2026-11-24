"""Regression tests for KDE bandwidths under changes of measurement units."""

import numpy as np
import pytest

from causationentropy.core.discovery import discover_network
from causationentropy.core.information.conditional_mutual_information import (
    kde_conditional_mutual_information,
)
from causationentropy.core.information.entropy import (
    _robust_feature_scales,
    kde_entropy,
)
from causationentropy.core.information.mutual_information import (
    kde_mutual_information,
)


def test_kde_mutual_information_matches_gaussian_reference():
    rng = np.random.default_rng(0)
    n, rho = 500, 0.6
    x = rng.normal(size=(n, 1))
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=(n, 1))

    estimate = kde_mutual_information(x, y)
    expected = -0.5 * np.log(1 - rho**2)

    # KDE is a finite-sample estimator; this checks the scale and sign of the
    # estimate without pretending it is exactly unbiased.
    assert estimate == pytest.approx(expected, abs=0.12)


def test_rule_bandwidth_mi_is_invariant_to_units():
    rng = np.random.default_rng(0)
    n, rho = 500, 0.6
    x = rng.normal(size=(n, 1))
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=(n, 1))
    z = rng.normal(size=(n, 1))

    for scale in (0.01, 1.0, 100.0):
        dependent = kde_mutual_information(scale * x, scale * y)
        independent = kde_mutual_information(scale * x, scale * z)
        assert dependent == pytest.approx(kde_mutual_information(x, y), abs=1e-10)
        assert independent == pytest.approx(kde_mutual_information(x, z), abs=1e-10)

    # Units may differ for each variable, not only by a common factor.
    assert kde_mutual_information(0.01 * x, 100.0 * y) == pytest.approx(
        kde_mutual_information(x, y), abs=1e-10
    )


def test_kde_entropy_includes_scale_jacobian():
    rng = np.random.default_rng(21)
    data = rng.normal(size=(300, 2))
    scales = np.array([0.01, 100.0])

    baseline = kde_entropy(data)
    rescaled = kde_entropy(data * scales)

    assert rescaled - baseline == pytest.approx(np.log(scales).sum(), abs=1e-10)


@pytest.mark.parametrize("bandwidth", ["silverman", "scott"])
def test_rule_bandwidth_cmi_is_invariant_to_per_variable_units(bandwidth):
    rng = np.random.default_rng(12)
    n = 400
    z = rng.normal(size=(n, 1))
    x = 0.7 * z + rng.normal(size=(n, 1))
    y = 0.5 * x - 0.4 * z + rng.normal(size=(n, 1))

    baseline = kde_conditional_mutual_information(x, y, z, bandwidth=bandwidth)
    rescaled = kde_conditional_mutual_information(
        0.01 * x, 100.0 * y, 7.0 * z, bandwidth=bandwidth
    )

    assert rescaled == pytest.approx(baseline, abs=1e-10)


def test_kde_rule_bandwidth_handles_heavy_tailed_link():
    rng = np.random.default_rng(0)
    n = 300
    source = rng.standard_cauchy(size=(n, 1))
    noise = rng.normal(scale=0.1, size=(n, 1))
    target = np.exp(3 * (np.arctan(source[:-1]) + noise[1:]))

    estimate = kde_mutual_information(source[:-1], target)
    rescaled = kde_mutual_information(100.0 * source[:-1], 0.01 * target)

    assert estimate > 0.5
    assert rescaled == pytest.approx(estimate, abs=1e-10)


def test_kde_discovery_recovers_same_link_after_unit_change():
    rng = np.random.default_rng(0)
    n = 300
    source = rng.normal(size=n)
    target = np.empty(n)
    target[0] = rng.normal()
    target[1:] = 0.9 * source[:-1] + 0.3 * rng.normal(size=n - 1)
    data = np.column_stack((source, target))

    edge_sets = []
    for scale in (1.0, 100.0):
        graph = discover_network(
            scale * data,
            information="kde",
            max_lag=1,
            n_shuffles=30,
            n_jobs=1,
            random_state=42,
        )
        edge_sets.append(
            {
                (source, target, attrs["lag"])
                for source, target, attrs in graph.edges(data=True)
            }
        )

    assert ("X0", "X1", 1) in edge_sets[0]
    assert edge_sets[1] == edge_sets[0]


def test_robust_scales_fall_back_for_zero_iqr_and_constant_features():
    data = np.column_stack(
        [np.r_[np.zeros(9), 10.0], np.ones(10), np.arange(10, dtype=float)]
    )

    scales = _robust_feature_scales(data)

    assert scales[0] == pytest.approx(np.std(data[:, 0], ddof=1))
    assert scales[1] == 1.0
    assert scales[2] > 0


def test_numeric_bandwidth_fits_original_units(monkeypatch):
    import causationentropy.core.information.entropy as entropy_module

    fitted = {}

    class RecordingKDE:
        def __init__(self, bandwidth, kernel):
            fitted["bandwidth"] = bandwidth
            fitted["kernel"] = kernel

        def fit(self, data):
            fitted["data"] = np.array(data, copy=True)
            return self

        def score_samples(self, data):
            return np.zeros(len(data))

    monkeypatch.setattr(entropy_module, "KernelDensity", RecordingKDE)
    data = 100.0 * np.arange(12, dtype=float).reshape(-1, 1)

    kde_entropy(data, bandwidth=0.5)

    assert fitted["bandwidth"] == 0.5
    np.testing.assert_array_equal(fitted["data"], data)
