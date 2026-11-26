"""Tie handling and default metrics of the k-NN (KSG) estimators."""

import warnings

import numpy as np
import pytest

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
    knn_conditional_mutual_information,
)
from causationentropy.core.information.mutual_information import (
    knn_mutual_information,
)
from causationentropy.core.stats import bootstrap_cmi_confidence_interval


class TestKNNRepeatedValues:
    """Repeated samples used to make the k-th neighbour distance 0."""

    def test_repeated_values_give_finite_estimates(self):
        X = np.repeat(np.arange(10.0), 20)[:, None]
        Y = X[::-1]

        assert np.isfinite(knn_mutual_information(X, Y, k=3))
        assert np.isfinite(knn_conditional_mutual_information(X, Y, X, k=3))
        assert np.isfinite(
            conditional_mutual_information(X, Y, None, method="knn", k=3)
        )
        assert np.isfinite(conditional_mutual_information(X, Y, X, method="knn", k=3))

    def test_independent_count_data_is_close_to_zero(self):
        rng = np.random.default_rng(0)
        X, Y, Z = [rng.poisson(3, (300, d)).astype(float) for d in (1, 1, 2)]

        mi = knn_mutual_information(X, Y, k=3)
        cmi = knn_conditional_mutual_information(X, Y, Z, k=3)

        assert np.isfinite(mi) and np.isfinite(cmi)
        assert abs(mi) < 0.1
        assert abs(cmi) < 0.1

    def test_estimates_are_deterministic(self):
        rng = np.random.default_rng(1)
        X, Y, Z = [rng.poisson(3, (200, 1)).astype(float) for _ in range(3)]

        first = knn_conditional_mutual_information(X, Y, Z, k=3)
        second = knn_conditional_mutual_information(X, Y, Z, k=3)

        assert first == second

    def test_inputs_are_not_modified(self):
        X = np.repeat(np.arange(5.0), 10)[:, None]
        Y = X[::-1].copy()
        X_before, Y_before = X.copy(), Y.copy()

        knn_conditional_mutual_information(X, Y, X, k=3)

        np.testing.assert_array_equal(X, X_before)
        np.testing.assert_array_equal(Y, Y_before)


class TestKNNDefaultMetric:
    """The KSG estimators assume the maximum norm, also when Z is None."""

    def test_marginal_case_defaults_to_chebyshev(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((300, 1))
        Y = 0.5 * X + rng.standard_normal((300, 1))

        assert knn_conditional_mutual_information(
            X, Y, None, k=5
        ) == knn_conditional_mutual_information(X, Y, None, metric="chebyshev", k=5)
        assert knn_mutual_information(X, Y, k=5) == knn_mutual_information(
            X, Y, metric="chebyshev", k=5
        )

    def test_marginal_default_is_close_to_known_value(self):
        rng = np.random.default_rng(1)
        r = 0.5
        truth = -0.5 * np.log(1 - r**2)  # about 0.144
        dependent, independent = [], []
        for _ in range(10):
            X = rng.standard_normal((500, 1))
            Y_dep = r * X + np.sqrt(1 - r**2) * rng.standard_normal((500, 1))
            Y_ind = rng.standard_normal((500, 1))
            dependent.append(knn_conditional_mutual_information(X, Y_dep, None, k=5))
            independent.append(knn_conditional_mutual_information(X, Y_ind, None, k=5))

        assert abs(np.mean(dependent) - truth) < 0.04
        assert abs(np.mean(independent)) < 0.04


class TestBootstrapKNNWarning:
    """Bootstrap resamples repeat rows, which biases k-NN estimates."""

    def _data(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 1))
        Z = rng.standard_normal((100, 1))
        Y = 0.6 * X + 0.8 * Z + rng.standard_normal((100, 1))
        return X, Y, Z

    def test_knn_bootstrap_warns_and_returns_finite_values(self):
        X, Y, Z = self._data()

        with pytest.warns(UserWarning, match="biased"):
            estimate, lower, upper, replicates = bootstrap_cmi_confidence_interval(
                X, Y, Z, method="knn", n_bootstraps=10, seed=0
            )

        assert np.isfinite(estimate)
        assert np.all(np.isfinite(replicates))
        assert lower <= upper

    def test_gaussian_bootstrap_does_not_warn(self):
        X, Y, Z = self._data()

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            bootstrap_cmi_confidence_interval(
                X, Y, Z, method="gaussian", n_bootstraps=10, seed=0
            )
