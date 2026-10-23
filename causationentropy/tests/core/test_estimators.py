"""Tests for entropic regression equation discovery (issue #8)."""

import numpy as np
import pytest

from causationentropy.core.estimators import (
    EntropicRegression,
    central_difference,
    entropic_regression,
    polynomial_library,
)


def _lorenz(T, dt, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Lorenz trajectory from fourth-order Runge-Kutta, and its exact vector field."""

    def F(x):
        return np.array(
            [
                sigma * (x[1] - x[0]),
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2],
            ]
        )

    X = np.empty((T, 3))
    x = np.array([-8.0, 7.0, 27.0])
    for t in range(T):
        X[t] = x
        k1 = F(x)
        k2 = F(x + dt / 2 * k1)
        k3 = F(x + dt / 2 * k2)
        k4 = F(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X, np.array([F(x) for x in X])


# Library order for 3 variables, degree 2:
# x, y, z, x^2, x y, x z, y^2, y z, z^2
LORENZ_COEF = np.array(
    [
        [-10.0, 10.0, 0, 0, 0, 0, 0, 0, 0],
        [28.0, -1.0, 0, 0, 0, -1.0, 0, 0, 0],
        [0, 0, -8.0 / 3.0, 0, 1.0, 0, 0, 0, 0],
    ]
)


class TestPolynomialLibrary:
    def test_names_and_values(self):
        Phi, names = polynomial_library(np.array([[2.0, 3.0]]), degree=3)
        assert names == [
            "x0", "x1", "x0^2", "x0 x1", "x1^2",
            "x0^3", "x0^2 x1", "x0 x1^2", "x1^3",
        ]  # fmt: skip
        assert Phi.tolist() == [[2, 3, 4, 6, 9, 8, 12, 18, 27]]

    def test_bias_and_feature_names(self):
        Phi, names = polynomial_library(
            np.array([1.0, 2.0]), degree=2, include_bias=True, feature_names=["u"]
        )
        assert names == ["1", "u", "u^2"]
        assert Phi.tolist() == [[1, 1, 1], [1, 2, 4]]

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            polynomial_library(np.ones((3, 2)), degree=0)
        with pytest.raises(ValueError):
            polynomial_library(np.ones((3, 2)), feature_names=["a"])


class TestCentralDifference:
    def test_exact_for_quadratics(self):
        t = np.arange(10) * 0.1
        X = np.column_stack([3 * t**2, 1 - t])
        expected = np.column_stack([6 * t[1:-1], -np.ones(8)])
        assert np.allclose(central_difference(X, dt=0.1), expected)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            central_difference(np.arange(5.0), dt=0.0)
        with pytest.raises(ValueError):
            central_difference(np.arange(2.0), dt=0.1)


class TestEntropicRegressionFunction:
    def test_recovers_sparse_linear_model(self):
        rng = np.random.default_rng(0)
        Phi = rng.normal(size=(400, 6))
        f = 2.0 * Phi[:, 0] - 3.0 * Phi[:, 3] + 0.1 * rng.normal(size=400)
        S, coef, b0 = entropic_regression(f, Phi, n_shuffles=50, rng=1)
        assert S == [0, 3]
        assert coef[[0, 3]] == pytest.approx([2.0, -3.0], abs=0.03)
        assert np.all(coef[[1, 2, 4, 5]] == 0)
        assert b0 == 0.0

    def test_pure_noise_selects_nothing(self):
        rng = np.random.default_rng(2)
        S, coef, _ = entropic_regression(
            rng.normal(size=300), rng.normal(size=(300, 4)), n_shuffles=50, rng=3
        )
        assert S == []
        assert np.all(coef == 0)

    def test_fit_intercept(self):
        rng = np.random.default_rng(4)
        Phi = rng.normal(size=(300, 3))
        f = 5.0 + 2.0 * Phi[:, 1] + 0.1 * rng.normal(size=300)
        S, coef, b0 = entropic_regression(
            f, Phi, n_shuffles=50, fit_intercept=True, rng=5
        )
        assert S == [1]
        assert coef[1] == pytest.approx(2.0, abs=0.03)
        assert b0 == pytest.approx(5.0, abs=0.03)

    def test_exact_fit_stops_forward_stage(self):
        rng = np.random.default_rng(6)
        Phi = rng.normal(size=(200, 5))
        f = Phi[:, 2] - 0.5 * Phi[:, 4]
        S, coef, _ = entropic_regression(f, Phi, n_shuffles=50, rng=7)
        assert S == [2, 4]
        assert coef[[2, 4]] == pytest.approx([1.0, -0.5])

    def test_knn_uses_chebyshev_in_every_step(self, monkeypatch):
        # The first forward step has no conditioning; it must use the same
        # metric as the conditional steps.
        import causationentropy.core.estimators as est

        metrics = []
        original = est.conditional_mutual_information

        def recording(*args, **kwargs):
            metrics.append((args[2] is None, kwargs["metric"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(est, "conditional_mutual_information", recording)
        rng = np.random.default_rng(8)
        Phi = rng.normal(size=(200, 3))
        f = Phi[:, 0] + 0.1 * rng.normal(size=200)
        entropic_regression(f, Phi, n_shuffles=10, rng=9)
        assert (True, "chebyshev") in metrics
        assert (False, "chebyshev") in metrics
        assert {m for _, m in metrics} == {"chebyshev"}

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            entropic_regression(np.zeros(10), np.zeros((9, 2)))


class TestEntropicRegressionModel:
    def test_lorenz_exact_derivatives(self):
        X, X_dot = _lorenz(400, 0.01)
        model = EntropicRegression(degree=2, n_shuffles=50, random_state=0)
        model.fit(X, X_dot=X_dot)
        assert model.support_ == [[0, 1], [0, 1, 5], [2, 4]]
        assert np.allclose(model.coef_, LORENZ_COEF, atol=1e-6)

    def test_lorenz_noisy_states(self):
        # Gaussian observation noise, derivatives from central differences
        # (the setting of Fig. 2 in AlMomani, Sun and Bollt, 2020).
        X, _ = _lorenz(500, 0.01)
        X = X + 0.01 * np.random.default_rng(1).normal(size=X.shape)
        model = EntropicRegression(degree=2, n_shuffles=50, random_state=0)
        model.fit(X, dt=0.01, feature_names=["x", "y", "z"])
        assert model.support_ == [[0, 1], [0, 1, 5], [2, 4]]
        nonzero = LORENZ_COEF != 0
        assert np.allclose(model.coef_[nonzero], LORENZ_COEF[nonzero], rtol=0.05)
        assert model.equations(precision=1) == [
            "x' = -10.0 x + 10.0 y",
            "y' = 27.8 x - 1.0 y - 1.0 x z",
            "z' = -2.7 z + 1.0 x y",
        ]

    def test_logistic_map(self):
        # X_dot can be any target, e.g. the next state of a map.
        x = np.empty(301)
        x[0] = 0.3
        for t in range(300):
            x[t + 1] = 3.9 * x[t] * (1 - x[t])
        x_next = x[1:] + 1e-4 * np.random.default_rng(2).normal(size=300)
        model = EntropicRegression(degree=3, n_shuffles=50, random_state=0)
        model.fit(x[:-1], X_dot=x_next)
        assert model.support_ == [[0, 1]]
        assert model.coef_[0, :2] == pytest.approx([3.9, -3.9], abs=1e-2)

    def test_predict(self):
        X, X_dot = _lorenz(300, 0.01)
        model = EntropicRegression(n_shuffles=30, random_state=0).fit(X, X_dot=X_dot)
        assert np.allclose(model.predict(X), X_dot, atol=1e-6)

    def test_reproducible_with_random_state(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(200, 2))
        X_dot = np.column_stack([X[:, 0] * X[:, 1], -X[:, 1]]) + 0.05 * rng.normal(
            size=(200, 2)
        )
        a = EntropicRegression(n_shuffles=30, random_state=9).fit(X, X_dot=X_dot)
        b = EntropicRegression(n_shuffles=30, random_state=9).fit(X, X_dot=X_dot)
        assert a.support_ == b.support_
        assert np.array_equal(a.coef_, b.coef_)

    def test_fit_errors(self):
        X = np.ones((10, 2))
        with pytest.raises(ValueError, match="dt or X_dot"):
            EntropicRegression().fit(X)
        with pytest.raises(ValueError, match="shape"):
            EntropicRegression().fit(X, X_dot=np.ones((9, 2)))
