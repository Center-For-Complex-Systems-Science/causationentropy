import numpy as np
import pytest
from scipy.stats import poisson

from causationentropy.core.information.entropy import (
    poisson_entropy,
    poisson_joint_entropy,
)


class TestPoissonEntropyAccuracy:
    """Regression tests for tail truncation and zero-probability terms."""

    @pytest.mark.parametrize(
        "lambdas",
        [
            np.array([0.1, 20.0]),
            np.array([0.5, 50.0]),
            np.array([1.0, 100.0]),
        ],
    )
    def test_heterogeneous_rate_array_matches_scipy(self, lambdas):
        """A small rate in an array must not truncate the tail of larger rates."""
        expected = np.array([float(poisson.entropy(lam)) for lam in lambdas])
        actual = poisson_entropy(lambdas)

        assert actual.shape == lambdas.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_zero_rate_in_array_and_all_zeros(self):
        """Zero rate has entropy 0 without NaN or shape collapse."""
        mixed = np.array([0.0, 5.0])
        actual_mixed = poisson_entropy(mixed)
        assert actual_mixed.shape == (2,)
        np.testing.assert_allclose(
            actual_mixed,
            np.array([0.0, float(poisson.entropy(5.0))]),
            rtol=1e-10,
            atol=1e-12,
        )

        zeros = np.array([0.0, 0.0])
        actual_zeros = poisson_entropy(zeros)
        assert actual_zeros.shape == (2,)
        np.testing.assert_allclose(actual_zeros, np.zeros(2), atol=1e-12)

    def test_large_lambda_where_exp_neg_lambda_underflows(self):
        """Large lambda where P(X=0) underflows to 0.0 must stay finite and accurate."""
        lam = 1000.0
        actual = poisson_entropy(lam)
        expected = float(poisson.entropy(lam))

        assert np.isfinite(actual)
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_poisson_joint_entropy_heterogeneous_diagonal(self):
        """Joint entropy uses accurate marginal entropies across mixed scales."""
        cov = np.array([[0.5, 0.1], [0.1, 50.0]])
        expected = float(poisson.entropy(0.5) + poisson.entropy(50.0) + 0.1)
        actual = poisson_joint_entropy(cov)

        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12)
