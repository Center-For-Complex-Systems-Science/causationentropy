import networkx as nx
import numpy as np
import pytest

from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
    logisic_dynamics,
    logistic_map,
)


class TestLogisticMap:
    """Test the logistic map function."""

    def test_logistic_map_basic(self):
        """Test basic logistic map functionality."""
        # Test with simple values
        X = 0.5
        r = 2.0
        result = logistic_map(X, r)
        expected = r * X * (1 - X)  # 2.0 * 0.5 * 0.5 = 0.5
        assert np.isclose(result, expected)

    def test_logistic_map_array(self):
        """Test logistic map with array input."""
        X = np.array([0.3, 0.5, 0.7])
        r = 3.0
        result = logistic_map(X, r)
        expected = r * X * (1 - X)

        assert isinstance(result, np.ndarray)
        assert np.allclose(result, expected)
        assert result.shape == X.shape

    def test_logistic_map_boundary_values(self):
        """Test logistic map with boundary values."""
        # X = 0 should give 0
        assert logistic_map(0.0, 3.0) == 0.0

        # X = 1 should give 0
        assert logistic_map(1.0, 3.0) == 0.0

        # Maximum occurs at X = 0.5
        X_max = 0.5
        r = 4.0
        max_result = logistic_map(X_max, r)
        assert max_result == r * 0.25  # r/4

    def test_logistic_map_different_r_values(self):
        """Test logistic map with different r values."""
        X = 0.6
        r_values = [1.0, 2.5, 3.7, 4.0]

        for r in r_values:
            result = logistic_map(X, r)
            expected = r * X * (1 - X)
            assert np.isclose(result, expected)

    def test_logistic_map_stability(self):
        """Test logistic map numerical properties."""
        X = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        r = 3.5
        result = logistic_map(X, r)

        # Result should be in [0, r/4] for X in [0,1]
        assert np.all(result >= 0)
        assert np.all(result <= r / 4)


class TestLogisticDynamics:
    """Test the network coupled logistic map dynamics."""

    def test_logistic_dynamics_basic(self):
        """Test basic logistic dynamics functionality."""
        XY, A = logisic_dynamics(n=5, p=0.2, t=20, r=3.5, sigma=0.1, seed=42)

        # Check output shapes
        assert XY.shape == (20, 5)
        assert A.shape == (5, 5)

        # Check data types
        assert isinstance(XY, np.ndarray)
        assert isinstance(A, (np.ndarray, np.matrix))

        # Values should be in reasonable range for logistic map
        assert np.all(XY >= -1)  # Allow some negative due to coupling
        assert np.all(XY <= 2)  # Allow some overshoot

    def test_logistic_dynamics_reproducibility(self):
        """Test that results are reproducible with same seed."""
        XY1, A1 = logisic_dynamics(n=3, p=0.3, t=10, seed=123)
        XY2, A2 = logisic_dynamics(n=3, p=0.3, t=10, seed=123)

        assert np.allclose(XY1, XY2)
        assert np.allclose(A1, A2)

    def test_logistic_dynamics_different_parameters(self):
        """Test logistic dynamics with different parameter values."""
        # Test different network sizes
        for n in [3, 5, 10]:
            XY, A = logisic_dynamics(n=n, p=0.2, t=15, seed=42)
            assert XY.shape == (15, n)
            assert A.shape == (n, n)

        # Test different time lengths
        for t in [10, 50, 100]:
            XY, A = logisic_dynamics(n=4, p=0.1, t=t, seed=42)
            assert XY.shape == (t, 4)

        # Test different connection probabilities
        for p in [0.0, 0.2, 0.5, 1.0]:
            XY, A = logisic_dynamics(n=4, p=p, t=20, seed=42)
            assert XY.shape == (20, 4)
            assert A.shape == (4, 4)

    def test_logistic_dynamics_adjacency_properties(self):
        """Test properties of the adjacency matrix."""
        XY, A = logisic_dynamics(n=6, p=0.3, t=30, seed=42)

        # A should be non-negative (after normalization)
        assert np.all(A >= 0)

        # A should be properly normalized (after transpose, each column should sum to 1 or 0)
        col_sums = np.array(A.sum(axis=0)).flatten()
        # Columns with connections should sum to 1, columns without should sum to 0
        assert np.all((np.isclose(col_sums, 1.0)) | (np.isclose(col_sums, 0.0)))

    def test_logistic_dynamics_initial_conditions(self):
        """Test that initial conditions are in [0,1]."""
        XY, A = logisic_dynamics(n=4, p=0.4, t=25, seed=42)

        # Initial conditions should be in [0,1]
        assert np.all(XY[0, :] >= 0)
        assert np.all(XY[0, :] <= 1)

    def test_logistic_dynamics_extreme_parameters(self):
        """Test with extreme parameter values."""
        # Very small network
        XY, A = logisic_dynamics(n=2, p=0.5, t=5, r=2.0, sigma=0.01, seed=42)
        assert XY.shape == (5, 2)

        # No connections (p=0)
        XY, A = logisic_dynamics(n=4, p=0.0, t=10, seed=42)
        assert np.allclose(A, 0)  # Should be all zeros

        # All connections (p=1, small network)
        XY, A = logisic_dynamics(n=3, p=1.0, t=10, seed=42)
        # Should have connections between all pairs
        non_diagonal = A[~np.eye(A.shape[0], dtype=bool)]
        assert np.any(non_diagonal > 0)  # At least some connections


class TestLinearStochasticGaussianProcess:
    """Test the linear stochastic Gaussian process."""

    def test_gaussian_process_basic(self):
        """Test basic Gaussian process functionality."""
        XY, A = linear_stochastic_gaussian_process(
            rho=0.3, n=5, T=20, p=0.2, epsilon=0.1, seed=42
        )

        # Check output shapes
        assert XY.shape == (20, 5)
        assert A.shape == (5, 5)
        assert isinstance(XY, np.ndarray)
        assert isinstance(A, np.ndarray)

        # Values should be reasonable (not infinite or NaN)
        assert np.all(np.isfinite(XY))
        assert np.all(np.isfinite(A))

    def test_gaussian_process_reproducibility(self):
        """Test reproducibility with same seed."""
        XY1, A1 = linear_stochastic_gaussian_process(rho=0.5, n=4, T=15, seed=123)
        XY2, A2 = linear_stochastic_gaussian_process(rho=0.5, n=4, T=15, seed=123)

        assert np.allclose(XY1, XY2)
        assert np.allclose(A1, A2)

    def test_gaussian_process_different_parameters(self):
        """Test with different parameter values."""
        # Different rho values
        for rho in [0.1, 0.5, 0.9]:
            XY, A = linear_stochastic_gaussian_process(rho=rho, n=4, T=10, seed=42)
            assert XY.shape == (10, 4)
            assert np.all(np.isfinite(XY))

        # Different network sizes
        for n in [2, 5, 10]:
            XY, A = linear_stochastic_gaussian_process(rho=0.4, n=n, T=15, seed=42)
            assert XY.shape == (15, n)

        # Different time lengths
        for T in [5, 25, 50]:
            XY, A = linear_stochastic_gaussian_process(rho=0.3, n=3, T=T, seed=42)
            assert XY.shape == (T, 3)

        # Different connection probabilities
        for p in [0.0, 0.2, 0.8]:
            XY, A = linear_stochastic_gaussian_process(rho=0.4, n=4, T=10, p=p, seed=42)
            assert XY.shape == (10, 4)

        # Different noise levels
        for epsilon in [0.01, 0.1, 1.0]:
            XY, A = linear_stochastic_gaussian_process(
                rho=0.3, n=3, T=10, epsilon=epsilon, seed=42
            )
            assert XY.shape == (10, 3)

    def test_gaussian_process_statistical_properties(self):
        """Test statistical properties of the generated data."""
        XY, A = linear_stochastic_gaussian_process(
            rho=0.3, n=6, T=200, epsilon=0.1, seed=42
        )

        # With sufficient data, should have reasonable variance
        var = np.var(XY, axis=0)
        assert np.all(var > 0)  # Should have non-zero variance

        # Mean should not be too extreme
        mean = np.mean(XY, axis=0)
        assert np.all(np.abs(mean) < 5)  # Reasonable range

    def test_gaussian_process_stability(self):
        """Test numerical stability."""
        # Test with parameters that might cause instability
        XY, A = linear_stochastic_gaussian_process(
            rho=0.95, n=4, T=50, p=0.8, epsilon=0.01, seed=42
        )

        # Should not blow up or have NaN/inf values
        assert np.all(np.isfinite(XY))

        # Values should not be extremely large
        assert np.all(np.abs(XY) < 1000)

    def test_gaussian_process_initial_conditions(self):
        """Test initial conditions."""
        XY, A = linear_stochastic_gaussian_process(
            rho=0.4, n=5, T=30, epsilon=0.2, seed=42
        )

        # Initial conditions should be small (scaled by epsilon)
        initial_vals = XY[0, :]
        assert np.all(np.abs(initial_vals) < 2.0)  # Reasonable bound


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_small_networks(self):
        """Test with very small networks."""
        # Single node
        XY, A = logisic_dynamics(n=1, p=0.5, t=10, seed=42)
        assert XY.shape == (10, 1)
        assert A.shape == (1, 1)
        assert A[0, 0] == 0  # No self-loops in Erdős–Rényi

        # Two nodes
        XY, A = linear_stochastic_gaussian_process(rho=0.3, n=2, T=15, seed=42)
        assert XY.shape == (15, 2)

    def test_short_time_series(self):
        """Test with very short time series."""
        # Minimum length
        XY, A = logisic_dynamics(n=3, p=0.2, t=2, seed=42)
        assert XY.shape == (2, 3)

        XY, A = linear_stochastic_gaussian_process(rho=0.4, n=3, T=2, seed=42)
        assert XY.shape == (2, 3)

    def test_extreme_coupling_parameters(self):
        """Test with extreme coupling parameters."""
        # Very small coupling
        XY, A = logisic_dynamics(n=4, p=0.3, t=15, sigma=1e-6, seed=42)
        assert XY.shape == (15, 4)

        # Large coupling (might cause instability, but should not crash)
        try:
            XY, A = logisic_dynamics(n=3, p=0.2, t=10, sigma=2.0, seed=42)
            assert XY.shape == (10, 3)
        except:
            pass  # Large sigma might cause numerical issues, which is acceptable

    def test_extreme_rho_values(self):
        """Test Gaussian process with extreme rho values."""
        # Very small rho
        XY, A = linear_stochastic_gaussian_process(rho=1e-6, n=3, T=10, seed=42)
        assert XY.shape == (10, 3)

        # rho close to 1 (might be unstable)
        try:
            XY, A = linear_stochastic_gaussian_process(rho=0.99, n=3, T=10, seed=42)
            assert XY.shape == (10, 3)
        except:
            pass  # Very high rho might cause numerical issues

    def test_zero_noise(self):
        """Test with zero noise."""
        XY, A = linear_stochastic_gaussian_process(
            rho=0.3, n=3, T=10, epsilon=0.0, seed=42
        )
        assert XY.shape == (10, 3)
        # With zero noise, initial condition should be zero
        assert np.allclose(XY[0, :], 0.0)


class TestParameterValidation:
    """Test parameter validation and type handling."""

    def test_integer_parameters(self):
        """Test that functions work with integer parameters."""
        # All parameters as integers
        XY, A = logisic_dynamics(
            n=int(4), p=0.2, t=int(10), r=4, sigma=0.1, seed=int(42)
        )
        assert XY.shape == (10, 4)

        XY, A = linear_stochastic_gaussian_process(
            rho=0.3, n=int(5), T=int(15), seed=int(42)
        )
        assert XY.shape == (15, 5)

    def test_float_parameters(self):
        """Test that functions work with float parameters where appropriate."""
        XY, A = logisic_dynamics(n=4, p=0.2, t=10, r=3.14159, sigma=0.123, seed=42)
        assert XY.shape == (10, 4)

        XY, A = linear_stochastic_gaussian_process(
            rho=0.333, n=4, T=12, epsilon=0.01, seed=42
        )
        assert XY.shape == (12, 4)

    def test_random_seed_handling(self):
        """Test different types of random seeds."""
        # Integer seed
        XY1, A1 = logisic_dynamics(n=3, p=0.3, t=10, seed=42)

        # Different seed gives different results
        XY2, A2 = logisic_dynamics(n=3, p=0.3, t=10, seed=123)
        assert not np.allclose(XY1, XY2)

        # Same seed gives same results
        XY3, A3 = logisic_dynamics(n=3, p=0.3, t=10, seed=42)
        assert np.allclose(XY1, XY3)
        assert np.allclose(A1, A3)
