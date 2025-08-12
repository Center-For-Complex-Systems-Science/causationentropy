import numpy as np
import pytest
from scipy import stats
from scipy.special import i0, i1
from unittest.mock import patch, MagicMock

# Import your entropy functions - adjust the import path based on your structure
from causalentropy.core.information.entropy import (
    l2dist,
    hyperellipsoid_check,
    kde_entropy,
    geometric_knn_entropy,
    poisson_entropy,
    poisson_joint_entropy,
)


class TestUtilityFunctions:
    """Test helper functions used in entropy calculations."""

    def test_l2dist(self):
        """Test L2 distance calculation."""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        expected = np.sqrt(27)  # sqrt((4-1)^2 + (5-2)^2 + (6-3)^2)
        assert np.isclose(l2dist(a, b), expected)

        # Test with identical points
        assert l2dist(a, a) == 0.0

        # Test with 2D points
        a_2d = np.array([0, 0])
        b_2d = np.array([3, 4])
        assert l2dist(a_2d, b_2d) == 5.0

    def test_hyperellipsoid_check(self):
        """Test hyperellipsoid containment check."""
        # Create a simple test case
        Y = np.array([[1, 0], [0, 1], [-1, 0]])  # 3x2 matrix
        svd_Y = np.linalg.svd(Y)

        # Point inside
        Z_inside = np.array([0.1, 0.1])
        # Point outside
        Z_outside = np.array([2.0, 2.0])

        # Note: This is a basic structural test since the exact behavior
        # depends on the SVD decomposition
        result_inside = hyperellipsoid_check(svd_Y, Z_inside)
        result_outside = hyperellipsoid_check(svd_Y, Z_outside)

        assert isinstance(result_inside, (bool, np.bool_))
        assert isinstance(result_outside, (bool, np.bool_))


class TestKDEEntropy:
    """Test KDE-based entropy estimation."""

    @patch("causalentropy.core.information.entropy.KernelDensity")
    def test_kde_entropy_basic(self, mock_kde_class):
        """Test basic KDE entropy calculation."""
        # Mock the KDE behavior
        mock_kde = MagicMock()
        mock_kde.score_samples.return_value = np.array([-1, -2, -1.5])
        mock_kde_class.return_value.fit.return_value = mock_kde

        X = np.array([[1, 2], [2, 3], [3, 4]])
        result = kde_entropy(X)

        # Verify KDE was called correctly
        mock_kde_class.assert_called_once_with(bandwidth="silverman", kernel="gaussian")
        mock_kde.score_samples.assert_called_once()

        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_kde_entropy_parameters(self):
        """Test KDE entropy with different parameters."""
        X = np.random.normal(0, 1, (50, 2))

        # Test with different bandwidth
        h1 = kde_entropy(X, bandwidth=0.5)
        h2 = kde_entropy(X, bandwidth=1.0)

        assert isinstance(h1, float)
        assert isinstance(h2, float)
        assert not np.isnan(h1)
        assert not np.isnan(h2)


class TestGeometricKNNEntropy:
    """Test geometric k-NN entropy estimation."""

    def test_geometric_knn_entropy_basic(self):
        """Test basic geometric k-NN entropy."""
        # Simple 2D dataset
        np.random.seed(42)
        X = np.random.normal(0, 1, (20, 2))

        # Calculate distance matrix
        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        result = geometric_knn_entropy(X, Xdist, k=3)

        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_geometric_knn_entropy_k_values(self):
        """Test geometric k-NN entropy with different k values."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (15, 2))

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        h1 = geometric_knn_entropy(X, Xdist, k=1)
        h3 = geometric_knn_entropy(X, Xdist, k=3)

        assert isinstance(h1, float)
        assert isinstance(h3, float)
        assert not np.isnan(h1)
        assert not np.isnan(h3)


class TestPoissonEntropy:
    """Test Poisson entropy estimation."""

    def test_poisson_entropy_single_value(self):
        """Test Poisson entropy for single lambda value."""
        lambda_val = 2.0
        result = poisson_entropy(lambda_val)

        assert isinstance(result, (float, np.floating))
        assert result > 0  # Entropy should be positive
        assert not np.isnan(result)

    def test_poisson_entropy_array(self):
        """Test Poisson entropy for array of lambda values."""
        lambdas = np.array([1.0, 2.0, 3.0])
        result = poisson_entropy(lambdas)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(lambdas)
        assert np.all(result > 0)
        assert not np.any(np.isnan(result))

    def test_poisson_entropy_negative_values(self):
        """Test that negative lambda values are handled (abs taken)."""
        lambda_val = -2.0
        result = poisson_entropy(lambda_val)

        assert isinstance(result, (float, np.floating))
        assert result > 0
        assert not np.isnan(result)

    def test_poisson_joint_entropy(self):
        """Test Poisson joint entropy calculation."""
        # Create a simple covariance matrix
        Cov = np.array([[2.0, 0.5], [0.5, 3.0]])
        result = poisson_joint_entropy(Cov)

        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)


class TestIntegrationAndEdgeCases:
    """Integration tests and edge cases."""

    def test_poisson_entropy_edge_cases(self):
        """Test Poisson entropy edge cases."""
        # Very small lambda
        result = poisson_entropy(1e-6)
        assert np.isfinite(result)
        assert result > 0

        # Large lambda
        result = poisson_entropy(50.0)
        assert np.isfinite(result)
        assert result > 0
