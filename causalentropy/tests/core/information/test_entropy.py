import numpy as np
import pytest
from scipy import stats
from scipy.special import i0, i1
from unittest.mock import patch, MagicMock

# Import your entropy functions - adjust the import path based on your structure
from causalentropy.core.information.entropy import (
    l2dist, hyperellipsoid_check, kde_entropy, geometric_knn_entropy,
    poisson_entropy, poisson_joint_entropy, negative_binomial_entropy,
    hawkes_entropy, von_mises_entropy, laplace_entropy, histogram_entropy
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

    @patch('causalentropy.core.information.entropy.KernelDensity')
    def test_kde_entropy_basic(self, mock_kde_class):
        """Test basic KDE entropy calculation."""
        # Mock the KDE behavior
        mock_kde = MagicMock()
        mock_kde.score_samples.return_value = np.array([-1, -2, -1.5])
        mock_kde_class.return_value.fit.return_value = mock_kde

        X = np.array([[1, 2], [2, 3], [3, 4]])
        result = kde_entropy(X)

        # Verify KDE was called correctly
        mock_kde_class.assert_called_once_with(bandwidth='silverman', kernel='gaussian')
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


class TestNegativeBinomialEntropy:
    """Test negative binomial entropy estimation."""

    def test_negative_binomial_entropy_basic(self):
        """Test basic negative binomial entropy."""
        r, p = 5, 0.3
        result = negative_binomial_entropy(r, p)

        assert isinstance(result, float)
        assert result > 0
        assert not np.isnan(result)

    def test_negative_binomial_entropy_different_bases(self):
        """Test negative binomial entropy with different bases."""
        r, p = 3, 0.4

        h_e = negative_binomial_entropy(r, p, base=np.e)
        h_2 = negative_binomial_entropy(r, p, base=2)
        h_10 = negative_binomial_entropy(r, p, base=10)

        # Base conversion should satisfy: H_b = H_e / ln(b)
        assert np.isclose(h_2, h_e / np.log(2), rtol=1e-10)
        assert np.isclose(h_10, h_e / np.log(10), rtol=1e-10)

    def test_negative_binomial_entropy_invalid_parameters(self):
        """Test negative binomial entropy with invalid parameters."""
        # Invalid p values
        with pytest.raises(ValueError, match="p must satisfy 0 < p < 1"):
            negative_binomial_entropy(5, 0)

        with pytest.raises(ValueError, match="p must satisfy 0 < p < 1"):
            negative_binomial_entropy(5, 1)

        with pytest.raises(ValueError, match="p must satisfy 0 < p < 1"):
            negative_binomial_entropy(5, -0.1)

        # Invalid r values
        with pytest.raises(ValueError, match="r must be positive"):
            negative_binomial_entropy(0, 0.5)

        with pytest.raises(ValueError, match="r must be positive"):
            negative_binomial_entropy(-1, 0.5)

    def test_negative_binomial_entropy_custom_max_k(self):
        """Test negative binomial entropy with custom max_k."""
        r, p = 2, 0.6

        h1 = negative_binomial_entropy(r, p, max_k=50)
        h2 = negative_binomial_entropy(r, p, max_k=100)

        # Higher max_k should give more accurate result
        assert isinstance(h1, float)
        assert isinstance(h2, float)
        assert h1 > 0 and h2 > 0


class TestHawkesEntropy:
    """Test Hawkes process entropy estimation."""

    def test_hawkes_entropy_basic(self):
        """Test basic Hawkes entropy calculation."""
        events = np.array([0.5, 1.2, 2.1, 3.5, 4.2])
        mu, alpha, beta = 0.5, 0.3, 1.0

        result = hawkes_entropy(events, mu, alpha, beta)

        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_hawkes_entropy_different_bases(self):
        """Test Hawkes entropy with different bases."""
        events = np.array([1.0, 2.0, 3.5])
        mu, alpha, beta = 1.0, 0.2, 0.8

        h_e = hawkes_entropy(events, mu, alpha, beta, base=np.e)
        h_2 = hawkes_entropy(events, mu, alpha, beta, base=2)

        assert np.isclose(h_2, h_e / np.log(2), rtol=1e-10)

    def test_hawkes_entropy_invalid_events(self):
        """Test Hawkes entropy with invalid event sequences."""
        mu, alpha, beta = 1.0, 0.5, 1.0

        # Empty events
        with pytest.raises(ValueError, match="events must be a non-empty 1-D array"):
            hawkes_entropy(np.array([]), mu, alpha, beta)

        # Non-increasing events
        with pytest.raises(ValueError, match="events must be strictly increasing"):
            hawkes_entropy(np.array([1.0, 2.0, 1.5]), mu, alpha, beta)

    def test_hawkes_entropy_invalid_parameters(self):
        """Test Hawkes entropy with invalid parameters."""
        events = np.array([1.0, 2.0, 3.0])

        # Invalid mu
        with pytest.raises(ValueError, match="require mu>0, alpha≥0, beta>0"):
            hawkes_entropy(events, -0.1, 0.5, 1.0)

        # Invalid alpha
        with pytest.raises(ValueError, match="require mu>0, alpha≥0, beta>0"):
            hawkes_entropy(events, 1.0, -0.1, 1.0)

        # Invalid beta
        with pytest.raises(ValueError, match="require mu>0, alpha≥0, beta>0"):
            hawkes_entropy(events, 1.0, 0.5, 0)

    def test_hawkes_entropy_with_T(self):
        """Test Hawkes entropy with explicit time horizon."""
        events = np.array([1.0, 2.0, 3.0])
        mu, alpha, beta = 1.0, 0.3, 0.8

        h1 = hawkes_entropy(events, mu, alpha, beta, T=5.0)
        h2 = hawkes_entropy(events, mu, alpha, beta, T=10.0)

        assert isinstance(h1, float)
        assert isinstance(h2, float)
        assert not np.isnan(h1)
        assert not np.isnan(h2)

        # Invalid T
        with pytest.raises(ValueError, match="T must be ≥ last event time"):
            hawkes_entropy(events, mu, alpha, beta, T=2.0)


class TestVonMisesEntropy:
    """Test von Mises entropy estimation."""

    def test_von_mises_entropy_basic(self):
        """Test basic von Mises entropy calculation."""
        kappa = 2.0
        result = von_mises_entropy(kappa)

        assert isinstance(result, float)
        assert result > 0
        assert not np.isnan(result)

    def test_von_mises_entropy_zero_kappa(self):
        """Test von Mises entropy with kappa=0 (uniform distribution)."""
        result = von_mises_entropy(0.0)
        expected = np.log(2 * np.pi)  # Uniform on circle

        assert np.isclose(result, expected)

    def test_von_mises_entropy_array(self):
        """Test von Mises entropy with array of kappa values."""
        kappas = np.array([0.0, 1.0, 2.0, 5.0])
        result = von_mises_entropy(kappas)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(kappas)
        assert np.all(result > 0)
        assert not np.any(np.isnan(result))

    def test_von_mises_entropy_different_bases(self):
        """Test von Mises entropy with different bases."""
        kappa = 1.5

        h_e = von_mises_entropy(kappa, base=np.e)
        h_2 = von_mises_entropy(kappa, base=2)

        assert np.isclose(h_2, h_e / np.log(2), rtol=1e-10)

    def test_von_mises_entropy_invalid_kappa(self):
        """Test von Mises entropy with invalid kappa values."""
        with pytest.raises(ValueError, match="kappa must be ≥ 0"):
            von_mises_entropy(-1.0)

        with pytest.raises(ValueError, match="kappa must be ≥ 0"):
            von_mises_entropy(np.array([1.0, -0.5, 2.0]))


class TestLaplaceEntropy:
    """Test Laplace entropy estimation."""

    def test_laplace_entropy_basic(self):
        """Test basic Laplace entropy calculation."""
        b = 1.0
        result = laplace_entropy(b)
        expected = 1.0 + np.log(2.0)  # Known formula

        assert np.isclose(result, expected)

    def test_laplace_entropy_different_scales(self):
        """Test Laplace entropy with different scale parameters."""
        b1, b2 = 0.5, 2.0

        h1 = laplace_entropy(b1)
        h2 = laplace_entropy(b2)

        # Larger scale should give higher entropy
        assert h2 > h1
        assert isinstance(h1, float)
        assert isinstance(h2, float)

    def test_laplace_entropy_array(self):
        """Test Laplace entropy with array of scale parameters."""
        bs = np.array([0.5, 1.0, 2.0])
        result = laplace_entropy(bs)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(bs)
        assert np.all(result > 0)

    def test_laplace_entropy_different_bases(self):
        """Test Laplace entropy with different bases."""
        b = 1.5

        h_e = laplace_entropy(b, base=np.e)
        h_2 = laplace_entropy(b, base=2)

        assert np.isclose(h_2, h_e / np.log(2), rtol=1e-10)

    def test_laplace_entropy_invalid_scale(self):
        """Test Laplace entropy with invalid scale parameters."""
        with pytest.raises(ValueError, match="scale parameter b must be positive"):
            laplace_entropy(0.0)

        with pytest.raises(ValueError, match="scale parameter b must be positive"):
            laplace_entropy(-1.0)

        with pytest.raises(ValueError, match="scale parameter b must be positive"):
            laplace_entropy(np.array([1.0, 0.0, 2.0]))


class TestHistogramEntropy:
    """Test histogram-based entropy estimation."""

    def test_histogram_entropy_basic(self):
        """Test basic histogram entropy calculation."""
        # Uniform data should have high entropy
        x = np.random.uniform(0, 1, 1000)
        result = histogram_entropy(x, bins=10)

        assert isinstance(result, float)
        assert result > 0
        assert not np.isnan(result)

    def test_histogram_entropy_deterministic(self):
        """Test histogram entropy with deterministic data."""
        # All same value should have zero entropy
        x = np.ones(100)
        result = histogram_entropy(x, bins=10)

        assert np.isclose(result, 0.0, atol=1e-15)

    def test_histogram_entropy_different_bins(self):
        """Test histogram entropy with different bin counts."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)

        h1 = histogram_entropy(x, bins=10)
        h2 = histogram_entropy(x, bins=50)

        assert isinstance(h1, float)
        assert isinstance(h2, float)
        assert h1 > 0 and h2 > 0

    def test_histogram_entropy_different_bases(self):
        """Test histogram entropy with different bases."""
        x = np.random.exponential(1.0, 500)

        h_e = histogram_entropy(x, base=np.e)
        h_2 = histogram_entropy(x, base=2)
        h_10 = histogram_entropy(x, base=10)

        assert np.isclose(h_2, h_e / np.log(2), rtol=1e-10)
        assert np.isclose(h_10, h_e / np.log(10), rtol=1e-10)

    def test_histogram_entropy_empty_data(self):
        """Test histogram entropy with empty data."""
        with pytest.raises(ValueError, match="x must contain at least one value"):
            histogram_entropy(np.array([]))

    def test_histogram_entropy_multidimensional(self):
        """Test histogram entropy flattens multidimensional input."""
        x = np.random.normal(0, 1, (10, 10))
        result = histogram_entropy(x, bins=20)

        assert isinstance(result, float)
        assert result > 0
        assert not np.isnan(result)


class TestIntegrationAndEdgeCases:
    """Integration tests and edge cases."""

    def test_entropy_monotonicity_properties(self):
        """Test that entropy behaves as expected under certain transformations."""
        # Von Mises: higher kappa should give lower entropy
        kappa_low, kappa_high = 0.5, 5.0
        h_low = von_mises_entropy(kappa_low)
        h_high = von_mises_entropy(kappa_high)
        assert h_low > h_high

        # Laplace: higher scale should give higher entropy
        b_low, b_high = 0.5, 2.0
        h_low = laplace_entropy(b_low)
        h_high = laplace_entropy(b_high)
        assert h_high > h_low

    def test_entropy_base_conversion_consistency(self):
        """Test that base conversion is consistent across functions."""
        functions_and_params = [
            (negative_binomial_entropy, (3, 0.4)),
            (hawkes_entropy, (np.array([1.0, 2.0, 3.0]), 1.0, 0.3, 0.8)),
            (von_mises_entropy, (1.5,)),
            (laplace_entropy, (1.2,)),
            (histogram_entropy, (np.random.normal(0, 1, 100),))
        ]

        for func, params in functions_and_params:
            h_e = func(*params, base=np.e)
            h_2 = func(*params, base=2)
            h_10 = func(*params, base=10)

            # Test base conversion relationships
            assert np.isclose(h_2, h_e / np.log(2), rtol=1e-10)
            assert np.isclose(h_10, h_e / np.log(10), rtol=1e-10)

    def test_numerical_stability(self):
        """Test numerical stability with extreme parameter values."""
        # Very small Laplace scale
        result = laplace_entropy(1e-10)
        assert np.isfinite(result)

        # Very large von Mises concentration
        result = von_mises_entropy(100.0)
        assert np.isfinite(result)
        # For very large kappa, von Mises entropy can be negative (high concentration)
        # This is mathematically correct behavior

        # Very small negative binomial p
        result = negative_binomial_entropy(1, 0.001)
        assert np.isfinite(result)
        assert result > 0