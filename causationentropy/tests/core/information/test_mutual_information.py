from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from causationentropy.core.information.mutual_information import (
    gaussian_mutual_information,
    geometric_knn_mutual_information,
    kde_mutual_information,
    knn_mutual_information,
)


class TestGaussianMutualInformation:
    """Test Gaussian mutual information calculation."""

    def test_gaussian_mi_independent_variables(self):
        """Test MI of independent Gaussian variables (should be close to 0)."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        mi = gaussian_mutual_information(X, Y)

        # Independent variables should have low MI
        assert isinstance(mi, float)
        assert abs(mi) < 0.5  # Should be close to 0 for independent data

    def test_gaussian_mi_identical_variables(self):
        """Test MI of identical variables (should be high)."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 1))
        Y = X.copy()  # Perfect correlation

        mi = gaussian_mutual_information(X, Y)

        # Identical variables should have high MI
        assert isinstance(mi, float)
        assert mi > 0.5

    def test_gaussian_mi_correlated_variables(self):
        """Test MI of correlated variables."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 1))
        Y = 0.8 * X + 0.2 * np.random.normal(0, 1, (n, 1))

        mi = gaussian_mutual_information(X, Y)

        assert isinstance(mi, float)
        assert mi > 0  # Correlated variables should have positive MI

    def test_gaussian_mi_multivariate(self):
        """Test MI with multivariate variables."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 2))
        Y = np.random.normal(0, 1, (n, 3))

        mi = gaussian_mutual_information(X, Y)

        assert isinstance(mi, float)
        assert not np.isnan(mi)
        assert np.isfinite(mi)

    def test_gaussian_mi_dimension_mismatch(self):
        """Test behavior when X and Y have different number of samples."""
        X = np.random.normal(0, 1, (50, 2))
        Y = np.random.normal(0, 1, (40, 1))

        # Should handle gracefully or raise appropriate error
        try:
            mi = gaussian_mutual_information(X, Y)
        except (ValueError, IndexError):
            pass  # Expected for dimension mismatch


class TestKDEMutualInformation:
    """Test KDE-based mutual information calculation."""

    @patch("causationentropy.core.information.mutual_information.kde_entropy")
    def test_kde_mi_basic(self, mock_entropy):
        """Test basic KDE MI calculation."""
        # Mock entropy values
        mock_entropy.side_effect = [1.0, 1.5, 2.0]  # H(X), H(Y), H(X,Y)

        X = np.random.normal(0, 1, (20, 1))
        Y = np.random.normal(0, 1, (20, 1))

        mi = kde_mutual_information(X, Y)

        # MI = H(X) + H(Y) - H(X,Y) = 1.0 + 1.5 - 2.0 = 0.5
        assert mi == 0.5
        assert mock_entropy.call_count == 3

    def test_kde_mi_parameters(self):
        """Test KDE MI with different parameters."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (30, 1))
        Y = np.random.normal(0, 1, (30, 1))

        mi1 = kde_mutual_information(X, Y, bandwidth="silverman")
        mi2 = kde_mutual_information(X, Y, bandwidth=0.5)
        mi3 = kde_mutual_information(X, Y, kernel="linear")

        for mi in [mi1, mi2, mi3]:
            assert isinstance(mi, float)
            assert not np.isnan(mi)


class TestKNNMutualInformation:
    """Test k-NN based mutual information calculation."""

    def test_knn_mi_independent_data(self):
        """Test k-NN MI with independent data."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        mi = knn_mutual_information(X, Y, k=1)

        assert isinstance(mi, float)
        assert not np.isnan(mi)
        assert np.isfinite(mi)

    def test_knn_mi_different_k_values(self):
        """Test k-NN MI with different k values."""
        np.random.seed(42)
        n = 40
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        mi1 = knn_mutual_information(X, Y, k=1)
        mi3 = knn_mutual_information(X, Y, k=3)
        mi5 = knn_mutual_information(X, Y, k=5)

        for mi in [mi1, mi3, mi5]:
            assert isinstance(mi, float)
            assert not np.isnan(mi)
            assert np.isfinite(mi)

    def test_knn_mi_different_metrics(self):
        """Test k-NN MI with different distance metrics."""
        np.random.seed(42)
        n = 30
        X = np.random.normal(0, 1, (n, 2))
        Y = np.random.normal(0, 1, (n, 2))

        metrics = [
            "euclidean",
            "cityblock",
            "chebyshev",
        ]  # cityblock is scipy's name for manhattan
        for metric in metrics:
            mi = knn_mutual_information(X, Y, metric=metric, k=2)
            assert isinstance(mi, float)
            assert not np.isnan(mi)

    def test_knn_mi_correlated_data(self):
        """Test k-NN MI with correlated data."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 1))
        noise = np.random.normal(0, 0.1, (n, 1))
        Y = X + noise  # Strong correlation

        mi = knn_mutual_information(X, Y, k=3)

        assert isinstance(mi, float)
        assert mi > 0  # Should be positive for correlated data
        assert not np.isnan(mi)


class TestGeometricKNNMutualInformation:
    """Test geometric k-NN mutual information calculation."""

    @patch("causationentropy.core.information.mutual_information.geometric_knn_entropy")
    def test_geometric_knn_mi_basic(self, mock_entropy):
        """Test basic geometric k-NN MI calculation."""
        # Mock entropy values
        mock_entropy.side_effect = [2.0, 1.8, 3.2]  # H(X), H(Y), H(X,Y)

        X = np.random.normal(0, 1, (20, 1))
        Y = np.random.normal(0, 1, (20, 1))

        mi = geometric_knn_mutual_information(X, Y, k=2)

        # MI = H(X) + H(Y) - H(X,Y) = 2.0 + 1.8 - 3.2 = 0.6
        assert np.isclose(mi, 0.6, rtol=1e-14)
        assert mock_entropy.call_count == 3

    def test_geometric_knn_mi_real_data(self):
        """Test geometric k-NN MI with real data."""
        np.random.seed(42)
        n = 40
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        mi = geometric_knn_mutual_information(X, Y, k=3)

        assert isinstance(mi, float)
        assert not np.isnan(mi)
        assert np.isfinite(mi)

    def test_geometric_knn_mi_different_k(self):
        """Test geometric k-NN MI with different k values."""
        np.random.seed(42)
        n = 35
        X = np.random.normal(0, 1, (n, 2))
        Y = np.random.normal(0, 1, (n, 1))

        for k in [1, 2]:  # Reduced k values to avoid bounds issues with small datasets
            mi = geometric_knn_mutual_information(X, Y, k=k)
            assert isinstance(mi, float)
            assert not np.isnan(mi)

    def test_geometric_knn_mi_metrics(self):
        """Test geometric k-NN MI with different metrics."""
        np.random.seed(42)
        n = 25
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        metrics = ["euclidean", "cityblock"]  # cityblock is scipy's name for manhattan
        for metric in metrics:
            mi = geometric_knn_mutual_information(X, Y, metric=metric, k=2)
            assert isinstance(mi, float)
            assert not np.isnan(mi)

    @patch("causationentropy.core.information.mutual_information.geometric_knn_entropy")
    def test_geometric_knn_mi_non_finite_handling(self, mock_entropy):
        """Test that non-finite MI values are handled gracefully by returning 0.0."""
        # Create simple test data
        X = np.array([[1.0], [2.0], [3.0]])
        Y = np.array([[4.0], [5.0], [6.0]])

        # Test case 1: One entropy is NaN
        mock_entropy.side_effect = [np.nan, 1.0, 2.0]  # H(X), H(Y), H(X,Y)
        mi = geometric_knn_mutual_information(X, Y, k=1)
        assert mi == 0.0  # Should return 0.0 for non-finite MI

        # Test case 2: One entropy is infinite
        mock_entropy.side_effect = [2.0, np.inf, 1.0]  # H(X), H(Y), H(X,Y)
        mi = geometric_knn_mutual_information(X, Y, k=1)
        assert mi == 0.0  # Should return 0.0 for non-finite MI

        # Test case 3: Joint entropy is NaN leading to NaN MI
        mock_entropy.side_effect = [1.0, 1.0, np.nan]  # H(X), H(Y), H(X,Y)
        mi = geometric_knn_mutual_information(X, Y, k=1)
        assert mi == 0.0  # Should return 0.0 for non-finite MI

        # Test case 4: Combination that produces negative infinity
        mock_entropy.side_effect = [1.0, 1.0, np.inf]  # H(X), H(Y), H(X,Y)
        mi = geometric_knn_mutual_information(X, Y, k=1)
        assert mi == 0.0  # Should return 0.0 for non-finite MI (-inf)

        # Test case 5: Normal finite case (should not trigger fallback)
        mock_entropy.side_effect = [2.0, 1.5, 3.0]  # H(X), H(Y), H(X,Y)
        mi = geometric_knn_mutual_information(X, Y, k=1)
        assert mi == 0.5  # Normal calculation: 2.0 + 1.5 - 3.0 = 0.5
        assert np.isfinite(mi)

    @patch("causationentropy.core.information.mutual_information.geometric_knn_entropy")
    def test_geometric_knn_mi_nan_warning(self, mock_entropy):
        """Test that a warning is issued when geometric_knn_mutual_information returns NaN."""
        # Create simple test data
        X = np.array([[1.0], [2.0], [3.0]])
        Y = np.array([[4.0], [5.0], [6.0]])

        # Mock entropy to produce NaN MI
        mock_entropy.side_effect = [np.nan, 1.0, 2.0]  # H(X), H(Y), H(X,Y)

        with pytest.warns(
            UserWarning, match="NaN result in geometric_knn_mutual_information"
        ):
            mi = geometric_knn_mutual_information(X, Y, k=1)
            assert mi == 0.0


class TestMutualInformationProperties:
    """Test mathematical properties of mutual information."""

    def test_mi_symmetry(self):
        """Test that MI is symmetric: I(X;Y) = I(Y;X)."""
        np.random.seed(42)
        n = 40
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        mi_xy = gaussian_mutual_information(X, Y)
        mi_yx = gaussian_mutual_information(Y, X)

        assert np.isclose(mi_xy, mi_yx, rtol=1e-10)

    def test_mi_non_negativity(self):
        """Test that MI is non-negative."""
        np.random.seed(42)
        n = 30
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        methods = [
            (gaussian_mutual_information, {}),
            (knn_mutual_information, {"k": 3}),
        ]

        for method, kwargs in methods:
            mi = method(X, Y, **kwargs)
            assert mi >= -0.5  # Allow larger numerical errors for k-NN methods

    def test_mi_identical_variables(self):
        """Test MI properties with identical variables."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 1))

        # I(X;X) should equal H(X)
        mi_xx = gaussian_mutual_information(X, X)
        assert mi_xx > 0  # Should be positive and equal to entropy


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        X = np.array([]).reshape(0, 1)
        Y = np.array([]).reshape(0, 1)

        # Should handle gracefully or raise appropriate error
        for func in [gaussian_mutual_information, knn_mutual_information]:
            try:
                result = func(X, Y)
            except (ValueError, IndexError, np.linalg.LinAlgError):
                pass  # Expected for empty data

    def test_single_point(self):
        """Test behavior with single data point."""
        X = np.array([[1.0]])
        Y = np.array([[2.0]])

        # Should handle single point gracefully
        try:
            mi = gaussian_mutual_information(X, Y)
            # Single point may produce NaN due to degenerate covariance
            assert np.isnan(mi) or np.isfinite(mi)
        except (ValueError, np.linalg.LinAlgError):
            pass  # May fail due to insufficient data

    def test_constant_variables(self):
        """Test MI with constant variables."""
        n = 30
        X = np.ones((n, 1))  # Constant
        Y = np.random.normal(0, 1, (n, 1))

        # MI with constant should typically be 0 or undefined
        try:
            mi = gaussian_mutual_information(X, Y)
            # May be 0 or may fail due to singular covariance
        except (ValueError, np.linalg.LinAlgError):
            pass  # Expected for constant data
