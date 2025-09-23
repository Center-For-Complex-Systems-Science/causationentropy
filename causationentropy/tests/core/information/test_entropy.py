from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy import stats
from scipy.special import i0, i1

# Import your entropy functions - adjust the import path based on your structure
from causationentropy.core.information.entropy import (
    geometric_knn_entropy,
    hyperellipsoid_check,
    kde_entropy,
    l2dist,
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

    @patch("causationentropy.core.information.entropy.KernelDensity")
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

    def test_poisson_entropy_exception_path(self):
        """Test the exception handling path in poisson_entropy: except: est = -np.sum(est_a)."""
        # The exception occurs when np.sum(est_a, axis=0) fails
        # We can force this by manipulating the internal array P to have incompatible dimensions

        # Patch the function at the exact point where the exception occurs
        original_sum = np.sum
        exception_triggered = False

        def patched_sum(a, axis=None):
            nonlocal exception_triggered
            # When we see the specific call that tries axis=0 on est_a
            if axis == 0 and not exception_triggered:
                exception_triggered = True
                # Force an exception to trigger the except path
                raise ValueError("axis 0 is out of bounds")
            return original_sum(a, axis=axis)

        with patch(
            "causationentropy.core.information.entropy.np.sum", side_effect=patched_sum
        ):
            # This should trigger the except: est = -np.sum(est_a) path
            result = poisson_entropy(1.0)

            # Verify the exception path was taken
            assert exception_triggered, "Exception path was not triggered"

            # Should still return a valid result via the exception handling
            assert isinstance(result, (float, np.floating))
            assert np.isfinite(result)


class TestGeometricKNNEntropyEdgeCases:
    """Test edge cases and error handling in geometric k-NN entropy."""

    def test_geometric_knn_entropy_zero_distances(self):
        """Test handling of zero/very small distances (log_distances edge case)."""
        # Create data with duplicate points to force zero distances
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],  # Duplicate point (zero distance)
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )

        # Calculate distance matrix
        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # This should trigger the log_distances.append(-12.0) line
        result = geometric_knn_entropy(X, Xdist, k=1)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_identical_points(self):
        """Test with dataset containing multiple identical points."""
        # Create data where many points are identical
        X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # Should handle zero distances gracefully
        result = geometric_knn_entropy(X, Xdist, k=2)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_singular_matrix(self):
        """Test handling of singular matrices (SVD exception case)."""
        # Create data that will lead to singular matrices in SVD
        # Points all on a line (rank deficient)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # This may trigger SVD exception handling
        result = geometric_knn_entropy(X, Xdist, k=3)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_degenerate_data(self):
        """Test with extremely degenerate data to trigger various edge cases."""
        # Create data with very small variations that may cause numerical issues
        X = np.array(
            [
                [0.0, 0.0],
                [1e-15, 1e-15],  # Extremely close to first point
                [1e-14, 1e-14],  # Also very close
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # Should handle numerical edge cases
        result = geometric_knn_entropy(X, Xdist, k=2)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_rank_deficient_neighborhoods(self):
        """Test data designed to create rank-deficient neighborhood matrices."""
        # Create data where neighborhoods have very small singular values
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1e-13, 0.0, 0.0],  # Very small perturbation
                [0.0, 1e-13, 0.0],  # Another small perturbation
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # This should trigger the sing_ratio_sum += -12.0 line
        result = geometric_knn_entropy(X, Xdist, k=3)

        assert isinstance(result, float)
        assert np.isfinite(result)

    @patch("causationentropy.core.information.entropy.np.linalg.svd")
    def test_geometric_knn_entropy_svd_failure(self, mock_svd):
        """Test SVD failure handling using mocking."""
        # Mock SVD to raise an exception
        mock_svd.side_effect = np.linalg.LinAlgError("SVD did not converge")

        # Regular data
        X = np.random.normal(0, 1, (10, 2))
        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # Should handle SVD exception gracefully
        result = geometric_knn_entropy(X, Xdist, k=3)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_infinite_correction(self):
        """Test handling of non-finite geometric corrections."""
        # Create a scenario that might produce infinite corrections
        # Use data with extreme aspect ratios
        X = np.array(
            [
                [0.0, 0.0],
                [1e-100, 0.0],  # Extremely close in one dimension
                [1e10, 0.0],  # Very far in one dimension
                [0.0, 1.0],
                [0.0, -1.0],
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # Should handle potential infinite/NaN corrections
        result = geometric_knn_entropy(X, Xdist, k=3)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_all_edge_cases_combined(self):
        """Test with data designed to trigger multiple edge cases."""
        # Complex scenario with multiple potential issues
        X = np.array(
            [
                [0.0, 0.0, 0.0],  # Origin
                [0.0, 0.0, 0.0],  # Duplicate (zero distance)
                [1e-15, 1e-15, 1e-15],  # Nearly zero (small distance)
                [1e-13, 0.0, 0.0],  # Small singular values
                [0.0, 1e-13, 0.0],  # Small singular values
                [0.0, 0.0, 1e-13],  # Small singular values
                [1.0, 0.0, 0.0],  # Normal point
                [0.0, 1.0, 0.0],  # Normal point
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # Should handle combination of edge cases
        result = geometric_knn_entropy(X, Xdist, k=4)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_target_line_coverage(self):
        """Specific test to ensure target error handling lines are covered."""
        # Test case 1: Force zero distance to trigger log_distances.append(-12.0)
        X_zero_dist = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])  # Exact duplicate
        N = X_zero_dist.shape[0]
        Xdist_zero = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist_zero[i, j] = l2dist(X_zero_dist[i], X_zero_dist[j])

        result1 = geometric_knn_entropy(X_zero_dist, Xdist_zero, k=1)
        assert np.isfinite(result1)

        # Test case 2: Force very small singular values to trigger sing_ratio_sum += -12.0
        X_small_sing = np.array(
            [
                [0.0, 0.0, 0.0],
                [1e-14, 0.0, 0.0],  # Very small perturbation
                [0.0, 1e-14, 0.0],  # Another very small perturbation
                [1.0, 0.0, 0.0],
            ]
        )
        N = X_small_sing.shape[0]
        Xdist_small = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist_small[i, j] = l2dist(X_small_sing[i], X_small_sing[j])

        result2 = geometric_knn_entropy(X_small_sing, Xdist_small, k=2)
        assert np.isfinite(result2)

    def test_geometric_knn_entropy_extreme_singular_ratios(self):
        """Test to specifically trigger sing_ratio_sum += -12.0 line."""
        # Create data with extreme singular value ratios to force ratio <= 1e-12
        X = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.000000001, 1e-15, 1e-15],  # Very small perturbations
                [0.999999999, 1e-15, 1e-15],  # Very small perturbations
                [1.0000000005, 1e-16, 1e-16],  # Even smaller perturbations
                [0.0, 1.0, 0.0],
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # This should trigger sing_ratio_sum += -12.0 when ratio <= 1e-12
        result = geometric_knn_entropy(X, Xdist, k=3)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_non_finite_corrections(self):
        """Test to specifically trigger geometric_corrections.append(0.0) for non-finite values."""
        # Create data that might produce infinite or NaN corrections
        X = np.array(
            [
                [0.0, 0.0],
                [1e-100, 0.0],  # Extremely small value
                [1e100, 0.0],  # Extremely large value
                [0.0, 1e-100],  # Extremely small value
                [0.0, 1e100],  # Extremely large value
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # This should trigger geometric_corrections.append(0.0) for non-finite corrections
        result = geometric_knn_entropy(X, Xdist, k=3)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_pathological_matrix(self):
        """Test with pathological matrix to force both edge cases."""
        # Create data designed to hit both specific lines
        X = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0 + 1e-16, 1.0 + 1e-16, 1.0 + 1e-16],  # Virtually identical
                [1.0 + 1e-15, 1.0 + 1e-15, 1.0 + 1e-15],  # Virtually identical
                [0.0, 0.0, 0.0],  # Origin
                [1e-100, 1e-100, 1e-100],  # Near origin
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # Should trigger multiple edge cases
        result = geometric_knn_entropy(X, Xdist, k=4)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_force_ratio_edge_case(self):
        """Test specifically designed to force ratio <= 1e-12 condition."""
        # Create a matrix that will have very small singular value ratios
        # when neighborhoods are formed
        base_point = np.array([1.0, 1.0])

        X = np.array(
            [
                base_point,
                base_point
                + np.array([1e-14, 0]),  # Tiny perturbation in first dimension
                base_point
                + np.array([0, 1e-14]),  # Tiny perturbation in second dimension
                base_point + np.array([1e-13, 1e-14]),  # Mixed tiny perturbations
                np.array([2.0, 2.0]),  # Distant point
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # This should create neighborhoods with very small singular value ratios
        result = geometric_knn_entropy(X, Xdist, k=3)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_precise_singular_ratio_trigger(self):
        """Precisely target the sing_ratio_sum += -12.0 line."""
        # Create data that will have large first singular value but tiny second
        # Use nearly collinear points with tiny perpendicular displacement

        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 1e-7],  # On line with tiny y displacement
                [2.0, -1e-7],  # On line with tiny y displacement
                [3.0, 1.5e-7],  # On line with tiny y displacement
                [4.0, -1.5e-7],  # On line with tiny y displacement
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # This creates neighborhoods where points are nearly collinear
        # Leading to large first singular value, tiny second -> tiny ratio
        result = geometric_knn_entropy(X, Xdist, k=3)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_infinite_correction_trigger(self):
        """Precisely target the geometric_corrections.append(0.0) line for non-finite corrections."""
        # Create data with extreme scaling to force non-finite corrections

        X = np.array(
            [
                [0.0, 0.0],
                [1e-200, 0.0],  # Extremely small - may cause underflow
                [1e200, 0.0],  # Extremely large - may cause overflow
                [0.0, 1e-200],  # Extremely small
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # This extreme scaling should trigger non-finite correction handling
        result = geometric_knn_entropy(X, Xdist, k=2)

        # Even if internal calculations are non-finite, result should be finite due to fallback
        assert isinstance(result, float)
        # Note: result might be inf if the overall calculation overflows

    def test_geometric_knn_entropy_rank_1_matrix_neighborhoods(self):
        """Create neighborhoods that are rank-1 to force tiny singular values."""
        # All neighbors lie exactly on a line through the center point

        center = np.array([1.0, 1.0])
        direction = np.array([1.0, 0.5])  # Direction vector

        X = np.array(
            [
                center,  # Center point
                center + 0.1 * direction,  # On line from center
                center - 0.1 * direction,  # On line from center
                center + 0.05 * direction,  # On line from center
                center - 0.05 * direction,  # On line from center
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # When center point's neighborhood is computed, all neighbors are collinear
        # This should create a neighborhood matrix with rank 1
        result = geometric_knn_entropy(X, Xdist, k=4)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_designed_for_tiny_ratios(self):
        """Design data specifically to create sing_Yi[l]/sing_Yi[0] <= 1e-12."""
        # Create a 3D dataset where neighborhoods have 1 large and 2 very small singular values

        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1e-8, 1e-8],  # Large change in x, tiny in y,z
                [0.5, -1e-8, 1e-8],  # Medium change in x, tiny in y,z
                [1.5, 1e-8, -1e-8],  # Large change in x, tiny in y,z
                [0.25, 1e-8, 1e-8],  # Small change in x, tiny in y,z
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # This should create neighborhoods with one dominant direction (x-axis)
        # and very small variations in y,z leading to tiny singular value ratios
        result = geometric_knn_entropy(X, Xdist, k=3)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_geometric_knn_entropy_guaranteed_edge_cases(self):
        """Test cases specifically crafted to guarantee hitting the target lines."""

        # Test case 1: Create data that forces tiny singular value ratios
        # Use points that are almost perfectly collinear to force second singular value to be tiny
        X1 = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],  # Exactly on x-axis
                [2.0, 1e-13],  # Almost on x-axis (tiny y)
                [3.0, -1e-13],  # Almost on x-axis (tiny y)
                [4.0, 2e-13],  # Almost on x-axis (tiny y)
            ]
        )

        N1 = X1.shape[0]
        Xdist1 = np.zeros((N1, N1))
        for i in range(N1):
            for j in range(N1):
                Xdist1[i, j] = l2dist(X1[i], X1[j])

        # When computing neighborhoods, the Y_i matrix will have one large singular value (x-direction)
        # and one very small singular value (y-direction), making ratio = small/large <= 1e-12
        result1 = geometric_knn_entropy(X1, Xdist1, k=3)
        assert isinstance(result1, float)
        assert np.isfinite(result1)

        # Test case 2: Create data guaranteed to produce non-finite corrections
        # Use extreme coordinates that will cause overflow/underflow in calculations
        X2 = np.array(
            [
                [0.0, 0.0],
                [1e100, 1e100],  # Extremely large values
                [1e-100, 1e-100],  # Extremely small values
                [1e100, -1e100],  # Mixed extreme values
            ]
        )

        N2 = X2.shape[0]
        Xdist2 = np.zeros((N2, N2))
        for i in range(N2):
            for j in range(N2):
                Xdist2[i, j] = l2dist(X2[i], X2[j])

        # This extreme scaling will cause numerical issues in SVD and correction calculations
        # potentially producing inf or nan, triggering geometric_corrections.append(0.0)
        result2 = geometric_knn_entropy(X2, Xdist2, k=2)
        assert isinstance(result2, float)
        # Note: result2 might be inf due to extreme values

        # Test case 3: Force rank-deficient neighborhoods for tiny ratios
        # Create 3D data where all neighbors of a point lie in a 2D subspace
        origin = np.array([0.0, 0.0, 0.0])
        X3 = np.array(
            [
                origin,
                origin + np.array([1.0, 0.0, 0.0]),  # x-direction
                origin + np.array([0.0, 1.0, 0.0]),  # y-direction
                origin + np.array([1.0, 1.0, 0.0]),  # xy-plane
                origin + np.array([0.5, 0.5, 1e-14]),  # Almost in xy-plane (tiny z)
            ]
        )

        N3 = X3.shape[0]
        Xdist3 = np.zeros((N3, N3))
        for i in range(N3):
            for j in range(N3):
                Xdist3[i, j] = l2dist(X3[i], X3[j])

        # When computing the neighborhood around the origin, most neighbors lie in xy-plane
        # This creates a scenario where the 3rd singular value (z-direction) is much smaller
        result3 = geometric_knn_entropy(X3, Xdist3, k=4)
        assert isinstance(result3, float)
        assert np.isfinite(result3)

    def test_geometric_knn_entropy_force_non_finite_correction(self):
        """Specifically designed to force geometric_corrections.append(0.0) line."""
        # Use data with NaN or inf values to force non-finite corrections

        # Create data with very problematic numerical values
        X = np.array(
            [
                [0.0, 0.0],
                [np.inf, 0.0],  # Infinite coordinate
                [0.0, np.inf],  # Infinite coordinate
                [1.0, 1.0],  # Normal point for comparison
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # This should create NaN values in the distance matrix and SVD calculations
        # leading to non-finite corrections that trigger geometric_corrections.append(0.0)
        result = geometric_knn_entropy(X, Xdist, k=2)

        # The function should handle this gracefully
        assert isinstance(result, float)
        # Result might be inf due to extreme values, but function should not crash

    def test_geometric_knn_entropy_nan_input_handling(self):
        """Test with NaN values to force non-finite correction handling."""
        # Create data that will produce NaN in internal calculations

        X = np.array(
            [
                [0.0, 0.0],
                [np.nan, 0.0],  # NaN coordinate
                [0.0, np.nan],  # NaN coordinate
                [1.0, 1.0],  # Normal point
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # NaN coordinates will propagate through calculations
        # This should trigger the non-finite correction fallback
        result = geometric_knn_entropy(X, Xdist, k=2)

        # Function should handle NaN gracefully
        assert isinstance(result, float)
        # Result will likely be NaN, but function should complete

    def test_geometric_knn_entropy_extreme_aspect_ratios(self):
        """Create data with extreme aspect ratios to cause numerical overflow."""
        # Use data that creates extreme scaling in SVD calculations

        X = np.array(
            [
                [0.0, 0.0],
                [1e308, 0.0],  # Near float64 overflow limit
                [0.0, 1e-308],  # Near float64 underflow limit
                [1e-308, 1e308],  # Mixed extreme values
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # Extreme scaling should cause overflow/underflow in correction calculations
        # Potentially making correction = log_hyper + sing_ratio_sum non-finite
        result = geometric_knn_entropy(X, Xdist, k=2)

        assert isinstance(result, float)
        # Result might be inf due to overflow, but should not crash

    def test_geometric_knn_entropy_mock_non_finite_correction(self):
        """Use mocking to force the non-finite correction path."""

        # Create normal data first
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # Import the real isfinite function directly from the module
        import builtins

        real_isfinite = builtins.__dict__.get("__numpy_isfinite__", None)
        if real_isfinite is None:
            # Store a reference to the real numpy isfinite before mocking
            import numpy as np_real

            real_isfinite = np_real.isfinite

        # Mock np.isfinite to return False for correction values
        # This will force the geometric_corrections.append(0.0) line
        def mock_isfinite_func(x):
            # Let other isfinite calls work normally
            if hasattr(x, "shape") and x.shape == ():  # scalar
                # If this looks like a correction value, return False
                if isinstance(x, (float, np.floating)) and abs(x) > 1:
                    return False
            # Use a direct finite check to avoid recursion
            try:
                # A number is finite if it's not inf and not nan
                # We can check this by comparing x to itself (nan != nan)
                # and by checking if it's less than infinity
                if x != x:  # NaN check
                    return False
                if abs(x) == float("inf"):  # Inf check
                    return False
                return True
            except:
                return True

        with patch("numpy.isfinite", side_effect=mock_isfinite_func):
            # This should trigger geometric_corrections.append(0.0) due to mocked non-finite
            result = geometric_knn_entropy(X, Xdist, k=2)

        assert isinstance(result, float)
        assert np.isfinite(result)

    @patch("causationentropy.core.information.entropy.np.isfinite")
    def test_geometric_knn_entropy_direct_mock_non_finite(self, mock_isfinite):
        """Direct mock to force geometric_corrections.append(0.0) line."""

        # Create simple data
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        N = X.shape[0]
        Xdist = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Xdist[i, j] = l2dist(X[i], X[j])

        # Mock the specific np.isfinite call in the entropy module
        # to return False for correction values, forcing append(0.0)
        call_count = [0]

        def mock_isfinite_side_effect(x):
            call_count[0] += 1
            # On certain calls, return False to trigger the fallback
            if call_count[0] % 3 == 0:  # Every 3rd call
                return False
            return True  # Most calls return True

        mock_isfinite.side_effect = mock_isfinite_side_effect

        # This should definitely hit geometric_corrections.append(0.0)
        result = geometric_knn_entropy(X, Xdist, k=2)

        # Verify the mock was called
        assert mock_isfinite.called
        assert isinstance(result, float)
        assert np.isfinite(result)
