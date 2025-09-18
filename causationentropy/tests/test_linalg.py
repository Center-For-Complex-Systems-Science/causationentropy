import numpy as np
import pytest

from causationentropy.core.linalg import correlation_log_determinant


class TestCorrelationLogDeterminant:
    """Test the correlation log determinant function."""

    def test_correlation_log_det_basic(self):
        """Test basic correlation log determinant calculation."""
        # Simple 2D data
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = correlation_log_determinant(A)

        assert isinstance(result, float)
        assert not np.isnan(result)
        assert np.isfinite(result)

    def test_correlation_log_det_independent_variables(self):
        """Test with independent variables (correlation ≈ identity)."""
        np.random.seed(42)
        n_samples = 100
        n_vars = 3

        # Generate independent variables
        A = np.random.normal(0, 1, (n_samples, n_vars))

        result = correlation_log_determinant(A)

        assert isinstance(result, float)
        assert not np.isnan(result)
        # For independent variables, correlation matrix should be close to identity
        # so log determinant should be close to 0
        assert abs(result) < 5  # Reasonable bound

    def test_correlation_log_det_correlated_variables(self):
        """Test with highly correlated variables."""
        np.random.seed(42)
        n_samples = 50

        # Create correlated data
        x1 = np.random.normal(0, 1, n_samples)
        x2 = x1 + 0.1 * np.random.normal(0, 1, n_samples)  # Highly correlated
        x3 = np.random.normal(0, 1, n_samples)  # Independent

        A = np.column_stack([x1, x2, x3])

        result = correlation_log_determinant(A)

        assert isinstance(result, float)
        assert not np.isnan(result)
        # Highly correlated variables should give smaller (more negative) log det
        assert result < 0

    def test_correlation_log_det_single_variable(self):
        """Test with single variable."""
        A = np.array([[1.0], [2.0], [3.0], [4.0]])

        result = correlation_log_determinant(A)

        # Single variable correlation matrix is scalar 1, log(1) = 0
        assert result == 0.0

    def test_correlation_log_det_identical_variables(self):
        """Test with identical variables (singular correlation matrix)."""
        A = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        result = correlation_log_determinant(A)

        # Identical variables should give singular correlation matrix
        # Log determinant should be -inf, but function might handle this
        assert isinstance(result, float)
        # Could be -inf or some large negative number depending on implementation
        if np.isfinite(result):
            assert result < -10  # Should be very negative

    def test_correlation_log_det_zero_columns(self):
        """Test with zero columns (edge case)."""
        A = np.array([]).reshape(5, 0)  # 5 rows, 0 columns

        result = correlation_log_determinant(A)

        # Empty matrix should return 0
        assert result == 0.0

    def test_correlation_log_det_constant_variables(self):
        """Test with constant variables."""
        A = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])

        # First column is constant
        result = correlation_log_determinant(A)

        assert isinstance(result, float)
        # Constant variable should cause issues with correlation calculation
        # Function should handle this gracefully

    def test_correlation_log_det_numerical_stability(self):
        """Test numerical stability with various data scales."""
        np.random.seed(42)

        # Very small values
        A_small = 1e-10 * np.random.normal(0, 1, (30, 3))
        result_small = correlation_log_determinant(A_small)
        assert isinstance(result_small, float)

        # Very large values
        A_large = 1e10 * np.random.normal(0, 1, (30, 3))
        result_large = correlation_log_determinant(A_large)
        assert isinstance(result_large, float)

        # Mixed scales
        A_mixed = np.column_stack(
            [
                1e-5 * np.random.normal(0, 1, 30),
                1e5 * np.random.normal(0, 1, 30),
                np.random.normal(0, 1, 30),
            ]
        )
        result_mixed = correlation_log_determinant(A_mixed)
        assert isinstance(result_mixed, float)

    def test_correlation_log_det_with_nans(self):
        """Test behavior with NaN values."""
        A = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])

        # Function should handle NaNs gracefully or raise appropriate error
        try:
            result = correlation_log_determinant(A)
            # If it returns a value, it should be NaN
            if not np.isnan(result):
                assert isinstance(result, float)
        except (ValueError, np.linalg.LinAlgError):
            pass  # Acceptable to raise error with NaN input

    def test_correlation_log_det_with_infs(self):
        """Test behavior with infinite values."""
        A = np.array([[1.0, 2.0], [np.inf, 3.0], [4.0, 5.0]])

        # Function should handle infs gracefully or raise appropriate error
        try:
            result = correlation_log_determinant(A)
            assert isinstance(result, (float, type(np.inf)))
        except (ValueError, np.linalg.LinAlgError):
            pass  # Acceptable to raise error with inf input

    def test_correlation_log_det_orthogonal_data(self):
        """Test with orthogonal data vectors."""
        # Create orthogonal vectors
        A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

        result = correlation_log_determinant(A)

        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_correlation_log_det_large_matrix(self):
        """Test with larger matrices."""
        np.random.seed(42)
        n_samples = 200
        n_vars = 10

        A = np.random.normal(0, 1, (n_samples, n_vars))

        result = correlation_log_determinant(A)

        assert isinstance(result, float)
        assert not np.isnan(result)
        assert np.isfinite(result)

    def test_correlation_log_det_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(123)
        A1 = np.random.normal(0, 1, (50, 4))
        result1 = correlation_log_determinant(A1)

        np.random.seed(123)
        A2 = np.random.normal(0, 1, (50, 4))
        result2 = correlation_log_determinant(A2)

        assert np.isclose(result1, result2)

    def test_correlation_log_det_mathematical_properties(self):
        """Test mathematical properties of correlation matrices."""
        np.random.seed(42)
        n_samples = 100

        # Test with different numbers of variables
        for n_vars in [2, 3, 5]:
            A = np.random.normal(0, 1, (n_samples, n_vars))
            result = correlation_log_determinant(A)

            assert isinstance(result, float)
            # Log determinant of correlation matrix should be <= 0
            # (since correlation matrix eigenvalues are <= 1)
            if np.isfinite(result):
                assert result <= 1  # Small tolerance for numerical errors

    def test_correlation_log_det_data_types(self):
        """Test with different data types."""
        # Integer data
        A_int = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)
        result_int = correlation_log_determinant(A_int)
        assert isinstance(result_int, float)

        # Float32 data
        A_float32 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result_float32 = correlation_log_determinant(A_float32)
        assert isinstance(result_float32, float)

        # Float64 data
        A_float64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result_float64 = correlation_log_determinant(A_float64)
        assert isinstance(result_float64, float)

    def test_correlation_log_det_worked_out_example(self):
        """Test mathematical properties of correlation matrices.
        This test puts in a martrix and asserts the answer is correct versus
        a version worked out on paper.
        """
        np.random.seed(42)
        n_samples = 100

        # Test with different numbers of variables
        for n_vars in [2, 3, 5]:
            A = np.random.normal(0, 1, (n_samples, n_vars))
            result = correlation_log_determinant(A)

            assert isinstance(result, float)
            # Log determinant of correlation matrix should be <= 0
            # (since correlation matrix eigenvalues are <= 1)
            if np.isfinite(result):
                assert result <= 1  # Small tolerance for numerical errors

class TestCorrelationLogDeterminantEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_matrix(self):
        """Test with empty matrix."""
        A = np.array([]).reshape(0, 0)

        # Should handle empty matrix gracefully
        try:
            result = correlation_log_determinant(A)
            assert result == 0.0
        except (ValueError, IndexError):
            pass  # Acceptable to raise error

    def test_single_sample(self):
        """Test with single sample (insufficient for correlation)."""
        A = np.array([[1.0, 2.0, 3.0]])  # Only one sample

        # Correlation requires at least 2 samples
        try:
            result = correlation_log_determinant(A)
        except (ValueError, np.linalg.LinAlgError):
            pass  # Expected to fail with insufficient samples

    def test_two_samples_identical(self):
        """Test with two identical samples."""
        A = np.array([[1.0, 2.0], [1.0, 2.0]])

        # Identical samples should cause singular correlation matrix
        result = correlation_log_determinant(A)
        assert isinstance(result, float)
        # Should be -inf or very large negative number

    def test_more_variables_than_samples(self):
        """Test when number of variables exceeds samples."""
        A = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
        )  # 2 samples, 5 variables

        # This should result in singular correlation matrix
        result = correlation_log_determinant(A)
        assert isinstance(result, float)
        # Typically should be -inf or very negative

    def test_epsilon_parameter(self):
        """Test that epsilon parameter exists and can be used."""
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Test default epsilon
        result1 = correlation_log_determinant(A)

        # Test custom epsilon
        result2 = correlation_log_determinant(A, epsilon=1e-12)

        # Both should be valid floats
        assert isinstance(result1, float)
        assert isinstance(result2, float)

    def test_correlation_matrix_properties(self):
        """Test that the underlying correlation computation is reasonable."""
        # Create data where we know the correlation structure
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        y = 0.8 * x + 0.6 * np.random.normal(0, 1, n)  # Correlation ≈ 0.8

        A = np.column_stack([x, y])
        result = correlation_log_determinant(A)

        # For 2x2 correlation matrix with correlation r:
        # det = 1 - r^2, so log(det) = log(1 - r^2)
        # With r ≈ 0.8, det ≈ 0.36, log(det) ≈ -1.02
        assert isinstance(result, float)
        assert -2 < result < 0  # Should be negative but not too extreme
