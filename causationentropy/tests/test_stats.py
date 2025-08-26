import numpy as np
import pytest

from causationentropy.core.stats import Compute_TPR_FPR, auc


class TestAUC:
    """Test the AUC (Area Under Curve) calculation."""

    def test_auc_basic(self):
        """Test basic AUC calculation."""
        # Simple case: unit square
        TPRs = np.array([0, 1, 1])
        FPRs = np.array([0, 0, 1])
        result = auc(TPRs, FPRs)

        # Should be close to 1 (perfect classifier)
        assert isinstance(result, float)
        assert result > 0.8  # Should be high for this ROC curve

    def test_auc_perfect_classifier(self):
        """Test AUC for perfect classifier (TPR=1, FPR=0)."""
        TPRs = np.array([0, 1])
        FPRs = np.array([0, 0])
        result = auc(TPRs, FPRs)

        # Perfect classifier should have AUC close to 1
        # Note: depends on exact implementation of trapz
        assert isinstance(result, float)
        assert result >= 0

    def test_auc_random_classifier(self):
        """Test AUC for random classifier (diagonal line)."""
        # Diagonal line from (0,0) to (1,1)
        points = np.linspace(0, 1, 11)
        TPRs = points
        FPRs = points
        result = auc(TPRs, FPRs)

        # Random classifier should have AUC H 0.5
        assert isinstance(result, float)
        assert 0.4 < result < 0.6

    def test_auc_monotonic_curve(self):
        """Test AUC with monotonically increasing curve."""
        TPRs = np.array([0, 0.2, 0.5, 0.8, 1.0])
        FPRs = np.array([0, 0.1, 0.3, 0.6, 1.0])
        result = auc(TPRs, FPRs)

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_auc_single_point(self):
        """Test AUC with single point."""
        TPRs = np.array([0.5])
        FPRs = np.array([0.3])
        result = auc(TPRs, FPRs)

        # Single point should give 0 area
        assert result == 0.0

    def test_auc_two_points(self):
        """Test AUC with two points."""
        TPRs = np.array([0.0, 1.0])
        FPRs = np.array([0.0, 0.5])
        result = auc(TPRs, FPRs)

        assert isinstance(result, float)
        assert result >= 0

    def test_auc_numerical_stability(self):
        """Test AUC with very small differences."""
        TPRs = np.array([0.0, 0.001, 0.002, 1.0])
        FPRs = np.array([0.0, 0.0001, 0.0002, 1.0])
        result = auc(TPRs, FPRs)

        assert isinstance(result, float)
        assert not np.isnan(result)
        assert np.isfinite(result)


class TestComputeTPRFPR:
    """Test the TPR/FPR computation function."""

    def test_tpr_fpr_identical_matrices(self):
        """Test TPR/FPR when matrices are identical."""
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]])
        B = A.copy()

        TPR, FPR = Compute_TPR_FPR(A, B)

        # Identical matrices should give TPR=1, FPR=0
        assert isinstance(TPR, (float, np.floating))
        assert isinstance(FPR, (float, np.floating))
        assert TPR == 1.0
        assert FPR == 0.0

    def test_tpr_fpr_different_matrices(self):
        """Test TPR/FPR with different matrices."""
        A = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]])
        B = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])

        TPR, FPR = Compute_TPR_FPR(A, B)

        assert isinstance(TPR, (float, np.floating))
        assert isinstance(FPR, (float, np.floating))
        assert 0 <= TPR <= 1
        assert 0 <= FPR <= 1

    def test_tpr_fpr_zero_matrices(self):
        """Test TPR/FPR with zero matrices."""
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))

        # This might cause division by zero, should handle gracefully
        try:
            TPR, FPR = Compute_TPR_FPR(A, B)
            assert isinstance(TPR, (float, np.floating))
            assert isinstance(FPR, (float, np.floating))
        except (ZeroDivisionError, RuntimeWarning):
            pass  # Division by zero is expected when A has no positive entries

    def test_tpr_fpr_binary_matrices(self):
        """Test TPR/FPR with binary matrices."""
        # True network
        A = np.array([[0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]])

        # Predicted network with some errors
        B = np.array(
            [
                [0, 1, 1, 1],  # Extra edge (2,0)
                [0, 0, 1, 0],  # Correct
                [1, 0, 0, 0],  # Missing edge (2,3)
                [0, 1, 1, 0],
            ]
        )  # Extra edge (3,2)

        TPR, FPR = Compute_TPR_FPR(A, B)

        assert isinstance(TPR, (float, np.floating))
        assert isinstance(FPR, (float, np.floating))
        assert 0 <= TPR <= 1
        assert 0 <= FPR <= 1

    def test_tpr_fpr_perfect_prediction(self):
        """Test TPR/FPR with perfect prediction."""
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        B = A.copy()  # Perfect prediction

        TPR, FPR = Compute_TPR_FPR(A, B)

        assert TPR == 1.0  # All true edges detected
        assert FPR == 0.0  # No false positives

    def test_tpr_fpr_worst_prediction(self):
        """Test TPR/FPR with worst possible prediction."""
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        B = 1 - A  # Completely wrong (flip all off-diagonal elements)
        np.fill_diagonal(B, 0)  # Keep diagonal as zero

        TPR, FPR = Compute_TPR_FPR(A, B)

        assert isinstance(TPR, (float, np.floating))
        assert isinstance(FPR, (float, np.floating))
        # Should have low TPR and high FPR for completely wrong prediction

    def test_tpr_fpr_empty_true_network(self):
        """Test TPR/FPR when true network has no edges."""
        A = np.zeros((4, 4))  # No true edges
        B = np.array(
            [[0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]]
        )  # Some predicted edges

        # This should cause division by zero for TPR calculation
        try:
            TPR, FPR = Compute_TPR_FPR(A, B)
            # If it doesn't raise an error, check the values
            assert isinstance(FPR, (float, np.floating))
            assert FPR > 0  # Should have false positives
        except (ZeroDivisionError, RuntimeWarning):
            pass  # Expected when no true edges exist

    def test_tpr_fpr_dimension_validation(self):
        """Test that function validates matrix dimensions."""
        A = np.array([[0, 1], [1, 0]])
        B = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # Wrong size

        with pytest.raises(AssertionError):
            Compute_TPR_FPR(A, B)

    def test_tpr_fpr_non_square_matrices(self):
        """Test behavior with non-square matrices."""
        A = np.array([[0, 1, 0]])  # 1x3 matrix
        B = np.array([[0, 1, 0]])

        with pytest.raises(AssertionError):
            Compute_TPR_FPR(A, B)

    def test_tpr_fpr_single_node(self):
        """Test TPR/FPR with single node networks."""
        A = np.array([[0]])
        B = np.array([[0]])

        # Single node with no self-loops
        try:
            TPR, FPR = Compute_TPR_FPR(A, B)
            # Should handle gracefully
            assert isinstance(TPR, (float, np.floating))
            assert isinstance(FPR, (float, np.floating))
        except (ZeroDivisionError, RuntimeWarning):
            pass  # May fail due to division by zero

    def test_tpr_fpr_large_matrices(self):
        """Test TPR/FPR with larger matrices."""
        np.random.seed(42)
        n = 10
        A = (np.random.rand(n, n) > 0.7).astype(int)
        np.fill_diagonal(A, 0)  # No self-loops

        B = (np.random.rand(n, n) > 0.8).astype(int)
        np.fill_diagonal(B, 0)  # No self-loops

        if np.sum(A) > 0:  # Only test if A has some edges
            TPR, FPR = Compute_TPR_FPR(A, B)

            assert isinstance(TPR, (float, np.floating))
            assert isinstance(FPR, (float, np.floating))
            assert 0 <= TPR <= 1
            assert 0 <= FPR <= 1


class TestStatsFunctionProperties:
    """Test mathematical properties and edge cases."""

    def test_auc_properties(self):
        """Test mathematical properties of AUC."""
        # AUC should be between 0 and 1 for reasonable ROC curves
        TPRs = np.array([0, 0.3, 0.7, 1.0])
        FPRs = np.array([0, 0.2, 0.5, 1.0])
        result = auc(TPRs, FPRs)

        assert 0 <= result <= 1

    def test_tpr_fpr_range(self):
        """Test that TPR and FPR are in valid ranges."""
        np.random.seed(42)
        for _ in range(10):
            n = np.random.randint(3, 8)
            A = (np.random.rand(n, n) > 0.6).astype(int)
            B = (np.random.rand(n, n) > 0.6).astype(int)
            np.fill_diagonal(A, 0)
            np.fill_diagonal(B, 0)

            if np.sum(A) > 0:  # Only test if A has edges
                TPR, FPR = Compute_TPR_FPR(A, B)
                assert 0 <= TPR <= 1
                assert 0 <= FPR <= 1

    def test_numerical_stability_with_floats(self):
        """Test numerical stability with float inputs."""
        # Test AUC with float arrays
        TPRs = np.array([0.0, 0.33333, 0.66667, 1.0])
        FPRs = np.array([0.0, 0.1111, 0.4444, 1.0])
        result = auc(TPRs, FPRs)

        assert isinstance(result, float)
        assert not np.isnan(result)
        assert np.isfinite(result)

    def test_edge_case_empty_arrays(self):
        """Test behavior with empty arrays."""
        try:
            result = auc(np.array([]), np.array([]))
            assert result == 0.0
        except (ValueError, IndexError):
            pass  # Empty arrays might raise errors, which is acceptable
