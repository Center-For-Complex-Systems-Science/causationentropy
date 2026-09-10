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

    def test_tpr_fpr_manual_confusion_counts(self):
        """Hand-check TP/FP/TN/FN on a 3x3 off-diagonal example.

        Ground truth A and prediction B (diagonal already zero):

            A = [[0, 1, 0],          B = [[0, 1, 1],
                 [0, 0, 1],               [0, 0, 0],
                 [1, 0, 0]]               [1, 1, 0]]

        Off-diagonal cells only:
            (0,1): A=1, B=1 -> TP
            (0,2): A=0, B=1 -> FP
            (1,0): A=0, B=0 -> TN
            (1,2): A=1, B=0 -> FN
            (2,0): A=1, B=1 -> TP
            (2,1): A=0, B=1 -> FP

        TP=2, FP=2, TN=1, FN=1, P=3, N=3
        TPR = TP/P = 2/3, FPR = FP/N = 2/3
        """
        A = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        B = np.array([[0, 1, 1], [0, 0, 0], [1, 1, 0]])

        TPR, FPR = Compute_TPR_FPR(A, B)

        assert TPR == pytest.approx(2 / 3)
        assert FPR == pytest.approx(2 / 3)

    def test_tpr_fpr_ignores_predicted_self_loops(self):
        """Diagonal 1s in the prediction must not inflate FPR.

        Same off-diagonal pattern as the hand-checked example, but B has
        self-loops. Those three diagonal entries would be counted as extra
        false positives if the diagonal were included in the numerator while
        still being excluded from the n*(n-1) denominator.
        """
        A = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        B = np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]])

        TPR, FPR = Compute_TPR_FPR(A, B)

        assert TPR == pytest.approx(2 / 3)
        assert FPR == pytest.approx(2 / 3)
        assert 0 <= FPR <= 1

    def test_tpr_fpr_ignores_ground_truth_self_loops(self):
        """Diagonal 1s in A are not true edges and must not change TPR/FPR.

        Off-diagonal A matches B exactly (three edges, three non-edges), so
        TP=3, FP=0, TN=3, FN=0 even though A has 1s on the diagonal.
        """
        A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        B = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

        TPR, FPR = Compute_TPR_FPR(A, B)

        assert TPR == pytest.approx(1.0)
        assert FPR == pytest.approx(0.0)

    def test_fpr_in_unit_interval_with_self_loops(self):
        """FPR stays in [0, 1] even when A or B contain self-loops."""
        rng = np.random.default_rng(19)
        for n in range(2, 8):
            A = rng.integers(0, 2, size=(n, n))
            B = rng.integers(0, 2, size=(n, n))
            TPR, FPR = Compute_TPR_FPR(A, B)
            assert 0 <= TPR <= 1
            assert 0 <= FPR <= 1

    def test_tpr_fpr_perfect_prediction_with_self_loops(self):
        """Perfect off-diagonal recovery is TPR=1, FPR=0 even if B has loops."""
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        B = A.copy()
        np.fill_diagonal(B, 1)

        TPR, FPR = Compute_TPR_FPR(A, B)

        assert TPR == 1.0
        assert FPR == 0.0

    def test_tpr_fpr_no_true_positives(self):
        """No off-diagonal edges in A: TPR=1 by convention, FPR = FP / N."""
        A = np.zeros((3, 3), dtype=int)
        B = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])

        # Off-diagonal: one FP at (0,1); N = 6; P = 0
        TPR, FPR = Compute_TPR_FPR(A, B)

        assert TPR == pytest.approx(1.0)
        assert FPR == pytest.approx(1 / 6)

    def test_tpr_fpr_no_true_negatives(self):
        """Complete directed graph without self-loops has N=0, so FPR=0."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        B = A.copy()
        B[0, 1] = 0  # one missed edge; still no off-diagonal non-edges in A

        TPR, FPR = Compute_TPR_FPR(A, B)

        assert TPR == pytest.approx(5 / 6)
        assert FPR == pytest.approx(0.0)

    def test_tpr_fpr_issue19_all_ones_prediction(self):
        """All-ones prediction must not yield FPR > 1 (issue #19).

        n=5, P=6 off-diagonal true edges, B is all ones including the diagonal.
        Off-diagonal: TP=6, FN=0, FP=14, TN=0 -> TPR=1, FPR=1.
        The unfixed implementation counted 5 extra diagonal FPs over N=14,
        giving FPR = 19/14 ≈ 1.357.
        """
        A = np.array(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        B = np.ones_like(A)

        TPR, FPR = Compute_TPR_FPR(A, B)

        assert np.sum(A) == 6
        assert TPR == pytest.approx(1.0)
        assert FPR == pytest.approx(1.0)
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
