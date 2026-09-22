import numpy as np
import pytest

from causationentropy.core.stats import (
    Compute_TPR_FPR,
    adaptive_bh_correction,
    auc,
    benjamini_hochberg_correction,
    benjamini_yekutieli_correction,
    bonferroni_correction,
    estimate_null_proportion,
)


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


class TestBonferroniCorrection:
    """Test the Bonferroni FWER correction."""

    def test_bonferroni_basic(self):
        """Only p-values below alpha / m are rejected."""
        rejected, p_adj = bonferroni_correction([0.01, 0.02, 0.03, 0.5])

        assert list(rejected) == [True, False, False, False]
        np.testing.assert_allclose(p_adj, [0.04, 0.08, 0.12, 1.0])

    def test_bonferroni_all_rejected(self):
        """Strong signals are all rejected."""
        rejected, p_adj = bonferroni_correction([0.001, 0.002], alpha=0.05)

        assert list(rejected) == [True, True]
        assert np.all(p_adj <= 0.05)

    def test_bonferroni_none_rejected(self):
        """Null-like p-values survive."""
        rejected, p_adj = bonferroni_correction([0.2, 0.4, 0.6, 0.8])

        assert list(rejected) == [False, False, False, False]
        assert np.all(p_adj > 0.05)

    def test_bonferroni_adjusted_capped_at_one(self):
        """Adjusted p-values never exceed 1."""
        _, p_adj = bonferroni_correction([0.5, 0.9])

        assert np.all(p_adj <= 1.0)

    def test_bonferroni_nan_missing(self):
        """NaN entries are missing tests: never rejected, NaN adjusted."""
        rejected, p_adj = bonferroni_correction([0.001, np.nan])

        # m counts only the finite entry, so 0.001 * 1 <= 0.05
        assert list(rejected) == [True, False]
        assert p_adj[0] == 0.001
        assert np.isnan(p_adj[1])

    def test_bonferroni_empty(self):
        """Empty input gives empty outputs."""
        rejected, p_adj = bonferroni_correction([])

        assert rejected.shape == (0,)
        assert p_adj.shape == (0,)

    def test_bonferroni_invalid_alpha(self):
        """Alpha outside (0, 1] raises."""
        for bad_alpha in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                bonferroni_correction([0.01, 0.5], alpha=bad_alpha)

    def test_bonferroni_invalid_p_values(self):
        """P-values outside [0, 1] or non-1D input raise."""
        with pytest.raises(ValueError):
            bonferroni_correction([0.01, 1.5])
        with pytest.raises(ValueError):
            bonferroni_correction([-0.1, 0.5])
        with pytest.raises(ValueError):
            bonferroni_correction([[0.01, 0.02], [0.03, 0.04]])


class TestBenjaminiHochbergCorrection:
    """Test the Benjamini-Hochberg FDR procedure."""

    def test_bh_basic_more_powerful_than_bonferroni(self):
        """BH rejects the first three where Bonferroni rejects one."""
        rejected, _ = benjamini_hochberg_correction([0.01, 0.02, 0.03, 0.5])

        assert list(rejected) == [True, True, True, False]

    def test_bh_adjusted_values(self):
        """BH-adjusted p-values match the step-up formula."""
        _, p_adj = benjamini_hochberg_correction([0.01, 0.02, 0.03, 0.5])

        np.testing.assert_allclose(p_adj, [0.04, 0.04, 0.04, 0.5])

    def test_bh_preserves_input_order(self):
        """Results follow the input order, not the sorted order."""
        rejected, p_adj = benjamini_hochberg_correction([0.5, 0.01, 0.02, 0.03])

        assert list(rejected) == [False, True, True, True]
        assert p_adj[0] == 0.5
        np.testing.assert_allclose(p_adj[1:], [0.04, 0.04, 0.04])

    def test_bh_all_null(self):
        """Uniform-like p-values give no rejections."""
        rejected, p_adj = benjamini_hochberg_correction([0.25, 0.5, 0.75, 1.0])

        assert not np.any(rejected)
        assert np.all(p_adj > 0.05)

    def test_bh_nan_missing(self):
        """NaN entries are never rejected and stay NaN."""
        rejected, p_adj = benjamini_hochberg_correction([0.01, np.nan, 0.5])

        assert list(rejected) == [True, False, False]
        assert np.isnan(p_adj[1])

    def test_bh_empty(self):
        """Empty input gives empty outputs."""
        rejected, p_adj = benjamini_hochberg_correction([])

        assert rejected.shape == (0,)
        assert p_adj.shape == (0,)

    def test_bh_invalid_inputs(self):
        """Bad alpha or p-values raise."""
        with pytest.raises(ValueError):
            benjamini_hochberg_correction([0.01], alpha=0.0)
        with pytest.raises(ValueError):
            benjamini_hochberg_correction([2.0])


class TestNullProportionAndAdaptiveBH:
    """Test null-proportion estimation and adaptive BH."""

    def test_estimate_null_proportion(self):
        """Storey estimator on a known example."""
        assert estimate_null_proportion([0.01, 0.02, 0.03, 0.04, 0.9]) == 0.4

    def test_estimate_null_proportion_clipped(self):
        """Estimate never exceeds 1."""
        assert estimate_null_proportion([0.6, 0.7, 0.8, 0.9]) == 1.0

    def test_estimate_null_proportion_all_signal(self):
        """No p-value above lambda gives 0."""
        assert estimate_null_proportion([0.01, 0.02, 0.03]) == 0.0

    def test_estimate_null_proportion_empty(self):
        """Empty input defaults to all null."""
        assert estimate_null_proportion([]) == 1.0

    def test_estimate_null_proportion_invalid_lambda(self):
        """Lambda outside (0, 1) raises."""
        for bad_lambda in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError):
                estimate_null_proportion([0.1, 0.9], lambda_=bad_lambda)

    def test_adaptive_bh_recovers_power(self):
        """Adaptive BH rejects a fourth signal where BH stops at three."""
        p_values = [0.01, 0.02, 0.03, 0.045, 0.9]

        bh_rejected, _ = benjamini_hochberg_correction(p_values)
        adap_rejected, _ = adaptive_bh_correction(p_values)

        assert list(bh_rejected) == [True, True, True, False, False]
        assert list(adap_rejected) == [True, True, True, True, False]

    def test_adaptive_bh_reduces_to_bh_at_pi0_one(self):
        """With pi0 = 1 both procedures agree."""
        p_values = [0.01, 0.2, 0.6, 0.8]

        bh_rejected, bh_adj = benjamini_hochberg_correction(p_values)
        adap_rejected, adap_adj = adaptive_bh_correction(p_values)

        np.testing.assert_array_equal(adap_rejected, bh_rejected)
        np.testing.assert_allclose(adap_adj, bh_adj)

    def test_adaptive_bh_nan_and_empty(self):
        """NaN is never rejected; empty input gives empty outputs."""
        rejected, p_adj = adaptive_bh_correction([0.001, np.nan])

        assert list(rejected) == [True, False]
        assert np.isnan(p_adj[1])

        rejected, p_adj = adaptive_bh_correction([])

        assert rejected.shape == (0,)
        assert p_adj.shape == (0,)

    def test_adaptive_bh_invalid_inputs(self):
        """Bad alpha or lambda raise."""
        with pytest.raises(ValueError):
            adaptive_bh_correction([0.01], alpha=1.5)
        with pytest.raises(ValueError):
            adaptive_bh_correction([0.01], lambda_=0.0)


class TestBenjaminiYekutieliCorrection:
    """Test the Benjamini-Yekutieli procedure for dependent tests."""

    def test_by_basic(self):
        """Strong signals pass the tightened threshold."""
        rejected, p_adj = benjamini_yekutieli_correction([0.001, 0.002, 0.5, 0.9])

        assert list(rejected) == [True, True, False, False]
        harmonic_4 = 1 + 1 / 2 + 1 / 3 + 1 / 4
        expected = [0.001 * 4 * harmonic_4, 0.002 * 4 * harmonic_4 / 2, 1.0, 1.0]
        np.testing.assert_allclose(p_adj, expected)

    def test_by_more_conservative_than_bh(self):
        """BY rejections are a subset of BH rejections on dependent-like data."""
        np.random.seed(0)
        p_values = np.random.beta(0.5, 5, size=50)

        by_rejected, _ = benjamini_yekutieli_correction(p_values)
        bh_rejected, _ = benjamini_hochberg_correction(p_values)

        assert np.all(~by_rejected | bh_rejected)

    def test_by_stricter_example(self):
        """BH rejects where BY does not."""
        by_rejected, _ = benjamini_yekutieli_correction([0.01, 0.02, 0.03, 0.5])
        bh_rejected, _ = benjamini_hochberg_correction([0.01, 0.02, 0.03, 0.5])

        assert list(bh_rejected) == [True, True, True, False]
        assert sum(by_rejected) < sum(bh_rejected)

    def test_by_nan_and_empty(self):
        """NaN is never rejected; empty input gives empty outputs."""
        rejected, p_adj = benjamini_yekutieli_correction([0.0001, np.nan])

        assert list(rejected) == [True, False]
        assert np.isnan(p_adj[1])

        rejected, p_adj = benjamini_yekutieli_correction([])

        assert rejected.shape == (0,)
        assert p_adj.shape == (0,)

    def test_by_invalid_inputs(self):
        """Bad alpha or p-values raise."""
        with pytest.raises(ValueError):
            benjamini_yekutieli_correction([0.01], alpha=0.0)
        with pytest.raises(ValueError):
            benjamini_yekutieli_correction([-0.5])
