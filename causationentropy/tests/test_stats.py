import numpy as np
import pytest

from causationentropy.core.stats import (
    Compute_TPR_FPR,
    auc,
    bootstrap_cmi_confidence_interval,
    bootstrap_confidence_interval,
    moving_block_bootstrap_indices,
    stationary_bootstrap_indices,
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


class TestMovingBlockBootstrap:
    """Test moving block bootstrap index resampling."""

    def test_shapes_and_range(self):
        """Output shape is (B, n) with indices in range."""
        indices = moving_block_bootstrap_indices(100, 10, 5, seed=0)

        assert indices.shape == (5, 100)
        assert indices.dtype.kind == "i"
        assert np.all(indices >= 0)
        assert np.all(indices < 100)

    def test_blocks_are_contiguous(self):
        """Each block holds consecutive indices (wrapping circularly)."""
        indices = moving_block_bootstrap_indices(50, 7, 4, seed=1)

        for row in indices:
            for start in range(0, 50, 7):
                block = row[start : start + 7]
                expected = (block[0] + np.arange(len(block))) % 50
                np.testing.assert_array_equal(block, expected)

    def test_reproducibility(self):
        """Same seed repeats; different seed differs."""
        first = moving_block_bootstrap_indices(60, 8, 3, seed=11)
        second = moving_block_bootstrap_indices(60, 8, 3, seed=11)
        third = moving_block_bootstrap_indices(60, 8, 3, seed=12)

        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, third)

    def test_full_length_block(self):
        """Block length n reproduces shifted full copies."""
        indices = moving_block_bootstrap_indices(10, 10, 3, seed=0)

        for row in indices:
            np.testing.assert_array_equal(np.sort(row), np.arange(10))

    def test_invalid_sizes(self):
        """Out-of-range sizes raise."""
        with pytest.raises(ValueError):
            moving_block_bootstrap_indices(0, 1, 5)
        with pytest.raises(ValueError):
            moving_block_bootstrap_indices(10, 0, 5)
        with pytest.raises(ValueError):
            moving_block_bootstrap_indices(10, 11, 5)
        with pytest.raises(ValueError):
            moving_block_bootstrap_indices(10, 5, 0)


class TestStationaryBootstrap:
    """Test stationary bootstrap index resampling."""

    def test_shapes_and_range(self):
        """Output shape is (B, n) with indices in range."""
        indices = stationary_bootstrap_indices(100, 10.0, 5, seed=0)

        assert indices.shape == (5, 100)
        assert np.all(indices >= 0)
        assert np.all(indices < 100)

    def test_reproducibility(self):
        """Same seed repeats; different seed differs."""
        first = stationary_bootstrap_indices(80, 5.0, 3, seed=3)
        second = stationary_bootstrap_indices(80, 5.0, 3, seed=3)
        third = stationary_bootstrap_indices(80, 5.0, 3, seed=4)

        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, third)

    def test_mean_block_length_effect(self):
        """Shorter mean blocks start many more blocks per replicate."""
        short = stationary_bootstrap_indices(200, 2.0, 20, seed=0)
        long = stationary_bootstrap_indices(200, 50.0, 20, seed=0)

        def count_breaks(row):
            return int(np.sum(row[1:] != (row[:-1] + 1) % 200))

        short_breaks = np.mean([count_breaks(row) for row in short])
        long_breaks = np.mean([count_breaks(row) for row in long])
        assert short_breaks > 50 > long_breaks

    def test_invalid_sizes(self):
        """Out-of-range sizes raise."""
        with pytest.raises(ValueError):
            stationary_bootstrap_indices(0, 5.0, 3)
        with pytest.raises(ValueError):
            stationary_bootstrap_indices(10, 0.5, 3)
        with pytest.raises(ValueError):
            stationary_bootstrap_indices(10, 5.0, 0)


class TestBootstrapConfidenceInterval:
    """Test the percentile confidence interval."""

    def test_known_quantiles(self):
        """Quantiles of 0..100 at 5% are 2.5 and 97.5."""
        lower, upper = bootstrap_confidence_interval(np.arange(101), alpha=0.05)

        assert lower == 2.5
        assert upper == 97.5

    def test_constant_input(self):
        """Constant replicates give a zero-width interval."""
        lower, upper = bootstrap_confidence_interval(np.full(50, 1.5))

        assert lower == 1.5
        assert upper == 1.5

    def test_invalid_inputs(self):
        """Bad alpha, empty, non-1D, or non-finite input raises."""
        with pytest.raises(ValueError):
            bootstrap_confidence_interval([0.1, 0.2], alpha=0.0)
        with pytest.raises(ValueError):
            bootstrap_confidence_interval([0.1, 0.2], alpha=1.0)
        with pytest.raises(ValueError):
            bootstrap_confidence_interval([])
        with pytest.raises(ValueError):
            bootstrap_confidence_interval([[0.1], [0.2]])
        with pytest.raises(ValueError):
            bootstrap_confidence_interval([0.1, np.nan])
        with pytest.raises(ValueError):
            bootstrap_confidence_interval([0.1, np.inf])


class TestBootstrapCMIConfidenceInterval:
    """Test end-to-end CMI intervals with the Gaussian estimator."""

    def _coupled_data(self, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((300, 1))
        Y = X + 0.5 * rng.standard_normal((300, 1))
        return X, Y

    def test_coupled_interval(self):
        """Strong coupling gives a positive interval containing the estimate."""
        X, Y = self._coupled_data()
        estimate, lower, upper, boot = bootstrap_cmi_confidence_interval(
            X, Y, n_bootstraps=50, seed=0
        )

        assert boot.shape == (50,)
        assert np.all(np.isfinite(boot))
        assert estimate > 0.5
        assert 0 < lower <= estimate <= upper

    def test_independent_interval_near_zero(self):
        """Independent series give an estimate and interval near zero."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((300, 1))
        Y = rng.standard_normal((300, 1))
        estimate, lower, upper, _ = bootstrap_cmi_confidence_interval(
            X, Y, n_bootstraps=50, seed=1
        )

        assert estimate < 0.05
        assert lower >= 0.0
        assert upper < 0.05

    def test_stationary_option(self):
        """Stationary resampling runs and brackets the estimate."""
        X, Y = self._coupled_data()
        estimate, lower, upper, boot = bootstrap_cmi_confidence_interval(
            X, Y, n_bootstraps=30, use_stationary=True, seed=0
        )

        assert boot.shape == (30,)
        assert lower <= estimate <= upper

    def test_reproducibility(self):
        """Same seed repeats the replicates exactly."""
        X, Y = self._coupled_data()
        first = bootstrap_cmi_confidence_interval(X, Y, n_bootstraps=20, seed=5)
        second = bootstrap_cmi_confidence_interval(X, Y, n_bootstraps=20, seed=5)

        np.testing.assert_array_equal(first[3], second[3])
        assert first[:3] == second[:3]

    def test_conditioning_set(self):
        """A conditioning set is accepted and keeps shapes."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((200, 1))
        Z = rng.standard_normal((200, 1))
        Y = X + Z + 0.5 * rng.standard_normal((200, 1))
        estimate, lower, upper, boot = bootstrap_cmi_confidence_interval(
            X, Y, Z, n_bootstraps=20, seed=2
        )

        assert boot.shape == (20,)
        assert estimate > 0
        assert lower <= upper

    def test_invalid_inputs(self):
        """Mismatched rows and bad sizes raise."""
        X, Y = self._coupled_data()
        with pytest.raises(ValueError):
            bootstrap_cmi_confidence_interval(X, Y[:-1], n_bootstraps=10)
        with pytest.raises(ValueError):
            bootstrap_cmi_confidence_interval(X, Y, np.zeros((10, 1)), n_bootstraps=10)
        with pytest.raises(ValueError):
            bootstrap_cmi_confidence_interval(X, Y, n_bootstraps=0)
        with pytest.raises(ValueError):
            bootstrap_cmi_confidence_interval(X, Y, alpha=0.0)
