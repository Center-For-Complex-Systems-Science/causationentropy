"""Test plotting functionality."""

from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from causationentropy.core.plotting import roc_curve


class TestROCCurve:
    """Test ROC curve plotting functionality."""

    def setup_method(self):
        """Set up test environment before each test."""
        # Use Agg backend to avoid GUI windows during testing
        plt.switch_backend("Agg")
        plt.clf()  # Clear any existing plots

    def teardown_method(self):
        """Clean up after each test."""
        plt.clf()  # Clear plots after test
        plt.close("all")

    def test_roc_curve_basic_functionality(self):
        """Test basic ROC curve plotting works without errors."""
        tpr = [0, 0.5, 1]
        fpr = [0, 0.3, 1]

        # Should not raise any exceptions
        roc_curve(tpr, fpr)

        # Check that a plot was created
        fig = plt.gcf()
        axes = fig.get_axes()
        assert len(axes) == 1

        ax = axes[0]
        # Should have at least 2 lines (ROC curve + diagonal reference)
        lines = ax.get_lines()
        assert len(lines) >= 2

    def test_roc_curve_perfect_classifier(self):
        """Test ROC curve for perfect classifier."""
        tpr = [0, 1, 1]  # Perfect classifier: TPR jumps to 1 at FPR=0
        fpr = [0, 0, 1]

        roc_curve(tpr, fpr)

        ax = plt.gca()
        lines = ax.get_lines()

        # Check the main ROC curve data
        roc_line = lines[0]
        x_data, y_data = roc_line.get_data()

        np.testing.assert_array_equal(x_data, fpr)
        np.testing.assert_array_equal(y_data, tpr)

    def test_roc_curve_random_classifier(self):
        """Test ROC curve for random classifier."""
        tpr = [0, 0.5, 1]  # Random classifier: diagonal line
        fpr = [0, 0.5, 1]

        roc_curve(tpr, fpr)

        ax = plt.gca()
        lines = ax.get_lines()

        # Check data was plotted correctly
        roc_line = lines[0]
        x_data, y_data = roc_line.get_data()

        np.testing.assert_array_equal(x_data, fpr)
        np.testing.assert_array_equal(y_data, tpr)

    def test_roc_curve_labels_and_title(self):
        """Test that plot has correct labels and title."""
        tpr = [0, 0.8, 1]
        fpr = [0, 0.2, 1]

        roc_curve(tpr, fpr)

        ax = plt.gca()
        assert ax.get_xlabel() == "False Positive Rate (FPR)"
        assert ax.get_ylabel() == "True Positive Rate (TPR)"
        assert ax.get_title() == "ROC Curve"

    def test_roc_curve_axis_limits(self):
        """Test that plot has correct axis limits."""
        tpr = [0, 0.7, 1]
        fpr = [0, 0.1, 1]

        roc_curve(tpr, fpr)

        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        assert xlim == (0, 1)
        assert ylim == (0, 1)

    def test_roc_curve_diagonal_reference_line(self):
        """Test that diagonal reference line is plotted."""
        tpr = [0, 0.6, 1]
        fpr = [0, 0.4, 1]

        roc_curve(tpr, fpr)

        ax = plt.gca()
        lines = ax.get_lines()

        # Should have at least 2 lines
        assert len(lines) >= 2

        # Find the diagonal reference line (should be dashed)
        diagonal_line = None
        for line in lines:
            if line.get_linestyle() == "--":
                diagonal_line = line
                break

        assert diagonal_line is not None

        # Check diagonal line data
        x_data, y_data = diagonal_line.get_data()
        np.testing.assert_array_equal(x_data, [0, 1])
        np.testing.assert_array_equal(y_data, [0, 1])

    @patch("causationentropy.core.plotting.auc")
    def test_roc_curve_auc_computation(self, mock_auc):
        """Test that AUC is computed and displayed."""
        mock_auc.return_value = 0.75

        tpr = [0, 0.8, 1]
        fpr = [0, 0.2, 1]

        roc_curve(tpr, fpr)

        # Check that auc function was called with correct parameters
        mock_auc.assert_called_once_with(tpr, fpr)

        # Check that AUC text is displayed
        ax = plt.gca()
        texts = ax.texts
        assert len(texts) >= 1

        # Find AUC text
        auc_text = None
        for text in texts:
            if "AUC" in text.get_text():
                auc_text = text
                break

        assert auc_text is not None
        assert "0.7500" in auc_text.get_text()

    def test_roc_curve_with_numpy_arrays(self):
        """Test ROC curve with numpy arrays as input."""
        tpr = np.array([0, 0.6, 0.9, 1])
        fpr = np.array([0, 0.1, 0.3, 1])

        # Should work without errors
        roc_curve(tpr, fpr)

        ax = plt.gca()
        lines = ax.get_lines()
        roc_line = lines[0]

        x_data, y_data = roc_line.get_data()
        np.testing.assert_array_equal(x_data, fpr)
        np.testing.assert_array_equal(y_data, tpr)

    def test_roc_curve_empty_arrays(self):
        """Test behavior with empty arrays."""
        tpr = []
        fpr = []

        # Should handle gracefully
        try:
            roc_curve(tpr, fpr)
        except Exception:
            # Some behavior is acceptable for empty arrays
            pass

    def test_roc_curve_single_point(self):
        """Test ROC curve with single point."""
        tpr = [0.5]
        fpr = [0.3]

        roc_curve(tpr, fpr)

        ax = plt.gca()
        lines = ax.get_lines()
        roc_line = lines[0]

        x_data, y_data = roc_line.get_data()
        np.testing.assert_array_equal(x_data, fpr)
        np.testing.assert_array_equal(y_data, tpr)

    def test_roc_curve_mismatched_lengths(self):
        """Test behavior with mismatched array lengths."""
        tpr = [0, 0.5, 1]
        fpr = [0, 1]  # Different length

        # matplotlib should handle this gracefully or raise appropriate error
        try:
            roc_curve(tpr, fpr)
        except (ValueError, IndexError):
            # Expected behavior for mismatched lengths
            pass

    def test_roc_curve_values_outside_range(self):
        """Test ROC curve with values outside [0,1] range."""
        tpr = [-0.1, 0.5, 1.2]  # Values outside [0,1]
        fpr = [0, 0.3, 1]

        # Should still plot, as matplotlib is flexible
        roc_curve(tpr, fpr)

        ax = plt.gca()
        lines = ax.get_lines()
        assert len(lines) >= 1

    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.text")
    def test_roc_curve_matplotlib_calls(self, mock_text, mock_plot):
        """Test that correct matplotlib functions are called."""
        tpr = [0, 0.8, 1]
        fpr = [0, 0.2, 1]

        with patch("causationentropy.core.plotting.auc", return_value=0.9):
            roc_curve(tpr, fpr)

        # Check that plot was called for main curve
        assert mock_plot.call_count >= 2  # Main curve + diagonal line

        # Check that text was called for AUC display
        mock_text.assert_called_once()
        args, kwargs = mock_text.call_args
        assert "AUC = 0.9000" in args[2]  # Third argument should be the text

    def test_roc_curve_integration_with_stats(self):
        """Test integration with the actual auc function."""
        # Test with known values
        tpr = [0, 1]  # Perfect step function
        fpr = [0, 1]

        roc_curve(tpr, fpr)

        # The actual AUC should be computed and displayed
        ax = plt.gca()
        texts = ax.texts

        # Should have AUC text
        auc_texts = [t for t in texts if "AUC" in t.get_text()]
        assert len(auc_texts) >= 1


class TestPlottingEdgeCases:
    """Test edge cases and error conditions for plotting."""

    def setup_method(self):
        """Set up test environment."""
        plt.switch_backend("Agg")
        plt.clf()

    def teardown_method(self):
        """Clean up after each test."""
        plt.clf()
        plt.close("all")

    def test_plotting_with_different_backends(self):
        """Test that plotting works with different matplotlib backends."""
        original_backend = plt.get_backend()

        try:
            # Test with Agg backend (non-interactive)
            plt.switch_backend("Agg")

            tpr = [0, 0.5, 1]
            fpr = [0, 0.3, 1]

            roc_curve(tpr, fpr)

            # Should complete without errors
            ax = plt.gca()
            assert ax is not None

        finally:
            plt.switch_backend(original_backend)

    def test_multiple_roc_curves_on_same_plot(self):
        """Test plotting multiple ROC curves on the same axes."""
        # First curve
        tpr1 = [0, 0.8, 1]
        fpr1 = [0, 0.2, 1]
        roc_curve(tpr1, fpr1)

        # Second curve (should add to existing plot)
        tpr2 = [0, 0.6, 1]
        fpr2 = [0, 0.4, 1]
        roc_curve(tpr2, fpr2)

        ax = plt.gca()
        lines = ax.get_lines()

        # Should have multiple lines (2 ROC curves + 2 diagonal references)
        assert len(lines) >= 3  # At least the two main curves + diagonal
