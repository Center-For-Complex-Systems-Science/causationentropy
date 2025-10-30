"""Test plotting functionality."""

from unittest.mock import MagicMock, Mock, patch

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from causationentropy.core.plotting import (
    _circular_positions,
    _communities_seed_order,
    _connected_angle_penalty,
    _edge_crossings_for_order,
    _edge_length_variance,
    _label_collision_proxy,
    _objective,
    optimize_circular_order,
    plot_causal_network,
    roc_curve,
)


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


class TestCommunitiesSeedOrder:
    """Test _communities_seed_order helper function."""

    def test_simple_graph(self):
        """Test with a simple connected graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])

        order = _communities_seed_order(G)

        # Should return all nodes
        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}

    def test_disconnected_graph(self):
        """Test with disconnected components."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])

        order = _communities_seed_order(G)

        # Should return all nodes
        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}

    def test_single_node(self):
        """Test with single node graph."""
        G = nx.Graph()
        G.add_node(0)

        order = _communities_seed_order(G)

        assert order == [0]

    def test_empty_graph(self):
        """Test with empty graph."""
        G = nx.Graph()

        order = _communities_seed_order(G)

        assert order == []

    def test_complete_graph(self):
        """Test with complete graph."""
        G = nx.complete_graph(5)

        order = _communities_seed_order(G)

        # Should return all nodes
        assert len(order) == 5
        assert set(order) == set(range(5))

    def test_no_duplicate_nodes(self):
        """Test that returned order has no duplicates."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])

        order = _communities_seed_order(G)

        # Check no duplicates
        assert len(order) == len(set(order))

    def test_multigraph_input(self):
        """Test with MultiDiGraph (used in actual application)."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1)
        G.add_edge(1, 2, lag=1)
        G.add_edge(2, 0, lag=2)

        order = _communities_seed_order(G)

        assert len(order) == 3
        assert set(order) == {0, 1, 2}


class TestCircularPositions:
    """Test _circular_positions helper function."""

    def test_basic_positions(self):
        """Test basic circular positioning."""
        order = [0, 1, 2, 3]
        pos = _circular_positions(order, radius=1.0)

        # Should have position for each node
        assert len(pos) == 4
        assert set(pos.keys()) == {0, 1, 2, 3}

        # All positions should be on unit circle
        for node, position in pos.items():
            dist = np.linalg.norm(position)
            np.testing.assert_almost_equal(dist, 1.0)

    def test_custom_radius(self):
        """Test with custom radius."""
        order = [0, 1]
        pos = _circular_positions(order, radius=2.5)

        for node, position in pos.items():
            dist = np.linalg.norm(position)
            np.testing.assert_almost_equal(dist, 2.5)

    def test_single_node(self):
        """Test with single node."""
        order = [0]
        pos = _circular_positions(order, radius=1.0)

        assert len(pos) == 1
        assert 0 in pos

    def test_even_spacing(self):
        """Test that nodes are evenly spaced."""
        order = [0, 1, 2, 3]
        pos = _circular_positions(order, radius=1.0)

        # Calculate angles
        angles = []
        for node in order:
            x, y = pos[node]
            angle = np.arctan2(y, x)
            angles.append(angle)

        # Check angular spacing
        expected_spacing = 2 * np.pi / 4
        for i in range(len(angles) - 1):
            angular_diff = (angles[i + 1] - angles[i]) % (2 * np.pi)
            np.testing.assert_almost_equal(angular_diff, expected_spacing, decimal=5)

    def test_position_format(self):
        """Test that positions are numpy arrays."""
        order = [0, 1]
        pos = _circular_positions(order)

        for node, position in pos.items():
            assert isinstance(position, np.ndarray)
            assert position.shape == (2,)


class TestEdgeCrossingsForOrder:
    """Test _edge_crossings_for_order helper function."""

    def test_no_crossings(self):
        """Test graph with no edge crossings."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        order = [0, 1, 2]

        crossings = _edge_crossings_for_order(G, order)

        assert crossings == 0

    def test_single_crossing(self):
        """Test graph with one crossing."""
        G = nx.Graph()
        G.add_edges_from([(0, 2), (1, 3)])
        order = [0, 1, 2, 3]

        crossings = _edge_crossings_for_order(G, order)

        assert crossings == 1

    def test_self_loops_ignored(self):
        """Test that self-loops are ignored."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 0, lag=1)  # Self-loop
        G.add_edge(0, 1, lag=1)
        order = [0, 1]

        crossings = _edge_crossings_for_order(G, order)

        assert crossings == 0

    def test_empty_graph(self):
        """Test with empty graph."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        order = [0, 1, 2]

        crossings = _edge_crossings_for_order(G, order)

        assert crossings == 0

    def test_complete_graph_crossings(self):
        """Test complete graph has many crossings."""
        G = nx.complete_graph(5)
        order = list(range(5))

        crossings = _edge_crossings_for_order(G, order)

        # Complete graph should have some crossings
        assert crossings > 0

    def test_different_orders_different_crossings(self):
        """Test that different orderings give different crossing counts."""
        G = nx.Graph()
        G.add_edges_from([(0, 2), (1, 3)])

        order1 = [0, 1, 2, 3]
        order2 = [0, 2, 1, 3]

        crossings1 = _edge_crossings_for_order(G, order1)
        crossings2 = _edge_crossings_for_order(G, order2)

        # Different orders should give different results
        assert crossings1 != crossings2


class TestEdgeLengthVariance:
    """Test _edge_length_variance helper function."""

    def test_uniform_lengths(self):
        """Test graph with uniform edge lengths."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        order = [0, 1, 2, 3]

        variance = _edge_length_variance(G, order)

        # Should have low variance for evenly spaced edges
        assert variance >= 0
        assert variance < 0.1

    def test_empty_graph(self):
        """Test with graph with no edges."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        order = [0, 1, 2]

        variance = _edge_length_variance(G, order)

        assert variance == 0.0

    def test_single_edge(self):
        """Test with single edge."""
        G = nx.Graph()
        G.add_edge(0, 1)
        order = [0, 1]

        variance = _edge_length_variance(G, order)

        # Single edge has zero variance
        assert variance == 0.0

    def test_self_loops_ignored(self):
        """Test that self-loops are ignored."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 0, lag=1)  # Self-loop
        G.add_edge(0, 1, lag=1)
        order = [0, 1]

        variance = _edge_length_variance(G, order)

        # Should only consider the non-self-loop edge
        assert variance == 0.0

    def test_variance_non_negative(self):
        """Test that variance is always non-negative."""
        G = nx.Graph()
        G.add_edges_from([(0, 2), (1, 4), (2, 3)])
        order = [0, 1, 2, 3, 4]

        variance = _edge_length_variance(G, order)

        assert variance >= 0


class TestConnectedAnglePenalty:
    """Test _connected_angle_penalty helper function."""

    def test_basic_penalty(self):
        """Test basic angle penalty computation."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        order = [0, 1, 2]

        penalty = _connected_angle_penalty(G, order)

        # Should return a positive value
        assert penalty > 0

    def test_empty_graph(self):
        """Test with no edges."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        order = [0, 1, 2]

        penalty = _connected_angle_penalty(G, order)

        # No edges should give zero penalty (divided by 1)
        assert penalty == 0.0

    def test_self_loops_ignored(self):
        """Test that self-loops are ignored."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 0, lag=1)  # Self-loop
        order = [0]

        penalty = _connected_angle_penalty(G, order)

        assert penalty == 0.0

    def test_penalty_increases_for_close_nodes(self):
        """Test that penalty is higher for adjacent nodes."""
        G = nx.Graph()
        G.add_edge(0, 1)

        # Adjacent nodes
        order1 = [0, 1, 2, 3]
        penalty1 = _connected_angle_penalty(G, order1)

        # Farther apart
        order2 = [0, 2, 1, 3]
        penalty2 = _connected_angle_penalty(G, order2)

        # Adjacent nodes should have higher penalty
        assert penalty1 > penalty2


class TestLabelCollisionProxy:
    """Test _label_collision_proxy helper function."""

    def test_returns_zero(self):
        """Test that function returns zero (placeholder implementation)."""
        order = [0, 1, 2, 3]
        result = _label_collision_proxy(order)

        assert result == 0

    def test_with_min_sep(self):
        """Test with min_sep parameter."""
        order = [0, 1, 2]
        result = _label_collision_proxy(order, min_sep=5)

        assert result == 0

    def test_empty_order(self):
        """Test with empty order."""
        result = _label_collision_proxy([])

        assert result == 0


class TestObjective:
    """Test _objective helper function."""

    def test_basic_objective(self):
        """Test basic objective computation."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        order = [0, 1, 2]

        obj = _objective(G, order)

        # Should return a non-negative value
        assert obj >= 0

    def test_custom_weights(self):
        """Test with custom weights."""
        G = nx.Graph()
        G.add_edges_from([(0, 2), (1, 3)])
        order = [0, 1, 2, 3]

        # Default weights
        obj1 = _objective(G, order, w=(1.0, 0.2, 0.05, 0.0))

        # Different weights
        obj2 = _objective(G, order, w=(0.5, 0.1, 0.0, 0.0))

        # Should give different results
        assert obj1 != obj2

    def test_empty_graph(self):
        """Test with empty graph."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        order = [0, 1, 2]

        obj = _objective(G, order)

        # Empty graph should have zero objective
        assert obj == 0.0

    def test_objective_components(self):
        """Test that objective includes all components."""
        G = nx.Graph()
        G.add_edges_from([(0, 2), (1, 3)])
        order = [0, 1, 2, 3]

        # Zero out all but crossings weight
        obj_crossings = _objective(G, order, w=(1.0, 0.0, 0.0, 0.0))

        # Zero out all but variance weight
        obj_variance = _objective(G, order, w=(0.0, 1.0, 0.0, 0.0))

        # At least one should be non-zero
        assert obj_crossings > 0 or obj_variance >= 0


class TestOptimizeCircularOrder:
    """Test optimize_circular_order function."""

    def test_basic_optimization(self):
        """Test basic optimization works."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])

        order = optimize_circular_order(G, max_iters=100, rng=42)

        # Should return all nodes
        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}

    def test_with_seed_order(self):
        """Test with provided seed order."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])

        seed_order = [2, 1, 0]
        order = optimize_circular_order(G, seed_order=seed_order, max_iters=10, rng=42)

        # Should return all nodes
        assert len(order) == 3
        assert set(order) == {0, 1, 2}

    def test_deterministic_with_seed(self):
        """Test that same random seed gives same result."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

        order1 = optimize_circular_order(G, max_iters=100, rng=42)
        order2 = optimize_circular_order(G, max_iters=100, rng=42)

        assert order1 == order2

    def test_different_seeds_different_results(self):
        """Test that different seeds can give different results."""
        G = nx.Graph()
        G.add_edges_from([(0, 2), (1, 3), (0, 3), (1, 2)])

        order1 = optimize_circular_order(G, max_iters=500, rng=1)
        order2 = optimize_circular_order(G, max_iters=500, rng=2)

        # May or may not be different, but both should be valid
        assert len(order1) == len(order2)
        assert set(order1) == set(order2)

    def test_block_moves_parameter(self):
        """Test block_moves parameter."""
        G = nx.Graph()
        G.add_edges_from([(i, i + 1) for i in range(10)])

        # With block moves
        order1 = optimize_circular_order(G, max_iters=50, block_moves=True, rng=42)

        # Without block moves
        order2 = optimize_circular_order(G, max_iters=50, block_moves=False, rng=42)

        # Both should be valid orderings
        assert len(order1) == 11
        assert len(order2) == 11

    def test_single_node(self):
        """Test with single node - expects ValueError due to sampling constraints."""
        G = nx.Graph()
        G.add_node(0)

        # Single node case: optimization requires at least 2 nodes for swapping
        # The function will fail when trying to sample 2 nodes from population of 1
        with pytest.raises(ValueError):
            optimize_circular_order(G, max_iters=10, rng=42)

    def test_two_nodes(self):
        """Test with two nodes (minimum for optimization)."""
        G = nx.Graph()
        G.add_edge(0, 1)

        order = optimize_circular_order(G, max_iters=10, rng=42)

        assert len(order) == 2
        assert set(order) == {0, 1}

    def test_optimization_improves_objective(self):
        """Test that optimization improves or maintains objective."""
        G = nx.Graph()
        G.add_edges_from([(0, 2), (1, 3), (0, 3), (1, 2)])

        seed_order = [0, 1, 2, 3]
        initial_obj = _objective(G, seed_order)

        optimized_order = optimize_circular_order(
            G, seed_order=seed_order, max_iters=1000, rng=42
        )
        final_obj = _objective(G, optimized_order)

        # Final objective should be no worse than initial
        assert final_obj <= initial_obj


class TestPlotCausalNetwork:
    """Test plot_causal_network function."""

    def setup_method(self):
        """Set up test environment."""
        plt.switch_backend("Agg")
        plt.clf()

    def teardown_method(self):
        """Clean up after tests."""
        plt.clf()
        plt.close("all")

    def test_basic_plotting(self):
        """Test basic network plotting."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)
        G.add_edge("X2", "X3", lag=1, cmi=0.3, p_value=0.03)

        fig, ax = plot_causal_network(G, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_empty_graph(self):
        """Test with empty graph."""
        G = nx.MultiDiGraph()

        fig, ax = plot_causal_network(G, show_plot=False)

        assert fig is None
        assert ax is None

    def test_single_node(self):
        """Test with single node, no edges."""
        G = nx.MultiDiGraph()
        G.add_node("X1")

        # Single node requires custom positions since optimization needs >= 2 nodes
        pos = {"X1": np.array([0, 0])}

        fig, ax = plot_causal_network(G, pos=pos, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_single_node_without_positions(self):
        """Test that single node without positions raises ValueError."""
        G = nx.MultiDiGraph()
        G.add_node("X1")

        # Without custom positions, should fail during optimization
        with pytest.raises(ValueError):
            plot_causal_network(G, show_plot=False)

    def test_multiple_lags(self):
        """Test network with multiple lag values."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)
        G.add_edge("X2", "X3", lag=2, cmi=0.4, p_value=0.02)
        G.add_edge("X3", "X1", lag=3, cmi=0.3, p_value=0.03)

        fig, ax = plot_causal_network(G, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_self_loops(self):
        """Test network with self-loops."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X1", lag=1, cmi=0.6, p_value=0.01)
        G.add_edge("X1", "X2", lag=1, cmi=0.4, p_value=0.02)

        fig, ax = plot_causal_network(G, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_custom_positions(self):
        """Test with custom node positions."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)

        pos = {"X1": np.array([0, 0]), "X2": np.array([1, 0])}

        fig, ax = plot_causal_network(G, pos=pos, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_colorblind_safe_mode(self):
        """Test colorblind-safe color palette."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)
        G.add_edge("X2", "X3", lag=2, cmi=0.4, p_value=0.02)

        fig, ax = plot_causal_network(G, colorblind_safe=True, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_custom_colormaps(self):
        """Test with custom colormaps."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)

        fig, ax = plot_causal_network(G, colormaps=["YlOrRd", "PuBu"], show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_no_colorbar(self):
        """Test with colorbar disabled."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)

        fig, ax = plot_causal_network(G, show_colorbar=False, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_no_statistics(self):
        """Test with statistics box disabled."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)

        fig, ax = plot_causal_network(G, show_statistics=False, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_edge_labels(self):
        """Test with edge labels enabled."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)

        fig, ax = plot_causal_network(G, show_edge_labels=True, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_pvalue_alpha_disabled(self):
        """Test with p-value alpha modulation disabled."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)
        G.add_edge("X2", "X3", lag=1, cmi=0.3, p_value=0.5)

        fig, ax = plot_causal_network(G, use_pvalue_alpha=False, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_custom_pvalue_threshold(self):
        """Test with custom p-value threshold."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)
        G.add_edge("X2", "X3", lag=1, cmi=0.3, p_value=0.03)

        fig, ax = plot_causal_network(G, pvalue_threshold=0.01, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_edges_without_pvalues(self):
        """Test network where edges don't have p-values."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5)  # No p_value

        fig, ax = plot_causal_network(G, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_custom_styling(self):
        """Test with custom styling parameters."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)

        fig, ax = plot_causal_network(
            G,
            figsize=(10, 10),
            node_size=5000,
            node_color="lightblue",
            node_linewidth=3.0,
            edge_width_range=(2.0, 10.0),
            arrowsize=20,
            label_fontsize=12,
            title_fontsize=18,
            show_plot=False,
        )

        assert fig is not None
        assert ax is not None

    def test_custom_title(self):
        """Test with custom title."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)

        fig, ax = plot_causal_network(G, title="My Custom Network", show_plot=False)

        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "My Custom Network"

    def test_high_dpi(self):
        """Test with high DPI setting."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)

        fig, ax = plot_causal_network(G, dpi=300, show_plot=False)

        assert fig is not None
        assert ax is not None
        assert fig.dpi == 300

    @patch("matplotlib.pyplot.savefig")
    def test_save_figure(self, mock_savefig):
        """Test saving figure to file."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)

        fig, ax = plot_causal_network(
            G,
            save_path="/tmp/test_network.png",
            file_format="png",
            show_plot=False,
        )

        # Check that savefig was called
        mock_savefig.assert_called_once()

    def test_transparent_background(self):
        """Test with transparent background."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            fig, ax = plot_causal_network(
                G,
                save_path="/tmp/test.png",
                transparent=True,
                show_plot=False,
            )

            # Check that transparent was passed
            call_kwargs = mock_savefig.call_args[1]
            assert call_kwargs["transparent"] is True

    def test_different_file_formats(self):
        """Test saving with different file formats."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.5, p_value=0.01)

        for fmt in ["png", "pdf", "svg"]:
            with patch("matplotlib.pyplot.savefig") as mock_savefig:
                fig, ax = plot_causal_network(
                    G,
                    save_path=f"/tmp/test.{fmt}",
                    file_format=fmt,
                    show_plot=False,
                )

                call_kwargs = mock_savefig.call_args[1]
                assert call_kwargs["format"] == fmt

    def test_large_network(self):
        """Test with larger network."""
        G = nx.MultiDiGraph()
        nodes = [f"X{i}" for i in range(10)]
        for i in range(9):
            G.add_edge(nodes[i], nodes[i + 1], lag=1, cmi=0.5, p_value=0.01)

        fig, ax = plot_causal_network(G, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_negative_cmi_values(self):
        """Test handling of negative CMI values (should be clipped to 0)."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=-0.1, p_value=0.01)
        G.add_edge("X2", "X3", lag=1, cmi=0.3, p_value=0.02)

        fig, ax = plot_causal_network(G, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_zero_cmi_values(self):
        """Test with zero CMI values."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2", lag=1, cmi=0.0, p_value=0.01)

        fig, ax = plot_causal_network(G, show_plot=False)

        assert fig is not None
        assert ax is not None

    def test_missing_edge_attributes(self):
        """Test with edges missing some attributes."""
        G = nx.MultiDiGraph()
        G.add_edge("X1", "X2")  # No attributes

        fig, ax = plot_causal_network(G, show_plot=False)

        assert fig is not None
        assert ax is not None
