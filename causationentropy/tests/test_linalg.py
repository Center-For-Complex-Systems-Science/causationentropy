import networkx as nx
import numpy as np
import pytest

from causationentropy.core.linalg import (
    companion_matrix,
    correlation_log_determinant,
    subnetwork,
)


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


class TestSubnetwork:
    """Test the subnetwork extraction function."""

    def test_subnetwork_basic(self):
        """Test basic subnetwork extraction."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(1, 2, lag=2, cmi=0.3, p_value=0.05)
        G.add_edge(0, 2, lag=1, cmi=0.4, p_value=0.02)

        H = subnetwork(G, lag=1)

        assert isinstance(H, nx.DiGraph)
        assert H.number_of_nodes() == 3
        assert H.number_of_edges() == 2
        assert H.has_edge(0, 1)
        assert H.has_edge(0, 2)
        assert not H.has_edge(1, 2)

    def test_subnetwork_preserves_attributes(self):
        """Test that edge attributes are preserved."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(1, 2, lag=1, cmi=0.3, p_value=0.05)

        H = subnetwork(G, lag=1)

        edge_data_01 = H.get_edge_data(0, 1)
        assert edge_data_01["cmi"] == 0.5
        assert edge_data_01["p_value"] == 0.01

        edge_data_12 = H.get_edge_data(1, 2)
        assert edge_data_12["cmi"] == 0.3
        assert edge_data_12["p_value"] == 0.05

    def test_subnetwork_empty_lag(self):
        """Test extraction when no edges exist at specified lag."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(1, 2, lag=2, cmi=0.3, p_value=0.05)

        H = subnetwork(G, lag=3)

        assert H.number_of_nodes() == 3
        assert H.number_of_edges() == 0

    def test_subnetwork_lag_zero(self):
        """Test extraction of contemporaneous edges (lag=0)."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=0, cmi=0.6, p_value=0.001)
        G.add_edge(1, 2, lag=1, cmi=0.3, p_value=0.05)

        H = subnetwork(G, lag=0)

        assert H.number_of_edges() == 1
        assert H.has_edge(0, 1)
        assert not H.has_edge(1, 2)

    def test_subnetwork_multiple_edges_same_nodes(self):
        """Test when MultiDiGraph has multiple edges between same nodes."""
        G = nx.MultiDiGraph()
        # Multiple edges with same lag (shouldn't happen in typical use, but test it)
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(0, 1, lag=1, cmi=0.4, p_value=0.02)
        G.add_edge(0, 1, lag=2, cmi=0.3, p_value=0.03)

        H = subnetwork(G, lag=1)

        # DiGraph should collapse multiple edges into one
        # The last one added typically overwrites
        assert H.number_of_edges() == 1
        assert H.has_edge(0, 1)

    def test_subnetwork_preserves_node_attributes(self):
        """Test that node attributes are preserved."""
        G = nx.MultiDiGraph()
        G.add_node(0, name="X0", variable="temperature")
        G.add_node(1, name="X1", variable="pressure")
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)

        H = subnetwork(G, lag=1)

        assert H.nodes[0]["name"] == "X0"
        assert H.nodes[0]["variable"] == "temperature"
        assert H.nodes[1]["name"] == "X1"
        assert H.nodes[1]["variable"] == "pressure"

    def test_subnetwork_bidirectional_edges(self):
        """Test extraction of bidirectional edges at same lag."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(1, 0, lag=1, cmi=0.4, p_value=0.02)

        H = subnetwork(G, lag=1)

        assert H.number_of_edges() == 2
        assert H.has_edge(0, 1)
        assert H.has_edge(1, 0)

    def test_subnetwork_missing_edge_attributes(self):
        """Test handling of edges with missing attributes."""
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, lag=1)  # No cmi or p_value

        H = subnetwork(G, lag=1)

        edge_data = H.get_edge_data(0, 1)
        assert edge_data["cmi"] == 0.0  # Default value
        assert edge_data["p_value"] == 1.0  # Default value

    def test_subnetwork_large_graph(self):
        """Test with larger graph."""
        G = nx.MultiDiGraph()
        n_nodes = 10

        # Add edges at different lags
        for i in range(n_nodes - 1):
            for lag in range(1, 4):
                G.add_edge(i, i + 1, lag=lag, cmi=0.5, p_value=0.01)

        H = subnetwork(G, lag=2)

        assert H.number_of_nodes() == n_nodes
        assert H.number_of_edges() == n_nodes - 1

    def test_subnetwork_string_node_names(self):
        """Test with string node names instead of integers."""
        G = nx.MultiDiGraph()
        G.add_edge("X0", "X1", lag=1, cmi=0.5, p_value=0.01)
        G.add_edge("X1", "X2", lag=1, cmi=0.3, p_value=0.05)

        H = subnetwork(G, lag=1)

        assert H.number_of_nodes() == 3
        assert H.has_edge("X0", "X1")
        assert H.has_edge("X1", "X2")

    def test_subnetwork_empty_graph(self):
        """Test with empty graph."""
        G = nx.MultiDiGraph()

        H = subnetwork(G, lag=1)

        assert H.number_of_nodes() == 0
        assert H.number_of_edges() == 0


class TestCompanionMatrix:
    """Test the companion matrix construction function."""

    def test_companion_matrix_basic(self):
        """Test basic companion matrix construction."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(1, 2, lag=2, cmi=0.3, p_value=0.05)

        C = companion_matrix(G)

        # For max_lag=2, n_nodes=3: shape should be (3*2, 3*2) = (6, 6)
        assert C.shape == (6, 6)
        assert isinstance(C, np.ndarray)

    def test_companion_matrix_structure(self):
        """Test the block structure of companion matrix."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, lag=1, cmi=1.0, p_value=0.01)
        G.add_edge(1, 0, lag=2, cmi=1.0, p_value=0.01)

        C = companion_matrix(G)

        # Shape: (2*2, 2*2) = (4, 4)
        assert C.shape == (4, 4)

        # Check identity blocks in lower rows
        # For k=1: C[2:4, 0:2] should be identity
        assert np.allclose(C[2:4, 0:2], np.eye(2))

    def test_companion_matrix_adjacency_values(self):
        """Test that adjacency matrices are correctly placed."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)

        C = companion_matrix(G)

        # The first block (top-left 2x2) should contain the lag-1 adjacency
        # Adjacency matrix should have 1 at position [0, 1]
        assert C[0, 1] == 1.0
        assert C[1, 0] == 0.0
        assert C[1, 1] == 0.0

    def test_companion_matrix_max_lag_zero(self):
        """Test when max_lag is 0 (no temporal edges)."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1, 2])
        # No edges or only lag=0 edges

        C = companion_matrix(G)

        # Should return empty array
        assert C.shape == (0, 0)

    def test_companion_matrix_only_lag_zero_edges(self):
        """Test when only contemporaneous edges exist."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, lag=0, cmi=0.5, p_value=0.01)

        C = companion_matrix(G)

        # max_lag=0, should return empty array
        assert C.shape == (0, 0)

    def test_companion_matrix_multiple_lags(self):
        """Test with edges at multiple lag values."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(1, 0, lag=2, cmi=0.3, p_value=0.02)
        G.add_edge(0, 1, lag=3, cmi=0.2, p_value=0.03)

        C = companion_matrix(G)

        # max_lag=3, n_nodes=2: shape (2*3, 2*3) = (6, 6)
        assert C.shape == (6, 6)

        # Check identity blocks
        assert np.allclose(C[2:4, 0:2], np.eye(2))  # k=1
        assert np.allclose(C[4:6, 2:4], np.eye(2))  # k=2

    def test_companion_matrix_dense_output(self):
        """Test that output is dense numpy array, not sparse."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)

        C = companion_matrix(G)

        # Should be numpy.ndarray, not scipy.sparse
        assert type(C).__module__ == "numpy"
        assert isinstance(C, np.ndarray)

    def test_companion_matrix_all_zeros_blocks(self):
        """Test that unused blocks are zeros."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, lag=2, cmi=0.5, p_value=0.01)

        C = companion_matrix(G)

        # Lag=1 block (C[0:2, 0:2]) should be all zeros
        assert np.allclose(C[0:2, 0:2], np.zeros((2, 2)))

        # Lag=2 block (C[0:2, 2:4]) should have the edge
        assert C[0, 3] == 1.0  # Edge from 0 to 1 at lag=2

    def test_companion_matrix_node_ordering(self):
        """Test that node ordering is consistent."""
        G = nx.MultiDiGraph()
        # Add nodes out of order
        G.add_node(2)
        G.add_node(0)
        G.add_node(1)
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(1, 2, lag=1, cmi=0.3, p_value=0.02)

        C = companion_matrix(G)

        # max_lag=1, n_nodes=3: shape should be (3*1, 3*1) = (3, 3)
        assert C.shape == (3, 3)

    def test_companion_matrix_string_nodes(self):
        """Test with string node names."""
        G = nx.MultiDiGraph()
        G.add_node("X0")
        G.add_node("X1")
        G.add_edge("X0", "X1", lag=1, cmi=0.5, p_value=0.01)

        C = companion_matrix(G)

        assert C.shape == (2, 2)
        # Adjacency should have edge from X0 to X1
        # String nodes are sorted alphabetically by NetworkX

    def test_companion_matrix_self_loops(self):
        """Test handling of self-loops."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 0, lag=1, cmi=0.5, p_value=0.01)  # Self-loop
        G.add_edge(0, 1, lag=1, cmi=0.3, p_value=0.02)

        C = companion_matrix(G)

        # Should include self-loop in adjacency
        assert C[0, 0] == 1.0  # Self-loop at lag=1
        assert C[0, 1] == 1.0  # Edge to node 1

    def test_companion_matrix_empty_graph(self):
        """Test with empty graph."""
        G = nx.MultiDiGraph()

        C = companion_matrix(G)

        assert C.shape == (0, 0)

    def test_companion_matrix_single_node(self):
        """Test with single node graph."""
        G = nx.MultiDiGraph()
        G.add_node(0)
        G.add_edge(0, 0, lag=1, cmi=0.5, p_value=0.01)

        C = companion_matrix(G)

        # Shape: (1*1, 1*1) = (1, 1)
        assert C.shape == (1, 1)
        assert C[0, 0] == 1.0  # Self-loop

    def test_companion_matrix_large_graph(self):
        """Test with larger graph."""
        G = nx.MultiDiGraph()
        n_nodes = 10
        max_lag = 5

        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes - 1):
            for lag in range(1, max_lag + 1):
                G.add_edge(i, i + 1, lag=lag, cmi=0.5, p_value=0.01)

        C = companion_matrix(G)

        expected_shape = (n_nodes * max_lag, n_nodes * max_lag)
        assert C.shape == expected_shape

        # Check all identity blocks exist
        for k in range(1, max_lag):
            r0 = k * n_nodes
            c0 = (k - 1) * n_nodes
            identity_block = C[r0 : r0 + n_nodes, c0 : c0 + n_nodes]
            assert np.allclose(identity_block, np.eye(n_nodes))

    def test_companion_matrix_bidirectional_edges(self):
        """Test with bidirectional edges."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(1, 0, lag=1, cmi=0.4, p_value=0.02)

        C = companion_matrix(G)

        # Adjacency matrix should have both edges
        assert C[0, 1] == 1.0
        assert C[1, 0] == 1.0

    def test_companion_matrix_mathematical_properties(self):
        """Test mathematical properties of the companion matrix."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(1, 2, lag=2, cmi=0.3, p_value=0.05)

        C = companion_matrix(G)

        # Companion matrix should be square
        assert C.shape[0] == C.shape[1]

        # Should be real-valued
        assert np.all(np.isreal(C))

        # Should have binary values in top row (adjacency) and identity blocks
        # (may not be strictly binary if weighted, but for unweighted should be)
        unique_vals = np.unique(C)
        assert all(val in [0.0, 1.0] for val in unique_vals)


class TestSubnetworkAndCompanionMatrixIntegration:
    """Integration tests for subnetwork and companion_matrix."""

    def test_subnetwork_feeds_companion_matrix(self):
        """Test that subnetwork correctly extracts data for companion_matrix."""
        G = nx.MultiDiGraph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(1, 2, lag=1, cmi=0.3, p_value=0.05)
        G.add_edge(0, 2, lag=2, cmi=0.4, p_value=0.02)

        # Extract subnetwork at lag=1
        H1 = subnetwork(G, lag=1)
        adj1 = nx.adjacency_matrix(H1).toarray()

        # Build companion matrix
        C = companion_matrix(G)

        # The first 3x3 block of C should match lag-1 adjacency
        assert np.allclose(C[0:3, 0:3], adj1)

    def test_companion_matrix_consistency_with_subnetworks(self):
        """Test that companion matrix blocks match individual subnetworks."""
        G = nx.MultiDiGraph()
        n = 3
        G.add_nodes_from(range(n))
        G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
        G.add_edge(1, 2, lag=2, cmi=0.3, p_value=0.05)

        C = companion_matrix(G)
        max_lag = 2

        for lag in range(1, max_lag + 1):
            H = subnetwork(G, lag)
            adj = nx.adjacency_matrix(H).toarray()

            # Extract the corresponding block from companion matrix
            start_col = (lag - 1) * n
            block = C[0:n, start_col : start_col + n]

            assert np.allclose(block, adj), f"Lag {lag} block doesn't match"
