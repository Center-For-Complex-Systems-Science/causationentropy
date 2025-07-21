import numpy as np
import pandas as pd
import networkx as nx
import pytest
from unittest.mock import patch, MagicMock

from causalentropy.core.discovery import discover_network


class TestDiscoverNetwork:
    """Test the main causal discovery function."""

    def test_discover_network_basic_numpy(self):
        """Test basic functionality with numpy array input."""
        # Create simple time series with causal relationship: X0 -> X1
        np.random.seed(42)
        T = 100
        X0 = np.random.normal(0, 1, T)
        X1 = np.zeros(T)
        # X1[t] depends on X0[t-1] plus noise
        for t in range(1, T):
            X1[t] = 0.7 * X0[t-1] + 0.3 * np.random.normal()
        
        data = np.column_stack([X0, X1])
        
        # Test the function runs without error
        G = discover_network(data, max_lag=2, n_shuffles=50)
        
        assert isinstance(G, nx.DiGraph)
        assert len(G.nodes()) == 2
        assert 'X0' in G.nodes()
        assert 'X1' in G.nodes()

    def test_discover_network_pandas_input(self):
        """Test functionality with pandas DataFrame input."""
        np.random.seed(42)
        T = 50
        data_dict = {
            'var1': np.random.normal(0, 1, T),
            'var2': np.random.normal(0, 1, T),
            'var3': np.random.normal(0, 1, T)
        }
        df = pd.DataFrame(data_dict)
        
        G = discover_network(df, max_lag=1, n_shuffles=20)
        
        assert isinstance(G, nx.DiGraph)
        assert len(G.nodes()) == 3
        assert 'var1' in G.nodes()
        assert 'var2' in G.nodes()
        assert 'var3' in G.nodes()

    def test_discover_network_parameter_validation(self):
        """Test parameter validation."""
        data = np.random.normal(0, 1, (20, 3))
        
        # Test invalid method
        with pytest.raises(NotImplementedError, match="method=invalid not supported"):
            discover_network(data, method="invalid")
        
        # Test invalid information type
        with pytest.raises(NotImplementedError, match="information=invalid not supported"):
            discover_network(data, information="invalid")
        
        # Test time series too short
        short_data = np.random.normal(0, 1, (5, 2))
        with pytest.raises(ValueError, match="Time series too short for chosen max_lag"):
            discover_network(short_data, max_lag=10)

    def test_discover_network_valid_methods(self):
        """Test that all valid methods are accepted."""
        data = np.random.normal(0, 1, (30, 2))
        
        valid_methods = ["standard", "alternative", "information_lasso", "lasso"]
        for method in valid_methods:
            G = discover_network(data, method=method, max_lag=1, n_shuffles=10)
            assert isinstance(G, nx.DiGraph)

    def test_discover_network_valid_information_types(self):
        """Test that all valid information types are accepted."""
        data = np.random.normal(0, 1, (30, 2))
        
        valid_info_types = ["gaussian", "knn", "kde", "poisson"]
        for info_type in valid_info_types:
            G = discover_network(data, information=info_type, max_lag=1, n_shuffles=10)
            assert isinstance(G, nx.DiGraph)

    def test_discover_network_edge_attributes(self):
        """Test that edges have proper attributes when found."""
        # Create data with strong causal relationship
        np.random.seed(123)
        T = 100
        X0 = np.random.normal(0, 1, T)
        X1 = np.zeros(T)
        # Strong linear relationship
        for t in range(1, T):
            X1[t] = 0.9 * X0[t-1] + 0.1 * np.random.normal()
        
        data = np.column_stack([X0, X1])
        G = discover_network(data, max_lag=2, alpha_forward=0.1, n_shuffles=100)
        
        # Check if any edges were found and have proper attributes
        for edge in G.edges(data=True):
            source, target, attrs = edge
            if 'lag' in attrs:
                assert isinstance(attrs['lag'], int)
                assert attrs['lag'] >= 0
            if 'cmi' in attrs:
                assert isinstance(attrs['cmi'], (int, float))

    def test_discover_network_different_parameters(self):
        """Test discover_network with different parameter values."""
        data = np.random.normal(0, 1, (50, 3))
        
        # Test different max_lag values
        G1 = discover_network(data, max_lag=1, n_shuffles=10)
        G2 = discover_network(data, max_lag=3, n_shuffles=10)
        assert isinstance(G1, nx.DiGraph)
        assert isinstance(G2, nx.DiGraph)
        
        # Test different alpha values
        G3 = discover_network(data, alpha_forward=0.01, alpha_backward=0.01, n_shuffles=10)
        G4 = discover_network(data, alpha_forward=0.1, alpha_backward=0.1, n_shuffles=10)
        assert isinstance(G3, nx.DiGraph)
        assert isinstance(G4, nx.DiGraph)

    def test_discover_network_empty_result(self):
        """Test behavior when no causal relationships are found."""
        # Pure noise should typically result in no edges
        np.random.seed(42)
        data = np.random.normal(0, 1, (30, 3))
        
        G = discover_network(data, max_lag=1, alpha_forward=0.001, n_shuffles=50)  # Very strict alpha
        
        assert isinstance(G, nx.DiGraph)
        assert len(G.nodes()) == 3
        # Edges may or may not be found depending on random chance

    def test_discover_network_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        data = np.random.normal(0, 1, (40, 2))
        
        # The function uses internal random seed (42), so should be reproducible
        G1 = discover_network(data, max_lag=1, n_shuffles=20)
        G2 = discover_network(data, max_lag=1, n_shuffles=20)
        
        # Should have same number of nodes
        assert len(G1.nodes()) == len(G2.nodes())
        assert set(G1.nodes()) == set(G2.nodes())
        
        # Should have same edges (given deterministic random seed)
        assert set(G1.edges()) == set(G2.edges())

    def test_discover_network_minimum_data_size(self):
        """Test behavior with minimum viable data size."""
        # Minimum data should be max_lag + 3 time points
        max_lag = 2
        T = max_lag + 3  # Minimum viable size
        data = np.random.normal(0, 1, (T, 2))
        
        G = discover_network(data, max_lag=max_lag, n_shuffles=10)
        assert isinstance(G, nx.DiGraph)
        assert len(G.nodes()) == 2

    def test_discover_network_single_variable(self):
        """Test behavior with single variable (should work but find no edges)."""
        data = np.random.normal(0, 1, (30, 1))
        
        G = discover_network(data, max_lag=1, n_shuffles=10)
        
        assert isinstance(G, nx.DiGraph)
        assert len(G.nodes()) == 1
        assert len(G.edges()) == 0  # No self-loops expected

    @patch('causalentropy.core.discovery.conditional_mutual_information')
    def test_discover_network_cmi_integration(self, mock_cmi):
        """Test integration with conditional mutual information function."""
        mock_cmi.return_value = 0.5  # Mock CMI value
        
        data = np.random.normal(0, 1, (20, 2))
        G = discover_network(data, max_lag=1, n_shuffles=10)
        
        # Verify CMI was called
        assert mock_cmi.called
        assert isinstance(G, nx.DiGraph)

    def test_discover_network_data_types(self):
        """Test different input data types."""
        T, n = 30, 3
        
        # Test with different numpy dtypes
        data_float32 = np.random.normal(0, 1, (T, n)).astype(np.float32)
        data_float64 = np.random.normal(0, 1, (T, n)).astype(np.float64)
        data_int = np.random.randint(0, 10, (T, n))
        
        for data in [data_float32, data_float64, data_int]:
            G = discover_network(data, max_lag=1, n_shuffles=10)
            assert isinstance(G, nx.DiGraph)
            assert len(G.nodes()) == n