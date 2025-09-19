"""Test package-level imports work correctly."""

import pytest


class TestPackageImports:
    """Test that main package imports work as documented."""

    def test_discover_network_import(self):
        """Test that discover_network can be imported from main package."""
        from causationentropy import discover_network

        # Check it's callable
        assert callable(discover_network)

    def test_synthetic_import(self):
        """Test that synthetic module can be imported."""
        from causationentropy.datasets import synthetic

        # Check the function exists
        assert hasattr(synthetic, "linear_stochastic_gaussian_process")
        assert callable(synthetic.linear_stochastic_gaussian_process)

    def test_readme_example_functionality(self):
        """Test the functionality shown in README example works."""
        from causationentropy import discover_network
        from causationentropy.datasets import synthetic

        # Generate small synthetic dataset for testing
        data, true_network = synthetic.linear_stochastic_gaussian_process(
            rho=0.5,  # coupling strength
            n=3,  # number of variables (small for testing)
            T=50,  # number of time samples (small for testing)
            p=0.3,  # edge probability (sparsity)
        )

        # Basic checks on generated data
        assert data.shape == (50, 3)
        assert true_network.shape == (3, 3)

        # Test discovery works without errors
        discovered = discover_network(data, max_lag=2)

        # Check result is a network with expected properties
        assert hasattr(discovered, "nodes")
        assert hasattr(discovered, "edges")
        assert len(discovered.nodes) == 3  # Should have 3 nodes
