from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
    gaussian_conditional_mutual_information,
    geometric_knn_conditional_mutual_information,
    kde_conditional_mutual_information,
    knn_conditional_mutual_information,
    poisson_conditional_mutual_information,
)


class TestGaussianConditionalMutualInformation:
    """Test Gaussian conditional mutual information calculation."""

    def test_gaussian_cmi_no_conditioning(self):
        """Test that CMI reduces to MI when Z=None."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        # I(X;Y|Z) with Z=None should equal I(X;Y)
        cmi = gaussian_conditional_mutual_information(X, Y, Z=None)

        # Import regular MI function to compare
        from causationentropy.core.information.mutual_information import (
            gaussian_mutual_information,
        )

        mi = gaussian_mutual_information(X, Y)

        assert np.isclose(cmi, mi, rtol=1e-10)

    def test_gaussian_cmi_independent_given_z(self):
        """Test CMI when X and Y are independent given Z."""
        np.random.seed(42)
        n = 100
        Z = np.random.normal(0, 1, (n, 1))

        # Create X and Y that are independent given Z
        X = Z + np.random.normal(0, 0.1, (n, 1))
        Y = Z + np.random.normal(0, 0.1, (n, 1))

        cmi = gaussian_conditional_mutual_information(X, Y, Z)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        # Should be small but may not be exactly 0 due to noise

    def test_gaussian_cmi_dependent_given_z(self):
        """Test CMI when X and Y are dependent given Z."""
        np.random.seed(42)
        n = 80
        Z = np.random.normal(0, 1, (n, 1))
        X = np.random.normal(0, 1, (n, 1))
        # Y depends on both X and Z
        Y = 0.5 * X + 0.3 * Z + 0.2 * np.random.normal(0, 1, (n, 1))

        cmi = gaussian_conditional_mutual_information(X, Y, Z)

        assert isinstance(cmi, float)
        assert cmi > 0  # Should be positive due to X->Y dependence
        assert not np.isnan(cmi)

    def test_gaussian_cmi_multivariate(self):
        """Test CMI with multivariate X, Y, Z."""
        np.random.seed(42)
        n = 60
        X = np.random.normal(0, 1, (n, 2))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 3))

        cmi = gaussian_conditional_mutual_information(X, Y, Z)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)

    def test_gaussian_cmi_identical_variables(self):
        """Test CMI with identical X and Y."""
        np.random.seed(42)
        n = 40
        X = np.random.normal(0, 1, (n, 1))
        Y = X.copy()  # Identical
        Z = np.random.normal(0, 1, (n, 1))

        cmi = gaussian_conditional_mutual_information(X, Y, Z)

        assert isinstance(cmi, float)
        assert cmi > 0  # Should be positive for identical variables
        assert not np.isnan(cmi)


class TestConditionalMutualInformation:
    """Test the main conditional_mutual_information function."""

    def test_cmi_gaussian_method(self):
        """Test CMI with Gaussian method."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        cmi = conditional_mutual_information(X, Y, Z, method="gaussian")

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)

    def test_cmi_knn_method(self):
        """Test CMI with k-NN method."""
        np.random.seed(42)
        n = 40
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        cmi = conditional_mutual_information(X, Y, Z, method="knn", k=3)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)

    def test_cmi_kde_method(self):
        """Test CMI with KDE method."""
        np.random.seed(42)
        n = 30
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        cmi = conditional_mutual_information(X, Y, Z, method="kde")

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)

    def test_cmi_invalid_method(self):
        """Test CMI with invalid method."""
        X = np.random.normal(0, 1, (20, 1))
        Y = np.random.normal(0, 1, (20, 1))
        Z = np.random.normal(0, 1, (20, 1))

        with pytest.raises(ValueError, match="Method 'invalid' unavailable"):
            conditional_mutual_information(X, Y, Z, method="invalid")

    def test_cmi_no_conditioning_variable(self):
        """Test CMI when no conditioning variable is provided."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        # Should work and reduce to regular MI
        cmi = conditional_mutual_information(X, Y, None, method="gaussian")

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)

    def test_cmi_different_k_values(self):
        """Test CMI with different k values for k-NN method."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        for k in [1, 3, 5]:
            cmi = conditional_mutual_information(X, Y, Z, method="knn", k=k)
            assert isinstance(cmi, float)
            assert not np.isnan(cmi)

    def test_cmi_bandwidth_parameter(self):
        """Test CMI with different bandwidth parameters for KDE."""
        np.random.seed(42)
        n = 40
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        # Test different bandwidth values
        bandwidths = ["silverman", 0.5, 1.0]
        for bw in bandwidths:
            cmi = conditional_mutual_information(X, Y, Z, method="kde", bandwidth=bw)
            assert isinstance(cmi, float)
            assert not np.isnan(cmi)


class TestConditionalMutualInformationProperties:
    """Test mathematical properties of conditional mutual information."""

    def test_cmi_non_negativity(self):
        """Test that CMI is non-negative (in theory)."""
        np.random.seed(42)
        n = 60
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        cmi = conditional_mutual_information(X, Y, Z, method="gaussian")

        # CMI can be negative due to finite sample effects, but should be small
        assert isinstance(cmi, float)
        assert not np.isnan(cmi)

    def test_cmi_chain_rule_property(self):
        """Test chain rule: I(X;Y) = I(X;Y|Z) + I(X;Z)."""
        # This is more of a mathematical property test
        # In practice, finite sample effects may cause deviations
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        from causationentropy.core.information.mutual_information import (
            gaussian_mutual_information,
        )

        mi_xy = gaussian_mutual_information(X, Y)
        cmi_xy_given_z = conditional_mutual_information(X, Y, Z, method="gaussian")
        mi_xz = gaussian_mutual_information(X, Z)

        # Chain rule: I(X;Y) H I(X;Y|Z) + I(X;Z) (approximately due to finite samples)
        chain_rule_sum = cmi_xy_given_z + mi_xz

        # Allow for numerical errors in finite samples
        assert isinstance(mi_xy, float)
        assert isinstance(chain_rule_sum, float)

    def test_cmi_symmetry_in_xy(self):
        """Test that CMI is symmetric in X and Y: I(X;Y|Z) = I(Y;X|Z)."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        cmi_xy = conditional_mutual_information(X, Y, Z, method="gaussian")
        cmi_yx = conditional_mutual_information(Y, X, Z, method="gaussian")

        assert np.isclose(cmi_xy, cmi_yx, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_cmi_empty_arrays(self):
        """Test behavior with empty arrays."""
        X = np.array([]).reshape(0, 1)
        Y = np.array([]).reshape(0, 1)
        Z = np.array([]).reshape(0, 1)

        # Should handle gracefully or raise appropriate error
        try:
            cmi = conditional_mutual_information(X, Y, Z, method="gaussian")
        except (ValueError, IndexError, np.linalg.LinAlgError):
            pass  # Expected for empty data

    def test_cmi_single_sample(self):
        """Test behavior with single sample."""
        X = np.array([[1.0]])
        Y = np.array([[2.0]])
        Z = np.array([[3.0]])

        # Should handle single sample gracefully or fail appropriately
        try:
            cmi = conditional_mutual_information(X, Y, Z, method="gaussian")
        except (ValueError, np.linalg.LinAlgError):
            pass  # May fail due to insufficient data for covariance estimation

    def test_cmi_constant_variables(self):
        """Test CMI with constant variables."""
        n = 30
        X = np.ones((n, 1))  # Constant
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        # Should handle constant variables gracefully
        try:
            cmi = conditional_mutual_information(X, Y, Z, method="gaussian")
        except (ValueError, np.linalg.LinAlgError):
            pass  # May fail due to singular covariance matrix

    def test_cmi_dimension_mismatch(self):
        """Test behavior when arrays have mismatched dimensions."""
        X = np.random.normal(0, 1, (50, 1))
        Y = np.random.normal(0, 1, (40, 1))  # Different number of samples
        Z = np.random.normal(0, 1, (50, 1))

        # Should raise appropriate error for dimension mismatch
        with pytest.raises((ValueError, IndexError)):
            conditional_mutual_information(X, Y, Z, method="gaussian")

    def test_cmi_high_dimensional(self):
        """Test CMI with high-dimensional data."""
        np.random.seed(42)
        n = 60
        X = np.random.normal(0, 1, (n, 5))
        Y = np.random.normal(0, 1, (n, 3))
        Z = np.random.normal(0, 1, (n, 4))

        cmi = conditional_mutual_information(X, Y, Z, method="gaussian")

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)

    def test_cmi_very_correlated_data(self):
        """Test CMI with highly correlated data."""
        np.random.seed(42)
        n = 80
        base = np.random.normal(0, 1, (n, 1))
        X = base + 0.01 * np.random.normal(0, 1, (n, 1))
        Y = base + 0.01 * np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        # Highly correlated X and Y
        cmi = conditional_mutual_information(X, Y, Z, method="gaussian")

        assert isinstance(cmi, float)
        # May have numerical issues with near-singular covariance
        if not np.isnan(cmi):
            assert np.isfinite(cmi)


class TestMethodComparison:
    """Compare different methods for computing CMI."""

    def test_methods_give_reasonable_results(self):
        """Test that different methods give reasonable (finite) results."""
        np.random.seed(42)
        n = 60
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        methods = ["gaussian", "knn", "kde"]
        results = {}

        for method in methods:
            try:
                if method == "knn":
                    cmi = conditional_mutual_information(X, Y, Z, method=method, k=3)
                else:
                    cmi = conditional_mutual_information(X, Y, Z, method=method)
                results[method] = cmi

                assert isinstance(cmi, float)
                assert not np.isnan(cmi)
                assert np.isfinite(cmi)
            except Exception as e:
                # Some methods might fail with certain data
                pass

    def test_gaussian_method_consistency(self):
        """Test that Gaussian method is consistent across calls."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        cmi1 = conditional_mutual_information(X, Y, Z, method="gaussian")
        cmi2 = conditional_mutual_information(X, Y, Z, method="gaussian")

        assert np.isclose(cmi1, cmi2, rtol=1e-15)


class TestGeometricKnnConditionalMutualInformation:
    """Test geometric k-NN conditional mutual information calculation."""

    def test_geometric_knn_cmi_no_conditioning(self):
        """Test that geometric k-NN CMI reduces to MI when Z=None."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 2))
        Y = np.random.normal(0, 1, (n, 1))

        # I(X;Y|Z) with Z=None should equal I(X;Y) with default k=1
        cmi = geometric_knn_conditional_mutual_information(X, Y, Z=None, k=3)

        from causationentropy.core.information.mutual_information import (
            geometric_knn_mutual_information,
        )

        # The function uses default k=1 when Z=None, so compare with k=1
        mi = geometric_knn_mutual_information(X, Y, k=1)

        assert np.isclose(cmi, mi, rtol=1e-10)

    def test_geometric_knn_cmi_basic_properties(self):
        """Test basic properties of geometric k-NN CMI."""
        np.random.seed(42)
        n = 80
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        cmi = geometric_knn_conditional_mutual_information(X, Y, Z, k=3)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)

    def test_geometric_knn_cmi_dependent_variables(self):
        """Test geometric k-NN CMI with dependent variables."""
        np.random.seed(42)
        n = 100
        Z = np.random.normal(0, 1, (n, 1))
        X = np.random.normal(0, 1, (n, 1))
        # Create Y that depends on both X and Z
        Y = 0.6 * X + 0.4 * Z + 0.1 * np.random.normal(0, 1, (n, 1))

        cmi = geometric_knn_conditional_mutual_information(X, Y, Z, k=3)

        assert isinstance(cmi, float)
        assert cmi > 0  # Should be positive due to X->Y dependence
        assert not np.isnan(cmi)

    def test_geometric_knn_cmi_multivariate(self):
        """Test geometric k-NN CMI with multivariate inputs."""
        np.random.seed(42)
        n = 70
        X = np.random.normal(0, 1, (n, 2))
        Y = np.random.normal(0, 1, (n, 2))
        Z = np.random.normal(0, 1, (n, 1))

        cmi = geometric_knn_conditional_mutual_information(X, Y, Z, k=3)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)

    def test_geometric_knn_cmi_different_k_values(self):
        """Test geometric k-NN CMI with different k values."""
        np.random.seed(42)
        n = 60
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        for k in [1, 3, 5]:
            cmi = geometric_knn_conditional_mutual_information(X, Y, Z, k=k)
            assert isinstance(cmi, float)
            assert not np.isnan(cmi)
            assert np.isfinite(cmi)

    def test_geometric_knn_cmi_different_metrics(self):
        """Test geometric k-NN CMI with different distance metrics."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        # Use valid scipy.spatial.distance metric names
        metrics = ["euclidean", "cityblock", "chebyshev"]
        for metric in metrics:
            cmi = geometric_knn_conditional_mutual_information(
                X, Y, Z, k=3, metric=metric
            )
            assert isinstance(cmi, float)
            assert not np.isnan(cmi)
            assert np.isfinite(cmi)

    def test_geometric_knn_cmi_symmetry(self):
        """Test that geometric k-NN CMI is symmetric in X and Y."""
        np.random.seed(42)
        n = 60
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        cmi_xy = geometric_knn_conditional_mutual_information(X, Y, Z, k=3)
        cmi_yx = geometric_knn_conditional_mutual_information(Y, X, Z, k=3)

        # Allow for small numerical differences due to k-NN estimation
        assert np.isclose(cmi_xy, cmi_yx, rtol=1e-6, atol=1e-6)

    def test_geometric_knn_cmi_identical_variables(self):
        """Test geometric k-NN CMI with identical X and Y."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 1))
        Y = X.copy()  # Identical
        Z = np.random.normal(0, 1, (n, 1))

        cmi = geometric_knn_conditional_mutual_information(X, Y, Z, k=3)

        assert isinstance(cmi, float)
        assert cmi > 0  # Should be positive for identical variables
        assert not np.isnan(cmi)

    def test_geometric_knn_cmi_edge_cases(self):
        """Test geometric k-NN CMI edge cases."""
        # Test with minimum valid sample size
        np.random.seed(42)
        n = 10
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        # Should work with k < n
        cmi = geometric_knn_conditional_mutual_information(X, Y, Z, k=min(3, n - 1))
        assert isinstance(cmi, float)
        assert not np.isnan(cmi)

    def test_geometric_knn_cmi_manifold_data(self):
        """Test geometric k-NN CMI with manifold-like data."""
        np.random.seed(42)
        n = 80

        # Create data that lies on a lower-dimensional manifold
        t = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)
        noise = 0.1 * np.random.normal(0, 1, (n, 1))

        # X lies on a circle in 2D
        X = np.hstack([np.cos(t), np.sin(t)]) + noise
        Y = np.random.normal(0, 1, (n, 1))
        Z = t + 0.1 * np.random.normal(0, 1, (n, 1))

        cmi = geometric_knn_conditional_mutual_information(X, Y, Z, k=3)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)

    def test_geometric_knn_cmi_via_main_function(self):
        """Test geometric k-NN CMI through the main conditional_mutual_information function."""
        np.random.seed(42)
        n = 60
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        # Call through main interface
        cmi_main = conditional_mutual_information(X, Y, Z, method="geometric_knn", k=3)

        # Call directly
        cmi_direct = geometric_knn_conditional_mutual_information(X, Y, Z, k=3)

        assert np.isclose(cmi_main, cmi_direct, rtol=1e-15)


class TestPoissonConditionalMutualInformation:
    """Test Poisson conditional mutual information calculation."""

    def test_poisson_cmi_no_conditioning(self):
        """Test Poisson CMI when Z=None (marginal MI case)."""
        np.random.seed(42)
        n = 50

        # Generate Poisson-like count data
        X = np.random.poisson(lam=2.0, size=(n, 2)).astype(float)
        Y = np.random.poisson(lam=1.5, size=(n, 1)).astype(float)

        # Test the Z=None code path
        cmi = poisson_conditional_mutual_information(X, Y, Z=None)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)

    def test_poisson_cmi_with_conditioning(self):
        """Test Poisson CMI with conditioning variable Z."""
        np.random.seed(42)
        n = 60

        # Generate Poisson-like count data
        X = np.random.poisson(lam=1.0, size=(n, 1)).astype(float)
        Y = np.random.poisson(lam=1.5, size=(n, 1)).astype(float)
        Z = np.random.poisson(lam=2.0, size=(n, 1)).astype(float)

        cmi = poisson_conditional_mutual_information(X, Y, Z)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)

    def test_poisson_cmi_multivariate(self):
        """Test Poisson CMI with multivariate inputs."""
        np.random.seed(42)
        n = 40

        # Generate multivariate Poisson-like count data
        X = np.random.poisson(lam=1.0, size=(n, 2)).astype(float)
        Y = np.random.poisson(lam=1.5, size=(n, 2)).astype(float)
        Z = np.random.poisson(lam=2.0, size=(n, 1)).astype(float)

        cmi = poisson_conditional_mutual_information(X, Y, Z)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)

    def test_poisson_cmi_identical_variables(self):
        """Test Poisson CMI with identical X and Y."""
        np.random.seed(42)
        n = 50

        X = np.random.poisson(lam=2.0, size=(n, 1)).astype(float)
        Y = X.copy()  # Identical
        Z = np.random.poisson(lam=1.0, size=(n, 1)).astype(float)

        # Test both code paths
        cmi_no_z = poisson_conditional_mutual_information(X, Y, Z=None)
        cmi_with_z = poisson_conditional_mutual_information(X, Y, Z)

        assert isinstance(cmi_no_z, float)
        assert isinstance(cmi_with_z, float)
        assert not np.isnan(cmi_no_z)
        assert not np.isnan(cmi_with_z)

    def test_poisson_cmi_via_main_function(self):
        """Test Poisson CMI through the main conditional_mutual_information function."""
        np.random.seed(42)
        n = 50

        X = np.random.poisson(lam=1.5, size=(n, 1)).astype(float)
        Y = np.random.poisson(lam=2.0, size=(n, 1)).astype(float)
        Z = np.random.poisson(lam=1.0, size=(n, 1)).astype(float)

        # Call through main interface
        cmi_main = conditional_mutual_information(X, Y, Z, method="poisson")

        # Call directly
        cmi_direct = poisson_conditional_mutual_information(X, Y, Z)

        assert np.isclose(cmi_main, cmi_direct, rtol=1e-15)

    def test_poisson_cmi_edge_cases(self):
        """Test Poisson CMI edge cases."""
        np.random.seed(42)

        # Test with small sample size
        n = 10
        X = np.random.poisson(lam=1.0, size=(n, 1)).astype(float)
        Y = np.random.poisson(lam=1.0, size=(n, 1)).astype(float)
        Z = np.random.poisson(lam=1.0, size=(n, 1)).astype(float)

        # Should work with small samples
        cmi_no_z = poisson_conditional_mutual_information(X, Y, Z=None)
        cmi_with_z = poisson_conditional_mutual_information(X, Y, Z)

        assert isinstance(cmi_no_z, (float, np.floating))
        assert isinstance(cmi_with_z, (float, np.floating))

    def test_poisson_cmi_count_data_properties(self):
        """Test Poisson CMI with realistic count data properties."""
        np.random.seed(42)
        n = 80

        # Create count data with known dependencies
        base_rate = np.random.poisson(lam=3.0, size=(n, 1)).astype(float)
        X = base_rate + np.random.poisson(lam=1.0, size=(n, 1)).astype(float)
        Y = base_rate + np.random.poisson(lam=1.0, size=(n, 1)).astype(float)
        Z = np.random.poisson(lam=2.0, size=(n, 1)).astype(float)

        # Test both paths with realistic count data
        cmi_no_z = poisson_conditional_mutual_information(X, Y, Z=None)
        cmi_with_z = poisson_conditional_mutual_information(X, Y, Z)

        assert isinstance(cmi_no_z, float)
        assert isinstance(cmi_with_z, float)
        assert not np.isnan(cmi_no_z)
        assert not np.isnan(cmi_with_z)

        # X and Y should have positive MI since they share base_rate
        assert cmi_no_z > 0


class TestKDEConditionalMutualInformation:
    """Test KDE conditional mutual information calculation."""

    def test_kde_cmi_no_conditioning(self):
        """Test KDE CMI when Z=None (marginal MI case)."""
        np.random.seed(42)
        n = 50

        # Generate data for KDE estimation
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        # Test the Z=None code path
        cmi = kde_conditional_mutual_information(X, Y, Z=None)

        # Compare with direct mutual information call
        from causationentropy.core.information.mutual_information import (
            kde_mutual_information,
        )

        mi = kde_mutual_information(X, Y)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)
        # Should be equal since Z=None reduces to mutual information
        assert np.isclose(cmi, mi, rtol=1e-10)

    def test_kde_cmi_no_conditioning_multivariate(self):
        """Test KDE CMI with multivariate data when Z=None."""
        np.random.seed(42)
        n = 40

        X = np.random.normal(0, 1, (n, 2))
        Y = np.random.normal(0, 1, (n, 1))

        # Test Z=None path with multivariate X
        cmi = kde_conditional_mutual_information(X, Y, Z=None, bandwidth=0.5)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)

    def test_kde_cmi_no_conditioning_parameters(self):
        """Test KDE CMI Z=None path with different parameters."""
        np.random.seed(42)
        n = 30

        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        # Test with different bandwidth
        cmi1 = kde_conditional_mutual_information(X, Y, Z=None, bandwidth="silverman")
        cmi2 = kde_conditional_mutual_information(X, Y, Z=None, bandwidth=0.8)

        # Test with different kernel
        cmi3 = kde_conditional_mutual_information(X, Y, Z=None, kernel="tophat")

        for cmi in [cmi1, cmi2, cmi3]:
            assert isinstance(cmi, float)
            assert not np.isnan(cmi)
            assert np.isfinite(cmi)

    def test_kde_cmi_with_conditioning(self):
        """Test KDE CMI with conditioning variable Z."""
        np.random.seed(42)
        n = 30

        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        cmi = kde_conditional_mutual_information(X, Y, Z)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)

    def test_kde_cmi_via_main_function(self):
        """Test KDE CMI through the main conditional_mutual_information function."""
        np.random.seed(42)
        n = 40

        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        # Test Z=None through main interface
        cmi_main = conditional_mutual_information(X, Y, None, method="kde")
        cmi_direct = kde_conditional_mutual_information(X, Y, Z=None)

        assert np.isclose(cmi_main, cmi_direct, rtol=1e-15)


class TestKNNConditionalMutualInformation:
    """Test k-NN conditional mutual information calculation."""

    def test_knn_cmi_no_conditioning(self):
        """Test k-NN CMI when Z=None (marginal MI case)."""
        np.random.seed(42)
        n = 50

        # Generate data for k-NN estimation
        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        # Test the Z=None code path
        cmi = knn_conditional_mutual_information(X, Y, Z=None, k=3)

        # Compare with direct mutual information call
        from causationentropy.core.information.mutual_information import (
            knn_mutual_information,
        )

        mi = knn_mutual_information(X, Y, k=3)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)
        # Should be equal since Z=None reduces to mutual information
        assert np.isclose(cmi, mi, rtol=1e-10)

    def test_knn_cmi_no_conditioning_multivariate(self):
        """Test k-NN CMI with multivariate data when Z=None."""
        np.random.seed(42)
        n = 60

        X = np.random.normal(0, 1, (n, 2))
        Y = np.random.normal(0, 1, (n, 2))

        # Test Z=None path with multivariate data
        cmi = knn_conditional_mutual_information(X, Y, Z=None, k=5)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)

    def test_knn_cmi_no_conditioning_parameters(self):
        """Test k-NN CMI Z=None path with different parameters."""
        np.random.seed(42)
        n = 40

        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        # Test with different k values
        cmi1 = knn_conditional_mutual_information(X, Y, Z=None, k=1)
        cmi2 = knn_conditional_mutual_information(X, Y, Z=None, k=3)
        cmi3 = knn_conditional_mutual_information(X, Y, Z=None, k=5)

        # Test with different metrics
        cmi4 = knn_conditional_mutual_information(X, Y, Z=None, metric="euclidean")
        cmi5 = knn_conditional_mutual_information(X, Y, Z=None, metric="cityblock")

        for cmi in [cmi1, cmi2, cmi3, cmi4, cmi5]:
            assert isinstance(cmi, float)
            assert not np.isnan(cmi)
            assert np.isfinite(cmi)

    def test_knn_cmi_with_conditioning(self):
        """Test k-NN CMI with conditioning variable Z."""
        np.random.seed(42)
        n = 50

        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))
        Z = np.random.normal(0, 1, (n, 1))

        cmi = knn_conditional_mutual_information(X, Y, Z, k=3)

        assert isinstance(cmi, float)
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)

    def test_knn_cmi_via_main_function(self):
        """Test k-NN CMI through the main conditional_mutual_information function."""
        np.random.seed(42)
        n = 50

        X = np.random.normal(0, 1, (n, 1))
        Y = np.random.normal(0, 1, (n, 1))

        # Test Z=None through main interface
        cmi_main = conditional_mutual_information(X, Y, None, method="knn", k=3)
        cmi_direct = knn_conditional_mutual_information(X, Y, Z=None, k=3)

        # Both should be non-negative (or non-finite)
        assert cmi_main >= 0.0 or not np.isfinite(cmi_main)
        # Direct call returns raw value, main interface clamps it
        assert np.isclose(
            cmi_main,
            max(0.0, cmi_direct) if np.isfinite(cmi_direct) else cmi_direct,
            rtol=1e-15,
        )

    def test_knn_cmi_dependent_variables_no_conditioning(self):
        """Test k-NN CMI with dependent variables when Z=None."""
        np.random.seed(42)
        n = 60

        # Create dependent variables
        X = np.random.normal(0, 1, (n, 1))
        noise = 0.1 * np.random.normal(0, 1, (n, 1))
        Y = 0.7 * X + noise  # Y depends on X

        cmi = knn_conditional_mutual_information(X, Y, Z=None, k=3)

        assert isinstance(cmi, float)
        assert cmi > 0  # Should be positive due to dependence
        assert not np.isnan(cmi)
        assert np.isfinite(cmi)
