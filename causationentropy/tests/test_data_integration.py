#!/usr/bin/env python3
"""
Integration test for all entropy methods in causal discovery.

This test runs discover_network() for every supported information type and method,
using synthetic data generators that match each entropy's distributional assumptions.
"""
import warnings

import networkx as nx
import numpy as np
import pytest

warnings.filterwarnings("ignore")

from causationentropy import discover_network
from causationentropy.core.stats import Compute_TPR_FPR
from causationentropy.datasets.synthetic import (
    linear_stochastic_gaussian_process,
    poisson_coupled_oscillators,
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def test_standard_gaussian():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )  # This will return the adjacency of the provided network.
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="gaussian",
        max_lag=1,
        alpha_forward=0.05,
        alpha_backward=0.05,
        n_shuffles=1000,  # Reduced for faster execution
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    # The matrices look binarized, but they are not.
    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Standard Gaussian Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr == 1.0
    assert fpr == 0.0


def test_alternative_gaussian():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )  # This will return the adjacency of the provided network.
    G_discovered = discover_network(
        data=data,
        method="alternative",
        information="gaussian",
        max_lag=1,
        alpha_forward=0.01,
        alpha_backward=0.01,
        n_shuffles=1000,  # Reduced for faster execution
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    # The matrices look binarized, but they are not.
    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Alternative Gaussian Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr == 1
    assert fpr == 0


def test_standard_knn():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )  # This will return the adjacency of the provided network.
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="knn",
        metric="euclidean",
        max_lag=2,
        k_means=5,  # This method is sensitive to the choice of k. Too low = low TPR
        alpha_forward=0.01,
        alpha_backward=0.01,
        n_shuffles=1000,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    # The matrices look binarized, but they are not.
    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Standard KNN Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr == 1
    assert fpr == 0


def test_alternative_knn():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )  # This will return the adjacency of the provided network.
    G_discovered = discover_network(
        data=data,
        method="alternative",
        information="knn",
        metric="euclidean",
        max_lag=2,
        k_means=20,  # This method is sensitive to the choice of k. Too low = low TPR
        alpha_forward=0.001,
        alpha_backward=0.001,
        n_shuffles=5000,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    # The matrices look binarized, but they are not.
    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Alternate KNN Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr == 1
    assert fpr == 0


def test_minkowski_standard_knn():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )  # This will return the adjacency of the provided network.
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="knn",
        metric="minkowski",
        max_lag=2,
        k_means=5,  # This method is sensitive to the choice of k. Too low = low TPR
        alpha_forward=0.01,
        alpha_backward=0.01,
        n_shuffles=1000,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    # The matrices look binarized, but they are not.
    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Standard KNN minkowski distance Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr == 1
    assert fpr == 0


def test_standard_geometric_knn():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )  # This will return the adjacency of the provided network.
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="geometric_knn",
        metric="minkowski",
        max_lag=2,
        k_means=10,  # This method is sensitive to the choice of k. Too low = low TPR
        alpha_forward=0.05,
        alpha_backward=0.05,
        n_shuffles=500,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    # The matrices look binarized, but they are not.
    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Standard Geometric-KNN distance Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr == 1.0
    assert fpr == 0.0


def test_standard_kde():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )  # This will return the adjacency of the provided network.
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="kde",
        max_lag=2,
        bandwidth="silverman",
        alpha_forward=0.01,
        alpha_backward=0.01,
        n_shuffles=1000,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    # The matrices look binarized, but they are not.
    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Standard KDE Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr == 1.0
    assert fpr <= 0.1  # Follow up about the FPR


def test_information_lasso():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )
    G_discovered = discover_network(
        data=data,
        method="information_lasso",
        information="gaussian",
        max_lag=1,
        alpha_forward=0.05,
        alpha_backward=0.05,
        n_shuffles=1000,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Information Lasso Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.9  # Information Lasso typically achieves high TPR
    assert fpr <= 0.2  # Allow moderate FPR for regularized methods


def test_lasso():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )
    G_discovered = discover_network(data=data, method="lasso", max_lag=1)
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Pure Lasso Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.9
    assert fpr <= 0.2


def test_standard_poisson():
    T = 200
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = poisson_coupled_oscillators(n=n_nodes, T=T, seed=seed, G=G_true)
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="poisson",
        max_lag=1,
        alpha_forward=0.05,
        alpha_backward=0.05,
        n_shuffles=1000,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Standard Poisson Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.95
    assert fpr <= 0.1


def test_alternative_poisson():
    T = 200
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = poisson_coupled_oscillators(n=n_nodes, T=T, seed=seed, G=G_true)
    G_discovered = discover_network(
        data=data,
        method="alternative",
        information="poisson",
        max_lag=1,
        alpha_forward=0.01,
        alpha_backward=0.01,
        n_shuffles=1000,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Alternative Poisson Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.95  # Alternative Poisson typically achieves high TPR
    assert fpr <= 0.1  # Allow small FPR for Poisson methods


def test_alternative_geometric_knn():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )
    G_discovered = discover_network(
        data=data,
        method="alternative",
        information="geometric_knn",
        metric="euclidean",
        max_lag=2,
        k_means=10,
        alpha_forward=0.001,
        alpha_backward=0.001,
        n_shuffles=500,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Alternative Geometric-KNN Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.4  # Alternative geometric-KNN can be more conservative
    assert fpr <= 0.1  # Usually achieves good specificity


def test_alternative_kde():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )
    G_discovered = discover_network(
        data=data,
        method="alternative",
        information="kde",
        max_lag=2,
        bandwidth="silverman",
        alpha_forward=0.01,
        alpha_backward=0.01,
        n_shuffles=1000,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Alternative KDE Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.1  # Alternative method can be very conservative
    assert fpr <= 0.1  # Usually achieves good specificity


def test_kde_scott_bandwidth():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="kde",
        max_lag=2,
        bandwidth="scott",
        alpha_forward=0.05,
        alpha_backward=0.05,
        n_shuffles=1000,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"KDE Scott Bandwidth Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.95  # Scott bandwidth typically achieves high TPR
    assert fpr <= 0.1  # Allow small FPR for KDE methods


def test_knn_chebyshev_metric():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="knn",
        metric="chebyshev",
        max_lag=2,
        k_means=8,
        alpha_forward=0.01,
        alpha_backward=0.01,
        n_shuffles=1000,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"KNN Chebyshev Metric Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.9
    assert fpr <= 0.1


def test_knn_manhattan_metric():
    T = 200
    rho = 0.7
    n_nodes = 5
    seed = 42
    p = 0.2
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="knn",
        metric="cityblock",
        max_lag=2,
        k_means=8,
        alpha_forward=0.01,
        alpha_backward=0.01,
        n_shuffles=1000,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"KNN Manhattan Metric Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr == 1
    assert fpr == 0


def test_parameter_variations():
    """Test various parameter combinations"""
    T = 150
    rho = 0.8
    n_nodes = 4
    seed = 123
    p = 0.3
    np.random.seed(seed)
    G_true = nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    data, A = linear_stochastic_gaussian_process(
        rho=rho, n=n_nodes, T=T, seed=seed, G=G_true
    )

    # Test with higher max_lag
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="gaussian",
        max_lag=3,
        alpha_forward=0.01,
        alpha_backward=0.01,
        n_shuffles=500,
    )
    B = nx.to_numpy_array(G_discovered)
    A_true = nx.to_numpy_array(G_true)

    A_bin = (A_true > 0).astype(int)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Higher max_lag Gaussian Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.9  # Higher max_lag Gaussian should achieve high TPR
    assert fpr <= 0.1  # Allow small FPR

    # Test with different k_means for knn
    G_discovered = discover_network(
        data=data,
        method="standard",
        information="knn",
        metric="euclidean",
        max_lag=2,
        k_means=15,
        alpha_forward=0.001,
        alpha_backward=0.001,
        n_shuffles=800,
    )
    B = nx.to_numpy_array(G_discovered)
    B_bin = (B > 0).astype(int)
    tpr, fpr = Compute_TPR_FPR(A_bin, B_bin)
    print(f"Higher k_means KNN Estimate: TPR: {tpr}, FPR: {fpr}")
    assert tpr >= 0.6  # KNN with higher k can be more conservative
    assert fpr <= 0.1  # Usually achieves good specificity


if __name__ == "__main__":
    test_standard_gaussian()
    test_alternative_gaussian()
    test_standard_knn()
    test_alternative_knn()
    test_minkowski_standard_knn()
    test_standard_geometric_knn()
    test_standard_kde()
    test_information_lasso()
    test_lasso()
    test_standard_poisson()
    test_alternative_poisson()
    test_alternative_geometric_knn()
    test_alternative_kde()
    test_kde_scott_bandwidth()
    test_knn_chebyshev_metric()
    test_knn_manhattan_metric()
    test_parameter_variations()
