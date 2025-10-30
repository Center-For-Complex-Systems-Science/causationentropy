import networkx as nx
import numpy as np


def logistic_map(X, r):
    return r * X * (1 - X)


def logisic_dynamics(n=20, p=0.1, t=100, r=3.99, sigma=0.1, seed=42):
    """Network coupled logistic map, r is the logistic map parameter
    and sigma is the coupling strength between oscillators"""

    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    A = nx.to_numpy_array(G)
    # Must adjust the adjacency matrix so that dynamics stay in [0,1]
    row_sums = np.sum(A, axis=1)
    # Avoid division by zero: only normalize rows that have connections
    non_zero_mask = row_sums > 0
    A[non_zero_mask] = A[non_zero_mask] / row_sums[non_zero_mask, np.newaxis]
    A = A.T

    # Since the row sums equal to 1 the Laplacian matrix is easy...
    L = np.eye(n) - A
    L = np.array(L)

    XY = np.zeros((t, n))
    XY[0, :] = rng.random(n)
    for i in range(1, t):
        XY[i, :] = (
            logistic_map(XY[i - 1, :], r)
            - sigma * np.dot(L, logistic_map(XY[i - 1, :], r)).T
        )

    return XY, A


def linear_stochastic_gaussian_process(
    rho, n=20, T=100, p=0.1, epsilon=1e-1, seed=42, G=None
):
    """Linear stochastic Gaussian process"""

    rng = np.random.default_rng(seed)
    if G is None:
        G = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    A = nx.to_numpy_array(G).T
    R = 2 * (rng.random((n, n)) - 0.5)
    A = A * R
    # Avoid division by zero in eigenvalue normalization
    eigvals = np.linalg.eigvals(A)
    max_eigval = np.max(np.abs(eigvals))
    if max_eigval > 1e-12:  # Only normalize if eigenvalue is significant
        A = A / max_eigval
    A = A * rho
    XY = np.zeros((T, n))
    XY[0, :] = epsilon * rng.standard_normal(n)
    for i in range(1, T):
        Xi = np.dot(A, XY[i - 1, :]) + epsilon * rng.standard_normal(n)
        XY[i, :] = Xi
    return XY, A


def poisson_coupled_oscillators(
    n=10, T=100, p=0.2, lambda_base=2.0, coupling_strength=0.3, seed=42, G=None
):
    """
    Coupled Poisson oscillators where each node's rate depends on its neighbors' previous states.

    Parameters
    ----------
    n : int
        Number of oscillators
    T : int
        Number of time steps
    p : float
        Edge probability for random graph
    lambda_base : float
        Base Poisson rate
    coupling_strength : float
        Strength of coupling between oscillators
    seed : int
        Random seed

    Returns
    -------
    X : array (T, n)
        Time series of Poisson counts
    A : array (n, n)
        True adjacency matrix

    References
    ------
    [1] Xanthi Pedeli, Dimitris Karlis, Some properties of multivariate INAR(1) processes,
    Computational Statistics & Data Analysis. (2013)
    """
    rng = np.random.default_rng(seed)
    if G is None:
        G = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    A = nx.to_numpy_array(G)

    X = np.zeros((T, n))
    X[0, :] = rng.poisson(lambda_base, n)

    for t in range(1, T):
        for i in range(n):
            # Rate depends on base rate plus coupled influence from neighbors
            neighbor_influence = coupling_strength * np.sum(A[:, i] * X[t - 1, :])
            rate = lambda_base + neighbor_influence
            rate = max(0.1, rate)  # Ensure positive rate
            X[t, i] = rng.poisson(rate)

    return X, A
