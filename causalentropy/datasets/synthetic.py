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
    A = A / np.sum(A, axis=1)
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
    R = 2 * (np.random.rand(n, n) - 0.5)
    A = A * R
    A = A / np.max(np.abs(np.linalg.eigvals(A)))
    A = A * rho
    XY = np.zeros((T, n))
    XY[0, :] = epsilon * np.random.randn(1, n)
    for i in range(1, T):
        Xi = np.dot(A, np.matrix(XY[i - 1, :]).T) + epsilon * np.random.randn(n, 1)
        XY[i, :] = Xi.T
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


def negative_binomial_coupled_oscillators(
    n=10, T=100, p=0.2, r_base=5, p_nb=0.3, coupling_strength=0.2, seed=42, G=None
):
    """
    Coupled negative binomial oscillators with overdispersed count dynamics.

    Parameters
    ----------
    n : int
        Number of oscillators
    T : int
        Number of time steps
    p : float
        Edge probability for random graph
    r_base : float
        Base number of failures parameter
    p_nb : float
        Success probability for negative binomial
    coupling_strength : float
        Strength of coupling between oscillators
    seed : int
        Random seed

    Returns
    -------
    X : array (T, n)
        Time series of negative binomial counts
    A : array (n, n)
        True adjacency matrix
    """
    rng = np.random.default_rng(seed)
    if G is None:
        G = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    A = nx.to_numpy_array(G)

    X = np.zeros((T, n))
    X[0, :] = rng.negative_binomial(r_base, p_nb, n)

    for t in range(1, T):
        for i in range(n):
            # r parameter varies based on neighbor influence
            neighbor_influence = coupling_strength * np.sum(A[:, i] * X[t - 1, :])
            r_effective = r_base + neighbor_influence
            r_effective = max(1.0, r_effective)  # Ensure positive r
            X[t, i] = rng.negative_binomial(int(r_effective), p_nb)

    return X, A


def hawkes_coupled_processes(
    n=10,
    T=100,
    p=0.2,
    mu_base=0.5,
    alpha=0.3,
    beta=1.0,
    coupling_strength=0.2,
    dt=0.1,
    seed=42,
    G=None,
):
    """
    Coupled Hawkes processes where each process can excite its neighbors.

    Parameters
    ----------
    n : int
        Number of processes
    T : int
        Number of time steps (in discrete time)
    p : float
        Edge probability for coupling graph
    mu_base : float
        Base intensity for each process
    alpha : float
        Self-excitation parameter
    beta : float
        Decay rate
    coupling_strength : float
        Strength of cross-excitation between processes
    dt : float
        Time step size
    seed : int
        Random seed

    Returns
    -------
    X : array (T, n)
        Event counts per time step for each process
    A : array (n, n)
        True adjacency matrix
    """
    rng = np.random.default_rng(seed)
    if G is None:
        G = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    A = nx.to_numpy_array(G)

    X = np.zeros((T, n))
    intensities = np.ones(n) * mu_base

    for t in range(T):
        # Generate events based on current intensities
        for i in range(n):
            # Probability of event in time interval dt
            prob = min(0.99, intensities[i] * dt)
            X[t, i] = rng.binomial(1, prob)

            # If event occurred, increase self-intensity
            if X[t, i] > 0:
                intensities[i] += alpha

                # Excite coupled neighbors
                for j in range(n):
                    if A[i, j] > 0:
                        intensities[j] += coupling_strength * A[i, j]

        # Decay all intensities
        intensities = mu_base + (intensities - mu_base) * np.exp(-beta * dt)
        intensities = np.maximum(intensities, mu_base)

    return X, A


def von_mises_coupled_oscillators(
    n=10,
    T=100,
    p=0.2,
    kappa_base=2.0,
    coupling_strength=0.5,
    freq_base=0.1,
    seed=42,
    G=None,
):
    """
    Coupled von Mises (circular) oscillators with phase coupling.

    Parameters
    ----------
    n : int
        Number of oscillators
    T : int
        Number of time steps
    p : float
        Edge probability for coupling graph
    kappa_base : float
        Base concentration parameter
    coupling_strength : float
        Strength of phase coupling
    freq_base : float
        Base frequency for phase evolution
    seed : int
        Random seed

    Returns
    -------
    X : array (T, n)
        Time series of circular phases [0, 2π]
    A : array (n, n)
        True adjacency matrix
    """
    rng = np.random.default_rng(seed)
    if G is None:
        G = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    A = nx.to_numpy_array(G)

    X = np.zeros((T, n))
    phases = rng.uniform(0, 2 * np.pi, n)
    X[0, :] = phases

    for t in range(1, T):
        new_phases = np.zeros(n)
        for i in range(n):
            # Natural frequency evolution
            phase_drift = phases[i] + freq_base

            # Coupling from neighbors (Kuramoto-like)
            neighbor_coupling = 0
            for j in range(n):
                if A[j, i] > 0:
                    phase_diff = phases[j] - phases[i]
                    neighbor_coupling += (
                        coupling_strength * A[j, i] * np.sin(phase_diff)
                    )

            # New phase with coupling and noise
            mean_phase = phase_drift + neighbor_coupling
            new_phases[i] = rng.vonmises(mean_phase, kappa_base)

        phases = new_phases % (2 * np.pi)
        X[t, :] = phases

    return X, A


def laplace_coupled_oscillators(
    n=10, T=100, p=0.2, scale_base=1.0, coupling_strength=0.3, seed=42, G=None
):
    """
    Coupled Laplace oscillators with heavy-tailed dynamics.

    Parameters
    ----------
    n : int
        Number of oscillators
    T : int
        Number of time steps
    p : float
        Edge probability for coupling graph
    scale_base : float
        Base scale parameter for Laplace distribution
    coupling_strength : float
        Strength of coupling between oscillators
    seed : int
        Random seed

    Returns
    -------
    X : array (T, n)
        Time series with Laplace-distributed increments
    A : array (n, n)
        True adjacency matrix
    """
    rng = np.random.default_rng(seed)
    if G is None:
        G = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    A = nx.to_numpy_array(G)

    X = np.zeros((T, n))
    X[0, :] = rng.laplace(0, scale_base, n)

    for t in range(1, T):
        for i in range(n):
            # Location parameter influenced by neighbors
            neighbor_influence = coupling_strength * np.sum(A[:, i] * X[t - 1, :])
            loc = neighbor_influence

            # Scale parameter can also be influenced
            scale_effective = scale_base

            X[t, i] = X[t - 1, i] + rng.laplace(loc, scale_effective)

    return X, A
