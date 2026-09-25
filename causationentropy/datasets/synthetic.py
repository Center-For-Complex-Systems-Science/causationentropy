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


def _companion_spectral_radius(lag_matrices):
    """Spectral radius of the companion matrix built from lag matrices.

    Parameters
    ----------
    lag_matrices : list of np.ndarray
        Non-empty list of ``(n, n)`` coefficient matrices for lags
        1 through ``p``.

    Returns
    -------
    radius : float
        Largest eigenvalue magnitude of the ``(n * p, n * p)`` companion
        matrix.
    """
    n = lag_matrices[0].shape[0]
    max_lag = len(lag_matrices)
    companion = np.zeros((n * max_lag, n * max_lag))
    companion[:n, :] = np.hstack(lag_matrices)
    if max_lag > 1:
        companion[n:, :-n] = np.eye(n * (max_lag - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _rescale_to_spectral_radius(lag_matrices, rho):
    """Rescale lag matrices so the companion spectral radius is ``<= rho``.

    A one-shot ``rho / radius`` scaling is only valid for lag-1 systems:
    the companion matrix contains fixed shift-identity blocks, so its
    spectral radius is not linear in the coefficient scale for multi-lag
    systems. Instead, bisect a scalar multiplier in ``[0, hi]`` (growing
    ``hi`` first when the system is already stable, mirroring the upward
    scaling of :func:`linear_stochastic_gaussian_process`), keeping the
    largest multiplier whose companion radius is ``<= rho``. The returned
    radius is always re-checked against ``rho``. Nilpotent systems (e.g.
    DAG couplings, radius 0 at any scale) keep their weights unchanged.

    Parameters
    ----------
    lag_matrices : list of np.ndarray
        Coefficient matrices for lags 1 through ``p``. May be empty.
    rho : float
        Target spectral radius. Must be positive.

    Returns
    -------
    scaled : list of np.ndarray
        Rescaled coefficient matrices.
    final_radius : float
        Companion spectral radius of ``scaled``, guaranteed ``<= rho``.

    Raises
    ------
    ValueError
        If the rescaled system still exceeds ``rho`` (safety net; scale 0
        is always stable, so this is unreachable in practice).
    """
    if not lag_matrices:
        return [], 0.0

    def radius_at(scale):
        return _companion_spectral_radius([matrix * scale for matrix in lag_matrices])

    # Nilpotent systems (e.g. DAG couplings, radius 0 at any scale) cannot
    # be rescaled meaningfully: keep the weights as given.
    initial = radius_at(1.0)
    if initial <= 1e-12:
        return list(lag_matrices), initial

    lo, hi = 0.0, 1.0
    if radius_at(hi) <= rho:
        for _ in range(64):
            lo = hi
            hi *= 2.0
            if radius_at(hi) > rho:
                break
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if radius_at(mid) <= rho:
            lo = mid
        else:
            hi = mid
    scaled = [matrix * lo for matrix in lag_matrices]
    final_radius = _companion_spectral_radius(scaled)
    if not final_radius <= rho:
        raise ValueError(
            f"Could not stabilize the system to rho={rho} "
            f"(final radius {final_radius})."
        )
    return scaled, final_radius


def linear_gaussian_from_graph(G, T=500, coupling=0.7, rho=0.9, epsilon=0.1, seed=42):
    r"""Simulate a stable vector autoregression from a directed lag graph.

    Each edge ``source -> sink`` with a ``lag`` attribute defines one term
    of a vector autoregression: the sink at time ``t`` is driven by the
    source at time ``t - lag``. This accepts the same
    :class:`networkx.MultiDiGraph` format produced by
    ``discover_network`` (nodes are variables, edges carry ``lag``), so a
    discovered network can be used directly as ground truth for unit and
    integration tests.

    The simulated process is:

    .. math::

        X_i(t) = \\sum_{(j, \\tau) \\to i} w_{j \\to i}^{(\\tau)}
        X_j(t - \\tau) + \\epsilon_i(t)

    where :math:`\epsilon_i(t)` is Gaussian noise with standard deviation
    ``epsilon``. Edge weights default to ``coupling`` unless the edge
    carries a ``weight`` attribute. All weights are jointly rescaled (by
    bisection on a scalar multiplier) so the companion matrix spectral
    radius is ``<= rho``, which keeps the process stationary for
    ``rho < 1``.

    Parameters
    ----------
    G : nx.MultiDiGraph or nx.DiGraph
        Directed graph defining the ground-truth couplings. Nodes are
        variables (any hashable labels); each edge must carry an integer
        ``lag >= 1`` attribute (missing lags default to 1) and may carry
        a numeric ``weight`` attribute. Parallel edges between the same
        nodes at the same lag have their weights summed.
    T : int, default=500
        Number of time steps. Must exceed the largest lag in ``G``.
    coupling : float, default=0.7
        Default per-edge weight used when an edge has no ``weight``
        attribute.
    rho : float, default=0.9
        Target spectral radius of the companion matrix. Must be positive;
        values below 1 give a stationary process.
    epsilon : float, default=0.1
        Standard deviation of the Gaussian innovations.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (T, n)
        Simulated multivariate time series. Column ``k`` corresponds to
        ``list(G.nodes())[k]``.
    G : nx.MultiDiGraph or nx.DiGraph
        The input graph, returned unchanged as ground truth.

    Raises
    ------
    ValueError
        If ``G`` is not directed, any lag is not an integer ``>= 1``,
        any weight is not finite, ``rho`` is not positive, or ``T`` does
        not exceed the largest lag.

    Examples
    --------
    >>> import networkx as nx
    >>> from causationentropy.datasets.synthetic import (
    ...     linear_gaussian_from_graph,
    ... )
    >>>
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, lag=1)
    >>> G.add_edge(1, 2, lag=2)
    >>> X, truth = linear_gaussian_from_graph(G, T=300, seed=0)
    >>> X.shape
    (300, 3)

    See Also
    --------
    linear_stochastic_gaussian_process : Random-graph Gaussian dynamics.
    """
    if not G.is_directed():
        raise ValueError("G must be a directed graph.")
    if rho <= 0:
        raise ValueError("rho must be positive.")
    if T < 1:
        raise ValueError("T must be at least 1.")

    nodes = list(G.nodes())
    n = len(nodes)
    index = {node: k for k, node in enumerate(nodes)}

    coefficients = {}  # lag -> (n, n) matrix with [sink, source] weights
    for u, v, data in G.edges(data=True):
        lag = data.get("lag", 1)
        if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)):
            raise ValueError(f"Edge {(u, v)} has non-integer lag={lag!r}.")
        lag = int(lag)
        if lag < 1:
            raise ValueError(f"Edge {(u, v)} has lag={lag}; lags start at 1.")
        weight = data.get("weight", coupling)
        if not np.isfinite(weight):
            raise ValueError(f"Edge {(u, v)} has non-finite weight={weight!r}.")
        matrix = coefficients.setdefault(lag, np.zeros((n, n)))
        matrix[index[v], index[u]] += float(weight)

    if n == 0:
        return np.zeros((T, 0)), G

    max_lag = max(coefficients) if coefficients else 0
    if T <= max_lag:
        raise ValueError(f"T={T} must exceed the largest lag {max_lag} in G.")

    rng = np.random.default_rng(seed)
    lag_matrices = [
        coefficients.get(tau, np.zeros((n, n))) for tau in range(1, max_lag + 1)
    ]
    if lag_matrices:
        lag_matrices, _ = _rescale_to_spectral_radius(lag_matrices, rho)

    X = np.zeros((T, n))
    warmup = max(max_lag, 1)
    X[:warmup, :] = epsilon * rng.standard_normal((warmup, n))
    for t in range(warmup, T):
        driven = np.zeros(n)
        for tau, matrix in enumerate(lag_matrices, start=1):
            driven += matrix @ X[t - tau, :]
        X[t, :] = driven + epsilon * rng.standard_normal(n)
    return X, G
