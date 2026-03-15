import networkx as nx
import numpy as np


# =============================================================================
# NETWORK TOPOLOGY GENERATION
# =============================================================================


def generate_graph_topology(topo_type, n_nodes, params, seed):
    """
    Generate various network topologies for benchmarking.

    Parameters
    ----------
    topo_type : str
        Type of topology: "Erdos-Renyi", "Scale-Free", or "Small-World"
    n_nodes : int
        Number of nodes in the graph
    params : dict
        Parameters specific to the topology type:
        - Erdos-Renyi: {"p_edge": float} edge probability
        - Scale-Free: {"m_attachment": int} number of edges to attach
        - Small-World: {"k_neighbors": int, "p_rewire": float}
    seed : int
        Random seed for reproducibility

    Returns
    -------
    G : nx.DiGraph
        Directed graph with the specified topology
    """
    if topo_type == "Erdos-Renyi":
        p = params.get("p_edge", 0.2)
        return nx.erdos_renyi_graph(n_nodes, p, seed=seed, directed=True)
    if topo_type == "Scale-Free":
        m = params.get("m_attachment", 1)
        G_und = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        return nx.DiGraph(G_und)
    if topo_type == "Small-World":
        k = params.get("k_neighbors", 2)
        p = params.get("p_rewire", 0.1)
        G_und = nx.connected_watts_strogatz_graph(n_nodes, k, p, tries=100, seed=seed)
        return nx.DiGraph(G_und)
    raise ValueError(f"Unknown topology: {topo_type}")


# =============================================================================
# KURAMOTO OSCILLATORS
# =============================================================================


def wrap_to_pi(x):
    """Wrap angles to [-π, π]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def simulate_kuramoto(
    G,
    T,
    dt=0.05,
    rho=0.7,
    seed=0,
    omega_mean=0.0,
    omega_std=1.0,
    phase_noise_std=0.02,
    burn_in=50,
    normalize_by_indegree=True,
):
    """
    Simulate the Kuramoto model on a given network.

    This function simulates the dynamics of coupled phase oscillators on a network G.
    The dynamics are governed by the Kuramoto equation:
    d(theta_i)/dt = omega_i + rho * sum_j(A_ij * sin(theta_j - theta_i))

    Parameters
    ----------
    G : nx.DiGraph
        The network on which to simulate.
    T : int
        Number of time steps to simulate after burn-in.
    dt : float, optional
        Integration time step.
    rho : float, optional
        Coupling strength.
    seed : int, optional
        Random seed for reproducibility.
    omega_mean : float, optional
        Mean of the natural frequencies of the oscillators.
    omega_std : float, optional
        Standard deviation of the natural frequencies.
    phase_noise_std : float, optional
        Standard deviation of the Gaussian noise added at each step.
    burn_in : int, optional
        Number of initial time steps to discard.
    normalize_by_indegree : bool, optional
        If True, normalize the coupling term by the in-degree of each node.

    Returns
    -------
    out : np.ndarray
        A (T, n) array of phase trajectories.
    dict
        A dictionary containing simulation parameters.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes).astype(float)

    omega = rng.normal(loc=omega_mean, scale=omega_std, size=n)
    theta = rng.uniform(low=-np.pi, high=np.pi, size=n)

    if normalize_by_indegree:
        indeg = A.sum(axis=0)
        indeg_safe = np.where(indeg > 0, indeg, 1.0)
    else:
        indeg_safe = np.ones(n)

    total_steps = burn_in + T
    out = np.zeros((T, n), dtype=float)

    for step in range(total_steps):
        coupling = (A.T * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)
        dtheta = omega + (rho * coupling / indeg_safe)
        noise = rng.normal(loc=0.0, scale=phase_noise_std, size=n)
        theta = theta + dt * dtheta + noise
        theta = wrap_to_pi(theta)

        if step >= burn_in:
            out[step - burn_in] = theta

    return out, {"omega": omega, "dt": dt, "rho": rho}


def phase_velocity(theta, dt):
    """Compute phase velocity from phase trajectories."""
    dtheta = wrap_to_pi(theta[1:] - theta[:-1])
    return dtheta / dt


def prepare_kuramoto_data_for_causal_discovery(theta, dt):
    """
    Transforms Kuramoto model phase data into a regression-ready format.

    This function converts the non-linear time series of oscillator phases into a
    higher-dimensional, but linear-approximable, dataset. The core idea is to
    predict the angular velocity of each oscillator based on the sine of phase
    differences between oscillators at the previous time step.

    The transformation is as follows:
    1.  Target variables (Y): The angular velocities `v_i(t+1)` for each oscillator `i`.
    2.  Predictor variables (X): The coupling terms `sin(theta_j(t) - theta_i(t))`
        for all pairs `(i, j)` where `i != j`.

    This creates a dataset where standard linear causal discovery methods can be
    applied to uncover the underlying network structure `j -> i`.

    Parameters
    ----------
    theta : np.ndarray
        The (T, n) array of phase trajectories from the simulation.
    dt : float
        The time step of the simulation.

    Returns
    -------
    X : np.ndarray
        The transformed data matrix of shape (T-2, n + n*(n-1)). The first `n`
        columns are the target velocities, and the remaining columns are the
        predictor sine-coupling terms.
    meta : dict
        A dictionary of metadata, including the number of nodes `n`, the time
        step `dt`, and a `basis_map` that links columns in `X` back to the
        original oscillator pairs.
    var_names : list of str
        A list of variable names corresponding to the columns of `X`.
    """
    T, n = theta.shape
    v = phase_velocity(theta, dt)  # shape (T-1, n)
    th = theta[:-1]  # shape (T-1, n)

    # build S aligned with v(t)
    S = []
    basis_map = []
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            S.append(np.sin(th[:, j] - th[:, i]))
            basis_map.append((i, j))
    S = np.stack(S, axis=1)  # (T-1, n*(n-1))

    # SHIFT: predictors at t, targets at t+1
    v_next = v[1:]  # (T-2, n)
    S_prev = S[:-1]  # (T-2, n*(n-1))

    X = np.concatenate([v_next, S_prev], axis=1)

    var_names = [f"v{i}" for i in range(n)] + [f"s_{i}<-{j}" for (i, j) in basis_map]
    meta = {"n": n, "dt": dt, "pred_offset": n, "basis_map": basis_map, "p": X.shape[1]}
    return X, meta, var_names


# =============================================================================
# RÖSSLER OSCILLATORS
# =============================================================================


def simulate_rossler(
    G,
    T,
    dt=0.02,
    rho=0.1,
    seed=0,
    # Rössler parameters (chaotic for many settings near these defaults)
    a=0.2,
    b=0.2,
    c=5.7,
    # Initial condition noise / heterogeneity
    init_scale=1.0,
    # Additive process noise (optional; keep small)
    noise_std=0.0,
    burn_in=200,
    normalize_by_indegree=True,
    coupling_on="x",  # "x" (default), or "y"/"z" if you want
):
    """
    Network-coupled Rössler oscillators with diffusive coupling on one coordinate.

    Node i dynamics (uncoupled):
        dx = -y - z
        dy =  x + a y
        dz =  b + z (x - c)

    Coupling (directed graph):
        + rho * sum_j A[j,i] * (s_j - s_i) / indeg_i   where s is chosen coordinate

    Parameters
    ----------
    G : nx.DiGraph
        The network on which to simulate.
    T : int
        Number of time steps to simulate after burn-in.
    dt : float, optional
        Integration time step.
    rho : float, optional
        Coupling strength.
    seed : int, optional
        Random seed for reproducibility.
    a : float, optional
        Rössler parameter a.
    b : float, optional
        Rössler parameter b.
    c : float, optional
        Rössler parameter c.
    init_scale : float, optional
        Scale of initial condition noise.
    noise_std : float, optional
        Standard deviation of additive process noise.
    burn_in : int, optional
        Number of initial time steps to discard.
    normalize_by_indegree : bool, optional
        If True, normalize the coupling term by the in-degree of each node.
    coupling_on : str, optional
        Which coordinate to couple on: "x", "y", or "z".

    Returns
    -------
    traj : np.ndarray
        A (T, n, 3) array with columns [x, y, z].
    meta : dict
        A dictionary containing simulation parameters.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes).astype(float)  # A[u,v]=1 means u->v
    # We want incoming neighbors for i: A[j,i], hence we use A[:, i]

    if normalize_by_indegree:
        indeg = A.sum(axis=0)
        indeg_safe = np.where(indeg > 0, indeg, 1.0)
    else:
        indeg_safe = np.ones(n)

    # Initial conditions
    x = rng.normal(0.0, init_scale, size=n)
    y = rng.normal(0.0, init_scale, size=n)
    z = rng.normal(0.0, init_scale, size=n)

    total_steps = burn_in + T
    out = np.zeros((T, n, 3), dtype=float)

    # helper: choose coupling coordinate array
    def coord_vec(name):
        if name == "x":
            return x
        if name == "y":
            return y
        if name == "z":
            return z
        raise ValueError("coupling_on must be one of {'x','y','z'}")

    for step in range(total_steps):
        s = coord_vec(coupling_on)

        # incoming diffusive coupling: sum_j A[j,i]*(s_j - s_i)
        # vector form: coupling_i = sum_j A[j,i]*s_j - s_i*sum_j A[j,i]
        incoming_sum = A.T @ s  # (n,) where i gets sum_j A[j,i]*s_j
        coupling = incoming_sum - s * A.sum(axis=0)
        coupling = coupling / indeg_safe

        # Rössler ODE + coupling applied to chosen coordinate
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)

        if coupling_on == "x":
            dx = dx + rho * coupling
        elif coupling_on == "y":
            dy = dy + rho * coupling
        else:
            dz = dz + rho * coupling

        # Euler step (simple + fast)
        if noise_std > 0:
            x = x + dt * dx + rng.normal(0.0, noise_std, size=n)
            y = y + dt * dy + rng.normal(0.0, noise_std, size=n)
            z = z + dt * dz + rng.normal(0.0, noise_std, size=n)
        else:
            x = x + dt * dx
            y = y + dt * dy
            z = z + dt * dz

        if step >= burn_in:
            out[step - burn_in, :, 0] = x
            out[step - burn_in, :, 1] = y
            out[step - burn_in, :, 2] = z

    return out, {
        "dt": dt,
        "rho": rho,
        "a": a,
        "b": b,
        "c": c,
        "coupling_on": coupling_on,
    }


def finite_diff(traj, dt):
    """Compute finite difference derivatives: traj: (T,n,d) -> dtraj/dt (T-1,n,d)."""
    return (traj[1:] - traj[:-1]) / dt


def prepare_rossler_data_for_causal_discovery(traj, dt, coupling_on="x"):
    """
    Build expanded data matrix with targets = next-step derivatives (dx,dy,dz per node),
    predictors include coupling terms for the coupled coordinate: (s_j - s_i) for all
    ordered pairs i!=j.

    This mirrors the Kuramoto basis approach:
      predictors(t) -> targets(t+1) alignment via lag1 shift.

    Parameters
    ----------
    traj : np.ndarray
        The (T, n, 3) array of trajectories from the simulation.
    dt : float
        The time step of the simulation.
    coupling_on : str, optional
        Which coordinate to couple on: "x", "y", or "z".

    Returns
    -------
    X : np.ndarray
        The transformed data matrix.
    meta : dict
        Metadata including basis_map.
    var_names : list of str
        Variable names.
    """
    T, n, d = traj.shape
    assert d == 3

    dtraj = finite_diff(traj, dt)  # (T-1, n, 3)

    # Use derivatives at their natural time indices. discover_network and
    # PCMCI handle lag-1 alignment internally; pre-shifting here would
    # create a double-lag that destroys the causal signal.
    dx = dtraj[:, :, 0]  # (T-1, n)
    dy = dtraj[:, :, 1]
    dz = dtraj[:, :, 2]
    Y = np.concatenate([dx, dy, dz], axis=1)  # (T-1, 3n)

    # Coupling coordinate at the same time steps as derivatives
    if coupling_on == "x":
        s = traj[:-1, :, 0]  # (T-1, n)
    elif coupling_on == "y":
        s = traj[:-1, :, 1]
    elif coupling_on == "z":
        s = traj[:-1, :, 2]
    else:
        raise ValueError("coupling_on must be one of {'x','y','z'}")

    # Build coupling basis: c_{i<-j}(t) = s_j(t) - s_i(t)
    # Shape: (T-1, n*(n-1))
    C_cols = []
    basis_map = []
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            C_cols.append(s[:, j] - s[:, i])
            basis_map.append((i, j))
    C = np.stack(C_cols, axis=1)

    # Full X = [derivatives | coupling basis]
    X = np.concatenate([Y, C], axis=1)

    # Names for first 3n nodes (derivative targets)
    var_names = []
    for i in range(n):
        var_names.append(f"dx{i}")
    for i in range(n):
        var_names.append(f"dy{i}")
    for i in range(n):
        var_names.append(f"dz{i}")

    # Names for coupling terms
    for i, j in basis_map:
        var_names.append(f"c_{i}<-{j}")

    meta = {
        "n": n,
        "dt": dt,
        "coupling_on": coupling_on,
        "pred_offset": 3 * n,
        "basis_map": basis_map,
        "p": X.shape[1],
        "target_dim": 3 * n,
    }
    return X, meta, var_names


# =============================================================================
# LEGACY FUNCTIONS (KEPT FOR BACKWARD COMPATIBILITY)
# =============================================================================


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
