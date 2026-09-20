import networkx as nx
import numpy as np


def correlation_log_determinant(A, epsilon=1e-10):
    """
    Compute the logarithm of the determinant of a correlation matrix.

    This function calculates the signed log-determinant of the correlation matrix
    derived from the input data matrix A. The correlation matrix is defined as:

    .. math::

        \\mathbf{R}_{ij} = \\frac{\\text{Cov}(X_i, X_j)}{\\sqrt{\\text{Var}(X_i) \\text{Var}(X_j)}}

    The log-determinant is computed using:

    .. math::

        \\log |\\mathbf{R}| = \\text{sign}(|\\mathbf{R}|) \\cdot \\log(||\\mathbf{R}||)

    This approach provides numerical stability for matrices that may be close to singular.

    Parameters
    ----------
    A : array-like of shape (n_samples, n_features)
        Input data matrix where rows are samples and columns are features.
    epsilon : float, default=1e-10
        Small regularization parameter (currently unused but reserved for
        potential numerical stabilization).

    Returns
    -------
    log_det : float
        Logarithm of the determinant of the correlation matrix.
        Returns 0.0 for degenerate cases (empty matrix or scalar).

    Notes
    -----
    **Special Cases:**
    - Empty matrix (n_features = 0): Returns 0.0
    - Scalar correlation (1x1 matrix): Returns 0.0
    - Singular matrix: May return -inf or raise warnings

    **Numerical Considerations:**
    - Uses `numpy.linalg.slogdet` for stable computation of log-determinant
    - Handles edge cases gracefully without exceptions
    - More stable than computing `log(det(R))` directly

    **Applications:**
    - Gaussian mutual information calculation
    - Model selection criteria (AIC, BIC)
    - Multivariate normality testing
    - Information-theoretic measures

    **Interpretation:**
    - Large positive values: High linear dependence among variables
    - Values near zero: Near-independence of variables
    - Large negative values: Multicollinearity, near-singular correlation matrix

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.linalg import correlation_log_determinant
    >>>
    >>> # Independent variables
    >>> A_indep = np.random.randn(100, 3)
    >>> log_det_indep = correlation_log_determinant(A_indep)
    >>> print(f"Independent variables log-det: {log_det_indep:.3f}")
    >>>
    >>> # Correlated variables
    >>> A_corr = np.random.randn(100, 1)
    >>> A_corr = np.hstack([A_corr, A_corr + 0.1*np.random.randn(100, 1)])
    >>> log_det_corr = correlation_log_determinant(A_corr)
    >>> print(f"Correlated variables log-det: {log_det_corr:.3f}")
    >>>
    >>> # Expected: log_det_corr < log_det_indep due to correlation

    See Also
    --------
    numpy.corrcoef : Compute correlation coefficients
    numpy.linalg.slogdet : Compute sign and log-determinant
    """
    if A.shape[1] == 0:
        return 0.0
    C = np.corrcoef(A.T)
    if C.ndim == 0:
        return 0.0

    # Handle numerical issues with correlation matrix
    sign, logdet = np.linalg.slogdet(C)

    # If the matrix is singular (sign=0), return a large negative value instead of -inf
    if sign == 0 or not np.isfinite(logdet):
        return -1000.0  # Large negative value for singular matrices

    return logdet


def subnetwork(G: nx.MultiDiGraph, lag: int) -> nx.DiGraph:
    r"""
    Extract a subgraph containing only edges at a specific lag.

    The general return value from `discover_network` is a NetworkX MultiDiGraph
    with lag, p-value, and cmi encoded as edge attributes. This method returns a
    DiGraph containing only edges at the specified lag value.

    Since the input is a MultiDiGraph, bidirectional connections at the same lag
    are represented as two separate directed edges: one from i to j and one from
    j to i.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The causal network graph from discover_network with edge attributes
        'lag', 'cmi', and 'p_value'.
    lag : int
        The time lag to extract. Only edges with this lag value will be included.

    Returns
    -------
    H : nx.DiGraph
        A directed graph containing only the edges at the specified lag.
        Edge attributes 'cmi' and 'p_value' are preserved.

    Examples
    --------
    >>> import networkx as nx
    >>> from causationentropy.core.linalg import subnetwork
    >>>
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
    >>> G.add_edge(1, 2, lag=2, cmi=0.3, p_value=0.05)
    >>>
    >>> H1 = subnetwork(G, lag=1)
    >>> H1.number_of_edges()
    1
    """
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, k, data in G.edges(keys=True, data=True):
        if data.get("lag") == lag:
            cmi = data.get("cmi", 0.0)
            p_value = data.get("p_value", 1.0)
            H.add_edge(u, v, cmi=cmi, p_value=p_value)
    return H


def companion_matrix(G: nx.MultiDiGraph) -> np.ndarray:
    r"""
    Construct the block companion matrix for a causal network.

    The purpose of this method is to store the causal graph in a structure that this
    library prefers and is not necessarily the graph theoretical construction.

    The companion matrix is a block-structured matrix used in vector autoregression (VAR)
    and dynamical systems analysis:

    .. math::

        C =
        \begin{bmatrix}
        A^{(1)} & A^{(2)} & \cdots & A^{(K-1)} & A^{(K)} \\
        I       & 0       & \cdots & 0         & 0       \\
        0       & I       & \cdots & 0         & 0       \\
        \vdots  & \vdots  & \ddots & \vdots    & \vdots  \\
        0       & 0       & \cdots & I         & 0
        \end{bmatrix}

    Each :math:`A^{(k)}` is the adjacency matrix of edges with lag = k, and I represents
    an identity matrix of size n x n.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The causal network graph from discover_network. Must contain edge attribute 'lag'.

    Returns
    -------
    C : np.ndarray of shape (n_nodes * max_lag, n_nodes * max_lag)
        The block companion matrix. Returns empty (0, 0) array if max_lag = 0.

    Notes
    -----
    - Nodes are ordered according to NetworkX's default ordering (sorted)
    - Edges with lag=0 are ignored (contemporaneous effects not included)
    - The matrix enables analysis of temporal dynamics via eigenvalue analysis

    Examples
    --------
    >>> import networkx as nx
    >>> import numpy as np
    >>> from causationentropy.core.linalg import companion_matrix
    >>>
    >>> G = nx.MultiDiGraph()
    >>> G.add_nodes_from([0, 1, 2])
    >>> G.add_edge(0, 1, lag=1, cmi=0.5, p_value=0.01)
    >>> G.add_edge(1, 2, lag=2, cmi=0.3, p_value=0.05)
    >>>
    >>> C = companion_matrix(G)
    >>> print(C.shape)
    (6, 6)
    """
    # Get max lag from all edges
    max_lag = max((data.get("lag", 0) for _, _, data in G.edges(data=True)), default=0)

    if max_lag == 0:
        return np.zeros((0, 0))

    n_nodes = G.number_of_nodes()

    # Block companion matrix: (n*K) x (n*K) for VAR(K) model
    C = np.zeros((n_nodes * max_lag, n_nodes * max_lag))

    # Top row: Fill with adjacency matrices A^(1), A^(2), ..., A^(K)
    for lag in range(1, max_lag + 1):
        H = subnetwork(G, lag)
        A_by_lag = nx.adjacency_matrix(H).toarray()
        start_col = (lag - 1) * n_nodes
        C[0:n_nodes, start_col : start_col + n_nodes] = A_by_lag

    # Fill in block diagonal identity matrices for temporal shift structure
    for k in range(1, max_lag):
        r0 = k * n_nodes
        c0 = (k - 1) * n_nodes
        C[r0 : r0 + n_nodes, c0 : c0 + n_nodes] = np.eye(n_nodes)

    return C
