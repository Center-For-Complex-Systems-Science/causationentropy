import copy
from typing import Dict, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoLarsIC

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)


def discover_network(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = "standard",
    information: str = "gaussian",
    max_lag: int = 5,
    alpha_forward: float = 0.05,
    alpha_backward: float = 0.05,
    metric: str = "euclidean",
    bandwidth="silverman",
    k_means: int = 5,
    n_shuffles: int = 200,
    n_jobs=-1,
) -> nx.MultiDiGraph:
    r"""
    Infer a causal graph via Optimal Causation Entropy (oCSE).

    This function implements the optimal Causation Entropy algorithm for causal network discovery
    from multivariate time series data. The algorithm uses conditional mutual information to
    identify causal relationships between variables across different time lags.

    The core principle is based on the Causation Entropy framework, which quantifies causal
    relationships using information-theoretic measures. For variables :math:`X_i` and :math:`X_j`
    with lag :math:`\\tau`, the conditional mutual information is computed as:

    .. math::

        I\!\left(X_j^{(t-\tau)}; X_i^{(t)} \,\middle|\, \mathbf{Z}_i^{(t)}\right)
        \;=\;
        H\!\left(X_i^{(t)} \,\middle|\, \mathbf{Z}_i^{(t)}\right)
        \;-\;
        H\!\left(X_i^{(t)} \,\middle|\, X_j^{(t-\tau)}, \mathbf{Z}_i^{(t)}\right)

    where :math:`\mathbf{Z}_i^{(t)}` represents the conditioning set for variable :math:`i` at time :math:`t`.

    The algorithm proceeds in two main phases:

    1. **Forward Selection**: Iteratively selects predictors that maximize conditional mutual
       information with the target variable, conditioned on already selected predictors.

    2. **Backward Elimination**: Removes predictors that do not maintain statistical significance
       when conditioned on all other selected predictors.

    Statistical significance is assessed via permutation tests, where the null hypothesis assumes
    no causal relationship exists between variables.

    Parameters
    ----------
    data : array-like of shape (T, n) or DataFrame
        Multivariate time series data where T is the number of time points and n is the number
        of variables. Variables correspond to columns.
    method : str, default='standard'
        Causal discovery algorithm variant. Options:

        - 'standard': Uses initial conditioning set of lagged target variables
        - 'alternative': No initial conditioning set
        - 'information_lasso': Information-theoretic variant with LASSO regularization
        - 'lasso': Pure LASSO-based selection
    information : str, default='gaussian'
        Information measure estimator type. Options:

        - 'gaussian': Assumes Gaussian distributions
        - 'knn': k-nearest neighbor estimator
        - 'kde': Kernel density estimation
        - 'geometric_knn': Geometric mean k-NN estimator
        - 'poisson': Poisson distribution assumption
    max_lag : int, default=5
        Maximum time lag to consider in causal relationships. The algorithm examines
        lags from 1 to max_lag (inclusive).
    k_means : int, default=5
        Number of clusters for k-means based estimators (when applicable).
    alpha_forward : float, default=0.05
        Significance level for forward selection permutation tests. Lower values
        require stronger evidence for causal relationships.
    alpha_backward : float, default=0.05
        Significance level for backward elimination permutation tests.
    metric : str, default='euclidean'
        Distance metric for k-NN based estimators.
    n_shuffles : int, default=200
        Number of permutations for statistical significance testing. Higher values
        provide more accurate p-value estimates but increase computational cost.
    n_jobs : int, default=-1
        Number of parallel jobs for computation. -1 uses all available processors.

    Returns
    -------
    G : networkx.MultiDiGraph
        Multi-directed graph representing the discovered causal network. Nodes correspond to
        variables and edges represent causal relationships. Multiple edges between the same
        node pair represent relationships at different time lags. Edge attributes include:

        - 'lag': Time delay :math:`\tau` of the causal relationship
        - 'cmi': Conditional mutual information value for this edge
        - 'p_value': Empirical p-value from permutation test

    Raises
    ------
    NotImplementedError
        If an unsupported method or information type is specified.
    ValueError
        If the time series is too short for the chosen max_lag.

    Notes
    -----
    The algorithm's computational complexity is approximately :math:`O(T \cdot n^2 \cdot \tau_{max} \cdot N_{shuffle})`,
    where :math:`T` is the time series length, :math:`n` is the number of variables,
    :math:`\tau_{max}` is the maximum lag, and :math:`N_{shuffle}` is the number of permutations.

    For optimal performance with high-dimensional data, consider:

    - Reducing max_lag for shorter time series
    - Using 'gaussian' information type for continuous data
    - Adjusting n_shuffles based on desired statistical precision

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.discovery import discover_network
    >>>
    >>> # Generate sample time series data
    >>> T, n = 1000, 3
    >>> data = np.random.randn(T, n)
    >>>
    >>> # Discover causal network
    >>> G = discover_network(data, max_lag=3, alpha_forward=0.01)

    References
    ----------
    .. [1] Sun, J., Bollt, E.M. Causation entropy identifies indirect influences, dominance of
           neighbors and anticipatory couplings. Physica D 267, 49-57 (2014).

    .. [2] Schreiber, T. Measuring information transfer. Physical Review Letters 85, 461 (2000).
    """
    rng = np.random.default_rng(42)

    if method not in ["standard", "alternative", "information_lasso", "lasso"]:
        raise NotImplementedError(f"discover_network: method={method} not supported.")
    supported_information_types = ["gaussian", "knn", "kde", "geometric_knn", "poisson"]
    if information not in supported_information_types:
        raise NotImplementedError(
            f"discover_network: information={information} not supported. "
            f"Supported types: {supported_information_types}"
        )

    # Convert DataFrame to ndarray while keeping column labels
    if isinstance(data, pd.DataFrame):
        series = data.values
        var_names = list(data.columns)
    else:
        series = np.asarray(data)
        var_names = [f"X{i}" for i in range(series.shape[1])]

    T, n = series.shape
    if T <= max_lag + 2:
        raise ValueError("Time series too short for chosen max_lag.")

    indices = np.arange(max_lag, T - 1)
    # Step 1: Create lagged predictors and corresponding labels
    X_lagged = []
    feature_names = []  # stores (var_idx, lag)
    for j in range(n):  # variable index
        for tau in range(1, max_lag + 1):  # lag from 1 to max_lag
            col = series[max_lag - tau : T - tau, j]
            X_lagged.append(col)
            feature_names.append((j, tau))

    X_lagged = np.column_stack(X_lagged)  # shape: (T - max_lag, n * max_lag)
    Y_all = series[max_lag:, :]  # aligned target matrix

    # Step 2: Initialize causal graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(var_names)

    # Step 3: Loop over each variable and infer parents from lagged predictors
    for i in range(n):
        print(f"Estimating edges for node {i} ({var_names[i]})")

        Y = Y_all[:, [i]]  # shape: (T - max_lag, 1)
        if method == "standard":
            Z_init = []
            for tau in range(1, max_lag + 1):
                Z_init.append(series[max_lag - tau : T - tau, i])  # lagged Y_i
            Z_init = np.column_stack(Z_init)  # shape: (T - max_lag, max_lag)
            S = standard_optimal_causation_entropy(
                X_lagged,
                Y,
                Z_init,
                rng,
                alpha_forward,
                alpha_backward,
                n_shuffles,
                information,
                metric,
                k_means,
                bandwidth,
            )
        if method == "alternative":
            S = alternative_optimal_causation_entropy(
                X_lagged,
                Y,
                rng,
                alpha_forward,
                alpha_backward,
                n_shuffles,
                information,
                metric,
                k_means,
                bandwidth,
            )
        if method == "information_lasso":
            S = information_lasso_optimal_causation_entropy(X_lagged, Y, rng)
        if method == "lasso":
            S = lasso_optimal_causation_entropy(X_lagged, Y, rng)
        for s in S:
            src_var, src_lag = feature_names[s]

            # Compute CMI and p-value for this edge
            X_predictor = X_lagged[:, [s]]  # predictor at this lag
            Y_target = Y  # target variable

            # Conditioning set: all other selected predictors for this target
            other_selected = [idx for idx in S if idx != s]
            Z_cond = X_lagged[:, other_selected] if other_selected else None

            # Compute conditional mutual information
            cmi = conditional_mutual_information(
                X_predictor,
                Y_target,
                Z_cond,
                method=information,
                metric=metric,
                k=k_means,
                bandwidth=bandwidth,
            )

            # Compute p-value using shuffle test
            test_result = shuffle_test(
                X_predictor,
                Y_target,
                Z_cond,
                cmi,
                alpha=alpha_backward,  # Use backward elimination alpha
                rng=rng,
                n_shuffles=n_shuffles,
                information=information,
                metric=metric,
                k_means=k_means,
                bandwidth=bandwidth,
            )

            G.add_edge(
                var_names[src_var],
                var_names[i],
                lag=src_lag,
                cmi=cmi,
                p_value=test_result["P_value"],
            )

    return G


def standard_optimal_causation_entropy(
    X,
    Y,
    Z_init,
    rng,
    alpha1=0.05,
    alpha2=0.05,
    n_shuffles=200,
    information="gaussian",
    metric="euclidean",
    k_means=5,
    bandwidth="silverman",
):
    r"""
    Execute the standard optimal Causation Entropy algorithm with initial conditioning set.

    This function implements the standard oCSE algorithm that begins with a non-empty
    initial conditioning set (typically lagged target variables). The algorithm combines
    forward selection and backward elimination phases to identify significant causal predictors.

    The conditional mutual information for candidate predictor :math:`X_j` given current
    conditioning set :math:`\mathbf{Z}` is:

    .. math::

        I(X_j; Y | \mathbf{Z}) = \sum_{x_j,y,\mathbf{z}} p(x_j,y,\mathbf{z}) \log \frac{p(x_j,y|\mathbf{z})}{p(x_j|\mathbf{z})p(y|\mathbf{z})}

    Parameters
    ----------
    X : array-like of shape (T, n)
        Predictor variables matrix.
    Y : array-like of shape (T, 1)
        Target variable column.
    Z_init : array-like of shape (T, p)
        Initial conditioning set (e.g., lagged target values).
    rng : numpy.random.Generator
        Random number generator for reproducible results.
    alpha1 : float, default=0.05
        Significance level for forward selection phase.
    alpha2 : float, default=0.05
        Significance level for backward elimination phase.
    n_shuffles : int, default=200
        Number of permutations for statistical testing.
    information : str, default='gaussian'
        Information measure estimator type.

    Returns
    -------
    S : list of int
        Indices of selected predictor variables that passed both forward and backward phases.
    """

    forward_pass = standard_forward(
        X, Y, Z_init, rng, alpha1, n_shuffles, information, metric, k_means, bandwidth
    )

    S = backward(
        X,
        Y,
        forward_pass,
        rng,
        alpha2,
        n_shuffles,
        information,
        metric,
        k_means,
        bandwidth,
    )

    return S


def alternative_optimal_causation_entropy(
    X,
    Y,
    rng,
    alpha1=0.05,
    alpha2=0.05,
    n_shuffles=200,
    information="gaussian",
    metric="euclidean",
    k_means=5,
    bandwidth="silverman",
):
    """
    Execute the alternative optimal Causation Entropy algorithm without initial conditioning.

    This variant of the oCSE algorithm starts with an empty conditioning set, building
    causal relationships purely from the forward selection process. This approach may
    be more suitable when no prior knowledge about lagged dependencies exists.

    Parameters
    ----------
    X : array-like of shape (T, n)
        Predictor variables matrix.
    Y : array-like of shape (T, 1)
        Target variable column.
    rng : numpy.random.Generator
        Random number generator for reproducible results.
    alpha1 : float, default=0.05
        Significance level for forward selection phase.
    alpha2 : float, default=0.05
        Significance level for backward elimination phase.
    n_shuffles : int, default=200
        Number of permutations for statistical testing.
    information : str, default='gaussian'
        Information measure estimator type.

    Returns
    -------
    S : list of int
        Indices of selected predictor variables.
    """

    forward_pass = alternative_forward(
        X, Y, rng, alpha1, n_shuffles, information, metric, k_means, bandwidth
    )

    S = backward(
        X,
        Y,
        forward_pass,
        rng,
        alpha2,
        n_shuffles,
        information,
        metric,
        k_means,
        bandwidth,
    )

    return S


def information_lasso_optimal_causation_entropy(
    X, Y, rng, criterion="bic", max_lambda=100, cross_val=10, information="gaussian"
):
    """
    Execute information-theoretic variant of oCSE with LASSO regularization.

    This method combines information-theoretic causal discovery with LASSO regularization
    to handle high-dimensional predictor spaces. The approach balances causal relationship
    strength with model complexity.

    Parameters
    ----------
    X : array-like of shape (T, n)
        Predictor variables matrix.
    Y : array-like of shape (T, 1)
        Target variable column.
    rng : numpy.random.Generator
        Random number generator.
    criterion : str, default='bic'
        Information criterion for model selection ('bic' or 'aic').
    max_lambda : int, default=100
        Maximum number of LASSO iterations.
    cross_val : int, default=10
        Cross-validation folds (currently unused).
    information : str, default='gaussian'
        Information measure estimator type.

    Returns
    -------
    S : list of int
        Indices of selected predictor variables.

    Notes
    -----
    This is a simplified implementation that delegates to LASSO. Future versions
    will incorporate information-theoretic weighting into the regularization.
    """

    # This is a simplified implementation - needs proper information-theoretic weighting
    return lasso_optimal_causation_entropy(X, Y, rng, criterion, max_lambda, cross_val)


def lasso_optimal_causation_entropy(
    X, Y, rng, criterion="bic", max_lambda=100, cross_val=10
):
    r"""
    Execute LASSO-based variable selection for causal discovery.

    This method uses LASSO (Least Absolute Shrinkage and Selection Operator) regression
    for variable selection in causal discovery. The LASSO objective function is:

    .. math::

        \min_{\boldsymbol{\beta}} \frac{1}{2n} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||_2^2 + \lambda ||\boldsymbol{\beta}||_1

    where :math:`\lambda` is the regularization parameter that controls sparsity.

    Parameters
    ----------
    X : array-like of shape (T, n)
        Predictor variables matrix.
    Y : array-like of shape (T, 1)
        Target variable column.
    rng : numpy.random.Generator
        Random number generator (unused in current implementation).
    criterion : str, default='bic'
        Information criterion for regularization parameter selection.
    max_lambda : int, default=100
        Maximum number of LASSO iterations.
    cross_val : int, default=10
        Cross-validation folds (currently unused).

    Returns
    -------
    S : list of int
        Indices of variables with non-zero LASSO coefficients.

    Notes
    -----
    Uses LassoLarsIC when the number of samples exceeds the number of predictors plus one,
    otherwise falls back to standard LASSO regression.
    """

    n = X.shape[1]
    if X.shape[0] > n + 1:
        lasso = LassoLarsIC(criterion=criterion, max_iter=max_lambda).fit(
            X, Y.flatten()
        )
    else:
        lasso = Lasso(max_iter=max_lambda).fit(X, Y.flatten())
    S = np.where(lasso.coef_ != 0)[0].tolist()
    return S


def alternative_forward(
    X_full,
    Y,
    rng,
    alpha=0.05,
    n_shuffles=200,
    information="gaussian",
    metric="euclidean",
    k_means=5,
    bandwidth="silverman",
):
    r"""
    Forward selection phase of oCSE without initial conditioning set.

    This function implements the forward selection phase starting with an empty conditioning
    set. At each step, it evaluates the conditional mutual information between each remaining
    candidate predictor and the target, conditioned on already selected predictors.

    The selection criterion at each step is:

    .. math::

        j^* = \arg\max_{j \in \text{candidates}} I(X_j^{(t)}; Y^{(t+\tau)} | \mathbf{S}^{(t)})

    where :math:`\mathbf{S}^{(t)}` represents the current set of selected predictors.

    Parameters
    ----------
    X_full : array-like of shape (T, n)
        Complete predictor matrix containing values at time t.
    Y : array-like of shape (T, 1)
        Target variable column containing values at time t+τ.
    rng : numpy.random.Generator
        Random number generator for permutation tests.
    alpha : float, default=0.05
        Significance level for permutation tests. Predictors must achieve
        conditional mutual information above the (1-α) percentile of the null distribution.
    n_shuffles : int, default=200
        Number of permutations to generate for statistical testing.
    information : str, default='gaussian'
        Information measure estimator type used for conditional mutual information computation.

    Returns
    -------
    S : list of int
        Indices of selected predictor variables that passed the significance test.

    Notes
    -----
    The algorithm terminates when no remaining candidate achieves statistical significance
    or when all candidates have been evaluated. Each selection updates the conditioning
    set for subsequent iterations.
    """
    n = X_full.shape[1]
    candidates = np.arange(n)
    S = []  # selected predictors
    Z = None  # current conditioning set

    while True:
        remaining = np.setdiff1d(candidates, S)
        if remaining.size == 0:
            break

        # 1. evaluate each remaining variable
        ent_values = np.zeros(remaining.size)
        for k, j in enumerate(remaining):
            Xj = X_full[:, [j]]  # keep 2-D shape
            ent_values[k] = conditional_mutual_information(
                Xj,
                Y,
                Z,
                method=information,
                metric=metric,
                k=k_means,
                bandwidth=bandwidth,
            )

        # 2. pick best
        j_best = remaining[ent_values.argmax()]
        X_best = X_full[:, [j_best]]
        mi_best = ent_values.max()

        # 3. permutation (shuffle) test
        passed = shuffle_test(
            X_best,
            Y,
            Z,
            mi_best,
            alpha,
            rng=rng,
            n_shuffles=n_shuffles,
            information=information,
            metric=metric,
            k_means=k_means,
            bandwidth=bandwidth,
        )["Pass"]
        if not passed:
            break

        # 4. accept and update conditioning set
        S.append(j_best)
        Z = X_full[:, S] if len(S) else None

    return S


def standard_forward(
    X_full,
    Y,
    Z_init,
    rng,
    alpha=0.05,
    n_shuffles=200,
    information="gaussian",
    metric="euclidean",
    k_means=5,
    bandwidth="silverman",
):
    r"""
    Standard forward selection phase of oCSE with initial conditioning set.

    This function implements forward selection starting with a non-empty initial conditioning
    set Z_init, typically consisting of lagged values of the target variable. This approach
    incorporates prior knowledge about temporal dependencies in the causal discovery process.

    At each iteration, the algorithm selects the predictor that maximizes conditional mutual
    information with the target, given the current conditioning set:

    .. math::

        j^* = \arg\max_{j \in \text{candidates}} I(X_j^{(t)}; Y^{(t+\tau)} | \mathbf{Z}^{(t)})

    where :math:`\mathbf{Z}^{(t)} = \mathbf{Z}_{\text{init}} \cup \mathbf{S}^{(t)}` combines the initial
    conditioning set with currently selected predictors.

    Parameters
    ----------
    X_full : array-like of shape (T, n)
        Complete predictor matrix at time t.
    Y : array-like of shape (T, 1)
        Target variable at time t+τ.
    Z_init : array-like of shape (T, p)
        Initial conditioning set, typically containing lagged target values.
    rng : numpy.random.Generator
        Random number generator for permutation tests.
    alpha : float, default=0.05
        Forward selection significance threshold for permutation tests.
    n_shuffles : int, default=200
        Number of shuffles for significance testing.
    information : str, default='gaussian'
        Information measure estimator type.

    Returns
    -------
    S : list of int
        Indices of selected predictor variables from X_full.

    Notes
    -----
    The initial conditioning set Z_init remains constant throughout the forward selection,
    while newly selected predictors are added to form the complete conditioning set for
    subsequent iterations.
    """
    n = X_full.shape[1]
    candidates = list(range(n))
    S = []
    Z = Z_init.copy() if Z_init is not None else None

    while candidates:
        # 1. compute CMI for every remaining candidate
        ent_values = np.empty(len(candidates))
        for k, j in enumerate(candidates):
            Xj = X_full[:, [j]]  # (T,1)  keep 2‑D
            ent_values[k] = conditional_mutual_information(
                Xj,
                Y,
                Z,
                method=information,
                metric=metric,
                k=k_means,
                bandwidth=bandwidth,
            )

        # 2. take the arg‑max
        k_best = int(ent_values.argmax())
        j_best = candidates[k_best]
        X_best = X_full[:, [j_best]]
        mi_best = ent_values[k_best]

        # 3. permutation (shuffle) test
        passed = shuffle_test(
            X_best,
            Y,
            Z,
            mi_best,
            alpha=alpha,
            rng=rng,
            n_shuffles=n_shuffles,
            information=information,
            metric=metric,
            k_means=k_means,
            bandwidth=bandwidth,
        )["Pass"]

        if not passed:
            candidates.pop(k_best)
            continue

        # 4. accept predictor, update conditioning set / candidate list
        S.append(j_best)
        Z = np.hstack([Z, X_best]) if Z is not None else X_best
        candidates.pop(k_best)

    return S


def backward(
    X_full,
    Y,
    S_init,
    rng,
    alpha=0.05,
    n_shuffles=200,
    information="gaussian",
    metric="euclidean",
    k_means=5,
    bandwidth="silverman",
):
    r"""
    Backward elimination phase of optimal Causation Entropy.

    This function performs backward elimination to remove spurious causal relationships
    identified during forward selection. For each predictor selected in the forward phase,
    it tests whether the predictor maintains statistical significance when conditioned on
    all other selected predictors.

    For each predictor :math:`X_j` in the selected set, the test evaluates:

    .. math::

        I(X_j^{(t)}; Y^{(t+\tau)} | \mathbf{S}_{-j}^{(t)}) > \text{threshold}

    where :math:`\mathbf{S}_{-j}^{(t)}` represents all selected predictors except :math:`X_j`.

    Parameters
    ----------
    X_full : array-like of shape (T, n)
        Complete predictor matrix at time t, unchanged throughout the process.
    Y : array-like of shape (T, 1)
        Target variable at time t+τ.
    S_init : list of int
        Indices of predictor variables selected during the forward phase.
    rng : numpy.random.Generator
        Random number generator for permutation order and significance testing.
    alpha : float, default=0.05
        Significance level for backward elimination tests.
    n_shuffles : int, default=200
        Number of permutation shuffles for statistical testing.
    information : str, default='gaussian'
        Information measure estimator type.

    Returns
    -------
    S_final : list of int
        Subset of S_init containing predictors that maintained statistical significance
        during backward elimination.

    Notes
    -----
    Predictors are evaluated in random order to avoid selection bias. A predictor is
    removed if its conditional mutual information with the target, given all other
    selected predictors, falls below the significance threshold.

    The backward phase is essential for controlling false positive rates in causal
    discovery, as forward selection may include predictors that become redundant
    when considered alongside other selected variables.
    """
    S = copy.deepcopy(S_init)  # working copy

    for j in rng.permutation(S_init):
        # conditioning set Z = S \ {j}
        Z = X_full[:, [k for k in S if k != j]] if len(S) > 1 else None

        Xj = X_full[:, [j]]
        cmij = conditional_mutual_information(
            Xj, Y, Z, method=information, metric=metric, k=k_means, bandwidth=bandwidth
        )

        passed = shuffle_test(
            Xj,
            Y,
            Z,
            cmij,
            alpha=alpha,
            rng=rng,
            n_shuffles=n_shuffles,
            information=information,
            metric=metric,
            k_means=k_means,
            bandwidth=bandwidth,
        )["Pass"]
        if not passed:
            S.remove(j)  # prune j

    return S


def shuffle_test(
    X,
    Y,
    Z,
    observed_cmi,
    alpha=0.05,
    n_shuffles=500,
    rng=None,
    information="gaussian",
    metric="euclidean",
    k_means=5,
    bandwidth="silverman",
):
    r"""
    Permutation test for conditional mutual information significance.

    This function performs a permutation test to assess the statistical significance of
    the conditional mutual information I(X;Y|Z). The test generates a null distribution
    by computing conditional mutual information on permuted versions of the predictor X,
    while keeping Y and Z unchanged.

    The null hypothesis is that X and Y are conditionally independent given Z:

    .. math::

        H_0: I(X; Y | Z) = 0

    The test statistic follows the distribution:

    .. math::

        \text{CMI}_{\text{null}} \sim \text{Distribution under } H_0

    Statistical significance is assessed by comparing the observed conditional mutual
    information to the (1-α) percentile of the null distribution.

    Parameters
    ----------
    X : array-like of shape (T, k_x)
        Predictor variable(s) under test. Must be 2-D even when k_x=1.
    Y : array-like of shape (T, 1)
        Target variable column.
    Z : array-like of shape (T, k_z) or None
        Current conditioning set. If None, tests marginal mutual information.
    observed_cmi : float
        Conditional mutual information value computed on original (unshuffled) data.
    alpha : float, default=0.05
        Significance level for the test. Lower values require stronger evidence.
    n_shuffles : int, default=500
        Number of random permutations to generate for the null distribution.
    rng : int, numpy.random.Generator, or None
        Random number generator or seed for reproducible results.
    information : str, default='gaussian'
        Information measure estimator type used for conditional mutual information.

    Returns
    -------
    result : dict
        Dictionary containing test results:

        - 'Threshold': float, the (1-α) percentile of the null distribution
        - 'Value': float, the observed conditional mutual information value
        - 'Pass': bool, True if observed_cmi >= threshold (statistically significant)
        - 'P_value': float, empirical p-value (proportion of null values >= observed)

    Notes
    -----
    The permutation test is based on the assumption that under the null hypothesis,
    the predictor X is exchangeable with respect to the target Y when conditioned on Z.
    This provides a non-parametric approach to significance testing that does not
    require distributional assumptions.

    For computational efficiency, consider reducing n_shuffles for preliminary analyses,
    though this may reduce the precision of p-value estimates.

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.discovery import shuffle_test
    >>>
    >>> # Generate sample data
    >>> X = np.random.randn(100, 1)
    >>> Y = np.random.randn(100, 1)
    >>> Z = np.random.randn(100, 2)
    >>> observed = 0.15
    >>>
    >>> # Perform permutation test
    >>> result = shuffle_test(X, Y, Z, observed, alpha=0.05, n_shuffles=1000)
    >>> print(f"Significant: {result['Pass']}, p-value ≈ {1 - result['Value']/result['Threshold']:.3f}")
    """
    rng = np.random.default_rng(rng)
    null_cmi = np.empty(n_shuffles)

    for i in range(n_shuffles):
        X_perm = X[rng.permutation(len(X)), :]  # shuffle rows
        null_cmi[i] = conditional_mutual_information(
            X_perm,
            Y,
            Z,
            method=information,
            metric=metric,
            k=k_means,
            bandwidth=bandwidth,
        )

    threshold = np.percentile(null_cmi, 100 * (1 - alpha))
    # Calculate p-value: proportion of null values >= observed value
    p_value = np.mean(null_cmi >= observed_cmi)
    return {
        "Threshold": threshold,
        "Value": observed_cmi,
        "Pass": observed_cmi >= threshold,
        "P_value": p_value,
    }


if __name__ == "__main__":
    from causationentropy.datasets.synthetic import logisic_dynamics

    data, A = logisic_dynamics()
    G = discover_network(data)
    print(data.shape)
    print(G)
