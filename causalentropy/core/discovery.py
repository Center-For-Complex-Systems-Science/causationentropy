import copy
import networkx as nx
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Tuple

from sklearn.linear_model import LassoLarsIC, Lasso

from causalentropy.core.information.conditional_mutual_information import conditional_mutual_information


def discover_network(
        data: Union[np.ndarray, pd.DataFrame],
        method: str = 'standard',
        information: str = "gaussian",
        max_lag: int = 5,
        k_means: int = 5,
        alpha_forward: float = 0.05,
        alpha_backward: float = 0.05,
        metric: str = "euclidean",
        n_shuffles: int = 200,
        n_jobs=-1,
) -> nx.DiGraph:
    """
    Infer a causal graph via Optimal Causation Entropy.

    Parameters
    ----------
    data : (T, n) ndarray or DataFrame
        Multivariate time series (variables = columns).
    method : str
        Entropy estimator.  Currently only 'gaussian' is implemented.
    information: str
        The information metric used.
    max_lag : int
        Consider lags 0 … max_lag (inclusive).
    alpha_forward : float
        α-level for permutation significance tests.
    alpha_backward : float
        α-level for permutation significance tests.
    n_shuffles: : int
        Number of permutations for each CMI test.

    Returns
    -------
    G : nx.DiGraph
        Directed graph with edge attributes
            lag   : int     (delay τ)
            cmi   : float   (conditional MI value at selection)
    """
    rng = np.random.default_rng(42)

    if method not in ["standard", 'alternative', 'information_lasso', "lasso"]:
        raise NotImplementedError(f"discover_network: method={method} not supported.")
    if information not in ["gaussian", "knn", "kde", "poisson"]:
        raise NotImplementedError(f"discover_network: information={information} not supported.")

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
    lagged: Dict[Tuple[int, int], np.ndarray] = {}
    for j in range(n):
        for tau in range(max_lag + 1):
            lagged[(j, tau)] = series[indices - tau, j]

    G = nx.DiGraph()
    G.add_nodes_from(var_names)

    # Step 1: Create lagged predictors and corresponding labels
    X_lagged = []
    feature_names = []  # stores (var_idx, lag)
    for j in range(n):  # variable index
        for tau in range(1, max_lag + 1):  # lag from 1 to max_lag
            col = series[max_lag - tau: T - tau, j]
            X_lagged.append(col)
            feature_names.append((j, tau))

    X_lagged = np.column_stack(X_lagged)  # shape: (T - max_lag, n * max_lag)
    Y_all = series[max_lag:, :]  # aligned target matrix

    # Step 2: Initialize causal graph
    G = nx.DiGraph()
    G.add_nodes_from(var_names)

    # Step 3: Loop over each variable and infer parents from lagged predictors
    for i in range(n):
        print(f"Estimating edges for node {i} ({var_names[i]})")

        Y = Y_all[:, [i]]  # shape: (T - max_lag, 1)
        if method == 'standard':
            Z_init = []
            for tau in range(1, max_lag + 1):
                Z_init.append(series[max_lag - tau:T - tau, i])  # lagged Y_i
            Z_init = np.column_stack(Z_init)  # shape: (T - max_lag, max_lag)
            S = standard_optimal_causation_entropy(X_lagged, Y, Z_init, rng, alpha_forward, alpha_backward, n_shuffles)
        if method == 'alternative':
            S = alternative_optimal_causation_entropy(X_lagged, Y, rng, alpha_forward, alpha_backward, n_shuffles)
        if method == 'information_lasso':
            S = information_lasso_optimal_causation_entropy(X_lagged, Y, rng, alpha_forward, alpha_backward, n_shuffles)
        if method == 'lasso':
            S = lasso_optimal_causation_entropy(X_lagged, Y, rng, alpha_forward, alpha_backward, n_shuffles)
        for s in S:
            src_var, src_lag = feature_names[s]
            G.add_edge(var_names[src_var], var_names[i], lag=src_lag)

    return G


def standard_optimal_causation_entropy(X, Y, Z_init, rng, alpha1=0.05, alpha2=0.05, n_shuffles=200, information='gaussian'):
    """Run the standard version of the oCSE algorithm. Note defaults to the
           KernelDensity plugin estimator if the method is not specified"""



    forward_pass = standard_forward(X, Y, Z_init, rng, alpha1, n_shuffles, information)

    S = backward(X, Y, forward_pass, rng, alpha2, n_shuffles, information)

    return S

def alternative_optimal_causation_entropy(X, Y, rng, alpha1=0.05, alpha2=0.05, n_shuffles=200, information='gaussian'):
    """Run the standard version of the oCSE algorithm. Note defaults to the
           KernelDensity plugin estimator if the method is not specified"""

    forward_pass = alternative_forward(X, Y, rng, alpha1, n_shuffles, information)

    S = backward(X, Y, forward_pass, rng, alpha2, n_shuffles, information)

    return S



def information_lasso_optimal_causation_entropy(X, Y, rng, criterion='bic', max_lambda=100, cross_val=10, information='gaussian'):
    """Run the standard version of the oCSE algorithm. Note defaults to the
           KernelDensity plugin estimator if the method is not specified"""

    X_entropy = conditional_mutual_information(X, information)
    return lasso_optimal_causation_entropy(X*X_entropy, Y, rng, criterion, max_lambda, cross_val)


def lasso_optimal_causation_entropy(X, Y, rng, criterion='bic', max_lambda=100, cross_val=10):
    """Run the standard version of the oCSE algorithm. Note defaults to the
           KernelDensity plugin estimator if the method is not specified"""

    n = X.shape[1]
    if X.shape[0] > n + 1:
        lasso = LassoLarsIC(criterion=criterion, max_iter=max_lambda).fit(X, Y.flatten())
    else:
        lasso = Lasso(criterion=criterion, max_lamba=max_lambda)
    S = np.where(lasso.coef_ != 0)
    return S


def alternative_forward(X_full, Y, rng, alpha=0.05, n_shuffles=200, information='gaussian'):
    """
    Forward step of oCSE *without* hidden state.

    Parameters
    ----------
    X_full : (T, n) predictor matrix  (values at time t)
    Y      : (T, 1) target column     (values at t+τ)
    alpha  : shuffle‐test significance level
    n_shuffles: int
        Number of shuffles for the shuffle test.
    method: str
        Conditional mutual information type
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
            ent_values[k] = conditional_mutual_information(Xj, Y, Z, information)

        # 2. pick best
        j_best = remaining[ent_values.argmax()]
        X_best = X_full[:, [j_best]]
        mi_best = ent_values.max()

        # 3. permutation (shuffle) test
        passed = shuffle_test(X_best, Y, Z, mi_best, alpha, rng=rng, n_shuffles=n_shuffles)['Pass']
        if not passed:
            break

        # 4. accept and update conditioning set
        S.append(j_best)
        Z = X_full[:, S] if len(S) else None

    return S


def standard_forward(X_full, Y, Z_init, rng, alpha=0.05, n_shuffles=200, information='gaussian'):
    """
    Standard forward oCSE with a non-empty initial conditioning set Z_init.

    Parameters
    ----------
    X_full : (T, n) predictors at time t
    Y      : (T, 1) target at time t+τ
    Z_init : (T, p) initial conditioning set (e.g., lagged Y)
    alpha: float
        forward significance threshold
    n_shuffles: int
        Number of shuffles for shuffle test.
    """
    n = X_full.shape[1]
    candidates = np.arange(n)
    S = []  # selected predictors
    Z = Z_init.copy() if Z_init is not None else None

    while True:
        remaining = np.setdiff1d(candidates, S)
        if remaining.size == 0:
            break

        # 1. evaluate each remaining variable
        ent_values = np.zeros(remaining.size)
        for k, j in enumerate(remaining):
            Xj = X_full[:, [j]]
            ent_values[k] = conditional_mutual_information(Xj, Y, Z, information)

        # 2. pick best
        j_best = remaining[ent_values.argmax()]
        X_best = X_full[:, [j_best]]
        mi_best = ent_values.max()

        # 3. permutation test
        passed = shuffle_test(X_best, Y, Z, mi_best, alpha, rng=rng, n_shuffles=n_shuffles)['Pass']
        if not passed:
            break

        # 4. accept and update conditioning set
        S.append(j_best)
        Z = np.hstack([Z, X_best]) if Z is not None else X_best

    return S


def backward(X_full, Y, S_init, rng, alpha=0.05, n_shuffles=200, information='gaussian'):
    """
    Backward pruning step of oCSE.

    Parameters
    ----------
    X_full : (T, n) ndarray
        Predictor matrix at time t (unchanged throughout).
    Y : (T, 1) ndarray
        Target variable at time t+τ.
    S_init : list[int]
        Predictor indices already selected by the forward pass.
    alpha : float
        Significance level of the shuffle test.
    n_shuffles: int
        Number of permutation test shuffles.
    Returns
    -------
    S_final : list[int]
        Subset of S_init that passed the backward significance test.
    """  # for permutation order
    S = copy.deepcopy(S_init)  # working copy

    for j in rng.permutation(S_init):
        # conditioning set Z = S \ {j}
        Z = X_full[:, [k for k in S if k != j]] if len(S) > 1 else None

        Xj = X_full[:, [j]]
        cmij = conditional_mutual_information(Xj, Y, Z, information)

        passed = shuffle_test(Xj, Y, Z, cmij, alpha=alpha, rng=rng, n_shuffles=n_shuffles)['Pass']
        if not passed:
            S.remove(j)  # prune j

    return S


def shuffle_test(X, Y, Z, observed_cmi, alpha=0.05, n_shuffles=500, rng=None):
    """
    Permutation test for I(X;Y|Z).

    Parameters
    ----------
    X : (T, kx) array   – predictor(s) under test (2-D even if kx=1)
    Y : (T, 1) array    – target column
    Z : (T, kz) array or None – current conditioning set
    observed_cmi : float
        The CMI value computed on the original (unshuffled) data.
    alpha : float
        Significance level (default 0.05).
    n_shuffles : int
        Number of random permutations.
    rng : int or np.random.Generator or None
        Seed for reproducibility.

    Returns
    -------
    dict with keys {'Threshold', 'Value', 'Pass'}
    """
    rng = np.random.default_rng(rng)
    null_cmi = np.empty(n_shuffles)

    for i in range(n_shuffles):
        X_perm = X[rng.permutation(len(X)), :]  # shuffle rows
        null_cmi[i] = conditional_mutual_information(X_perm, Y, Z)

    threshold = np.percentile(null_cmi, 100 * (1 - alpha))
    return {
        "Threshold": threshold,
        "Value": observed_cmi,
        "Pass": observed_cmi >= threshold,
    }


if __name__ == '__main__':
    from causalentropy.datasets.synthetic import logisic_dynamics

    data, A = logisic_dynamics()
    G = discover_network(data)
    print(data.shape)
    print(G)
