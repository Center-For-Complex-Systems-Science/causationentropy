# Copyright 2025 Kevin Slote
# SPDX-License-Identifier: MIT
import copy
from typing import Union

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
    early_stopping: bool = True,
    min_shuffles: int = 50,
    confidence: float = 0.95,
    kd_tree: bool = True,
) -> nx.MultiDiGraph:
    r"""
    Infer a causal graph via Optimal Causation Entropy (oCSE).
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
                early_stopping,
                min_shuffles,
                confidence,
                kd_tree,
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
                early_stopping,
                min_shuffles,
                confidence,
                kd_tree,
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
                kd_tree=kd_tree,
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
                early_stopping=early_stopping,
                min_shuffles=min_shuffles,
                confidence=confidence,
                kd_tree=kd_tree,
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
    early_stopping=True,
    min_shuffles=50,
    confidence=0.95,
    kd_tree=True,
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
        X,
        Y,
        Z_init,
        rng,
        alpha1,
        n_shuffles,
        information,
        metric,
        k_means,
        bandwidth,
        early_stopping,
        min_shuffles,
        confidence,
        kd_tree,
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
        early_stopping,
        min_shuffles,
        confidence,
        kd_tree,
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
    early_stopping=True,
    min_shuffles=50,
    confidence=0.95,
    kd_tree=True,
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
        X,
        Y,
        rng,
        alpha1,
        n_shuffles,
        information,
        metric,
        k_means,
        bandwidth,
        early_stopping,
        min_shuffles,
        confidence,
        kd_tree,
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
        early_stopping,
        min_shuffles,
        confidence,
        kd_tree,
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
    early_stopping=True,
    min_shuffles=50,
    confidence=0.95,
    kd_tree=True,
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
                kd_tree=kd_tree,
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
            early_stopping=early_stopping,
            min_shuffles=min_shuffles,
            confidence=confidence,
            kd_tree=kd_tree,
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
    early_stopping=True,
    min_shuffles=50,
    confidence=0.95,
    kd_tree=True,
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
                kd_tree=kd_tree,
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
            early_stopping=early_stopping,
            min_shuffles=min_shuffles,
            confidence=confidence,
            kd_tree=kd_tree,
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
    early_stopping=True,
    min_shuffles=50,
    confidence=0.95,
    kd_tree=True,
):
    r"""
    Backward elimination phase of optimal Causation Entropy.
    """
    S = copy.deepcopy(S_init)  # working copy

    for j in rng.permutation(S_init):
        # conditioning set Z = S \ {j}
        Z = X_full[:, [k for k in S if k != j]] if len(S) > 1 else None

        Xj = X_full[:, [j]]
        cmij = conditional_mutual_information(
            Xj,
            Y,
            Z,
            method=information,
            metric=metric,
            k=k_means,
            bandwidth=bandwidth,
            kd_tree=kd_tree,
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
            early_stopping=early_stopping,
            min_shuffles=min_shuffles,
            confidence=confidence,
            kd_tree=kd_tree,
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
    early_stopping=True,
    min_shuffles=50,
    confidence=0.95,
    kd_tree=True,
):
    r"""
    Permutation test for conditional mutual information significance.
    """
    from scipy import stats

    from causationentropy.core.information.conditional_mutual_information import (
        cached_detcorr,
    )

    rng = np.random.default_rng(rng)
    null_cmi = []

    # Variables for early stopping
    n_shuffles_used = 0

    if information == "gaussian" and Z is not None:
        # Pre-compute invariant parts for Gaussian case
        log_det_Z = cached_detcorr(Z)
        log_det_YZ = cached_detcorr(np.hstack((Y, Z)))

        for i in range(n_shuffles):
            X_perm = X[rng.permutation(len(X)), :]  # shuffle rows
            log_det_XZ_perm = cached_detcorr(np.hstack((X_perm, Z)))
            log_det_XYZ_perm = cached_detcorr(np.hstack((X_perm, Y, Z)))
            null_val = 0.5 * (
                log_det_XZ_perm + log_det_YZ - log_det_Z - log_det_XYZ_perm
            )
            null_cmi.append(null_val)
            n_shuffles_used = i + 1

            # Early stopping logic...
            if early_stopping and n_shuffles_used >= min_shuffles:
                n_greater_equal = np.sum(np.array(null_cmi) >= observed_cmi)
                current_p_value = n_greater_equal / n_shuffles_used
                if current_p_value <= alpha / 10:
                    binom_p = stats.binom.cdf(n_greater_equal, n_shuffles_used, alpha)
                    if binom_p >= confidence:
                        break
                elif current_p_value >= alpha * 3:
                    binom_p = 1 - stats.binom.cdf(
                        n_greater_equal - 1, n_shuffles_used, alpha
                    )
                    if binom_p >= confidence:
                        break

    else:
        # Original implementation for other information methods
        for i in range(n_shuffles):
            X_perm = X[rng.permutation(len(X)), :]  # shuffle rows
            null_val = conditional_mutual_information(
                X_perm,
                Y,
                Z,
                method=information,
                metric=metric,
                k=k_means,
                bandwidth=bandwidth,
                kd_tree=kd_tree,
            )
            null_cmi.append(null_val)
            n_shuffles_used = i + 1

            # Early stopping logic...
            if early_stopping and n_shuffles_used >= min_shuffles:
                n_greater_equal = np.sum(np.array(null_cmi) >= observed_cmi)
                current_p_value = n_greater_equal / n_shuffles_used
                if current_p_value <= alpha / 10:
                    binom_p = stats.binom.cdf(n_greater_equal, n_shuffles_used, alpha)
                    if binom_p >= confidence:
                        break
                elif current_p_value >= alpha * 3:
                    binom_p = 1 - stats.binom.cdf(
                        n_greater_equal - 1, n_shuffles_used, alpha
                    )
                    if binom_p >= confidence:
                        break

    # Convert to numpy array for final calculations
    null_cmi = np.array(null_cmi)

    # Calculate final statistics
    threshold = np.percentile(null_cmi, 100 * (1 - alpha))
    p_value = np.mean(null_cmi >= observed_cmi)

    return {
        "Threshold": threshold,
        "Value": observed_cmi,
        "Pass": observed_cmi >= threshold,
        "P_value": p_value,
        "n_shuffles_used": n_shuffles_used,
    }


if __name__ == "__main__":
    from causationentropy.datasets.synthetic import logisic_dynamics

    data, A = logisic_dynamics()
    G = discover_network(data)
    print(data.shape)
    print(G)
