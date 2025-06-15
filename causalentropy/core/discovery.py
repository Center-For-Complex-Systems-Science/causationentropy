import copy
import networkx as nx
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Tuple

from causalentropy.core.information.conditional_mutual_information import conditional_mutual_information



def discover_network(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = "gaussian",
    max_lag: int = 5,
    significance_forward: float = 0.05,
    backwards_significance: float = 0.5,
    metric: str = "euclidean",
    n_permutations: int = 200,
) -> nx.DiGraph:
    """
    Infer a causal graph via Optimal Causation Entropy.

    Parameters
    ----------
    data : (T, n) ndarray or DataFrame
        Multivariate time series (variables = columns).
    method : str
        Entropy estimator.  Currently only 'gaussian' is implemented.
    max_lag : int
        Consider lags 0 … max_lag (inclusive).
    significance_level : float
        α-level for permutation significance tests.
    n_permutations : int
        Number of permutations for each CMI test.

    Returns
    -------
    G : nx.DiGraph
        Directed graph with edge attributes
            lag   : int     (delay τ)
            cmi   : float   (conditional MI value at selection)
    """

    if method != "gaussian":
        raise NotImplementedError("discover_network: only method='gaussian' supported.")

    # Convert DataFrame → ndarray while keeping column labels
    if isinstance(data, pd.DataFrame):
        series = data.values
        var_names = list(data.columns)
    else:
        series = np.asarray(data)
        var_names = [f"X{i}" for i in range(series.shape[1])]

    T, n = series.shape
    if T <= max_lag + 2:
        raise ValueError("Time series too short for chosen max_lag.")


    indices = np.arange(max_lag, T - 1)            # t used in algorithm
    lagged: Dict[Tuple[int, int], np.ndarray] = {}
    for j in range(n):
        for tau in range(max_lag + 1):
            lagged[(j, tau)] = series[indices - tau, j]

    G = nx.DiGraph()
    G.add_nodes_from(var_names)



    tau = 1 # Figure it out later
    #XY = data.to_numpy() # Figure it out typing later
    XY = data
    XY_1 = XY[0:T - tau, :]
    XY_2 = XY[tau:, :]
    B = np.zeros((n, n))
    for i in range(n):
        print("Estimating edges for node number: ", i)
        Y = XY_2[:, [i]]
        X = XY_1
        S = standard_optimal_causation_entropy(X, Y)
        B[i, S] = 1

    for i in range(n):
        for j in range(n):
            if B[i, j] == 1:
                G.add_edge(var_names[j], var_names[i], lag=tau)
    return G


def standard_optimal_causation_entropy(X, Y):

    """Run the standard version of the oCSE algorithm. Note defaults to the
           KernelDensity plugin estimator if the method is not specified"""


    forward_pass = forward(X, Y)

    S = backward(X, Y, forward_pass)

    return S

def forward(X_full, Y, alpha=0.05):
    """
    Forward step of oCSE *without* hidden state.

    Parameters
    ----------
    X_full : (T, n) predictor matrix  (values at time t)
    Y      : (T, 1) target column     (values at t+τ)
    alpha  : shuffle‐test significance level
    """
    n = X_full.shape[1]
    candidates  = np.arange(n)
    S = []           # selected predictors
    Z = None         # current conditioning set

    while True:
        remaining = np.setdiff1d(candidates, S)
        if remaining.size == 0:
            break

        # 1. evaluate each remaining variable
        ent_values = np.zeros(remaining.size)
        for k, j in enumerate(remaining):
            Xj  = X_full[:, [j]]                       # keep 2-D shape
            ent_values[k] = conditional_mutual_information(
                                Xj, Y, Z)

        # 2. pick best
        j_best   = remaining[ent_values.argmax()]
        X_best   = X_full[:, [j_best]]
        mi_best  = ent_values.max()

        # 3. permutation (shuffle) test
        passed = shuffle_test(X_best, Y, Z, mi_best, alpha)['Pass']
        if not passed:
            break

        # 4. accept and update conditioning set
        S.append(j_best)
        Z = X_full[:, S] if len(S) else None

    return S


def backward(X_full, Y, S_init, alpha=0.05):
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
    shuffle_kwargs : dict or None
        Extra args forwarded to `shuffle_test` (e.g. n_bootstraps).

    Returns
    -------
    S_final : list[int]
        Subset of S_init that passed the backward significance test.
    """
    rng = np.random.default_rng()                   # for permutation order
    S   = copy.deepcopy(S_init)                     # working copy

    for j in rng.permutation(S_init):
        # conditioning set Z = S \ {j}
        Z = X_full[:, [k for k in S if k != j]] if len(S) > 1 else None

        Xj = X_full[:, [j]]
        cmij = conditional_mutual_information(Xj, Y, Z)

        passed  = shuffle_test(Xj, Y, Z, cmij, alpha=alpha)['Pass']
        if not passed:
            S.remove(j)  # prune j

    return S


def shuffle_test(X, Y, Z, observed_cmi,
                 alpha=0.05, n_bootstraps=500, rng=None):
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
    n_bootstraps : int
        Number of random permutations.
    rng : int or np.random.Generator or None
        Seed for reproducibility.

    Returns
    -------
    dict with keys {'Threshold', 'Value', 'Pass'}
    """
    rng = np.random.default_rng(rng)
    null_cmi = np.empty(n_bootstraps)

    for i in range(n_bootstraps):
        X_perm = X[rng.permutation(len(X)), :]   # shuffle rows
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