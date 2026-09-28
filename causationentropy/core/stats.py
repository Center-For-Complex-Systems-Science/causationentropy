from typing import Tuple, Union

import numpy as np
from scipy.integrate import trapezoid

from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)


def auc(TPRs, FPRs):
    r"""
    Compute Area Under the ROC Curve (AUC) using trapezoidal integration.

    The Area Under the Curve provides a single scalar measure of classifier performance
    across all classification thresholds. It is computed as:

    .. math::

        \text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})

    where TPR (True Positive Rate) and FPR (False Positive Rate) define the ROC curve.
    The integral is approximated using the trapezoidal rule:

    .. math::

        \text{AUC} \approx \sum_{i=1}^{n-1} \frac{1}{2}[\text{TPR}_i + \text{TPR}_{i+1}][\text{FPR}_{i+1} - \text{FPR}_i]

    Parameters
    ----------
    TPRs : array-like
        True Positive Rates (sensitivities) corresponding to different thresholds.
        Should be sorted in ascending order of FPR.
    FPRs : array-like
        False Positive Rates (1 - specificities) corresponding to different thresholds.
        Should be sorted in ascending order.

    Returns
    -------
    AUC : float
        Area under the ROC curve. Values range from 0 to 1, where:
        - 0.5: Random classifier performance
        - 1.0: Perfect classifier performance
        - 0.0: Perfectly wrong classifier (can be inverted)

    Notes
    -----
    The AUC metric provides several interpretations:

    1. **Probabilistic**: Probability that a randomly chosen positive instance
       ranks higher than a randomly chosen negative instance

    2. **Geometric**: Area under the ROC curve in TPR-FPR space

    3. **Performance**: Single-number summary of classifier quality across thresholds

    **Advantages:**
    - Scale-invariant: Measures prediction quality regardless of classification threshold
    - Aggregated: Provides performance summary across all thresholds

    **Limitations:**
    - Can be overly optimistic for imbalanced datasets
    - Doesn't reflect class distribution in deployment
    - May not align with specific cost considerations

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.stats import auc
    >>>
    >>> # Perfect classifier
    >>> tpr_perfect = np.array([0, 1, 1])
    >>> fpr_perfect = np.array([0, 0, 1])
    >>> print(f"Perfect AUC: {auc(tpr_perfect, fpr_perfect)}")
    >>>
    >>> # Random classifier
    >>> tpr_random = np.array([0, 0.5, 1])
    >>> fpr_random = np.array([0, 0.5, 1])
    >>> print(f"Random AUC: {auc(tpr_random, fpr_random)}")
    """

    AUC = trapezoid(TPRs, FPRs)
    return AUC


def Compute_TPR_FPR(A, B):
    r"""
    Compute True Positive Rate and False Positive Rate for binary adjacency matrices.

    This function evaluates the performance of a predicted network (B) against
    a ground truth network (A) by computing standard classification metrics:

    .. math::

       \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{\text{TP}}{P}

    .. math::

       \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{\text{FP}}{N}

    where:
    - TP: True positives (correctly predicted edges)
    - FN: False negatives (missed edges)
    - FP: False positives (incorrectly predicted edges)
    - TN: True negatives (correctly predicted non-edges)
    - P: Total positive edges in ground truth
    - N: Total negative edges in ground truth

    Parameters
    ----------
    A : array-like of shape (n, n)
        Ground truth binary adjacency matrix. Should contain only 0s and 1s.
    B : array-like of shape (n, n)
        Predicted binary adjacency matrix. Should contain only 0s and 1s
        and have the same shape as A.

    Returns
    -------
    TPR : float
        True Positive Rate (Sensitivity, Recall). Fraction of actual edges
        that were correctly identified.
    FPR : float
        False Positive Rate (1 - Specificity). Fraction of actual non-edges
        that were incorrectly predicted as edges.

    Notes
    -----
    This implementation assumes:
    - Matrices are square and binary
    - Self-loops are excluded (diagonal elements ignored)
    - Matrices represent undirected graphs (symmetric)

    **Interpretation:**
    - **TPR (Sensitivity)**: How well the method detects true connections
    - **FPR (1-Specificity)**: How often the method falsely detects connections

    **Performance Assessment:**
    - High TPR, Low FPR: Excellent performance
    - High TPR, High FPR: Sensitive but not specific
    - Low TPR, Low FPR: Conservative approach
    - Low TPR, High FPR: Poor performance

    **Applications:**
    - Network reconstruction evaluation
    - Causal discovery validation
    - ROC curve generation
    - Method comparison and benchmarking

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.stats import Compute_TPR_FPR
    >>>
    >>> # Ground truth: simple 3-node chain
    >>> A = np.array([[0, 1, 0],
    ...               [1, 0, 1],
    ...               [0, 1, 0]])
    >>>
    >>> # Perfect prediction
    >>> B_perfect = A.copy()
    >>> tpr, fpr = Compute_TPR_FPR(A, B_perfect)
    >>> print(f"Perfect: TPR={tpr:.2f}, FPR={fpr:.2f}")
    >>>
    >>> # Overprediction (extra edge)
    >>> B_over = np.array([[0, 1, 1],
    ...                    [1, 0, 1],
    ...                    [1, 1, 0]])
    >>> tpr, fpr = Compute_TPR_FPR(A, B_over)
    >>> print(f"Overpredicted: TPR={tpr:.2f}, FPR={fpr:.2f}")
    """
    n = A.shape[0]
    assert A.shape[0] == A.shape[1] == B.shape[0] == B.shape[1]

    off_diag = ~np.eye(n, dtype=bool)
    diff = (A - B)[off_diag]

    false_negatives = np.sum(diff > 0)
    false_positives = np.sum(diff < 0)

    total_positives = np.sum(A[off_diag])
    total_negatives = off_diag.sum() - total_positives

    TPR = 1 - (false_negatives / total_positives) if total_positives > 0 else 1.0
    FPR = false_positives / total_negatives if total_negatives > 0 else 0.0

    return (TPR, FPR)


def moving_block_bootstrap_indices(
    n: int,
    block_length: int,
    n_bootstraps: int,
    seed: Union[int, np.random.Generator, None] = 42,
) -> np.ndarray:
    r"""
    Draw moving-block bootstrap resamples of ``range(n)``.

    The standard bootstrap assumes i.i.d. data, which time series violate.
    The moving block bootstrap instead resamples overlapping blocks of
    consecutive observations, preserving local temporal dependence within
    each block. With ``n`` time points and block length ``l``, a bootstrap
    replicate is ``[B_1, B_2, ...]`` where each ``B_i`` is a block of
    length ``l`` starting at a uniformly drawn time point (wrapping around
    the end circularly); blocks are concatenated and truncated to ``n``.

    Parameters
    ----------
    n : int
        Number of time points. Must be at least 1.
    block_length : int
        Length ``l`` of each overlapping block. Must satisfy
        ``1 <= block_length <= n``. Larger values preserve more dependence
        but give fewer independent blocks.
    n_bootstraps : int
        Number of bootstrap replicates. Must be at least 1.
    seed : int, numpy.random.Generator, or None, default=42
        Random seed or generator for reproducibility.

    Returns
    -------
    indices : np.ndarray of shape (n_bootstraps, n) with dtype int
        Resampled time indices; ``indices[b]`` is the ``b``-th replicate.

    Raises
    ------
    ValueError
        If the sizes are out of range.

    Examples
    --------
    >>> from causationentropy.core.stats import moving_block_bootstrap_indices
    >>>
    >>> indices = moving_block_bootstrap_indices(100, 10, 5, seed=0)
    >>> indices.shape
    (5, 100)

    References
    ----------
    .. [1] Künsch, H. R. "The jackknife and the bootstrap for general
           stationary observations." Annals of Statistics 17, 1217-1241
           (1989).

    See Also
    --------
    stationary_bootstrap_indices : Bootstrap with random block lengths.
    bootstrap_confidence_interval : Percentile interval from replicates.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    if not 1 <= block_length <= n:
        raise ValueError("block_length must satisfy 1 <= block_length <= n.")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be at least 1.")
    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_bootstraps, n_blocks))
    offsets = np.arange(block_length)
    blocks = (starts[..., None] + offsets) % n
    return blocks.reshape(n_bootstraps, -1)[:, :n]


def stationary_bootstrap_indices(
    n: int,
    mean_block_length: float,
    n_bootstraps: int,
    seed: Union[int, np.random.Generator, None] = 42,
) -> np.ndarray:
    r"""
    Draw stationary-bootstrap resamples of ``range(n)``.

    Unlike the moving block bootstrap, block lengths here are random with
    a geometric distribution of mean ``mean_block_length``, which makes the
    resampled series itself stationary and preserves temporal dependence.
    Blocks start at uniformly drawn time points (wrapping circularly) and
    are concatenated until ``n`` indices are collected, then truncated.

    Parameters
    ----------
    n : int
        Number of time points. Must be at least 1.
    mean_block_length : float
        Mean block length; the per-step continuation probability is
        ``1 - 1 / mean_block_length``. Must be at least 1.
    n_bootstraps : int
        Number of bootstrap replicates. Must be at least 1.
    seed : int, numpy.random.Generator, or None, default=42
        Random seed or generator for reproducibility.

    Returns
    -------
    indices : np.ndarray of shape (n_bootstraps, n) with dtype int
        Resampled time indices; ``indices[b]`` is the ``b``-th replicate.

    Raises
    ------
    ValueError
        If the sizes are out of range.

    Examples
    --------
    >>> from causationentropy.core.stats import stationary_bootstrap_indices
    >>>
    >>> indices = stationary_bootstrap_indices(100, 10.0, 5, seed=0)
    >>> indices.shape
    (5, 100)

    References
    ----------
    .. [1] Politis, D. N., Romano, J. P. "The stationary bootstrap."
           Journal of the American Statistical Association 89, 1303-1313
           (1994).

    See Also
    --------
    moving_block_bootstrap_indices : Bootstrap with fixed block lengths.
    bootstrap_confidence_interval : Percentile interval from replicates.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    if mean_block_length < 1:
        raise ValueError("mean_block_length must be at least 1.")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be at least 1.")
    rng = np.random.default_rng(seed)

    indices = np.zeros((n_bootstraps, n), dtype=int)
    for b in range(n_bootstraps):
        filled = 0
        while filled < n:
            start = int(rng.integers(0, n))
            length = int(rng.geometric(1.0 / mean_block_length))
            block = (start + np.arange(length)) % n
            take = min(length, n - filled)
            indices[b, filled : filled + take] = block[:take]
            filled += take
    return indices


def bootstrap_confidence_interval(
    bootstrap_estimates, alpha: float = 0.05
) -> Tuple[float, float]:
    r"""
    Build a percentile bootstrap confidence interval.

    Given ``B`` bootstrap replicates
    :math:`\{\\hat{I}^{(b)}\\}_{b=1}^B` of an information estimate, the
    interval is:

    .. math::

        [\\hat{I}_{(\\alpha/2)}, \\hat{I}_{(1-\\alpha/2)}]

    where the bounds are the corresponding quantiles of the bootstrap
    distribution.

    Parameters
    ----------
    bootstrap_estimates : array-like of shape (B,)
        One-dimensional finite bootstrap replicate values.
    alpha : float, default=0.05
        Significance level in (0, 1); the interval covers
        ``1 - alpha`` of the bootstrap distribution.

    Returns
    -------
    lower : float
        Lower interval bound (``100 * alpha / 2`` percentile).
    upper : float
        Upper interval bound (``100 * (1 - alpha / 2)`` percentile).

    Raises
    ------
    ValueError
        If the input is not a non-empty one-dimensional array of finite
        values, or ``alpha`` is not in (0, 1).

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.stats import bootstrap_confidence_interval
    >>>
    >>> bootstrap_confidence_interval(np.arange(101), alpha=0.05)
    (2.5, 97.5)

    References
    ----------
    .. [1] Efron, B. "Bootstrap methods: another look at the jackknife."
           Annals of Statistics 7, 1-26 (1979).

    See Also
    --------
    moving_block_bootstrap_indices : Resample time series in blocks.
    stationary_bootstrap_indices : Resample with random block lengths.
    bootstrap_cmi_confidence_interval : Interval for conditional MI.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    values = np.asarray(bootstrap_estimates, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap_estimates must be a non-empty 1-D array-like.")
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap_estimates must contain only finite values.")
    lower = float(np.quantile(values, alpha / 2))
    upper = float(np.quantile(values, 1 - alpha / 2))
    return lower, upper


def bootstrap_cmi_confidence_interval(
    X,
    Y,
    Z=None,
    method: str = "gaussian",
    n_bootstraps: int = 200,
    alpha: float = 0.05,
    block_length: int = None,
    mean_block_length: float = 10.0,
    use_stationary: bool = False,
    metric: str = "euclidean",
    k: int = 5,
    bandwidth: str = "silverman",
    seed: Union[int, np.random.Generator, None] = 42,
) -> Tuple[float, float, float, np.ndarray]:
    r"""
    Estimate conditional mutual information with a bootstrap confidence interval.

    Computes :math:`I(X; Y \\mid Z)` on the observed rows plus on
    ``n_bootstraps`` block-bootstrap resamples of the rows, then reports
    the percentile interval (see :func:`bootstrap_confidence_interval`).
    Rows of ``X``, ``Y`` and ``Z`` are resampled jointly so each replicate
    preserves the contemporaneous joint distribution, while blocks preserve
    local temporal dependence. Use the stationary bootstrap for strongly
    autocorrelated series.

    Parameters
    ----------
    X : array-like of shape (T, k_x)
        Predictor variable(s). Must be 2-D even when ``k_x=1``.
    Y : array-like of shape (T, k_y)
        Target variable(s).
    Z : array-like of shape (T, k_z) or None
        Conditioning set. If None, intervals marginal mutual information.
    method : str, default='gaussian'
        Information estimator passed to ``conditional_mutual_information``.
    n_bootstraps : int, default=200
        Number of bootstrap replicates. Must be at least 1.
    alpha : float, default=0.05
        Significance level in (0, 1) for the interval.
    block_length : int, optional
        Fixed block length for the moving block bootstrap. If None,
        defaults to ``max(1, round(T ** (1/3)))``. Ignored when
        ``use_stationary`` is True.
    mean_block_length : float, default=10.0
        Mean block length for the stationary bootstrap. Only used when
        ``use_stationary`` is True.
    use_stationary : bool, default=False
        If True, use the stationary bootstrap; otherwise the moving block
        bootstrap.
    metric : str, default='euclidean'
        Distance metric for k-NN based estimators.
    k : int, default=5
        Number of neighbors for k-NN based estimators.
    bandwidth : str, default='silverman'
        Bandwidth selection method for KDE.
    seed : int, numpy.random.Generator, or None, default=42
        Random seed or generator for reproducibility.

    Returns
    -------
    estimate : float
        Conditional mutual information on the observed data.
    lower : float
        Lower confidence bound.
    upper : float
        Upper confidence bound.
    bootstrap_estimates : np.ndarray of shape (n_bootstraps,)
        The per-replicate estimates forming the interval.

    Raises
    ------
    ValueError
        If ``X``/``Y``/``Z`` disagree on ``T``, or bootstrap sizes are out
        of range.

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.stats import (
    ...     bootstrap_cmi_confidence_interval,
    ... )
    >>>
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((300, 1))
    >>> Y = X + 0.5 * rng.standard_normal((300, 1))
    >>> est, lo, hi, _ = bootstrap_cmi_confidence_interval(
    ...     X, Y, n_bootstraps=50, seed=0
    ... )
    >>> lo <= est <= hi
    True

    See Also
    --------
    moving_block_bootstrap_indices : Fixed-length block resampling.
    stationary_bootstrap_indices : Random-length block resampling.
    bootstrap_confidence_interval : Percentile interval from replicates.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be at least 1.")
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must be 2-D with the same number of rows.")
    if Z is not None:
        Z = np.asarray(Z)
        if Z.ndim != 2 or Z.shape[0] != X.shape[0]:
            raise ValueError("Z must be None or 2-D with the same rows as X.")
    n = X.shape[0]

    if use_stationary:
        indices = stationary_bootstrap_indices(
            n, mean_block_length, n_bootstraps, seed=seed
        )
    else:
        if block_length is None:
            block_length = max(1, int(round(n ** (1.0 / 3.0))))
        indices = moving_block_bootstrap_indices(
            n, block_length, n_bootstraps, seed=seed
        )

    estimate = float(
        conditional_mutual_information(
            X, Y, Z, method=method, metric=metric, k=k, bandwidth=bandwidth
        )
    )
    bootstrap_estimates = np.array(
        [
            conditional_mutual_information(
                X[rows],
                Y[rows],
                None if Z is None else Z[rows],
                method=method,
                metric=metric,
                k=k,
                bandwidth=bandwidth,
            )
            for rows in indices
        ],
        dtype=float,
    )
    lower, upper = bootstrap_confidence_interval(bootstrap_estimates, alpha=alpha)
    return estimate, lower, upper, bootstrap_estimates
