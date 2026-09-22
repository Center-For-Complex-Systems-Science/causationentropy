from typing import Tuple

import numpy as np
from scipy.integrate import trapezoid


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

    # Count true positives, false negatives, false positives
    # A - B > 0: edges in A but not in B (false negatives)
    # A - B < 0: edges in B but not in A (false positives)

    false_negatives = np.sum((A - B) > 0)
    false_positives = np.sum((A - B) < 0)

    total_positives = np.sum(A)  # Total edges in ground truth
    total_negatives = (
        n * (n - 1) - total_positives
    )  # Total non-edges (excluding diagonal)

    # Compute TPR and FPR
    TPR = 1 - (false_negatives / total_positives) if total_positives > 0 else 1.0
    FPR = false_positives / total_negatives if total_negatives > 0 else 0.0

    return (TPR, FPR)


def _validate_p_values(p_values) -> np.ndarray:
    """Validate and normalize a collection of p-values.

    Parameters
    ----------
    p_values : array-like of shape (m,)
        One-dimensional collection of p-values. ``NaN`` entries are allowed
        and treated as missing tests.

    Returns
    -------
    p : np.ndarray of shape (m,)
        Validated float array.

    Raises
    ------
    ValueError
        If the input is not one-dimensional or any finite entry lies
        outside [0, 1].
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be a one-dimensional array-like.")
    finite = p[~np.isnan(p)]
    if finite.size > 0 and (np.any(finite < 0.0) or np.any(finite > 1.0)):
        raise ValueError("All finite p-values must lie in [0, 1].")
    return p


def _validate_alpha(alpha: float) -> None:
    """Validate a significance level.

    Raises
    ------
    ValueError
        If alpha is not in (0, 1].
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1].")


def bonferroni_correction(
    p_values, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Apply the Bonferroni correction for multiple testing.

    In causal discovery, ``m`` hypotheses of the form
    :math:`H_{0,k}: I(X_{j_k}^{(t-\\tau_k)}; X_i^{(t)} \\mid \\mathbf{Z}_i^{(t)}) = 0`
    are tested simultaneously. Testing each at level ``alpha`` inflates the
    family-wise error rate (FWER):

    .. math::

        \\text{FWER} = P(\\text{at least one false rejection})

    The Bonferroni correction rejects :math:`H_{0,k}` when
    :math:`p_k \\leq \\alpha / m`, which controls the FWER at ``alpha``. It
    is conservative (low power) when ``m`` is large, and most appropriate
    when few true relationships exist or strong FWER control is required.

    Parameters
    ----------
    p_values : array-like of shape (m,)
        One-dimensional collection of p-values, e.g. the ``P_Value`` column
        of :func:`causationentropy.graph.utils.network_to_dataframe` output.
        ``NaN`` entries are treated as missing tests: they are excluded from
        ``m``, never rejected, and get an adjusted p-value of ``NaN``.
    alpha : float, default=0.05
        Desired family-wise error rate. Must lie in (0, 1].

    Returns
    -------
    rejected : np.ndarray of shape (m,) with dtype bool
        Whether each null hypothesis is rejected, in input order.
    p_adjusted : np.ndarray of shape (m,)
        Bonferroni-adjusted p-values, ``min(p * m, 1)``, in input order.

    Notes
    -----
    For stage-wise application in oCSE forward/backward selection, apply
    this correction separately to each stage's family of p-values.

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.stats import bonferroni_correction
    >>>
    >>> rejected, p_adj = bonferroni_correction([0.01, 0.02, 0.03, 0.5])
    >>> print(rejected)
    [ True False False False]

    References
    ----------
    .. [1] Bonferroni, C. "Teoria statistica delle classi e calcolo delle
           probabilita." Pubblicazioni del R. Istituto Superiore di Scienze
           Economiche e Commerciali di Firenze 8, 3-62 (1936).

    See Also
    --------
    benjamini_hochberg_correction : Control the false discovery rate.
    benjamini_yekutieli_correction : FDR control under arbitrary dependence.
    adaptive_bh_correction : Adaptive FDR control with null estimation.
    """
    _validate_alpha(alpha)
    p = _validate_p_values(p_values)
    m = int(np.sum(~np.isnan(p)))

    rejected = np.zeros(p.shape, dtype=bool)
    p_adjusted = np.full(p.shape, np.nan)
    if m == 0:
        return rejected, p_adjusted

    finite = ~np.isnan(p)
    p_adjusted[finite] = np.minimum(p[finite] * m, 1.0)
    rejected[finite] = p[finite] <= alpha / m
    return rejected, p_adjusted


def benjamini_hochberg_correction(
    p_values, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Apply the Benjamini-Hochberg procedure to control the false discovery rate.

    The false discovery rate (FDR) is the expected proportion of false
    rejections among all rejections:

    .. math::

        \\text{FDR} = \\mathbb{E}\\left[\\frac{V}{\\max(R, 1)}\\right]

    where :math:`V` is the number of false rejections and :math:`R` is the
    total number of rejections. The procedure orders the p-values
    :math:`p_{(1)} \\leq p_{(2)} \\leq \\cdots \\leq p_{(m)}`, finds the
    largest :math:`k` with :math:`p_{(k)} \\leq \\frac{k}{m}\\alpha`, and
    rejects :math:`H_{0,(1)}, \\ldots, H_{0,(k)}`. This controls the FDR at
    ``alpha`` while offering substantially more power than Bonferroni when
    many true relationships exist.

    Parameters
    ----------
    p_values : array-like of shape (m,)
        One-dimensional collection of p-values. ``NaN`` entries are treated
        as missing tests: they are excluded from ``m``, never rejected, and
        get an adjusted p-value of ``NaN``.
    alpha : float, default=0.05
        Desired false discovery rate. Must lie in (0, 1].

    Returns
    -------
    rejected : np.ndarray of shape (m,) with dtype bool
        Whether each null hypothesis is rejected, in input order.
    p_adjusted : np.ndarray of shape (m,)
        BH-adjusted p-values in input order, i.e. the smallest FDR level at
        which each hypothesis would be rejected.

    Notes
    -----
    For stage-wise application in oCSE forward/backward selection, apply
    this correction separately to each stage's family of p-values.

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.stats import benjamini_hochberg_correction
    >>>
    >>> rejected, p_adj = benjamini_hochberg_correction([0.01, 0.02, 0.03, 0.5])
    >>> print(rejected)
    [ True  True  True False]

    References
    ----------
    .. [1] Benjamini, Y., Hochberg, Y. "Controlling the false discovery
           rate: a practical and powerful approach to multiple testing."
           Journal of the Royal Statistical Society B 57, 289-300 (1995).

    See Also
    --------
    bonferroni_correction : Control the family-wise error rate.
    benjamini_yekutieli_correction : FDR control under arbitrary dependence.
    adaptive_bh_correction : Adaptive FDR control with null estimation.
    """
    _validate_alpha(alpha)
    p = _validate_p_values(p_values)

    rejected = np.zeros(p.shape, dtype=bool)
    p_adjusted = np.full(p.shape, np.nan)
    finite_idx = np.flatnonzero(~np.isnan(p))
    m = finite_idx.size
    if m == 0:
        return rejected, p_adjusted

    order = np.argsort(p[finite_idx], kind="mergesort")
    sorted_p = p[finite_idx[order]]
    ranks = np.arange(1, m + 1)

    # Largest k with p_(k) <= k / m * alpha
    passing = sorted_p <= ranks / m * alpha
    if np.any(passing):
        k_max = int(np.flatnonzero(passing)[-1]) + 1
        rejected[finite_idx[order[:k_max]]] = True

    # Adjusted p-values: min over j >= i of p_(j) * m / j, capped at 1
    raw_adjusted = sorted_p * m / ranks
    bh_adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    p_adjusted[finite_idx[order]] = np.minimum(bh_adjusted, 1.0)
    return rejected, p_adjusted


def estimate_null_proportion(p_values, lambda_: float = 0.5) -> float:
    r"""
    Estimate the proportion of true null hypotheses.

    Uses Storey's estimator:

    .. math::

        \\hat{\\pi}_0 = \\frac{\\text{num}\\{p_i > \\lambda\\}}{m(1 - \\lambda)}

    True null p-values are (approximately) uniform, so p-values above
    ``lambda_`` are attributed to the null fraction. The estimate is
    clipped to [0, 1].

    Parameters
    ----------
    p_values : array-like of shape (m,)
        One-dimensional collection of p-values. ``NaN`` entries are treated
        as missing tests and excluded from ``m``.
    lambda_ : float, default=0.5
        Tuning parameter in (0, 1). Larger values reduce bias but increase
        variance.

    Returns
    -------
    pi0 : float
        Estimated null proportion in [0, 1].

    Raises
    ------
    ValueError
        If ``lambda_`` is not in (0, 1).

    Examples
    --------
    >>> from causationentropy.core.stats import estimate_null_proportion
    >>>
    >>> estimate_null_proportion([0.01, 0.02, 0.03, 0.04, 0.9])
    0.4

    References
    ----------
    .. [1] Storey, J. D. "A direct approach to false discovery rates."
           Journal of the Royal Statistical Society B 64, 479-498 (2002).

    See Also
    --------
    adaptive_bh_correction : Use this estimate for adaptive FDR control.
    """
    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must lie in (0, 1).")
    p = _validate_p_values(p_values)
    finite = p[~np.isnan(p)]
    if finite.size == 0:
        return 1.0
    pi0 = float(np.sum(finite > lambda_) / (finite.size * (1.0 - lambda_)))
    return float(np.clip(pi0, 0.0, 1.0))


def adaptive_bh_correction(
    p_values, alpha: float = 0.05, lambda_: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Apply adaptive Benjamini-Hochberg FDR control.

    The standard BH procedure implicitly assumes all hypotheses are null.
    When many true relationships exist, estimating the null proportion
    :math:`\\hat{\\pi}_0` (see :func:`estimate_null_proportion`) and using
    the threshold

    .. math::

        p_{(k)} \\leq \\frac{k}{m \\hat{\\pi}_0}\\alpha

    recovers power while still controlling the FDR at ``alpha``. When
    :math:`\\hat{\\pi}_0 = 1` this reduces exactly to
    :func:`benjamini_hochberg_correction`.

    Parameters
    ----------
    p_values : array-like of shape (m,)
        One-dimensional collection of p-values. ``NaN`` entries are treated
        as missing tests: they are excluded from ``m``, never rejected, and
        get an adjusted p-value of ``NaN``.
    alpha : float, default=0.05
        Desired false discovery rate. Must lie in (0, 1].
    lambda_ : float, default=0.5
        Tuning parameter in (0, 1) for the null-proportion estimate.

    Returns
    -------
    rejected : np.ndarray of shape (m,) with dtype bool
        Whether each null hypothesis is rejected, in input order.
    p_adjusted : np.ndarray of shape (m,)
        Adaptively adjusted p-values in input order.

    Notes
    -----
    For stage-wise application in oCSE forward/backward selection, apply
    this correction separately to each stage's family of p-values.

    Examples
    --------
    >>> from causationentropy.core.stats import adaptive_bh_correction
    >>>
    >>> rejected, _ = adaptive_bh_correction([0.01, 0.02, 0.03, 0.04, 0.9])
    >>> print(rejected)
    [ True  True  True  True False]

    References
    ----------
    .. [1] Benjamini, Y., Hochberg, Y. "Controlling the false discovery
           rate: a practical and powerful approach to multiple testing."
           Journal of the Royal Statistical Society B 57, 289-300 (1995).
    .. [2] Storey, J. D. "A direct approach to false discovery rates."
           Journal of the Royal Statistical Society B 64, 479-498 (2002).

    See Also
    --------
    benjamini_hochberg_correction : Non-adaptive FDR control.
    benjamini_yekutieli_correction : FDR control under arbitrary dependence.
    estimate_null_proportion : The null-proportion estimator used here.
    """
    _validate_alpha(alpha)
    p = _validate_p_values(p_values)
    pi0 = estimate_null_proportion(p, lambda_=lambda_)

    rejected = np.zeros(p.shape, dtype=bool)
    p_adjusted = np.full(p.shape, np.nan)
    finite_idx = np.flatnonzero(~np.isnan(p))
    m = finite_idx.size
    if m == 0:
        return rejected, p_adjusted

    order = np.argsort(p[finite_idx], kind="mergesort")
    sorted_p = p[finite_idx[order]]
    ranks = np.arange(1, m + 1)

    if pi0 <= 0.0:
        # Every finite p-value is at most lambda_: all signal.
        rejected[finite_idx] = True
        p_adjusted[finite_idx] = np.minimum(sorted_p, 1.0)[
            np.argsort(order, kind="mergesort")
        ]
        return rejected, p_adjusted

    passing = sorted_p <= ranks / (m * pi0) * alpha
    if np.any(passing):
        k_max = int(np.flatnonzero(passing)[-1]) + 1
        rejected[finite_idx[order[:k_max]]] = True

    raw_adjusted = sorted_p * m * pi0 / ranks
    adaptive_adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    p_adjusted[finite_idx[order]] = np.minimum(adaptive_adjusted, 1.0)
    return rejected, p_adjusted


def benjamini_yekutieli_correction(
    p_values, alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Apply the Benjamini-Yekutieli procedure for FDR control under dependence.

    Shuffle-test p-values from the same data reuse are dependent, so the
    standard Benjamini-Hochberg guarantee (independence or positive
    dependence) may not hold. The Benjamini-Yekutieli procedure controls
    the FDR under arbitrary dependence by tightening the threshold with
    the harmonic number :math:`c(m) = \\sum_{j=1}^{m} 1/j`:

    .. math::

        p_{(k)} \\leq \\frac{k}{m \\, c(m)}\\alpha

    rejecting :math:`H_{0,(1)}, \\ldots, H_{0,(k)}` for the largest such
    :math:`k`. It is more conservative than
    :func:`benjamini_hochberg_correction` and appropriate when the
    dependence structure between tests is unknown.

    Parameters
    ----------
    p_values : array-like of shape (m,)
        One-dimensional collection of p-values. ``NaN`` entries are treated
        as missing tests: they are excluded from ``m``, never rejected, and
        get an adjusted p-value of ``NaN``.
    alpha : float, default=0.05
        Desired false discovery rate. Must lie in (0, 1].

    Returns
    -------
    rejected : np.ndarray of shape (m,) with dtype bool
        Whether each null hypothesis is rejected, in input order.
    p_adjusted : np.ndarray of shape (m,)
        BY-adjusted p-values in input order, i.e. the smallest FDR level at
        which each hypothesis would be rejected.

    Examples
    --------
    >>> from causationentropy.core.stats import benjamini_yekutieli_correction
    >>>
    >>> rejected, _ = benjamini_yekutieli_correction([0.001, 0.002, 0.5, 0.9])
    >>> print(rejected)
    [ True  True False False]

    References
    ----------
    .. [1] Benjamini, Y., Yekutieli, D. "The control of the false
           discovery rate in multiple testing under dependency."
           Annals of Statistics 29, 1165-1188 (2001).

    See Also
    --------
    benjamini_hochberg_correction : FDR control under independence.
    bonferroni_correction : Control the family-wise error rate.
    """
    _validate_alpha(alpha)
    p = _validate_p_values(p_values)

    rejected = np.zeros(p.shape, dtype=bool)
    p_adjusted = np.full(p.shape, np.nan)
    finite_idx = np.flatnonzero(~np.isnan(p))
    m = finite_idx.size
    if m == 0:
        return rejected, p_adjusted

    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p[finite_idx], kind="mergesort")
    sorted_p = p[finite_idx[order]]
    ranks = np.arange(1, m + 1)

    passing = sorted_p <= ranks / (m * harmonic) * alpha
    if np.any(passing):
        k_max = int(np.flatnonzero(passing)[-1]) + 1
        rejected[finite_idx[order[:k_max]]] = True

    raw_adjusted = sorted_p * m * harmonic / ranks
    by_adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    p_adjusted[finite_idx[order]] = np.minimum(by_adjusted, 1.0)
    return rejected, p_adjusted
