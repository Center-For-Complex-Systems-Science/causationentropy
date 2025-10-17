import numpy as np


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

    AUC = np.trapz(TPRs, FPRs)
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
