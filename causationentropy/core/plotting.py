import matplotlib.pyplot as plt
import networkx as nx

from causationentropy.core.stats import auc
from causationentropy.core.linalg import companion_matrix


def roc_curve(TPRs, FPRs):
    """
    Plot Receiver Operating Characteristic (ROC) curve.

    This function creates a ROC curve visualization, which is a graphical plot
    that illustrates the diagnostic ability of a binary classifier system.
    The ROC curve plots the True Positive Rate against the False Positive Rate
    at various threshold settings.

    The ROC curve is defined by the parametric equations:

    .. math::

        \\text{TPR}(t) = \\frac{\\text{TP}(t)}{\\text{TP}(t) + \\text{FN}(t)} = \\frac{\\text{TP}(t)}{P}

        \\text{FPR}(t) = \\frac{\\text{FP}(t)}{\\text{FP}(t) + \\text{TN}(t)} = \\frac{\\text{FP}(t)}{N}

    where t is the classification threshold, P is the total number of positives,
    and N is the total number of negatives.

    Parameters
    ----------
    TPRs : array-like
        True Positive Rates (Sensitivity, Recall) for different thresholds.
        Values should be in [0, 1].
    FPRs : array-like
        False Positive Rates (1 - Specificity) for different thresholds.
        Values should be in [0, 1].

    Notes
    -----
    **ROC Curve Interpretation:**
    - Perfect classifier: Curve passes through (0, 1) - high TPR, zero FPR
    - Random classifier: Diagonal line from (0, 0) to (1, 1)
    - Useless classifier: Curve below the diagonal

    **Key Points:**
    - (0, 0): No false positives, but also no true positives (very conservative)
    - (1, 1): All positives detected, but all negatives misclassified (very liberal)
    - (0, 1): Perfect classification (ideal classifier)

    **AUC (Area Under Curve):**
    - AUC = 1.0: Perfect classifier
    - AUC = 0.5: Random classifier
    - AUC < 0.5: Worse than random (can be inverted)

    **Applications:**
    - Medical diagnosis evaluation
    - Network reconstruction assessment
    - Causal discovery method comparison
    - Binary classification performance analysis

    The function automatically computes and displays the AUC value on the plot
    using the trapezoidal integration rule.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from causationentropy.core.plotting import roc_curve
    >>>
    >>> # Perfect classifier example
    >>> tpr_perfect = [0, 1, 1]
    >>> fpr_perfect = [0, 0, 1]
    >>>
    >>> plt.figure(figsize=(8, 6))
    >>> roc_curve(tpr_perfect, fpr_perfect)
    >>> plt.legend(['Perfect Classifier'])
    >>> plt.show()
    >>>
    >>> # Random classifier comparison
    >>> tpr_random = [0, 0.5, 1]
    >>> fpr_random = [0, 0.5, 1]
    >>> roc_curve(tpr_random, fpr_random)
    >>> plt.legend(['Random Classifier'])

    See Also
    --------
    causationentropy.core.stats.auc : Compute area under ROC curve
    causationentropy.core.stats.Compute_TPR_FPR : Compute TPR and FPR from confusion matrix
    """
    plt.plot(FPRs, TPRs)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add diagonal reference line for random classifier
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Classifier")

    AUC = auc(TPRs, FPRs)
    plt.text(
        0.4,
        0.1,
        f"AUC = {AUC:.4f}",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

def plot_causal_network(G: nx.MultiDiGraph):
    expanded_adjacecny = companion_matrix(G)