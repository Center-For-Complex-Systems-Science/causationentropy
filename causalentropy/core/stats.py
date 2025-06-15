import numpy as np


def auc(TPRs, FPRs):
    """Estimate the area under the curve (AUC) using the trapezoidal rule
       for integration..."""

    AUC = np.trapz(TPRs, FPRs)
    return AUC


def Compute_TPR_FPR(A, B):
    n = A.shape[0]
    assert A.shape[0] == A.shape[1] == B.shape[0] == B.shape[1]

    TPR = 1 - (np.sum((A - B) > 0)) / np.sum(A)
    FPR = np.sum((A - B) < 0) / (n * (n - 1) - np.sum(A))
    TPR = TPR
    FPR = FPR
    return (TPR, FPR)