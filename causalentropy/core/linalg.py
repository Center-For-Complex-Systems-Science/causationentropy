import numpy as np
import networkx as nx
from typing import Tuple, List, Optional

def correlation_log_determinant(A, epsilon=1e-10):
    """Compute log-determinant of correlation matrix."""
    if A.shape[1] == 0:
        return 0.0
    C = np.corrcoef(A.T)
    if C.ndim == 0:
        return 0.0
    return np.linalg.slogdet(C)[1]
