import numpy as np
import pandas as pd
import networkx as nx
import pytest
from unittest.mock import patch, MagicMock
import networkx as nx
from causalentropy.datasets.synthetic import linear_stochastic_gaussian_process


from causalentropy.core.discovery import discover_network

# tests/test_standard_forward.py
import numpy as np
import pytest

from causalentropy.core.discovery import standard_forward

def make_forward_case(T=1500, seed=0):
    """
    Helper: build a toy system where Y(t) depends on X1(t-1) and X2(t-1)
    but NOT on X0(t-1).  The signal‑to‑noise ratio is strong enough that
    the Gaussian‑CMI estimator should flag 1 and 2 every time.
    """
    rng = np.random.default_rng(seed)

    X1, X2, X0 = rng.normal(size=(3, T))

    # Y(t) depends on X1(t‑1) and X2(t‑1)
    noise = 0.05 * rng.normal(size=T)
    Y = np.empty(T)
    Y[0] = rng.normal()                       # throwaway first value
    Y[1:] = 0.9 * X1[:-1] + 1.1 * X2[:-1] + noise[1:]

    # build matrices aligned on index *t*
    X_full = np.column_stack([X0[:-1], X1[:-1], X2[:-1]])   # (T‑1, 3)
    Y = Y[1:].reshape(-1, 1)                           # (T‑1, 1)
    Z_init = Y[:-1]                                         # (T‑2, 1)

    X_full = X_full[1:]   # keep rows where Z_init is defined  (T‑2, 3)
    Y = Y[1:] # (T‑2, 1)
    return X_full, Y, Z_init


@pytest.mark.parametrize("alpha, n_shuffles", [(0.01, 1000)])
def test_standard_forward_recovers_parents(alpha, n_shuffles):
    """
    The forward phase should (almost) always select columns 1 and 2,
    never column 0.
    """
    X_full, Y, Z_init = make_forward_case()
    rng = np.random.default_rng(2024)          # single generator for reproducibility

    selected = standard_forward(
        X_full,
        Y,
        Z_init,
        rng=rng,
        alpha=alpha,
        n_shuffles=n_shuffles,
        information="gaussian",
    )

    assert set(selected) == {1, 2}, (
        f"Expected {{1, 2}}, got {set(selected)}"
    )


