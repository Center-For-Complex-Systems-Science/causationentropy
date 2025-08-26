import numpy as np
import pytest

from causationentropy.core.discovery import (
    alternative_forward,
    alternative_optimal_causation_entropy,
    backward,
    lasso_optimal_causation_entropy,
    standard_forward,
    standard_optimal_causation_entropy,
)


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
    Y[0] = rng.normal()  # throwaway first value
    Y[1:] = 0.9 * X1[:-1] + 1.1 * X2[:-1] + noise[1:]

    # build matrices aligned on index *t*
    X_full = np.column_stack([X0[:-1], X1[:-1], X2[:-1]])  # (T‑1, 3)
    Y = Y[1:].reshape(-1, 1)  # (T‑1, 1)
    Z_init = Y[:-1]  # (T‑2, 1)

    X_full = X_full[1:]  # keep rows where Z_init is defined  (T‑2, 3)
    Y = Y[1:]  # (T‑2, 1)
    return X_full, Y, Z_init


@pytest.mark.parametrize("alpha, n_shuffles", [(0.01, 1000)])
def test_standard_forward_recovers_parents(alpha, n_shuffles):
    """
    The forward phase should (almost) always select columns 1 and 2,
    never column 0.
    """
    X_full, Y, Z_init = make_forward_case()
    rng = np.random.default_rng(2024)  # single generator for reproducibility

    selected = standard_forward(
        X_full,
        Y,
        Z_init,
        rng=rng,
        alpha=alpha,
        n_shuffles=n_shuffles,
        information="gaussian",
    )

    assert set(selected) == {1, 2}, f"Expected {{1, 2}}, got {set(selected)}"


@pytest.mark.parametrize("alpha, n_shuffles", [(0.01, 1000)])
def test_standard_backward_maintains_true_parents(alpha, n_shuffles):
    """
    The backward phase should maintain the true causal parents (columns 1 and 2)
    and not eliminate them as spurious.
    """
    X_full, Y, Z_init = make_forward_case()
    rng = np.random.default_rng(2024)

    # First get forward selection results
    forward_selected = standard_forward(
        X_full,
        Y,
        Z_init,
        rng=rng,
        alpha=alpha,
        n_shuffles=n_shuffles,
        information="gaussian",
    )

    # Apply backward elimination
    backward_selected = backward(
        X_full,
        Y,
        forward_selected,
        rng=rng,
        alpha=alpha,
        n_shuffles=n_shuffles,
        information="gaussian",
    )

    # Backward should maintain the true causal relationships
    assert set(backward_selected) == {
        1,
        2,
    }, f"Backward phase expected {{1, 2}}, got {set(backward_selected)}"


@pytest.mark.parametrize("alpha, n_shuffles", [(0.01, 1000)])
def test_standard_full_method_recovers_parents(alpha, n_shuffles):
    """
    The complete standard optimal causation entropy method (forward + backward)
    should recover the true causal parents.
    """
    X_full, Y, Z_init = make_forward_case()
    rng = np.random.default_rng(2024)

    selected = standard_optimal_causation_entropy(
        X_full,
        Y,
        Z_init,
        rng=rng,
        alpha1=alpha,
        alpha2=alpha,
        n_shuffles=n_shuffles,
        information="gaussian",
    )

    assert set(selected) == {
        1,
        2,
    }, f"Standard method expected {{1, 2}}, got {set(selected)}"


@pytest.mark.parametrize("alpha, n_shuffles", [(0.01, 1000)])
def test_alternative_forward_recovers_parents(alpha, n_shuffles):
    """
    The alternative forward phase should recover causal parents without initial conditioning.
    """
    X_full, Y, Z_init = make_forward_case()
    rng = np.random.default_rng(2024)

    selected = alternative_forward(
        X_full,
        Y,
        rng=rng,
        alpha=alpha,
        n_shuffles=n_shuffles,
        information="gaussian",
    )

    assert set(selected) == {
        1,
        2,
    }, f"Alternative forward expected {{1, 2}}, got {set(selected)}"


@pytest.mark.parametrize("alpha, n_shuffles", [(0.01, 1000)])
def test_alternative_full_method_recovers_parents(alpha, n_shuffles):
    """
    The complete alternative optimal causation entropy method (forward + backward)
    should recover the true causal parents.
    """
    X_full, Y, Z_init = make_forward_case()
    rng = np.random.default_rng(2024)

    selected = alternative_optimal_causation_entropy(
        X_full,
        Y,
        rng=rng,
        alpha1=alpha,
        alpha2=alpha,
        n_shuffles=n_shuffles,
        information="gaussian",
    )

    assert set(selected) == {
        1,
        2,
    }, f"Alternative method expected {{1, 2}}, got {set(selected)}"


@pytest.mark.parametrize("criterion", ["aic", "bic"])
def test_lasso_method_recovers_parents(criterion):
    """
    The LASSO optimal causation entropy method should recover the true causal parents
    using different information criteria. LASSO may be more liberal in selection.
    """
    X_full, Y, Z_init = make_forward_case()
    rng = np.random.default_rng(2024)

    selected = lasso_optimal_causation_entropy(
        X_full, Y, rng=rng, criterion=criterion, max_lambda=100, cross_val=10
    )

    # Assert that the method returns valid indices and includes the true causal parents
    assert isinstance(
        selected, list
    ), f"LASSO method should return a list, got {type(selected)}"
    assert all(
        isinstance(idx, (int, np.integer)) for idx in selected
    ), "Selected indices should be integers"
    assert all(
        0 <= idx < X_full.shape[1] for idx in selected
    ), "Selected indices should be valid column indices"

    # Assert that true causal parents (1, 2) are included in the selection
    true_parents = {1, 2}
    selected_set = set(selected)
    assert true_parents.issubset(selected_set), (
        f"LASSO method with {criterion} should include true parents {{1, 2}}, "
        f"but got {selected_set}. Missing: {true_parents - selected_set}"
    )
