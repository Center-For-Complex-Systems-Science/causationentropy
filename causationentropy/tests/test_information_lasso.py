from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from causationentropy.core.discovery import (
    discover_network,
    information_lasso_optimal_causation_entropy,
)


def test_information_lasso_scales_features_by_information_weights():
    """CMI weights must alter the sparse-selection design matrix."""
    X = np.arange(10, dtype=float).reshape(5, 2) + 1.0
    Y = np.arange(5, dtype=float).reshape(-1, 1)
    rng = np.random.default_rng(0)

    fitted = MagicMock()
    fitted.fit.return_value = fitted
    fitted.coef_ = np.array([1.0, 0.0])

    with patch(
        "causationentropy.core.discovery.conditional_mutual_information",
        side_effect=[3.0, 1.0],
    ) as cmi, patch(
        "causationentropy.core.discovery.LassoLarsIC",
        return_value=fitted,
    ):
        selected = information_lasso_optimal_causation_entropy(
            X,
            Y,
            rng,
            information="kde",
            metric="cityblock",
            k_means=7,
            bandwidth=0.5,
        )

    assert selected == [0]
    np.testing.assert_allclose(
        fitted.fit.call_args.args[0],
        X * np.array([0.75, 0.25]),
    )
    for call in cmi.call_args_list:
        assert call.args[2] is None
        assert call.kwargs == {
            "method": "kde",
            "metric": "cityblock",
            "k": 7,
            "bandwidth": 0.5,
        }


def test_information_lasso_uses_cross_validation_when_high_dimensional():
    """p >= n uses the existing cross_val parameter instead of fixed-alpha LASSO."""
    X = np.arange(20, dtype=float).reshape(4, 5) + 1.0
    Y = np.arange(4, dtype=float).reshape(-1, 1)
    rng = np.random.default_rng(0)

    fitted = MagicMock()
    fitted.fit.return_value = fitted
    fitted.coef_ = np.array([0.0, 1.0, 0.0, 0.0, -1.0])

    with patch(
        "causationentropy.core.discovery.conditional_mutual_information",
        side_effect=[5.0, 4.0, 3.0, 2.0, 1.0],
    ), patch(
        "causationentropy.core.discovery.LassoCV",
        return_value=fitted,
    ) as lasso_cv:
        selected = information_lasso_optimal_causation_entropy(
            X,
            Y,
            rng,
            max_lambda=77,
            cross_val=10,
        )

    assert selected == [1, 4]
    lasso_cv.assert_called_once_with(cv=4, max_iter=1000)


def test_information_lasso_returns_empty_when_all_information_is_zero():
    """No information signal should not fall back to ordinary LASSO."""
    X = np.ones((6, 3))
    Y = np.arange(6, dtype=float).reshape(-1, 1)

    with patch(
        "causationentropy.core.discovery.conditional_mutual_information",
        return_value=0.0,
    ), patch("causationentropy.core.discovery.LassoLarsIC") as lars, patch(
        "causationentropy.core.discovery.LassoCV"
    ) as lasso_cv:
        selected = information_lasso_optimal_causation_entropy(
            X,
            Y,
            np.random.default_rng(0),
        )

    assert selected == []
    lars.assert_not_called()
    lasso_cv.assert_not_called()


def test_information_lasso_rejects_nonfinite_information_weight():
    """Estimator failures should surface rather than silently changing selection."""
    X = np.ones((6, 2))
    Y = np.arange(6, dtype=float).reshape(-1, 1)

    with patch(
        "causationentropy.core.discovery.conditional_mutual_information",
        side_effect=[1.0, np.nan],
    ):
        with pytest.raises(ValueError, match="non-finite information weight"):
            information_lasso_optimal_causation_entropy(
                X,
                Y,
                np.random.default_rng(0),
            )


def test_discover_network_forwards_information_lasso_estimator_options():
    """The public API estimator settings must reach Information-LASSO."""
    data = np.arange(36, dtype=float).reshape(12, 3)

    with patch(
        "causationentropy.core.discovery.information_lasso_optimal_causation_entropy",
        return_value=[],
    ) as information_lasso:
        graph = discover_network(
            data,
            method="information_lasso",
            information="kde",
            max_lag=1,
            metric="cityblock",
            bandwidth=0.4,
            k_means=9,
            n_shuffles=2,
        )

    assert graph.number_of_nodes() == 3
    assert information_lasso.call_count == 3
    for call in information_lasso.call_args_list:
        assert call.kwargs == {
            "information": "kde",
            "metric": "cityblock",
            "k_means": 9,
            "bandwidth": 0.4,
        }
