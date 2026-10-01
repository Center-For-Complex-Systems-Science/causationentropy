import numpy as np
import pytest

import causationentropy.core.discovery as discovery


def test_information_screened_rescue_then_maps_refined_support(monkeypatch):
    X = np.column_stack(
        [
            np.linspace(0.0, 1.0, 20),
            np.linspace(1.0, 2.0, 20),
            np.linspace(2.0, 3.0, 20),
            np.linspace(3.0, 4.0, 20),
        ]
    )
    Y = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    Z_init = np.ones((20, 1))

    monkeypatch.setattr(
        discovery,
        "information_lasso_optimal_causation_entropy",
        lambda *args, **kwargs: [0],
    )

    rescue_conditioning = []

    def fake_cmi(x, _y, z, **kwargs):
        rescue_conditioning.append(z.copy())
        # Candidate 2 is the strongest excluded rescue candidate.
        first = float(x[0, 0])
        return {1.0: 1.0, 2.0: 5.0, 3.0: 2.0}.get(first, 0.0)

    monkeypatch.setattr(discovery, "conditional_mutual_information", fake_cmi)

    captured = {}

    def fake_standard(x, _y, z_init, _rng, **kwargs):
        captured["screened"] = x.copy()
        captured["z_init"] = z_init.copy()
        # Return the second column of the screened matrix.
        return [1]

    monkeypatch.setattr(
        discovery,
        "standard_optimal_causation_entropy",
        fake_standard,
    )

    selected = discovery.information_screened_optimal_causation_entropy(
        X,
        Y,
        Z_init,
        np.random.default_rng(0),
        retention=0.5,
    )

    assert selected == [2]
    np.testing.assert_allclose(captured["screened"], X[:, [0, 2]])
    np.testing.assert_allclose(captured["z_init"], Z_init)

    expected_rescue_conditioning = np.hstack((Z_init, X[:, [0]]))
    assert len(rescue_conditioning) == 3
    for z in rescue_conditioning:
        np.testing.assert_allclose(z, expected_rescue_conditioning)


def test_information_screened_rescue_uses_history_when_proposal_is_empty(
    monkeypatch,
):
    X = np.column_stack(
        [
            np.linspace(0.0, 1.0, 20),
            np.linspace(1.0, 2.0, 20),
        ]
    )
    Y = np.linspace(-1.0, 1.0, 20).reshape(-1, 1)
    Z_init = np.column_stack(
        [
            np.linspace(4.0, 5.0, 20),
            np.linspace(5.0, 6.0, 20),
        ]
    )
    rescue_conditioning = []

    monkeypatch.setattr(
        discovery,
        "information_lasso_optimal_causation_entropy",
        lambda *args, **kwargs: [],
    )

    def fake_cmi(_x, _y, z, **kwargs):
        rescue_conditioning.append(z.copy())
        return 1.0

    monkeypatch.setattr(discovery, "conditional_mutual_information", fake_cmi)
    monkeypatch.setattr(
        discovery,
        "standard_optimal_causation_entropy",
        lambda *args, **kwargs: [],
    )

    discovery.information_screened_optimal_causation_entropy(
        X,
        Y,
        Z_init,
        np.random.default_rng(0),
        retention=0.5,
    )

    assert len(rescue_conditioning) == 2
    for z in rescue_conditioning:
        np.testing.assert_allclose(z, Z_init)


def test_information_screened_rejects_invalid_retention():
    X = np.ones((20, 2))
    Y = np.ones((20, 1))
    Z = np.ones((20, 1))

    for value in (0.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="retention"):
            discovery.information_screened_optimal_causation_entropy(
                X,
                Y,
                Z,
                np.random.default_rng(0),
                retention=value,
            )


def test_discover_information_screened_uses_external_candidates(monkeypatch):
    captured = []

    def fake_screened(X, Y, Z_init, rng, **kwargs):
        captured.append((X.shape, Y.shape, Z_init.shape, kwargs["retention"]))
        return []

    monkeypatch.setattr(
        discovery,
        "information_screened_optimal_causation_entropy",
        fake_screened,
    )

    data = np.random.default_rng(0).normal(size=(40, 3))
    graph = discovery.discover_network(
        data,
        method="information_screened",
        max_lag=2,
        screen_retention=0.35,
        n_shuffles=3,
        random_state=0,
    )

    assert len(graph) == 3
    assert len(captured) == 3
    for x_shape, y_shape, z_shape, retention in captured:
        assert x_shape == (38, 4)  # (n-1) * max_lag external candidates
        assert y_shape == (38, 1)
        assert z_shape == (38, 2)
        assert retention == 0.35


def test_backward_preserves_initial_conditioning_for_screened_refinement(monkeypatch):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 3))
    Y = rng.normal(size=(50, 1))
    Z_init = rng.normal(size=(50, 2))
    seen = []

    def fake_cmi(_x, _y, z, **kwargs):
        seen.append(z.copy())
        return 1.0

    monkeypatch.setattr(discovery, "conditional_mutual_information", fake_cmi)
    monkeypatch.setattr(
        discovery,
        "shuffle_test",
        lambda *args, **kwargs: {
            "Threshold": 0.0,
            "Value": 1.0,
            "Pass": True,
            "P_value": 0.0,
        },
    )

    selected = discovery.backward(
        X,
        Y,
        [0, 1],
        np.random.default_rng(11),
        n_shuffles=3,
        Z_init=Z_init,
    )

    assert selected == [0, 1]
    assert len(seen) == 2
    for z in seen:
        np.testing.assert_allclose(z[:, :2], Z_init)


def test_standard_discovery_does_not_retest_target_history(monkeypatch):
    calls = []

    def fake_standard(X, Y, Z_init, rng, *args, **kwargs):
        calls.append((X.copy(), Z_init.copy()))
        return []

    monkeypatch.setattr(
        discovery,
        "standard_optimal_causation_entropy",
        fake_standard,
    )

    data = np.arange(120, dtype=float).reshape(40, 3)
    discovery.discover_network(
        data,
        method="standard",
        max_lag=2,
        n_shuffles=3,
    )

    assert len(calls) == 3
    for X_candidates, Z_init in calls:
        assert X_candidates.shape == (38, 4)
        assert Z_init.shape == (38, 2)
        for candidate in X_candidates.T:
            assert not any(
                np.array_equal(candidate, conditioned)
                for conditioned in Z_init.T
            )
