import inspect
from unittest.mock import Mock

import networkx as nx
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


def test_information_screened_retention_is_soft_and_preserves_proposal(monkeypatch):
    X = np.column_stack(
        [
            np.linspace(0.0, 1.0, 24),
            np.linspace(1.0, 2.0, 24),
            np.linspace(2.0, 3.0, 24),
            np.linspace(3.0, 4.0, 24),
            np.linspace(4.0, 5.0, 24),
        ]
    )
    Y = np.linspace(-1.0, 1.0, 24).reshape(-1, 1)
    Z_init = np.linspace(10.0, 11.0, 24).reshape(-1, 1)

    # The proposal already exceeds the nominal 40% target (3 > ceil(0.4 * 5)=2).
    # Preserve the proposal and leave one rescue slot rather than truncating it.
    monkeypatch.setattr(
        discovery,
        "information_lasso_optimal_causation_entropy",
        lambda *args, **kwargs: [0, 1, 2],
    )

    rescue_conditioning = []

    def fake_cmi(x, _y, z, **kwargs):
        rescue_conditioning.append(z.copy())
        # Candidate 4 wins the single rescue slot over candidate 3.
        return 2.0 if np.isclose(x[0, 0], 4.0) else 1.0

    monkeypatch.setattr(discovery, "conditional_mutual_information", fake_cmi)

    captured = {}

    def fake_standard(x, _y, z_init, _rng, **kwargs):
        captured["screened"] = x.copy()
        captured["z_init"] = z_init.copy()
        return []

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
        retention=0.40,
    )

    assert selected == []
    np.testing.assert_allclose(captured["screened"], X[:, [0, 1, 2, 4]])
    np.testing.assert_allclose(captured["z_init"], Z_init)

    expected_conditioning = np.hstack((Z_init, X[:, [0, 1, 2]]))
    assert len(rescue_conditioning) == 2
    for z in rescue_conditioning:
        np.testing.assert_allclose(z, expected_conditioning)


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
                np.array_equal(candidate, conditioned) for conditioned in Z_init.T
            )


def test_discover_network_preserves_legacy_positional_arguments():
    """Old calls must still bind n_jobs, random_state and reporting correctly."""
    legacy = [
        "data",
        "method",
        "information",
        "max_lag",
        "alpha_forward",
        "alpha_backward",
        "metric",
        "bandwidth",
        "k_means",
        "n_shuffles",
        "n_jobs",
        "random_state",
        "only_return_significant",
    ]
    assert list(inspect.signature(discovery.discover_network).parameters) == (
        legacy + ["screen_retention"]
    )
    data = np.random.default_rng(17).normal(size=(40, 3))
    positional = discovery.discover_network(
        data,
        "standard",
        "gaussian",
        1,
        0.05,
        0.05,
        None,
        "silverman",
        5,
        5,
        1,
        17,
        False,
    )
    named = discovery.discover_network(
        data,
        method="standard",
        information="gaussian",
        max_lag=1,
        n_shuffles=5,
        n_jobs=1,
        random_state=17,
        only_return_significant=False,
    )
    assert nx.utils.graphs_equal(positional, named)
    assert positional.number_of_edges() == 6


@pytest.mark.parametrize("method", ["standard", "information_screened"])
@pytest.mark.parametrize("only_significant", [True, False])
def test_reporting_preserves_each_targets_history(
    monkeypatch, method, only_significant
):
    """Deferred reporting must not reuse the final target's baseline or mask."""
    data = np.random.default_rng(33).normal(size=(40, 3))
    calls = []

    def check_conditioning(x, y, z, method, metric, k, bandwidth):
        target = next(i for i in range(3) if np.array_equal(y[:, 0], data[2:, i]))
        expected = np.column_stack((data[1:-1, target], data[:-2, target]))
        np.testing.assert_array_equal(z[:, :2], expected)
        assert not any(np.array_equal(x[:, 0], col) for col in expected.T)
        calls.append(target)
        return 0.5

    monkeypatch.setattr(
        discovery, "standard_optimal_causation_entropy", Mock(return_value=[0])
    )
    monkeypatch.setattr(
        discovery,
        "information_screened_optimal_causation_entropy",
        Mock(return_value=[0]),
    )
    monkeypatch.setattr(discovery, "conditional_mutual_information", check_conditioning)
    monkeypatch.setattr(discovery, "shuffle_test", Mock(return_value={"P_value": 0.01}))
    graph = discovery.discover_network(
        data,
        method=method,
        max_lag=2,
        only_return_significant=only_significant,
    )
    per_target = 1 if only_significant else 4
    assert graph.number_of_edges() == 3 * per_target
    assert all(calls.count(target) == per_target for target in range(3))
    assert all(source != target for source, target in graph.edges())


def test_standard_passes_history_to_backward(monkeypatch):
    """The standard method, including restricted refinement, forwards its baseline."""
    data = np.random.default_rng(0).normal(size=(40, 4))
    backward = Mock(return_value=[0])
    monkeypatch.setattr(discovery, "standard_forward", Mock(return_value=[0]))
    monkeypatch.setattr(discovery, "backward", backward)
    discovery.standard_optimal_causation_entropy(
        data[:, :2],
        data[:, [2]],
        data[:, [3]],
        np.random.default_rng(1),
    )
    np.testing.assert_array_equal(backward.call_args.kwargs["Z_init"], data[:, [3]])


@pytest.mark.parametrize("method", ["standard", "information_screened"])
def test_reporting_mode_keeps_selected_edges_identical(method):
    """Compose the history correction with deferred reporting's RNG guarantee."""
    rng = np.random.default_rng(12)
    data = rng.normal(size=(120, 3))
    data[1:, 1] = 0.9 * data[:-1, 0] + 0.1 * rng.normal(size=119)
    selected = discovery.discover_network(data, method=method, max_lag=1, n_shuffles=19)
    all_edges = discovery.discover_network(
        data,
        method=method,
        max_lag=1,
        n_shuffles=19,
        only_return_significant=False,
    )
    retained = nx.MultiDiGraph()
    retained.add_nodes_from(all_edges.nodes(data=True))
    for source, target, attrs in all_edges.edges(data=True):
        if attrs["significant"]:
            attrs = {key: value for key, value in attrs.items() if key != "significant"}
            retained.add_edge(source, target, **attrs)
    assert nx.utils.graphs_equal(selected, retained)
