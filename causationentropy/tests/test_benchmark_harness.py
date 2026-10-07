"""Smoke coverage for the issue #18 benchmark harness (CI-fast)."""

import networkx as nx
import pytest

from benchmarks.adapters import generate_case, run_causation_entropy
from benchmarks.metrics import normalize_ce_edges, score_edges
from benchmarks.run import aggregate, markdown_table


def test_generate_case_truth_matches_graph():
    data, truth, node_order = generate_case(4, 80, seed=0)
    assert data.shape == (80, 4)
    assert node_order == ["X0", "X1", "X2", "X3"]
    G = nx.erdos_renyi_graph(4, 0.2, seed=0, directed=True)
    assert truth == {(u, v, 1) for u, v in G.edges()}


def test_score_edges_exact_match():
    out = score_edges({(0, 1, 1), (1, 2, 1)}, {(0, 1, 1), (0, 2, 2)})
    assert (out["TP"], out["FP"], out["FN"]) == (1, 1, 1)
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(0.5)


def test_tiny_end_to_end_ce_only():
    data, truth, node_order = generate_case(3, 60, seed=0)
    graph, runtime = run_causation_entropy(data, max_lag=1, seed=0, n_shuffles=5)
    assert runtime > 0
    predicted = normalize_ce_edges(graph, node_order)
    assert all(isinstance(e[0], int) and e[2] >= 1 for e in predicted)
    out = score_edges(truth, predicted)
    assert out["TP"] + out["FN"] == len(truth)


def test_pcmci_adapter_smoke():
    pytest.importorskip("tigramite")
    from benchmarks.adapters import run_pcmci_parcorr
    from benchmarks.metrics import normalize_pcmci_edges
    from causationentropy.graph.utils import pcmci_to_networkx

    data, truth, _ = generate_case(3, 60, seed=0)
    results, runtime = run_pcmci_parcorr(data, tau_max=1, pc_alpha=0.05)
    assert runtime > 0
    predicted = normalize_pcmci_edges(pcmci_to_networkx(results))
    out = score_edges(truth, predicted)
    assert out["TP"] + out["FN"] == len(truth)


def test_aggregate_and_markdown():
    rows = [
        {
            "method": "m",
            "n": 5,
            "T": 200,
            "seed": 0,
            "runtime_s": 2.0,
            "TP": 4,
            "FP": 1,
            "FN": 1,
            "precision": 0.8,
            "recall": 0.8,
            "F1": 0.8,
            "n_truth": 5,
            "n_predicted": 5,
        },
        {
            "method": "m",
            "n": 5,
            "T": 200,
            "seed": 1,
            "runtime_s": 4.0,
            "TP": 3,
            "FP": 3,
            "FN": 2,
            "precision": 0.5,
            "recall": 0.6,
            "F1": 0.545,
            "n_truth": 5,
            "n_predicted": 6,
        },
    ]
    summary = aggregate(rows)
    assert len(summary) == 1
    entry = summary[0]
    assert entry["runtime_median_s"] == pytest.approx(3.0)
    assert (entry["TP"], entry["FP"], entry["FN"]) == (7, 4, 3)
    table = markdown_table(summary, max_lag=2, n_shuffles=30, pc_alpha=0.05)
    assert "| m | 5 | 200 | 2 | 3.0 | 7 | 4 | 3 |" in table
