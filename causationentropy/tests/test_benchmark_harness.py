"""Smoke coverage for the issue #18 benchmark harness (CI-fast)."""

import networkx as nx
import pytest

pytest.importorskip("benchmarks")

from benchmarks.adapters import generate_case, run_causation_entropy
from benchmarks.metrics import normalize_ce_edges, score_edges
from benchmarks.run import aggregate, markdown_table, run_profile


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


def test_score_edges_self_lag_breakdown():
    truth = {(0, 0, 1), (0, 1, 1), (1, 2, 1)}
    predicted = {(0, 0, 1), (0, 1, 1), (2, 2, 1)}
    out = score_edges(truth, predicted)
    assert (out["cross_TP"], out["cross_FP"], out["cross_FN"]) == (1, 0, 1)
    assert (out["self_TP"], out["self_FP"], out["self_FN"]) == (1, 1, 0)
    assert out["cross_TP"] + out["self_TP"] == out["TP"]


def test_run_profile_continues_without_tigramite(monkeypatch):
    def missing_tigramite(*args, **kwargs):
        raise ImportError("tigramite is required for the PCMCI baseline.")

    import benchmarks.run as bench_run

    monkeypatch.setattr(bench_run, "run_pcmci_parcorr", missing_tigramite)
    rows = run_profile(
        sizes=[(3, 60), (4, 60)],
        seeds=[0, 1],
        max_lag=1,
        n_shuffles=5,
        pc_alpha=0.05,
        skip_pcmci=False,
        verbose=False,
    )
    assert len(rows) == 4
    assert all(r["method"] == "causationentropy-gaussian-oCSE" for r in rows)


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


def _row(
    seed, runtime, tp, fp, fn, cross_tp, cross_fp, cross_fn, self_tp, self_fp, self_fn
):
    return {
        "method": "m",
        "n": 5,
        "T": 200,
        "seed": seed,
        "runtime_s": runtime,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": 0.0,
        "recall": 0.0,
        "F1": 0.0,
        "n_truth": tp + fn,
        "n_predicted": tp + fp,
        "cross_TP": cross_tp,
        "cross_FP": cross_fp,
        "cross_FN": cross_fn,
        "cross_precision": 0.0,
        "cross_recall": 0.0,
        "cross_F1": 0.0,
        "self_TP": self_tp,
        "self_FP": self_fp,
        "self_FN": self_fn,
    }


def test_aggregate_and_markdown():
    rows = [
        _row(0, 2.0, 4, 1, 1, 4, 1, 1, 0, 0, 0),
        _row(1, 4.0, 3, 3, 2, 2, 2, 2, 1, 1, 0),
    ]
    summary = aggregate(rows)
    assert len(summary) == 1
    entry = summary[0]
    assert entry["runtime_median_s"] == pytest.approx(3.0)
    assert (entry["TP"], entry["FP"], entry["FN"]) == (7, 4, 3)
    assert (entry["cross_TP"], entry["cross_FP"], entry["cross_FN"]) == (6, 3, 3)
    assert (entry["self_TP"], entry["self_FP"], entry["self_FN"]) == (1, 1, 0)
    assert entry["cross_precision"] == pytest.approx(6 / 9)
    table = markdown_table(summary, max_lag=2, n_shuffles=30, pc_alpha=0.05)
    assert "| m | 5 | 200 | 2 | 3.0 | 6 | 3 | 3 |" in table
    assert "self TP" in table
