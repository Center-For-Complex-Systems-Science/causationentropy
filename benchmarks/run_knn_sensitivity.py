"""Benchmark kNN causal-discovery sensitivity to ``k`` and series length.

Usage::

    python -m benchmarks.run_knn_sensitivity --profile quick
    python -m benchmarks.run_knn_sensitivity --profile full

The full sweep is intentionally opt-in; kNN discovery can be expensive.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from benchmarks.adapters import generate_case
from benchmarks.metrics import normalize_ce_edges, score_edges

PROFILES = {
    "quick": {
        "n": 5,
        "lengths": [120],
        "ks": [2],
        "seeds": [0],
        "n_shuffles": 10,
    },
    "full": {
        "n": 5,
        "lengths": [300, 500, 1000],
        "ks": [2, 5, 10],
        "seeds": [0, 1, 2, 3, 4],
        "n_shuffles": 100,
    },
}


def run_sweep(n, lengths, ks, seeds, max_lag, n_shuffles, verbose=True):
    """Run all kNN discovery cases and return per-seed result rows."""
    from benchmarks.adapters import run_causation_entropy

    rows = []
    for T in lengths:
        for k in ks:
            for seed in seeds:
                data, truth, node_order = generate_case(n, T, seed)
                graph, runtime = run_causation_entropy(
                    data,
                    max_lag=max_lag,
                    seed=seed,
                    n_shuffles=n_shuffles,
                    information="knn",
                    k=k,
                )
                scores = score_edges(truth, normalize_ce_edges(graph, node_order))
                row = {
                    "n": n,
                    "T": T,
                    "k": k,
                    "max_lag": max_lag,
                    "seed": seed,
                    "method": "causationentropy-knn-oCSE",
                    "metric": "chebyshev",
                    "n_shuffles": n_shuffles,
                    "runtime_s": runtime,
                    **scores,
                }
                rows.append(row)
                if verbose:
                    print(
                        f"kNN n={n} T={T} k={k} seed={seed}: "
                        f"{runtime:.1f}s TP={scores['cross_TP']} "
                        f"FP={scores['cross_FP']} FN={scores['cross_FN']}",
                        flush=True,
                    )
    return rows


def aggregate(rows):
    """Aggregate seed-level results by ``(n, T, k)``."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["n"], row["T"], row["k"]), []).append(row)

    summary = []
    for (n, T, k), group in sorted(grouped.items()):
        tp = sum(row["cross_TP"] for row in group)
        fp = sum(row["cross_FP"] for row in group)
        fn = sum(row["cross_FN"] for row in group)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        summary.append(
            {
                "n": n,
                "T": T,
                "k": k,
                "n_seeds": len(group),
                "runtime_median_s": statistics.median(
                    row["runtime_s"] for row in group
                ),
                "cross_TP": tp,
                "cross_FP": fp,
                "cross_FN": fn,
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
        )
    return summary


def markdown_table(summary, max_lag, n_shuffles):
    lines = [
        "# kNN discovery sensitivity benchmark",
        "",
        f"Generator: `linear_stochastic_gaussian_process(rho=0.7, p=0.2)`, "
        f"`max_lag={max_lag}`; kNN metric: Chebyshev; "
        f"`n_shuffles={n_shuffles}`.",
        "Runtime is median across seeds; graph metrics are micro-averaged. "
        "Exact `(source, target, lag)` matches are required; cross-links are "
        "reported, with self-lags excluded from the main metrics.",
        "",
        "| n | T | k | seeds | median runtime s | TP | FP | FN | precision | recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['n']} | {row['T']} | {row['k']} | {row['n_seeds']} | "
            f"{row['runtime_median_s']:.2f} | {row['cross_TP']} | "
            f"{row['cross_FP']} | {row['cross_FN']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['F1']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument("--output", default="benchmark-results/knn-sensitivity")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--lengths", type=int, nargs="+", default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument("--n-shuffles", type=int, default=None)
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]
    n = args.n if args.n is not None else profile["n"]
    lengths = args.lengths if args.lengths is not None else profile["lengths"]
    ks = args.ks if args.ks is not None else profile["ks"]
    seeds = args.seeds if args.seeds is not None else profile["seeds"]
    n_shuffles = (
        args.n_shuffles if args.n_shuffles is not None else profile["n_shuffles"]
    )
    if n < 2 or any(T <= args.max_lag for T in lengths):
        parser.error("n must be >= 2 and every T must be greater than --max-lag")
    if not lengths or not ks or not seeds:
        parser.error(
            "lengths, k values, and seeds must each contain at least one value"
        )
    if any(k < 1 for k in ks) or n_shuffles < 1:
        parser.error("all k values and --n-shuffles must be >= 1")

    rows = run_sweep(n, lengths, ks, seeds, args.max_lag, n_shuffles)
    summary = aggregate(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.json").open("w") as stream:
        json.dump(
            {
                "config": {
                    "profile": args.profile,
                    "n": n,
                    "lengths": lengths,
                    "ks": ks,
                    "seeds": seeds,
                    "max_lag": args.max_lag,
                    "n_shuffles": n_shuffles,
                    "information": "knn",
                    "metric": "chebyshev",
                },
                "summary": summary,
            },
            stream,
            indent=2,
        )
    table = markdown_table(summary, args.max_lag, n_shuffles)
    (output / "SUMMARY.md").write_text(table)
    print()
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
