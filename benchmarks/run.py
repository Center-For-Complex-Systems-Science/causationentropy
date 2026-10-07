"""Reproducible runtime/accuracy benchmark for issue #18.

Usage::

    python -m benchmarks.run --profile quick --output benchmark-results
    python -m benchmarks.run --profile full --output benchmark-results

The full benchmark runs locally (it takes several minutes); CI should only
exercise the tiny smoke path via ``causationentropy/tests/test_benchmark_harness.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from benchmarks.adapters import generate_case, run_causation_entropy, run_pcmci_parcorr
from benchmarks.metrics import (
    normalize_ce_edges,
    normalize_pcmci_edges,
    score_edges,
)

PROFILES = {
    # Smoke: seconds, exercises the whole pipeline.
    "quick": {
        "sizes": [(5, 200)],
        "seeds": [0, 1],
        "n_shuffles": 30,
    },
    # Reviewable accuracy signal without the n=20 cost.
    "small": {
        "sizes": [(5, 500), (10, 500)],
        "seeds": [0, 1, 2],
        "n_shuffles": 50,
    },
    # The issue #18 table: n=5, 10, 20 at T=500.
    "full": {
        "sizes": [(5, 500), (10, 500), (20, 500)],
        "seeds": [0, 1, 2, 3, 4],
        "n_shuffles": 200,
    },
}

CE_CONFIG = {
    "method": "standard",
    "information": "gaussian",
    "alpha_forward": 0.05,
    "alpha_backward": 0.05,
}
PCMCI_CONFIG = {"pc_alpha": 0.05}


def run_profile(
    sizes,
    seeds,
    max_lag: int,
    n_shuffles: int,
    pc_alpha: float,
    skip_pcmci: bool,
    verbose: bool = True,
):
    """Run all cases; return a list of per-seed result rows."""
    from causationentropy.graph.utils import pcmci_to_networkx

    rows = []
    for n, T in sizes:
        for seed in seeds:
            data, truth, node_order = generate_case(n, T, seed)
            ce_graph, ce_time = run_causation_entropy(
                data,
                max_lag=max_lag,
                seed=seed,
                n_shuffles=n_shuffles,
                alpha_forward=CE_CONFIG["alpha_forward"],
                alpha_backward=CE_CONFIG["alpha_backward"],
            )
            ce_scores = score_edges(truth, normalize_ce_edges(ce_graph, node_order))
            row = {
                "n": n,
                "T": T,
                "max_lag": max_lag,
                "seed": seed,
                "method": "causationentropy-gaussian-oCSE",
                "runtime_s": ce_time,
                **ce_scores,
            }
            rows.append(row)
            if verbose:
                print(
                    f"CE      n={n} T={T} seed={seed}: "
                    f"{ce_time:.1f}s TP={ce_scores['TP']} "
                    f"FP={ce_scores['FP']} FN={ce_scores['FN']}",
                    flush=True,
                )

            if skip_pcmci:
                continue
            try:
                results, pcmci_time = run_pcmci_parcorr(
                    data, tau_max=max_lag, pc_alpha=pc_alpha
                )
            except ImportError as exc:
                print(f"WARNING: {exc} Skipping PCMCI baseline.", file=sys.stderr)
                return rows
            pcmci_graph = pcmci_to_networkx(results)
            pcmci_scores = score_edges(truth, normalize_pcmci_edges(pcmci_graph))
            row = {
                "n": n,
                "T": T,
                "max_lag": max_lag,
                "seed": seed,
                "method": "tigramite-pcmci-parcorr",
                "runtime_s": pcmci_time,
                **pcmci_scores,
            }
            rows.append(row)
            if verbose:
                print(
                    f"PCMCI   n={n} T={T} seed={seed}: "
                    f"{pcmci_time:.1f}s TP={pcmci_scores['TP']} "
                    f"FP={pcmci_scores['FP']} FN={pcmci_scores['FN']}",
                    flush=True,
                )
    return rows


def aggregate(rows):
    """Aggregate per-seed rows to one row per (method, n, T).

    Runtime is the median across seeds; accuracy is micro-averaged
    (TP/FP/FN summed, then precision/recall/F1 derived).
    """
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["n"], row["T"]), []).append(row)
    summary = []
    for (method, n, T), group in sorted(
        groups.items(), key=lambda kv: (kv[0][1], kv[0][0])
    ):
        runtimes = sorted(r["runtime_s"] for r in group)
        tp = sum(r["TP"] for r in group)
        fp = sum(r["FP"] for r in group)
        fn = sum(r["FN"] for r in group)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        summary.append(
            {
                "method": method,
                "n": n,
                "T": T,
                "n_seeds": len(group),
                "runtime_median_s": statistics.median(runtimes),
                "runtime_min_s": runtimes[0],
                "runtime_max_s": runtimes[-1],
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "n_truth": tp + fn,
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
        )
    return summary


def markdown_table(summary, max_lag: int, n_shuffles: int, pc_alpha: float) -> str:
    lines = [
        "# Issue #18 benchmark: CausationEntropy vs Tigramite PCMCI",
        "",
        f"Generator: `linear_stochastic_gaussian_process(rho=0.7, p=0.2)`, "
        f"`max_lag`/`tau_max`={max_lag}.",
        f"CE: Gaussian oCSE (`method='standard'`), serial (`n_jobs=1`), "
        f"`n_shuffles={n_shuffles}`. PCMCI: `ParCorr`, `pc_alpha={pc_alpha}`.",
        "Matching: exact `(source, target, lag)`; runtime is the median "
        "across seeds, accuracy micro-averaged (TP/FP/FN summed).",
        "",
        "| method | n | T | seeds | runtime median s | TP | FP | FN | precision | recall | F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in summary:
        lines.append(
            f"| {entry['method']} | {entry['n']} | {entry['T']} | "
            f"{entry['n_seeds']} | {entry['runtime_median_s']:.1f} | "
            f"{entry['TP']} | {entry['FP']} | {entry['FN']} | "
            f"{entry['precision']:.3f} | {entry['recall']:.3f} | {entry['F1']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    parser.add_argument(
        "--output",
        default="benchmark-results",
        help="Directory for run_results.csv, summary.json, SUMMARY.md",
    )
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument(
        "--n-shuffles", type=int, default=None, help="Override the profile default"
    )
    parser.add_argument("--pc-alpha", type=float, default=PCMCI_CONFIG["pc_alpha"])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Override the profile seed list",
    )
    parser.add_argument("--skip-pcmci", action="store_true")
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]
    seeds = args.seeds if args.seeds is not None else profile["seeds"]
    n_shuffles = (
        args.n_shuffles if args.n_shuffles is not None else profile["n_shuffles"]
    )

    rows = run_profile(
        profile["sizes"],
        seeds,
        args.max_lag,
        n_shuffles,
        args.pc_alpha,
        args.skip_pcmci,
    )
    summary = aggregate(rows)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run_results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out / "summary.json", "w") as fh:
        json.dump(
            {
                "config": {
                    "profile": args.profile,
                    "max_lag": args.max_lag,
                    "n_shuffles": n_shuffles,
                    "pc_alpha": args.pc_alpha,
                    "seeds": seeds,
                    **CE_CONFIG,
                },
                "summary": summary,
            },
            fh,
            indent=2,
        )
    table = markdown_table(summary, args.max_lag, n_shuffles, args.pc_alpha)
    with open(out / "SUMMARY.md", "w") as fh:
        fh.write(table)
    print()
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
