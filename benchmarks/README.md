# Benchmark harness for issue #18

Reproducible runtime/accuracy comparison: CausationEntropy Gaussian oCSE
vs Tigramite PCMCI + ParCorr on identical synthetic data.

## Quick start

```bash
python -m benchmarks.run --profile quick --output benchmark-results
python -m benchmarks.run --profile full --output benchmark-results
```

## Conventions (answers to the open questions on #18)

* **Generator:** `linear_stochastic_gaussian_process(rho=0.7, p=0.2)`,
  explicit `erdos_renyi_graph(n, p, seed, directed=True)` ground truth.
  With `p=0.2`, `n=20` yields ~77 true edges, matching the issue report.
* **Edge matching:** exact `(source, target, lag)`; all ground-truth
  edges are lag 1, so lag-2 discoveries count as false positives.
* **Self-lags:** included in the main counts (the generator produces no
  self-loops, so any self-lag discovery is a false positive).
* **CE config:** `method="standard"`, Gaussian, serial (`n_jobs=1`),
  `alpha_forward=alpha_backward=0.05`, profile-dependent `n_shuffles`.
* **PCMCI config:** `ParCorr`, `tau_max=max_lag`, `pc_alpha=0.05`.
  Only directed links at lag >= 1 count; undirected/conflicting and
  contemporaneous links are ignored.
* **Significance comparability:** end-to-end defaults (permutation vs
  analytical). An analytical Gaussian backend is out of scope here.
* **Repetitions:** fixed seeds; runtime = median, accuracy =
  micro-averaged (TP/FP/FN summed across seeds).
* **Execution:** serial comparison is primary. Parallel (`n_jobs`) results
  from #28 can be added as a separate table later.
* **Profiles:** `quick` (smoke), `small` (accuracy signal),
  `full` (the issue table: `n=5,10,20`, `T=500`, 5 seeds).

## Outputs

`--output` receives `run_results.csv` (per-seed rows), `summary.json`
(config + aggregates), and `SUMMARY.md` (the comparison table).
Raw result directories are local artifacts (see `.gitignore`); copy the
final table to `benchmarks/RESULTS_issue18.md` when publishing.

## CI

The full benchmark does not run in CI. CI covers the generators,
adapters, and metrics on a tiny dataset via
`causationentropy/tests/test_benchmark_harness.py`.
