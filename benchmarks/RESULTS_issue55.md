# Preliminary kNN discovery sensitivity results

**Status:** preliminary; rerun after PR #53 is merged. These measurements use
the kNN implementation on `main` at `271e1da`, before the tie-breaking and
Chebyshev-default fixes in #53. Chebyshev distance was explicitly selected.

Command:

```bash
python -m benchmarks.run_knn_sensitivity \
  --n 5 --lengths 300 500 --ks 2 5 --seeds 0 1 2 \
  --max-lag 2 --n-shuffles 50 \
  --output benchmark-results/knn-sensitivity-pre53
```

Generator: `linear_stochastic_gaussian_process(rho=0.7, p=0.2)`, with the
same seeded ground-truth graph for each estimator setting. Edge matches require
exact `(source, target, lag)` equality. Counts and precision/recall/F1 are
micro-averaged across seeds; runtime is the median. The metrics below are
cross-links only; self-lags are excluded.

| n | T | k | seeds | median runtime (s) | TP | FP | FN | precision | recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 300 | 2 | 3 | 4.61 | 2 | 1 | 6 | 0.667 | 0.250 | 0.364 |
| 5 | 300 | 5 | 3 | 4.67 | 4 | 0 | 4 | 1.000 | 0.500 | 0.667 |
| 5 | 500 | 2 | 3 | 13.73 | 3 | 0 | 5 | 1.000 | 0.375 | 0.545 |
| 5 | 500 | 5 | 3 | 13.90 | 4 | 0 | 4 | 1.000 | 0.500 | 0.667 |

In this small preliminary sweep, increasing `T` from 300 to 500 increased
median runtime substantially, while `k=5` had higher recall and F1 than `k=2`
at both lengths. Three seeds are not enough to draw a general conclusion;
the results are descriptive and should be refreshed against the finalized
estimator before being treated as the issue's final benchmark.
