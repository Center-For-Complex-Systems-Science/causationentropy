# Issue #18 benchmark: CausationEntropy vs Tigramite PCMCI

Generator: `linear_stochastic_gaussian_process(rho=0.7, p=0.2)`, `max_lag`/`tau_max`=2.
CE: Gaussian oCSE (`method='standard'`), serial (`n_jobs=1`), `n_shuffles=200`. PCMCI: `ParCorr`, `pc_alpha=0.05`.
Matching: exact `(source, target, lag)`; runtime is the median across seeds, accuracy micro-averaged (TP/FP/FN summed).

| method | n | T | seeds | runtime median s | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| causationentropy-gaussian-oCSE | 5 | 500 | 5 | 1.6 | 14 | 7 | 3 | 0.667 | 0.824 | 0.737 |
| tigramite-pcmci-parcorr | 5 | 500 | 5 | 0.0 | 14 | 11 | 3 | 0.560 | 0.824 | 0.667 |
| causationentropy-gaussian-oCSE | 10 | 500 | 5 | 7.0 | 69 | 32 | 3 | 0.683 | 0.958 | 0.798 |
| tigramite-pcmci-parcorr | 10 | 500 | 5 | 0.2 | 68 | 47 | 4 | 0.591 | 0.944 | 0.727 |
| causationentropy-gaussian-oCSE | 20 | 500 | 5 | 31.1 | 329 | 135 | 55 | 0.709 | 0.857 | 0.776 |
| tigramite-pcmci-parcorr | 20 | 500 | 5 | 0.9 | 324 | 190 | 60 | 0.630 | 0.844 | 0.722 |
