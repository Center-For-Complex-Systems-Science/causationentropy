# Issue #18 benchmark: CausationEntropy vs Tigramite PCMCI

Generator: `linear_stochastic_gaussian_process(rho=0.7, p=0.2)`, `max_lag`/`tau_max`=2.
CE: Gaussian oCSE (`method='standard'`), serial (`n_jobs=1`), `n_shuffles=200`. PCMCI: `ParCorr`, `pc_alpha=0.05`.
Matching: exact `(source, target, lag)`. Cross-links are the main metric; self-lags are reported separately because standard oCSE conditions on the target's own lags and cannot discover them. Runtime is the median across seeds, accuracy micro-averaged (TP/FP/FN summed).

| method | n | T | seeds | runtime median s | cross TP | cross FP | cross FN | cross precision | cross recall | cross F1 | self TP | self FP | self FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| causationentropy-gaussian-oCSE | 5 | 500 | 5 | 1.6 | 14 | 7 | 3 | 0.667 | 0.824 | 0.737 | 0 | 0 | 0 |
| tigramite-pcmci-parcorr | 5 | 500 | 5 | 0.0 | 14 | 8 | 3 | 0.636 | 0.824 | 0.718 | 0 | 3 | 0 |
| causationentropy-gaussian-oCSE | 10 | 500 | 5 | 6.9 | 69 | 32 | 3 | 0.683 | 0.958 | 0.798 | 0 | 0 | 0 |
| tigramite-pcmci-parcorr | 10 | 500 | 5 | 0.2 | 68 | 45 | 4 | 0.602 | 0.944 | 0.735 | 0 | 2 | 0 |
| causationentropy-gaussian-oCSE | 20 | 500 | 5 | 31.9 | 329 | 135 | 55 | 0.709 | 0.857 | 0.776 | 0 | 0 | 0 |
| tigramite-pcmci-parcorr | 20 | 500 | 5 | 0.9 | 324 | 179 | 60 | 0.644 | 0.844 | 0.731 | 0 | 11 | 0 |
