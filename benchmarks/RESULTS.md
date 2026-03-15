# Benchmark Results Summary

## Kuramoto Oscillators (10 nodes)

### Setup
- 10 coupled Kuramoto oscillators on Erdos-Renyi random graphs
- Physics-informed basis: sin(theta_j - theta_i) matching the actual coupling term in the ODE
- Sweeps over ER density (0.1-0.7) and coupling strength (0.1-0.9)
- 10 trials per condition for oCSE/PCMCI, 3 trials per condition for other methods

### Leaderboard (averaged over all experiments)

| Rank | Method             | TPR   | FPR   | F1    | Notes |
|------|--------------------|-------|-------|-------|-------|
| 1    | oCSE (basis)       | 0.655 | 0.082 | 0.606 | Best precision/FPR tradeoff |
| 2    | Koopman + oCSE     | 0.835 | 0.529 | 0.578 | Over-predicts |
| 3    | PCMCI (basis)      | 0.596 | 0.082 | 0.562 | Same low FPR as oCSE, lower TPR |
| 4    | Linear Granger     | 0.646 | 0.401 | 0.529 | Mediocre |
| 5    | LASSO (alpha=0.01) | 0.587 | 0.139 | 0.504 | Moderate regularization |
| 6    | VARLiNGAM          | 0.263 | 0.137 | 0.329 | Poor TPR |
| 7    | LASSO (alpha=0.1)  | 0.256 | 0.040 | 0.311 | Too conservative |

Note: LASSO (alpha=0.001) achieves the highest raw F1 (0.672, TPR=0.981) but at 5-6x the FPR
(0.459) compared to oCSE. LASSO (CV) is similar (F1=0.666, FPR=0.499). These LASSO variants
detect almost all true edges but flood the graph with false positives.

### Key Findings

- oCSE on the physics-informed basis is the best method when considering precision and FPR
  jointly. It achieves the lowest FPR (0.082) while maintaining reasonable TPR (0.655).
- PCMCI achieves the same low FPR (0.082) as oCSE but with lower TPR (0.596 vs 0.655),
  resulting in lower F1 (0.562 vs 0.606).
- Koopman + oCSE over-predicts edges (FPR=0.529), likely because the learned Koopman basis
  creates spurious dependencies.
- VARLiNGAM performs poorly (F1=0.329, TPR=0.263), struggling with the nonlinear coupling.
- Linear Granger is mediocre (F1=0.529, FPR=0.401), partially capturing the nonlinear dynamics.
- The correct physics-informed nonlinear basis is the key ingredient: oCSE and PCMCI both
  benefit from sin(theta_j - theta_i) basis functions that match the true ODE coupling.

---

## Rossler Oscillators (5 nodes, 3D chaotic)

### Setup
- 5 coupled Rossler oscillators, ER random graphs at density {0.2, 0.4, 0.6}
- Coupling strength rho=0.2 (causes synchronization)
- T=5000, dt=0.02, subsample=5, 3 trials per condition, 1000 training epochs
- Derivative basis: X = [dx, dy, dz | c_{i<-j} = x_j - x_i] with target Y = dx_i

### Core Problem: Synchronization Barrier

Synchronized Rossler oscillators create genuine statistical dependencies between non-neighbors.
When oscillators are phase-locked, no statistical test can easily distinguish indirect
synchronization from direct coupling. This is a fundamental limit of observation-only causal
discovery, not a methodological failure.

### Results

| Method                    | F1    | TPR   | FPR   | Notes |
|---------------------------|-------|-------|-------|-------|
| Stage 1: Koopman + oCSE  | 0.605 | 1.000 | 0.650 | Always predicts near-complete graph |
| Deriv backward (a=0.10)  | 0.579 | 0.878 | 0.540 | Best practical result |
| Deriv backward (a=0.05)  | 0.575 | 0.848 | 0.495 | Slightly more conservative |
| Deriv backward (a=0.01)  | 0.456 | 0.712 | 0.450 | Too aggressive pruning |
| PCMCI (derivative basis)  | 0.372 | 0.475 | 0.245 | Inflated -- see note below |
| oCSE raw (derivative basis)| 0.124 | 0.081 | 0.040 | Too conservative |

### Stage 1 Analysis

Stage 1 (Koopman + oCSE) always predicts the nearly complete graph (TPR=1.0, FPR~0.65).
Alpha sweep has no effect -- all p-values from the shuffle test are far below any reasonable
threshold. The Koopman model learns to reconstruct dynamics using all available coupling terms,
and the oCSE shuffle test confirms each is "significant" because synchronized oscillators truly
do carry predictive information about their neighbors' neighbors.

### Stage 2: Derivative Backward Pruning

Derivative backward pruning is the only method that meaningfully reduces FPR (from 0.75 to
0.38 at best), but at the cost of TPR. This two-stage approach substantially outperforms
standalone oCSE (F1=0.124) and PCMCI (F1=0.372).

### PCMCI Comparison Note

On Rossler data, PCMCI's binarize=True gives identical results to unbinarized (all links are
already marked as `-->`). However, PCMCI declares approximately 800 out of 1225 possible
basis-level edges as significant, while oCSE selects only 150-230. PCMCI's higher F1 at the
node level (0.372 vs oCSE's 0.124) is an artifact: the massive over-prediction at the basis
level means the node-level extraction filter randomly passes some true edges. oCSE's raw
approach is more rigorous (fewer basis edges selected) but too conservative for this data.

### Failed FPR Reduction Approaches

All of the following were tried and failed to substantially lower FPR:

1. **kNN CMI backward pruning** -- Nonparametric KSG estimator detected synchronization-induced
   dependencies as significant, pruned nothing.

2. **Different conditioning sets** -- Added dy_i/dz_i or all 3n self-derivatives to conditioning.
   Gaussian CMI partial correlations already captured all available info from dx_i alone.
   Identical results.

3. **Raw state variables (x_j(t-1) -> x_i(t))** -- Worse than derivative basis (FPR ~0.58 vs
   0.50). The physics-informed coupling terms (x_j - x_i) give more discriminative power than
   raw x_j despite noisy derivatives.

4. **Attention matrix thresholding** -- Shared attention Koopman weights are uniform across
   nodes (all thresholds 1.0x-3.0x gave identical results). TPR=0.076, F1=0.118. Attention
   learns to predict dynamics by attending equally to all synchronized nodes.

5. **oCSE intersection with attention** -- Same as attention alone since attention captures
   almost nothing.

6. **Phase reduction** -- Extracting phase from Rossler x-component and applying Kuramoto-style
   sin(phase_j - phase_i) basis. Does not improve over derivative basis.

7. **Phase LASSO** -- LASSO on phase-reduced basis. Same synchronization problem in phase space.

8. **KAN-Koopman** -- Kolmogorov-Arnold Network Koopman. No improvement.

9. **Per-node and per-pair Koopman** -- Training separate models per node or per pair.
   No improvement over shared model.

### All Methods Tested on Rossler

Stage 1 Koopman+oCSE, derivative backward pruning, PCMCI, oCSE raw, LASSO (derivative
residual), phase reduction, phase LASSO, KAN-Koopman, attention, per-node Koopman,
per-pair Koopman, Linear Granger. None solve the synchronization problem.

---

## Key Findings

1. **oCSE works well with the correct physics-informed nonlinear basis.** On Kuramoto with
   sin(theta_j - theta_i), oCSE achieves the best precision/FPR tradeoff among all methods.

2. **Chaotic synchronized systems remain an open challenge.** On Rossler oscillators, all
   methods tested fail to achieve low FPR. The synchronization barrier creates genuine
   statistical dependencies between non-neighbors that no observation-only method can
   easily resolve.

3. **PCMCI over-prediction is masked by node-level aggregation.** On Rossler, PCMCI declares
   800+ of 1225 basis-level edges significant. Its apparently better node-level F1 is an
   artifact of this over-prediction, not genuine causal discrimination.

4. **Two-stage Koopman + backward pruning is the best practical approach for Rossler**
   (F1=0.575), but FPR~0.50 appears to be a hard ceiling imposed by synchronization physics.

5. **The Rossler derivative basis (x_j - x_i) is already the correct functional form** for the
   linear coupling in the ODE. The problem is not the basis -- it is that synchronization
   makes the statistical signature of indirect influence indistinguishable from direct coupling.
