# Causal discovery on emergent regimes — findings

Setup: coupled Kuramoto–Sakaguchi oscillators driven into four emergent regimes
(cyclops, metastability, traveling wave, chimera) with a known ground-truth
coupling graph. Methods scored on node-edge recovery: oCSE, oCSE+OP (oCSE
conditioning additionally on the global order parameter Z_m(t), m=1,2), PCMCI,
Linear Granger, VARLiNGAM — all on the sine basis. 2 seeds/regime.

## Mean over seeds

| Regime | metric | oCSE | oCSE+OP | PCMCI | Granger | VARLiNGAM |
|---|---|---|---|---|---|---|
| metastable | FPR | 0.022 | 0.023 | 0.615 | 0.902 | 0.015 |
|            | TPR | 0.177 | 0.156 | 0.837 | 0.963 | 0.049 |
| traveling-wave | FPR | 0.019 | **0.006** | 0.508 | 0.735 | 0.004 |
|                | TPR | 0.250 | 0.219 | 0.875 | 1.000 | 0.000 |
| chimera | FPR | 0.025 | **0.012** | (skip) | 0.189 | (skip) |
|         | TPR | 0.014 | 0.010 | (skip) | 0.281 | (skip) |
| cyclops | FPR | — complete graph, FPR undefined — |
|         | TPR | 0.114 | 0.091 | 0.468 | 0.491 | 0.014 |

PCMCI/VARLiNGAM skipped on chimera (intractable on the basis at this N).

## Findings

1. **oCSE keeps its low-false-positive discipline on every emergent regime**
   (FPR 0.01–0.03), 20–40x below PCMCI (0.5–0.6) and Granger (0.7–0.9). PCMCI
   and Granger reach high TPR but only by over-connecting — the same artifact
   seen on partial-sync Kuramoto.

2. **Order-parameter conditioning lowers oCSE's FPR further whenever there are
   false positives to remove**: traveling-wave 0.019 -> 0.006 (3x), chimera
   0.025 -> 0.012 (2x, both seeds); neutral on metastable where oCSE is already
   at the floor (~0.02). It is a precision-favouring knob, not free — TPR drops
   slightly in every regime (e.g. traveling-wave 0.250 -> 0.219), consistent with
   adding a common-mode variable to the conditioning set.

3. **Emergent regimes are hard for every method, and chimera is the synchronization
   barrier in the open.** oCSE's TPR collapses to ~1% on the chimera: inside the
   coherent domain the oscillators are phase-locked, so the coupling features
   sin(theta_j - theta_i) are ~constant and carry no information — direct links
   are statistically invisible. This is the same obstruction documented for
   synchronized Rössler. Conditioning on the order parameter cannot manufacture
   signal the dynamics destroyed; it only trims the residual false positives.

4. **VARLiNGAM recovers essentially nothing** (TPR ~0) across all regimes;
   **cyclops** has a complete-graph ground truth, so FPR is undefined and only
   recall is meaningful there.

Net: oCSE (optionally + order parameter) is the method that does not flood the
graph under emergent dynamics, but the emergent regimes — chimera especially —
expose a hard limit of observational causal discovery wherever synchronization
removes the statistical signature of a real coupling.
