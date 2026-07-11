# Methods

## A. Coupled phase-oscillator model

We study networks of \(N\) coupled phase oscillators governed by a generalized
Kuramoto–Sakaguchi system,
\[
\dot\theta_i \;=\; \omega_i \;+\; K\sum_{j=1}^{N} \widetilde W_{ij}\,
\Big[\sin\!\big(\theta_j-\theta_i-\alpha\big)
\;+\; a_2\sin\!\big(2(\theta_j-\theta_i)-\beta\big)\Big]\;+\;\xi_i(t),
\tag{1}
\]
where \(\theta_i\) is the phase of oscillator \(i\), \(\omega_i\) its natural
frequency, \(K\) the global coupling strength, \(\alpha\) a phase lag (Sakaguchi
term), and \(a_2,\beta\) the amplitude and lag of a second coupling harmonic.
\(\widetilde W\) is the row-normalized coupling matrix,
\(\widetilde W_{ij}=W_{ij}/\sum_k W_{ik}\), and \(\xi_i\) is independent Gaussian
phase noise of standard deviation \(\sigma\). Equation (1) is integrated with a
fourth-order Runge–Kutta scheme at step \(dt=0.05\) after discarding a transient;
\(W_{ij}>0\) denotes a directed coupling \(j\to i\), and the binary support of
\(W\) defines the **ground-truth causal graph** to be reconstructed.

## B. Emergent dynamical regimes

By choosing the coupling topology \(W\), the phase lag \(\alpha\), the harmonic
content \((a_2,\beta)\), and the frequency distribution, Eq. (1) is driven into
four qualitatively distinct emergent regimes (Table I). Parameters follow the
established literature for each phenomenon; cyclops states in particular require
an odd number of identical units and biharmonic repulsive coupling [Munyayev et
al., Phys. Rev. Lett. 130, 107201 (2023)].

**Table I. Emergent-regime parameters.**

| Regime | Coupling \(W\) | \(\omega_i\) | \(\alpha\) | \(a_2,\beta\) | \(\sigma\) |
|---|---|---|---|---|---|
| Chimera | nonlocal ring, radius \(r\) | \(0\) (identical) | \(\pi/2-0.1\) | — | 0.02 |
| Traveling wave | ring, radius \(2\) | \(0\) (identical) | \(0.1\) | — | 0.08 |
| Metastability | modular (3-block SBM) | \(\mathcal N(0,0.5)\) | \(0\) | — | 0.02 |
| Cyclops | all-to-all (\(N\) odd) | \(0\) (identical) | \(1.7\) | \(0.08,-0.3\) | 0.02 |

Chimera states are seeded from a split initial condition (one ring arc coherent,
the complementary arc randomized). Traveling waves are initialized from a
\(q=2\) twisted state \(\theta_i=2\pi q i/N\). For metastability, mild frequency
heterogeneity on a modular graph yields communities that transiently
(de)synchronize.

## C. Regime verification via order parameters

Emergence is confirmed before any causal analysis. The standard Kuramoto order
parameter \(R_1(t)=\big|N^{-1}\sum_j e^{i\theta_j}\big|\) is *blind* to two of
these states — it vanishes for a twisted (traveling-wave) state and is constant
for a rigidly rotating cluster (cyclops) state — so we additionally use the
\(m\)-th Daido order parameter \(R_m(t)=\big|N^{-1}\sum_j e^{im\theta_j}\big|\)
and the generalized twisted order parameter
\(Z_q(t)=N^{-1}\sum_j e^{i(\theta_j-2\pi q j/N)}\). We verified: a chimera by the
spatial local order parameter ranging from \(\sim\!0.1\) (incoherent arc) to
\(\sim\!1\) (coherent arc); a traveling wave by \(|Z_2|\approx1\) with a nonzero
drift of \(\arg Z_2\); metastability by sustained fluctuations of \(R_1\); and a
cyclops state by a \((N{-}1)/2,(N{-}1)/2,1\) cluster partition (two coherent
clusters and a solitary oscillator) with frequency-locked rotation.

## D. Causal network reconstruction

All methods operate on a physics-informed linearization of the dynamics. From
the phase time series we build, for each oscillator \(i\), the regression target
\(v_i(t{+}1)\) (phase velocity) and the coupling features
\(s_{i\leftarrow j}(t)=\sin\!\big(\theta_j(t)-\theta_i(t)\big)\); a basis feature
\(s_{i\leftarrow j}\) selected for target \(i\) maps to the directed edge
\(j\to i\).

- **oCSE.** Optimal causation entropy with forward/backward selection on the
  velocity targets, conditioning on the target's own past. Candidate
  significance is assessed by a Gaussian conditional-mutual-information
  permutation test (\(S\) surrogate shuffles, level \(\alpha\)).
- **oCSE+OP (this work).** Identical to oCSE, but the conditioning set of every
  causation-entropy test is augmented with the global mean-field order
  parameter, supplied as the real time series
  \(\{\mathrm{Re}\,Z_1,\mathrm{Im}\,Z_1,\mathrm{Re}\,Z_2,\mathrm{Im}\,Z_2\}\).
  Because oCSE conditions on its full set, including the order parameter is
  intended to block the common-cause path through the mean field that produces
  spurious correlations between non-adjacent oscillators. The order parameter is
  supplied as a conditioning variable only and is never selectable as an edge.
- **PCMCI** (ParCorr, \(\tau_{\max}=1\)), **Linear Granger** (pairwise \(F\)-test
  conditioning on the target's own past), and **VARLiNGAM** (lag-1) are applied
  to the same basis. Node-level adjacencies are extracted from the basis-level
  results by the same feature-to-edge map used for oCSE.

## E. Evaluation protocol

Reconstructed graphs are compared to the ground-truth support of \(W\) by true-
and false-positive rate (TPR, FPR), precision, and \(F_1\). We use \(N=11\)
(cyclops, the only regime requiring odd \(N\)) and \(N=24\) for the remaining
regimes; for the chimera, a coupling radius \(r=3\) is used so that the
ground-truth graph is sparse (\(\approx\!26\%\) density) and the FPR is
well-posed. Each regime is evaluated over two independent realizations at
significance level \(\alpha=0.05\) with \(S=20\) surrogate shuffles for oCSE and
\(T\approx2000\) retained samples. Two caveats apply. (i) The cyclops coupling is
all-to-all, so the ground-truth graph is complete; it has no true negatives and
the FPR is undefined (only TPR/precision are reported). (ii) On the chimera,
PCMCI and VARLiNGAM are computationally intractable on the expanded basis and are
omitted; oCSE, oCSE+OP, and Granger are compared there.

---

# Results

Figure X summarizes false- and true-positive rates for all methods across the
four regimes; mean values over realizations are collected in Table II.

**Table II. Reconstruction performance (mean over two realizations).**

| Regime | metric | oCSE | oCSE+OP | PCMCI | Granger | VARLiNGAM |
|---|---|---|---|---|---|---|
| Metastable | FPR | 0.022 | 0.023 | 0.615 | 0.902 | 0.015 |
|            | TPR | 0.177 | 0.156 | 0.837 | 0.963 | 0.049 |
| Traveling wave | FPR | 0.019 | **0.006** | 0.508 | 0.735 | 0.004 |
|                | TPR | 0.250 | 0.219 | 0.875 | 1.000 | 0.000 |
| Chimera | FPR | 0.025 | **0.012** | — | 0.189 | — |
|         | TPR | 0.014 | 0.010 | — | 0.281 | — |
| Cyclops\(^\dagger\) | TPR | 0.114 | 0.091 | 0.468 | 0.491 | 0.014 |

\(^\dagger\)Complete ground-truth graph; FPR undefined.

## A. oCSE preserves a low false-positive rate under emergent dynamics

Across every regime, oCSE and oCSE+OP hold the false-positive rate near the floor
(FPR \(=0.01\text{–}0.03\)), one to two orders of magnitude below PCMCI
(\(0.5\text{–}0.6\)) and Linear Granger (\(0.7\text{–}0.9\)). The high recall of
PCMCI and Granger (TPR up to \(1.0\)) is therefore an artifact of
over-connection: both declare a large fraction of non-edges significant. This
reproduces, under emergent dynamics, the favorable precision/false-positive
trade-off of oCSE seen for partially synchronized networks. VARLiNGAM sits at the
opposite extreme, recovering almost no true edges (TPR \(\lesssim0.05\)) in every
regime.

## B. Order-parameter conditioning lowers oCSE's false positives

Conditioning oCSE on the global order parameter reduces the false-positive rate
wherever there are false positives to remove. On the traveling wave the FPR falls
\(0.019\to0.006\) (a \(3\times\) reduction); on the chimera it falls
\(0.025\to0.012\) (\(2\times\), consistently across realizations). On the
metastable network, where oCSE already operates at FPR \(\approx0.02\), the effect
is neutral. The reduction is accompanied by a small, systematic decrease in TPR
(e.g. \(0.250\to0.219\) on the traveling wave), consistent with adding a
common-mode variable to the conditioning set: oCSE+OP is a precision-favoring
variant that trades a little recall for fewer spurious links, rather than a free
improvement. The effect is the expected signature of blocking the mean-field
common-cause path.

## C. The chimera exposes a synchronization barrier

The emergent regimes are demanding for every method, and the chimera is the
extreme case: oCSE recovers only \(\sim\!1\%\) of true couplings (TPR \(=0.014\)).
This is not a failure of significance calibration but an intrinsic obstruction.
Within the coherent arc the oscillators are phase-locked, so the coupling feature
\(\sin(\theta_j-\theta_i)\) is essentially constant and carries no statistical
information; a direct link there is observationally indistinguishable from no
link. Conditioning on the order parameter cannot recover signal the dynamics have
removed — it only trims the residual false positives (FPR \(0.025\to0.012\)) while
TPR remains near zero. The same obstruction underlies the difficulty of causal
reconstruction in synchronized chaotic systems and marks a fundamental limit of
observational causal discovery: wherever synchronization erases the temporal
signature of a coupling, no conditional-independence method can restore it.

Taken together, these results show that oCSE — optionally augmented with
order-parameter conditioning — is the method that does not flood the inferred
network under chimera, cyclops, metastable, and traveling-wave dynamics, while
also delineating, through the chimera, where emergent synchronization renders the
underlying interaction structure unidentifiable from observation alone.
