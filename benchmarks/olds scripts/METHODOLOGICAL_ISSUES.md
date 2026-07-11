# Methodological Issues in Current Benchmarks

## The Fundamental Problem: Circular Reasoning

Both `kuramoto.py` and `rossler.py` have a critical flaw: **they encode the ground truth causal structure into the feature space**, then test whether methods can identify which features are relevant. This reduces causal discovery to variable selection, not true causal inference.

---

## Detailed Analysis

### What These Benchmarks Actually Test

#### Kuramoto Benchmark (kuramoto.py)

**Ground truth dynamics:**
```
dθᵢ/dt = ωᵢ + ρ Σⱼ Aᵢⱼ sin(θⱼ - θᵢ)
```

**What the benchmark does (lines 127-187):**
1. Explicitly constructs features `sin(θⱼ(t) - θᵢ(t))` for all pairs (i,j)
2. Asks: "Which `sin(θⱼ - θᵢ)` terms predict `vᵢ`?"
3. Maps back: "If `sin(θⱼ - θᵢ) → vᵢ` then j → i"

**Why this is circular:**
- The Kuramoto equation **literally contains** the term `sin(θⱼ - θᵢ)`
- By constructing these exact features, you've already done the hard work
- The "causal discovery" task becomes: linear variable selection among pre-labeled features

**Analogy:**
```python
# Suppose the true model is: Y = β₀ + β₁*X + β₂*X² + ε
# Testing if a method can discover X → Y by:

# CIRCULAR (what current benchmarks do):
features = [X, X², X³, Z, Z²]  # You explicitly include X and X²
"Can the method detect that X and X² predict Y?"
✗ This tests variable selection, not causal discovery

# PROPER:
features = [X, Z]  # Raw observations only
"Can the method detect that X → Y and recover the nonlinear relationship?"
✓ This tests true causal discovery
```

---

#### Rössler Benchmark (rossler.py)

**Ground truth dynamics:**
```
Coupling on x: dxᵢ/dt = -yᵢ - zᵢ + ρ Σⱼ Aᵢⱼ(xⱼ - xᵢ)
```

**What the benchmark does (lines 155-236):**
1. Explicitly constructs coupling differences `cᵢ←ⱼ = xⱼ(t) - xᵢ(t)` for all pairs
2. Asks: "Which coupling terms `cᵢ←ⱼ` predict `dxᵢ/dt`?"
3. Maps back: "If `cᵢ←ⱼ → dxᵢ/dt` then j → i"

**Why this is circular:**
- Same issue: the coupling equation contains the difference term `(xⱼ - xᵢ)`
- You've encoded the functional form into the features
- Methods just need to do linear regression with labeled predictors

---

## What True Causal Discovery Should Test

### Proper Benchmark Design

**Input:** Raw observations only
- Kuramoto: θᵢ(t) — just the phases
- Rössler: (xᵢ(t), yᵢ(t), zᵢ(t)) — just the state variables
- VAR: Xᵢ(t) — just the time series

**Task:** Discover both:
1. **Network structure** (which j → i)
2. **Functional form** (linear? nonlinear? what kind?)

**Methods:**
- Use nonlinear conditional independence tests (GPDC, kernel CI, neural CI)
- Let methods discover relationships from data, not pre-engineered features

---

## Specific Issues in Current Benchmarks

### 1. **Sample Size vs Dimensionality**

**kuramoto.py:**
- Original: T=500 → 498 effective samples after differencing
- 9-dimensional feature space (3 nodes + 6 coupling terms)
- ~55 samples per dimension
- **Now fixed:** T=2000 (you updated this)

**rossler.py:**
- T=4000, dt=0.02
- 15-dimensional feature space (3*3 derivatives + 6 coupling terms)
- ~267 samples per dimension (better, but still engineered basis)

### 2. **Scale Limitations**

Both benchmarks:
- Only test n=3 nodes (tiny networks)
- Real-world networks: hundreds/thousands of nodes
- Many causal methods show qualitatively different behavior at scale
- Computational complexity changes dramatically

**Why this matters:**
- At n=3, there are only 6 possible directed edges
- Easy to get lucky with small search spaces
- Doesn't test scalability or curse of dimensionality

### 3. **Limited Topology Diversity**

**What's tested:**
- Erdős-Rényi random graphs only

**What's NOT tested:**
- Scale-free networks (hubs, power-law degree distribution)
- Small-world networks (clustering + short paths)
- Modular/hierarchical structures
- Sparse vs dense regimes

**Why this matters:**
- Real networks have structure (social, biological, neural)
- Some methods exploit topology (e.g., assume sparsity)
- Performance can vary dramatically by graph type

### 4. **Missing Baselines**

**Current comparisons:**
- OCE (gaussian) vs PCMCI (ParCorr)
- Both use the **same engineered basis**
- Both are essentially doing linear regression

**Missing comparisons:**
- PC algorithm (constraint-based)
- GES (score-based)
- FCI (with latent confounders)
- LiNGAM (linear non-Gaussian)
- Neural methods (e.g., TCDF, NRI)
- Granger causality
- Transfer entropy

### 5. **Continuous-Time Issues**

**Kuramoto/Rössler are continuous-time systems:**
```
dθ/dt = f(θ)  ← Continuous SDE/ODE
```

**But methods assume discrete time:**
```
X(t+1) = f(X(t))  ← Discrete VAR-like
```

**The mismatch:**
- True causation is instantaneous (lag=0)
- Methods look for lag-1 or lag-τ relationships
- Discretization can create artifacts
- Results depend on sampling rate (dt)

**What's missing:**
- Continuous-time causal discovery methods
- Analysis of how dt affects recovery
- Score-based methods designed for ODEs/SDEs

### 6. **No Failure Analysis**

**Questions not answered:**
- Which graph structures are hardest to recover?
- What coupling strengths are too weak to detect?
- When do methods produce false positives vs false negatives?
- Which edges are systematically missed?
- How does synchronization affect detectability?

---

## The Fix: principled_comparison.py

### Three Testing Conditions

#### 1. **RAW OBSERVATIONS** (principled test)
```python
Input: θ(t) or X(t)  # No feature engineering
Method: GPDC (Gaussian Process CI test)
Test: Can methods discover j → i from raw nonlinear time series?
```

**This is the real test:** No knowledge of functional form.

#### 2. **ORACLE BASIS** (upper bound)
```python
Input: sin(θⱼ - θᵢ) or (xⱼ - xᵢ)  # Correct features
Method: ParCorr (linear CI test)
Test: How much does knowing the functional form help?
```

**This is your current benchmarks:** Shows best-case when you know the answer.

#### 3. **WRONG BASIS** (negative control)
```python
Input: (θⱼ - θᵢ)² or polynomial terms  # Incorrect features
Method: ParCorr
Test: What happens when model assumptions are wrong?
```

**This shows brittleness:** If functional form is wrong, does it fail catastrophically?

---

## Expected Results

### For Kuramoto (nonlinear):

| Condition | Expected F1 | Reason |
|-----------|------------|--------|
| Raw + GPDC | 0.4-0.6 | GPDC can handle nonlinearity but needs more samples |
| Raw + ParCorr | 0.2-0.3 | Linear CI test misses sine coupling |
| Oracle + ParCorr | 0.7-0.9 | Optimal: correct features + appropriate test |
| Wrong + ParCorr | 0.1-0.3 | Garbage in, garbage out |

**Key insight:** Large gap between Raw and Oracle reveals the value of domain knowledge.

### For VAR (linear):

| Condition | Expected F1 | Reason |
|-----------|------------|--------|
| Raw + GPDC | 0.6-0.8 | GPDC handles linear case fine (but slower) |
| Raw + ParCorr | 0.6-0.8 | Optimal: data is linear, test is linear |
| Oracle N/A | — | VAR is already linear, no "oracle" basis |
| Wrong N/A | — | Not applicable |

**Key insight:** Small gap indicates methods can handle the system naturally.

---

## Recommendations for Proper Benchmarking

### 1. **Primary Test: Raw Observations**
- Always start with raw data
- Use nonlinear CI tests (GPDC, kernel, neural)
- This is the only true test of causal discovery

### 2. **Include Oracle as Reference**
- Show best-case performance with perfect knowledge
- Quantify value of domain expertise
- But DON'T claim this tests causal discovery

### 3. **Test Multiple Systems**
- Linear (VAR): Sanity check
- Nonlinear smooth (Kuramoto, Lorenz)
- Nonlinear chaotic (Rössler)
- Regime-switching (different dynamics in different regimes)
- Unknown ground truth (real data)

### 4. **Scale Up**
- Test n ∈ {5, 10, 20, 50} nodes
- Report computational complexity
- Show where methods break down

### 5. **Comprehensive Comparisons**
- Include classical methods (PC, GES, FCI)
- Include domain-specific methods (Granger, transfer entropy)
- Include modern ML methods (neural, causal VAE)

### 6. **Report Failure Modes**
- Per-edge analysis (which edges missed?)
- Dependence on hyperparameters
- Sensitivity to noise, sample size, coupling strength
- Runtime vs accuracy tradeoffs

---

## How to Use the New Benchmark

### Run it:
```bash
cd /home/kslote/Desktop/causalentropy/benchmarks
python principled_comparison.py
```

### What it will show:

1. **For Kuramoto:**
   - Raw observations likely struggle (nonlinearity is hard)
   - Oracle basis does much better (you encoded the answer)
   - Gap quantifies value of knowing functional form

2. **For VAR:**
   - Raw observations work fine (it's linear)
   - No oracle needed
   - Validates methods work on appropriate data

### Interpret results:

- **Large Raw-Oracle gap → System is intrinsically hard**
  - Nonlinearity matters
  - Feature engineering is critical
  - Methods need better nonlinear tests

- **Small Raw-Oracle gap → System is tractable**
  - Methods handle it naturally
  - No need for specialized features

---

## Bottom Line

### Current benchmarks (kuramoto.py, rossler.py):
- ✗ Test variable selection, not causal discovery
- ✗ Encode answer in features
- ✓ Show upper bound performance (if you know the model)
- ✓ Good software engineering

### Proper benchmarks should:
- ✓ Use raw observations as primary test
- ✓ Include oracle condition for reference
- ✓ Test across multiple systems and scales
- ✓ Report comprehensive metrics and failure modes
- ✓ Compare to established baselines

### Action items:

1. **Acknowledge limitation** in current benchmark papers
2. **Use principled_comparison.py** for main results
3. **Keep oracle results** but label them correctly:
   - "Upper bound with known functional form"
   - NOT "Causal discovery performance"
4. **Scale up:** Test on n=10-20 nodes minimum
5. **Add baselines:** PC, GES, other standard methods

---

## References

- Pearl, J. (2009). *Causality: Models, Reasoning and Inference* — Foundations
- Peters, J., et al. (2017). *Elements of Causal Inference* — Modern methods
- Runge, J., et al. (2019). "Detecting and quantifying causal associations" — PCMCI
- Schölkopf, B., et al. (2021). "Toward Causal Representation Learning" — ML perspective
