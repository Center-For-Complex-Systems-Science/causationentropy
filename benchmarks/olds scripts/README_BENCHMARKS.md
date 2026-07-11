# Causal Discovery Benchmarks

This directory contains three benchmark implementations with fundamentally different testing philosophies.

---

## File Overview

### 1. **kuramoto.py** and **rossler.py** — Original Benchmarks
**What they test:** Variable selection with pre-engineered features

**Approach:**
- Construct features that explicitly encode the causal structure
- Example: For Kuramoto, create `sin(θⱼ - θᵢ)` terms (the exact functional form from the ground truth equation)
- Test which features predict which targets
- Map feature-target relationships back to network edges

**Pros:**
- ✓ Fast execution (linear methods on engineered features)
- ✓ High F1 scores (when functional form is correct)
- ✓ Good software structure
- ✓ Nice visualizations

**Cons:**
- ✗ **Not testing causal discovery** — testing variable selection
- ✗ Circular reasoning: answer is encoded in feature names
- ✗ Doesn't test ability to discover functional forms
- ✗ Only n=3 nodes (tiny networks)
- ✗ Missing standard baselines (PC, GES, etc.)

**Use case:**
- Upper bound performance (shows best case when functional form is known)
- Fast iteration for algorithm development
- **Should NOT be used** to claim causal discovery capability

---

### 2. **principled_comparison.py** — Fixed Benchmark
**What it tests:** True causal discovery from raw observations

**Approach:**
- **Condition 1: RAW OBSERVATIONS** (primary test)
  - Input: θ(t) or X(t) — no feature engineering
  - Method: Nonlinear CI tests (GPDC)
  - Tests ability to discover both structure AND functional form

- **Condition 2: ORACLE BASIS** (reference)
  - Input: Correctly engineered features (like original benchmarks)
  - Shows upper bound with perfect knowledge

- **Condition 3: WRONG BASIS** (negative control)
  - Input: Incorrectly engineered features
  - Shows brittleness to model misspecification

**Pros:**
- ✓ **Tests actual causal discovery**
- ✓ No circular reasoning
- ✓ Quantifies value of domain knowledge (Raw vs Oracle gap)
- ✓ Tests multiple systems (Kuramoto, VAR)
- ✓ Comprehensive condition coverage

**Cons:**
- ✗ Slower (GPDC is expensive)
- ✗ Lower F1 on raw data (but that's honest!)
- ✗ Still only n=3 nodes (inherited from originals)

**Use case:**
- **Primary results** for papers
- Honest assessment of method capabilities
- Understanding when feature engineering is critical

---

## Quick Comparison

| Aspect | Original (kuramoto.py/rossler.py) | Fixed (principled_comparison.py) |
|--------|-----------------------------------|----------------------------------|
| **Input data** | Engineered features | Raw observations (+ oracle for reference) |
| **Tests** | Variable selection | Causal discovery |
| **CI tests** | Linear (ParCorr, Gaussian) | Nonlinear (GPDC) + Linear |
| **Systems** | Kuramoto, Rössler | Kuramoto, VAR, (Rössler ready) |
| **F1 scores** | High (0.7-0.9) | Honest (0.4-0.6 raw, 0.7-0.9 oracle) |
| **Runtime** | Fast (~minutes) | Slower (~30-60 min with GPDC) |
| **Interpretation** | "Upper bound with known model" | "True discovery capability" |

---

## Usage Guide

### For Development / Fast Iteration:
```bash
# Use original benchmarks for quick testing
python kuramoto.py    # ~5-10 minutes
python rossler.py     # ~10-20 minutes
```
- Good for debugging algorithm changes
- Fast feedback loop
- BUT: Don't report these as causal discovery results

### For Publication / Honest Evaluation:
```bash
# Use principled benchmark for main results
python principled_comparison.py  # ~30-60 minutes
```
- Reports performance on raw observations (primary metric)
- Includes oracle condition for context
- Scientifically rigorous

### For Comprehensive Analysis:
```bash
# Run all three
python principled_comparison.py  # Main results
python kuramoto.py              # Kuramoto oracle reference
python rossler.py               # Rössler oracle reference

# Compare:
# - principled_comparison.py "Raw" conditions → True performance
# - principled_comparison.py "Oracle" conditions → Should match kuramoto.py/rossler.py
# - Gap between Raw and Oracle → Value of domain knowledge
```

---

## Expected Results

### Kuramoto (Nonlinear Phase Oscillators)

| Condition | Expected F1 | Interpretation |
|-----------|------------|----------------|
| **Raw + GPDC** | 0.4-0.6 | True discovery performance (nonlinearity is hard) |
| Raw + ParCorr | 0.2-0.3 | Linear test fails on nonlinear data |
| **Oracle + ParCorr** | 0.7-0.9 | Upper bound with correct functional form |
| Wrong + ParCorr | 0.1-0.3 | Wrong model assumptions fail |

**Key insight:** Large Raw-Oracle gap shows that **knowing** `sin(θⱼ - θᵢ)` is critical.

### VAR (Linear System)

| Condition | Expected F1 | Interpretation |
|-----------|------------|----------------|
| **Raw + ParCorr** | 0.6-0.8 | Linear test works on linear data |
| Raw + GPDC | 0.6-0.8 | Nonlinear test also works (but slower) |

**Key insight:** Small gap shows methods handle linear systems naturally (no oracle needed).

---

## Methodological Details

See **METHODOLOGICAL_ISSUES.md** for:
- Detailed analysis of circular reasoning problem
- Why engineered features ≠ causal discovery
- Recommendations for proper benchmarking
- Expected results and interpretation guide

---

## File Structure

```
benchmarks/
├── kuramoto.py                    # Original: Engineered basis, fast
├── rossler.py                     # Original: Engineered basis, chaotic
├── principled_comparison.py       # Fixed: Raw observations + oracle
├── METHODOLOGICAL_ISSUES.md       # Detailed critique
├── README_BENCHMARKS.md          # This file
│
├── figs/                          # Outputs from kuramoto.py/rossler.py
├── figs_principled/               # Outputs from principled_comparison.py
│
├── kuramoto_simplified_results.csv
├── rossler_simplified_results.csv
└── principled_results.csv
```

---

## Recommendations

### For Papers:

**Main Results:**
- Use `principled_comparison.py` "Raw" conditions
- Report: "Our method achieves F1=0.X on raw observations"

**Context / Upper Bound:**
- Use `principled_comparison.py` "Oracle" conditions
- Report: "With correct functional form (oracle), F1=0.Y"
- Calculate and report gap: "Domain knowledge provides ΔF1=0.Y-0.X improvement"

**DO NOT:**
- ✗ Report kuramoto.py/rossler.py results as "causal discovery performance"
- ✗ Claim high F1 scores without mentioning engineered features
- ✗ Compare to baselines that use raw data (unfair comparison)

### For Method Development:

1. **Fast iteration:** Use kuramoto.py/rossler.py during development
2. **Final validation:** Always test on principled_comparison.py before claiming success
3. **Scale testing:** Extend to n=10-20 nodes for realistic evaluation
4. **Baseline comparison:** Add PC, GES, FCI to principled_comparison.py

---

## Future Improvements

### Short term:
- [x] Add Accuracy + AUC to kuramoto.py
- [ ] Add Accuracy + AUC to rossler.py
- [x] Create principled_comparison.py
- [x] Document methodological issues

### Medium term:
- [ ] Add Rössler to principled_comparison.py (need score extraction)
- [ ] Increase to n=10 nodes
- [ ] Add PC, GES baselines
- [ ] Add continuous-time methods

### Long term:
- [ ] Scale to n=50-100 nodes
- [ ] Test on real data (fMRI, climate, finance)
- [ ] Add latent confounder scenarios
- [ ] Add measurement noise / interventions

---

## Citation

If you use these benchmarks, please acknowledge:

```
The original benchmarks (kuramoto.py, rossler.py) test variable selection
with engineered features that encode the ground truth functional form.
For true causal discovery evaluation, we use principled_comparison.py
which tests on raw observations without pre-specified functional forms.
```

---

## Questions?

**Q: Why do kuramoto.py/rossler.py have high F1 scores?**
A: Because they encode the answer in the feature names. They test variable selection, not causal discovery.

**Q: Should I delete kuramoto.py/rossler.py?**
A: No! They're useful for:
- Fast development iteration
- Upper bound reference
- Understanding value of domain knowledge
Just don't report them as causal discovery results.

**Q: Why is principled_comparison.py slower?**
A: GPDC (Gaussian Process CI test) is computationally expensive but necessary for handling nonlinearity without pre-specified functional forms.

**Q: Can I add my method to these benchmarks?**
A: Yes! For principled_comparison.py, add your method to the CONDITIONS dict. Make sure to test on RAW observations, not pre-engineered features.

**Q: What F1 score is "good"?**
A: Depends on condition:
- Raw observations: 0.4-0.6 is honest for nonlinear systems
- Oracle basis: 0.7-0.9 is expected (you encoded the answer)
- The gap between them matters more than absolute scores

---

## Contact

For questions about these benchmarks, see METHODOLOGICAL_ISSUES.md or contact the repository maintainers.
