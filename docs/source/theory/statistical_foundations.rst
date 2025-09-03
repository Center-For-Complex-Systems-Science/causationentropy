=======================
Statistical Foundations
=======================

This section covers the statistical principles underlying the Causation Entropy 
framework, including hypothesis testing, multiple comparisons, bootstrap methods, 
and theoretical guarantees. Understanding these foundations is essential for proper 
application and interpretation of causal discovery results.

Hypothesis Testing Framework
===========================

Null and Alternative Hypotheses
-------------------------------

In causal discovery, the fundamental hypothesis test is:

.. math::

   H_0: I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_i^{(t)}) = 0

.. math::

   H_1: I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_i^{(t)}) > 0

**Interpretation:**
- :math:`H_0`: No causal relationship (conditional independence)
- :math:`H_1`: Causal relationship exists (conditional dependence)

**Key Insight:** Conditional independence testing forms the backbone of 
information-theoretic causal discovery.

Test Statistics and Distributions
---------------------------------

The test statistic is the conditional mutual information:

.. math::

   T = \hat{I}(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_i^{(t)})

**Distribution Under Null:**
For most information estimators, the null distribution is not analytically tractable. 
This motivates non-parametric approaches like permutation testing.

**Asymptotic Properties:**
Under regularity conditions, for the Gaussian estimator:

.. math::

   2n \cdot \hat{I}(X;Y|Z) \xrightarrow{d} \chi^2_{df}

where :math:`df$ depends on the dimensionalities of :math:`X$, :math:`Y$, and :math:`Z$.

Permutation Testing
==================

Theoretical Foundation
---------------------

**Exchangeability Principle:**
Under :math:`H_0: X \perp Y | Z`, the joint distribution of :math:`(X,Y,Z)` 
is invariant to permutations of :math:`X$ given :math:`Z$.

**Permutation Distribution:**
Generate :math:`B$ permutations :math:`\{X^{(b)}\}_{b=1}^B$ and compute:

.. math::

   \{T^{(b)}\}_{b=1}^B = \{\hat{I}(X^{(b)}; Y | Z)\}_{b=1}^B

**P-value Calculation:**

.. math::

   p = \frac{1 + \sum_{b=1}^B \mathbb{I}(T^{(b)} \geq T_{\text{obs}})}{B + 1}

Permutation Strategies
---------------------

**Simple Permutation:**
Randomly shuffle :math:`X$ across all observations.

**Conditional Permutation:**
For continuous :math:`Z$, this is challenging. Alternatives include:

1. **Residual Permutation:** Permute residuals from :math:`X \sim f(Z)$
2. **Local Permutation:** Permute within neighborhoods of similar :math:`Z$ values
3. **Model-Based Permutation:** Fit :math:`p(X|Z)$ and generate synthetic data

**Block Permutation:**
For time series data, preserve temporal structure:

.. math::

   \text{Block}(X, l) = [X_{i:i+l-1}, X_{j:j+l-1}, \ldots]

where blocks of length :math:`l$ are permuted rather than individual observations.

Statistical Properties
---------------------

**Exactness:**
Permutation tests provide exact control of Type I error under :math:`H_0$.

**Power:**
Power depends on:
- Effect size (true conditional mutual information)
- Sample size :math:`n$
- Number of permutations :math:`B$
- Quality of information estimator

**Computational Cost:**
Total cost is :math:`O((B+1) \cdot C_{\text{estimator}})$ where :math:`C_{\text{estimator}}$ 
is the cost of computing one conditional mutual information estimate.

Multiple Testing Corrections
============================

The Multiple Testing Problem
---------------------------

In causal discovery, we typically test :math:`m$ hypotheses simultaneously:

.. math::

   H_{0,k}: I(X_{j_k}^{(t-\tau_k)}; X_i^{(t)} | \mathbf{Z}_i^{(t)}) = 0, \quad k = 1, \ldots, m

**Family-Wise Error Rate (FWER):**

.. math::

   \text{FWER} = P(\text{at least one false rejection}) = P\left(\bigcup_{k \in \mathcal{H}_0} \{p_k \leq \alpha\}\right)

**False Discovery Rate (FDR):**

.. math::

   \text{FDR} = \mathbb{E}\left[\frac{V}{\max(R, 1)}\right]

where :math:`V$ is the number of false rejections and :math:`R$ is the total number of rejections.

Bonferroni Correction
--------------------

**Method:** Reject :math:`H_{0,k}$ if :math:`p_k \leq \alpha/m$

**Properties:**
- Controls FWER exactly: :math:`\text{FWER} \leq \alpha$
- Conservative (low power) when :math:`m$ is large
- Appropriate when few true relationships exist

**Application in oCSE:**
Use when testing a small number of pre-specified relationships or 
when strong FWER control is required.

False Discovery Rate Control
---------------------------

**Benjamini-Hochberg Procedure:**
1. Order p-values: :math:`p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find largest :math:`k$ such that :math:`p_{(k)} \leq \frac{k}{m}\alpha$
3. Reject :math:`H_{0,(1)}, \ldots, H_{0,(k)}$

**Adaptive FDR:**
Estimate the proportion of true nulls :math:`\pi_0$:

.. math::

   \hat{\pi}_0 = \frac{\#\{p_i > \lambda\}}{m(1-\lambda)}

Then use threshold: :math:`p_{(k)} \leq \frac{k}{m\hat{\pi}_0}\alpha$

**By-Stage Methods:**
Control FDR at each stage of forward/backward selection.

Sequential Testing in oCSE
==========================

Forward Selection Testing
-------------------------

At each forward selection step :math:`s$:

1. Test all remaining candidates: :math:`\{H_{0,k}\}_{k \in \mathcal{R}_s}$
2. Apply multiple testing correction within the step
3. Select the most significant candidate (if any pass the threshold)

**Step-wise FDR Control:**
.. math::
   \alpha_s = \alpha \cdot \frac{|\mathcal{R}_s|}{|\mathcal{R}_1|}

This allocates the error budget proportionally across steps.

Backward Elimination Testing
----------------------------

Test each selected predictor for continued significance:

.. math::

   H_0: I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{S}_i \setminus \{j\}) = 0

**Challenges:**
- Dependencies between tests (same target, overlapping conditioning sets)
- Multiple testing across different removal orders

**Solutions:**
- Use more conservative :math:`\alpha$ for backward phase
- Apply FDR control across all backward tests
- Use stability-based selection criteria

Bootstrap Methods
=================

Bootstrap Confidence Intervals
------------------------------

**Procedure:**
1. Generate :math:`B$ bootstrap samples :math:`\{(\mathbf{X}^{(b)}, \mathbf{Y}^{(b)})\}_{b=1}^B$
2. Compute :math:`\{\hat{I}^{(b)}\}_{b=1}^B$ for each bootstrap sample
3. Construct confidence interval: :math:`[\hat{I}_{(\alpha/2)}, \hat{I}_{(1-\alpha/2)}]$

**Time Series Bootstrap:**
Standard bootstrap assumes i.i.d. data. For time series:

**Block Bootstrap:**
.. math::
   \text{Bootstrap Sample} = [B_1, B_2, \ldots, B_k]

where :math:`B_i$ are overlapping blocks of length :math:`l$.

**Stationary Bootstrap:**
Random block lengths with geometric distribution.

Bootstrap-based Variable Selection
----------------------------------

**Stability Selection:**
For each bootstrap sample, perform variable selection and compute 
selection probability:

.. math::

   \Pi_j = P(\text{variable } j \text{ selected}) = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(j \in \hat{\mathbf{S}}^{(b)})

Select variables with :math:`\Pi_j \geq \pi_{\text{threshold}}$ (typically 0.6-0.8).

**Theoretical Guarantees:**
Under appropriate conditions, stability selection provides FDR control:

.. math::

   \mathbb{E}[\text{FDR}] \leq \frac{1}{2\pi_{\text{threshold}} - 1} \cdot \frac{\mathbb{E}[V]}{|\hat{\mathbf{S}}|}

Theoretical Guarantees
=====================

Consistency Properties
---------------------

**Selection Consistency:**
An estimator is selection consistent if:

.. math::

   P(\hat{\mathbf{S}} = \mathbf{S}_{\text{true}}) \to 1 \text{ as } n \to \infty

**Conditions for oCSE:**
1. **Information Estimator Consistency:** :math:`\hat{I} \xrightarrow{P} I$
2. **Significance Level Scaling:** :math:`\alpha_n \to 0$ appropriately
3. **Sparsity:** :math:`|\mathbf{S}_{\text{true}}| = o(n)$
4. **Signal Strength:** Minimum true CMI bounded away from 0

Estimation Error Bounds
----------------------

For Gaussian estimators, the estimation error satisfies:

.. math::

   |\hat{I} - I| = O_p\left(\sqrt{\frac{d \log n}{n}}\right)

where :math:`d$ is the effective dimensionality.

**Implications for Causal Discovery:**
- Need :math:`n \gg d \log n$ for reliable estimation
- True relationships must have CMI significantly larger than :math:`\sqrt{\frac{d \log n}{n}}$

High-Dimensional Theory
----------------------

**Conditions for :math:`p > n$:**
When the number of potential predictors exceeds sample size:

1. **Sparsity:** :math:`s = |\mathbf{S}_{\text{true}}| \ll n$
2. **Restricted Eigenvalue Condition:** For information matrices
3. **Signal-to-Noise Ratio:** True CMI values sufficiently large

**Phase Transitions:**
In high-dimensional regimes, there are sharp phase transitions where 
selection becomes possible/impossible based on the scaling of :math:`n$, :math:`p$, and :math:`s$.

Power Analysis
=============

Theoretical Power
----------------

The power of a conditional independence test is:

.. math::

   \text{Power} = P(\text{reject } H_0 | H_1 \text{ true}) = P(T > t_{\alpha} | I > 0)

**Factors Affecting Power:**
- **Effect Size:** Larger true CMI increases power
- **Sample Size:** Power increases with :math:`n$
- **Dimensionality:** Higher dimensions reduce power (curse of dimensionality)
- **Information Estimator:** Different estimators have different power characteristics

Sample Size Calculations
------------------------

**Rule of Thumb for Gaussian Estimator:**
To detect CMI of size :math:`\delta$ with power :math:`1-\beta$:

.. math::

   n \gtrsim \frac{(z_{\alpha} + z_{\beta})^2}{\delta^2} \cdot d

where :math:`d$ is the effective dimensionality.

**Simulation-Based Power Analysis:**
1. Specify effect sizes of interest
2. Generate synthetic data under alternative hypothesis
3. Apply testing procedure and compute empirical power
4. Repeat for different sample sizes to find required :math:`n$

Robustness and Sensitivity
==========================

Robustness to Outliers
---------------------

**Impact of Outliers:**
Information estimators vary in sensitivity to outliers:
- **Gaussian:** Highly sensitive (based on sample covariance)
- **k-NN:** Moderately sensitive (distance-based)
- **Histogram:** Least sensitive (discretization reduces impact)

**Robust Estimators:**
- **Trimmed estimators:** Remove extreme observations
- **M-estimators:** Downweight outliers in computation
- **Robust covariance:** Use robust estimates in Gaussian methods

Model Misspecification
----------------------

**Gaussian Assumption Violations:**
When data is non-Gaussian but Gaussian estimator is used:
- May detect only linear relationships
- Power reduced for nonlinear dependencies
- Type I error control generally maintained

**Non-stationarity:**
Time-varying relationships violate stationarity assumptions:
- Use adaptive window methods
- Apply tests for structural breaks
- Consider time-varying parameter models

Sensitivity Analysis
-------------------

**Parameter Sensitivity:**
Assess robustness to hyperparameter choices:
- Information estimator parameters (bandwidth, k)
- Significance levels (:math:`\alpha$)
- Maximum lag (:math:`\tau_{\max}$)

**Cross-Validation:**
Use held-out data to validate discovered relationships:

.. math::

   \text{CV-Score} = \frac{1}{K} \sum_{k=1}^K I_{\text{test},k}(\hat{\mathbf{S}}_{\text{train},k})

Practical Guidelines
===================

Sample Size Requirements
-----------------------

**Minimum Sample Sizes by Estimator:**

.. list-table:: Sample Size Guidelines
   :widths: 25 25 25 25
   :header-rows: 1

   * - Estimator
     - Low Dim (d≤5)
     - Medium Dim (5<d≤20)
     - High Dim (d>20)
   * - Gaussian
     - n ≥ 50
     - n ≥ 100
     - n ≥ 500
   * - k-NN
     - n ≥ 100
     - n ≥ 500
     - n ≥ 1000+
   * - KDE
     - n ≥ 200
     - n ≥ 1000
     - Not recommended
   * - Histogram
     - n ≥ 500
     - n ≥ 2000
     - Not recommended

Significance Level Selection
---------------------------

**Forward Selection:** Use more stringent :math:`\alpha$ to control false positives
- Conservative: :math:`\alpha = 0.01$
- Standard: :math:`\alpha = 0.05$
- Liberal: :math:`\alpha = 0.10$ (exploratory analysis)

**Backward Elimination:** Can use less stringent :math:`\alpha$ for pruning
- Typical: :math:`\alpha_{\text{backward}} = 1.5 \times \alpha_{\text{forward}}$

**Multiple Testing:** Always apply appropriate corrections when testing multiple relationships simultaneously.

Diagnostic Procedures
====================

Model Checking
--------------

**Residual Analysis:**
After variable selection, examine residuals for:
- Independence (serial correlation tests)
- Normality (if using Gaussian methods)
- Heteroscedasticity

**Information Criteria:**
Compare model performance using information-theoretic criteria:

.. math::

   \text{AIC}_{\text{info}} = -2 \sum_{j \in \hat{\mathbf{S}}} \hat{I}_j + 2|\hat{\mathbf{S}}|

Stability Analysis
-----------------

**Bootstrap Stability:**
Assess selection stability across bootstrap samples:

.. math::

   \text{Stability Score} = \frac{1}{B} \sum_{b=1}^B \frac{|\hat{\mathbf{S}}^{(b)} \cap \hat{\mathbf{S}}|}{|\hat{\mathbf{S}}^{(b)} \cup \hat{\mathbf{S}}|}

**Cross-Validation Stability:**
Use K-fold CV to assess robustness to data splitting.

Future Directions
================

Methodological Advances
----------------------

1. **Adaptive Testing:** Data-driven significance level selection
2. **Sequential FDR:** Improved multiple testing for sequential selection
3. **Robust Information Measures:** Estimators less sensitive to outliers
4. **High-Dimensional Theory:** Better understanding of :math:`p >> n$ regimes
5. **Causal-Specific Tests:** Tests designed specifically for causal relationships

Computational Improvements
--------------------------

1. **Parallel Testing:** Efficient parallel algorithms for permutation tests
2. **Approximate Methods:** Fast approximate significance testing
3. **Online Methods:** Sequential testing for streaming data
4. **GPU Acceleration:** Hardware acceleration for large-scale problems

Conclusion
=========

The statistical foundations of optimal Causal Entropy provide the theoretical framework 
for reliable causal discovery. Key principles include:

- **Rigorous Hypothesis Testing:** All causal claims should be statistically validated
- **Multiple Testing Awareness:** Control for multiple comparisons when testing many relationships
- **Bootstrap Methods:** Use resampling for uncertainty quantification and stability assessment
- **Power Considerations:** Ensure sufficient sample sizes for reliable detection
- **Robustness Checks:** Validate methods across different assumptions and parameter choices

Understanding these statistical foundations is crucial for proper application and 
interpretation of causal discovery results. Practitioners should always validate 
their findings through appropriate statistical testing and sensitivity analysis.