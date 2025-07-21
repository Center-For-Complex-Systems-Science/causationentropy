=======================
Standard Causal Entropy
=======================

The Standard optimal Causal Entropy (standard oCSE) represents the canonical implementation 
of the causation entropy framework. This method begins with an initial conditioning set, 
typically consisting of lagged values of the target variable, and systematically builds 
the causal predictor set through forward selection and backward elimination phases.

Mathematical Foundation
=======================

The standard oCSE algorithm is built around the conditional mutual information measure:

.. math::

   I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_i^{(t)}) = 
   \sum_{x_j,x_i,\mathbf{z}} p(x_j,x_i,\mathbf{z}) \log \frac{p(x_i,x_j|\mathbf{z})}{p(x_i|\mathbf{z})p(x_j|\mathbf{z})}

where:
- :math:`X_j^{(t-\tau)}` is the candidate predictor variable :math:`j` at lag :math:`\tau`
- :math:`X_i^{(t)}` is the target variable :math:`i` at the current time
- :math:`\mathbf{Z}_i^{(t)} = \mathbf{Z}_{\text{init}} \cup \mathbf{S}_i^{(t)}` is the conditioning set

The conditioning set consists of two components:
- :math:`\mathbf{Z}_{\text{init}}`: Initial conditioning variables (usually lagged target values)
- :math:`\mathbf{S}_i^{(t)}`: Previously selected predictor variables

Algorithm Description
====================

The standard oCSE algorithm proceeds in two main phases:

Phase 1: Forward Selection with Initial Conditioning
---------------------------------------------------

**Input:**
- Time series data :math:`\mathbf{X} \in \mathbb{R}^{T \times n}`
- Maximum lag :math:`\tau_{\max}`
- Significance level :math:`\alpha_{\text{forward}}`
- Number of permutations :math:`N_{\text{perm}}`

**Initialization:**
For each target variable :math:`i`, construct the initial conditioning set:

.. math::

   \mathbf{Z}_{\text{init},i} = \{X_i^{(t-1)}, X_i^{(t-2)}, \ldots, X_i^{(t-\tau_{\max})}\}

This incorporates the autoregressive structure of the target variable.

**Forward Selection Loop:**

1. **Candidate Evaluation:** For each remaining candidate predictor :math:`X_j^{(t-\tau)}`:

   .. math::

      \text{CMI}_{j,\tau} = I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_i^{(t)})

2. **Best Candidate Selection:** Choose the predictor with maximum conditional mutual information:

   .. math::

      (j^*, \tau^*) = \arg\max_{j,\tau} \text{CMI}_{j,\tau}

3. **Significance Testing:** Perform permutation test for :math:`H_0: I(X_{j^*}^{(t-\tau^*)}; X_i^{(t)} | \mathbf{Z}_i^{(t)}) = 0`

   - Generate :math:`N_{\text{perm}}` permutations of :math:`X_{j^*}^{(t-\tau^*)}`
   - Compute null distribution: :math:`\{\text{CMI}_{\text{perm}}^{(k)}\}_{k=1}^{N_{\text{perm}}}`
   - Determine threshold: :math:`\theta = \text{percentile}(\{\text{CMI}_{\text{perm}}^{(k)}\}, 100(1-\alpha_{\text{forward}}))`

4. **Selection Decision:**

   .. math::

      \text{Accept } X_{j^*}^{(t-\tau^*)} \text{ if } \text{CMI}_{j^*,\tau^*} \geq \theta

5. **Conditioning Set Update:** If accepted, update:

   .. math::

      \mathbf{Z}_i^{(t)} \leftarrow \mathbf{Z}_i^{(t)} \cup \{X_{j^*}^{(t-\tau^*)}\}

Phase 2: Backward Elimination
-----------------------------

**Objective:** Remove spurious predictors that may have been selected due to transitivity 
or confounding effects.

**Backward Elimination Loop:**
For each selected predictor :math:`X_j^{(t-\tau)} \in \mathbf{S}_i` (in random order):

1. **Conditioning Set Construction:**

   .. math::

      \mathbf{Z}_{-j} = \mathbf{Z}_{\text{init},i} \cup (\mathbf{S}_i \setminus \{X_j^{(t-\tau)}\})

2. **Conditional Mutual Information:**

   .. math::

      \text{CMI}_{j,\tau} = I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_{-j})

3. **Significance Testing:** Test :math:`H_0: I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_{-j}) = 0`

4. **Elimination Decision:**

   .. math::

      \text{Remove } X_j^{(t-\tau)} \text{ if } \text{CMI}_{j,\tau} < \theta_{\text{backward}}

Key Properties
==============

Initial Conditioning Benefits
-----------------------------

The initial conditioning set :math:`\mathbf{Z}_{\text{init}}` provides several advantages:

1. **Autoregressive Control:** Controls for the natural temporal dependencies in the target variable
2. **Enhanced Specificity:** Identifies predictors that provide information beyond autoregressive patterns
3. **Confounding Mitigation:** Reduces spurious relationships due to common trends or cycles

Mathematical Formulation:

.. math::

   I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_{\text{init}}) = 
   H(X_i^{(t)} | \mathbf{Z}_{\text{init}}) - H(X_i^{(t)} | X_j^{(t-\tau)}, \mathbf{Z}_{\text{init}})

This measures the additional predictive information provided by :math:`X_j^{(t-\tau)}` beyond 
what is already captured by the autoregressive terms.

Conditioning Set Evolution
--------------------------

The conditioning set evolves as:

.. math::

   \mathbf{Z}_i^{(0)} &= \mathbf{Z}_{\text{init},i} \\
   \mathbf{Z}_i^{(k+1)} &= \mathbf{Z}_i^{(k)} \cup \{X_{j^*}^{(t-\tau^*)}\}

where :math:`X_{j^*}^{(t-\tau^*)}` is the predictor selected at iteration :math:`k+1`.

Information-Theoretic Interpretation
====================================

The standard oCSE framework can be understood through the lens of information decomposition.
For a target variable :math:`X_i^{(t)}` with autoregressive history :math:`\mathbf{H}_i` and 
external predictor :math:`X_j^{(t-\tau)}`:

.. math::

   I(X_j^{(t-\tau)}; X_i^{(t)}) = I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{H}_i) + I(X_j^{(t-\tau)}; \mathbf{H}_i)

The first term represents the direct causal influence, while the second represents shared 
information with the autoregressive structure. Standard oCSE focuses on the first term.

Advantages and Limitations
==========================

Advantages
----------

1. **Autoregressive Control:** Explicitly accounts for temporal dependencies in the target
2. **Theoretical Foundation:** Grounded in information theory with clear interpretations
3. **Flexible Information Measures:** Supports various entropy estimators (Gaussian, k-NN, KDE, etc.)
4. **Statistical Rigor:** Permutation-based significance testing controls false positives
5. **Multivariate Conditioning:** Properly handles confounding through conditioning sets

Limitations
-----------

1. **Computational Complexity:** :math:`O(n^2 \tau_{\max} N_{\text{perm}} T)` scaling
2. **Initial Conditioning Assumption:** Assumes autoregressive structure is relevant
3. **Greedy Selection:** Forward selection may miss globally optimal solutions
4. **Sample Size Requirements:** Information estimators require sufficient data
5. **Stationarity Assumptions:** Most effective on stationary time series

Implementation Considerations
============================

Hyperparameter Selection
-----------------------

**Significance Levels:**
- :math:`\alpha_{\text{forward}}`: Controls Type I error in forward selection (typically 0.01-0.05)
- :math:`\alpha_{\text{backward}}`: Controls Type I error in backward elimination (typically 0.05-0.10)

**Maximum Lag:**
- Should reflect domain knowledge of system dynamics
- Computational cost scales linearly with :math:`\tau_{\max}`
- Rule of thumb: :math:`\tau_{\max} \approx \sqrt{T}` for exploratory analysis

**Permutation Count:**
- Minimum 100 for rough estimates
- 1000+ for publication-quality significance tests
- Precision scales as :math:`1/\sqrt{N_{\text{perm}}}`

Information Estimator Choice
---------------------------

.. list-table:: Estimator Selection Guide
   :widths: 25 25 50
   :header-rows: 1

   * - Data Type
     - Recommended Estimator
     - Notes
   * - Continuous Gaussian
     - Gaussian
     - Exact under normality assumption
   * - Continuous Non-Gaussian
     - k-NN or KDE
     - k-NN more robust to dimensionality
   * - Mixed/Discrete
     - Histogram or k-NN
     - Careful binning for histogram
   * - High-Dimensional
     - Geometric k-NN
     - Accounts for manifold structure
   * - Small Sample
     - Gaussian (if appropriate)
     - Parametric methods more sample-efficient

Example Implementation
=====================

Here's a conceptual implementation of the standard oCSE forward selection:

.. code-block:: python

   def standard_forward_selection(X, Y, Z_init, alpha=0.05, n_shuffles=200):
       """
       Standard oCSE forward selection with initial conditioning.
       
       Parameters
       ----------
       X : array (T, n*tau_max)
           Lagged predictor matrix
       Y : array (T, 1)
           Target variable
       Z_init : array (T, p)
           Initial conditioning set
       """
       n_predictors = X.shape[1]
       selected = []
       Z_current = Z_init.copy()
       
       while True:
           # Evaluate remaining candidates
           remaining = [i for i in range(n_predictors) if i not in selected]
           if not remaining:
               break
               
           cmi_values = []
           for j in remaining:
               X_j = X[:, [j]]
               cmi = conditional_mutual_information(X_j, Y, Z_current)
               cmi_values.append(cmi)
           
           # Select best candidate
           best_idx = remaining[np.argmax(cmi_values)]
           best_cmi = max(cmi_values)
           
           # Significance test
           X_best = X[:, [best_idx]]
           significant = permutation_test(X_best, Y, Z_current, 
                                        best_cmi, alpha, n_shuffles)
           
           if not significant:
               break
               
           # Accept and update
           selected.append(best_idx)
           Z_current = np.hstack([Z_current, X_best])
           
       return selected

Comparison with Alternative Methods
==================================

The standard oCSE can be compared with its variants:

.. list-table:: Method Comparison
   :widths: 20 25 25 30
   :header-rows: 1

   * - Method
     - Initial Conditioning
     - Advantages
     - Use Cases
   * - Standard oCSE
     - Yes (lagged target)
     - Controls autoregression
     - Time series with strong temporal structure
   * - Alternative oCSE
     - No
     - Simpler, fewer assumptions
     - Exploratory analysis, weak temporal structure
   * - Information LASSO
     - Variable
     - Handles high dimensions
     - Large predictor spaces
   * - Pure LASSO
     - No
     - Computationally efficient
     - Linear relationships, benchmarking

Theoretical Connections
======================

The standard oCSE framework connects to several established concepts:

**Granger Causality:** Standard oCSE generalizes linear Granger causality to 
information-theoretic measures with flexible conditioning.

**Transfer Entropy:** Related but distinct - transfer entropy typically uses uniform 
conditioning across all variables, while oCSE uses targeted conditioning sets.

**Partial Correlation:** The Gaussian version of standard oCSE is closely related to 
partial correlation analysis but extends to nonlinear relationships.

Conclusion
==========

Standard oCSE provides a principled, information-theoretic approach to causal discovery 
that explicitly accounts for autoregressive structure in time series data. The method's 
strength lies in its theoretical foundation, flexibility in information measures, and 
rigorous statistical testing. However, users should be aware of its computational 
requirements and the assumptions inherent in the initial conditioning approach.

The method is particularly well-suited for time series with strong temporal dependencies 
where controlling for autoregressive effects is crucial for accurate causal inference.