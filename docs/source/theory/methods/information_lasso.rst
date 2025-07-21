============================
Information-Theoretic LASSO
============================

The Information-Theoretic LASSO (info-LASSO) method combines the information-theoretic 
foundation of optimal Causal Entropy with the regularization framework of LASSO regression. 
This hybrid approach aims to handle high-dimensional predictor spaces while maintaining 
the nonlinear relationship detection capabilities of information measures.

Mathematical Foundation
=======================

Traditional LASSO Objective
---------------------------

The standard LASSO optimization problem is:

.. math::

   \hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1

where:
- :math:`\mathbf{y} \in \mathbb{R}^n` is the response vector
- :math:`\mathbf{X} \in \mathbb{R}^{n \times p}` is the predictor matrix  
- :math:`\boldsymbol{\beta} \in \mathbb{R}^p` are the regression coefficients
- :math:`\lambda \geq 0` is the regularization parameter

Information-Theoretic Extension
------------------------------

The info-LASSO modifies this framework by incorporating information-theoretic weights 
and measures. The conceptual objective becomes:

.. math::

   \hat{\mathbf{S}} = \arg\min_{\mathbf{S} \subseteq \{1,\ldots,p\}} \left[ -\sum_{j \in \mathbf{S}} w_j \cdot I_j + \lambda |\mathbf{S}| \right]

where:
- :math:`\mathbf{S}` is the selected predictor subset
- :math:`w_j` are information-based weights
- :math:`I_j` represents the information contribution of predictor :math:`j`
- :math:`\lambda` controls the sparsity-information tradeoff

Information-Theoretic Weights
=============================

The weights :math:`w_j` are derived from conditional mutual information measures:

Base Weights
-----------

For each potential predictor :math:`X_j^{(t-\tau)}`:

.. math::

   w_{j,\tau} = \frac{I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_i)}{\sum_{k,\tau'} I(X_k^{(t-\tau')}; X_i^{(t)} | \mathbf{Z}_i)}

This normalizes the conditional mutual information values to create relative importance weights.

Adaptive Weighting
------------------

The weights can be updated iteratively based on current selections:

.. math::

   w_{j,\tau}^{(k+1)} = w_{j,\tau}^{(k)} \cdot \exp\left(\alpha \cdot I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{S}_i^{(k)})\right)

where:
- :math:`\mathbf{S}_i^{(k)}` is the current selection set at iteration :math:`k`
- :math:`\alpha` controls the adaptation rate

Significance-Based Weighting
----------------------------

Incorporate statistical significance from permutation tests:

.. math::

   w_{j,\tau} = I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_i) \cdot \mathbb{I}(\text{p-value}_{j,\tau} < \alpha_{\text{thresh}})

where :math:`\mathbb{I}(\cdot)` is the indicator function and :math:`\text{p-value}_{j,\tau}` 
comes from permutation testing.

Algorithmic Approaches
=====================

There are several ways to implement information-theoretic LASSO:

Approach 1: Weighted LASSO with Information Weights
--------------------------------------------------

1. **Weight Computation:** Calculate information-theoretic weights for all predictors
2. **Weighted LASSO:** Solve the modified LASSO problem:

   .. math::

      \hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \sum_{j=1}^p \frac{1}{w_j} |\beta_j|

3. **Selection:** Variables with :math:`\hat{\beta}_j \neq 0` are selected

Approach 2: Information-Guided Regularization Path
--------------------------------------------------

1. **Information Ranking:** Rank predictors by conditional mutual information
2. **Adaptive :math:`\lambda`:** Use different regularization for different predictors:

   .. math::

      \lambda_j = \lambda_0 \cdot \exp(-\gamma \cdot \text{rank}(I_j))

3. **Group Selection:** Apply group-wise regularization based on information content

Approach 3: Iterative Information-LASSO
---------------------------------------

Alternate between information computation and LASSO selection:

.. code-block:: none

   Initialize: S = ∅, Z = Z_init
   Repeat:
       1. Compute CMI for all candidates given current Z
       2. Update weights based on CMI values  
       3. Solve weighted LASSO with current weights
       4. Update selection S and conditioning set Z
   Until convergence

Implementation Framework
=======================

Two-Stage Implementation
-----------------------

**Stage 1: Information Assessment**

.. code-block:: python

   def compute_information_weights(X, Y, Z, method='gaussian'):
       """Compute information-theoretic weights for all predictors."""
       n_predictors = X.shape[1]
       weights = np.zeros(n_predictors)
       
       for j in range(n_predictors):
           X_j = X[:, [j]]
           cmi = conditional_mutual_information(X_j, Y, Z, method=method)
           weights[j] = cmi
           
       # Normalize weights
       weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
       return weights

**Stage 2: Weighted LASSO**

.. code-block:: python

   def information_lasso(X, Y, weights, lambda_reg=1.0):
       """Solve weighted LASSO problem with information weights."""
       from sklearn.linear_model import Lasso
       
       # Create penalty weights (inverse of information weights)
       penalty_weights = 1.0 / (weights + 1e-8)  # Add small constant for stability
       
       # Weighted features (approximate weighted penalty via feature scaling)
       X_weighted = X / penalty_weights.reshape(1, -1)
       
       # Fit LASSO
       lasso = Lasso(alpha=lambda_reg, fit_intercept=True)
       lasso.fit(X_weighted, Y.ravel())
       
       # Recover original coefficients
       beta_original = lasso.coef_ / penalty_weights
       
       # Select non-zero coefficients
       selected = np.where(np.abs(beta_original) > 1e-6)[0]
       return selected, beta_original

Adaptive Implementation
----------------------

.. code-block:: python

   def adaptive_information_lasso(X, Y, Z_init, max_iter=10, tol=1e-6):
       """Iterative information-guided LASSO selection."""
       n_predictors = X.shape[1]
       selected = []
       Z_current = Z_init.copy() if Z_init is not None else None
       
       for iteration in range(max_iter):
           # Compute current information weights
           weights = compute_information_weights(X, Y, Z_current)
           
           # Apply weighted LASSO
           new_selected, _ = information_lasso(X, Y, weights)
           
           # Check convergence
           if set(new_selected) == set(selected):
               break
               
           # Update selection and conditioning set
           selected = list(new_selected)
           if len(selected) > 0:
               Z_current = X[:, selected]
               if Z_init is not None:
                   Z_current = np.hstack([Z_init, Z_current])
           
       return selected

Theoretical Properties
=====================

Sparsity-Information Tradeoff
----------------------------

The info-LASSO balances two competing objectives:

.. math::

   \text{Information Gain} = \sum_{j \in \mathbf{S}} I(X_j^{(t-\tau_j)}; X_i^{(t)} | \mathbf{Z}_i)

.. math::

   \text{Complexity Cost} = \lambda |\mathbf{S}|

The optimal selection satisfies:

.. math::

   \mathbf{S}^* = \arg\max_{\mathbf{S}} \left[ \sum_{j \in \mathbf{S}} I_j - \lambda |\mathbf{S}| \right]

Consistency Properties
---------------------

Under appropriate conditions, the info-LASSO estimator has similar consistency properties 
to standard LASSO:

1. **Selection Consistency:** :math:`P(\hat{\mathbf{S}} = \mathbf{S}_{\text{true}}) \to 1` as :math:`n \to \infty`
2. **Estimation Consistency:** :math:`\|\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}_{\text{true}}\|_2 \to 0` 

The key difference is that "true" relationships are defined by information-theoretic 
rather than linear relationships.

Oracle Properties
-----------------

With proper choice of :math:`\lambda_n`, the info-LASSO can achieve oracle properties:

.. math::

   \lambda_n = o(n^{-1/2}) \quad \text{and} \quad \lambda_n \sqrt{\log p} \to \infty

Advantages and Limitations
=========================

Advantages
----------

1. **High-Dimensional Capability:** Handles :math:`p >> n` scenarios better than pure oCSE
2. **Regularization:** Built-in protection against overfitting
3. **Computational Efficiency:** Leverages fast LASSO solvers
4. **Information Preservation:** Maintains information-theoretic relationships
5. **Flexibility:** Can incorporate various information measures
6. **Path Solutions:** Can explore entire regularization path

Limitations
-----------

1. **Linear Approximation:** LASSO stage assumes linear relationships
2. **Weight Sensitivity:** Performance depends on quality of information weights
3. **Parameter Tuning:** Requires selection of :math:`\lambda` parameter
4. **Implementation Complexity:** More complex than pure LASSO or pure oCSE
5. **Theoretical Gaps:** Limited theoretical analysis for information-theoretic variant

Hyperparameter Selection
=======================

Cross-Validation for :math:`\lambda`
-----------------------------------

Use information-theoretic criteria for model selection:

.. math::

   \lambda^* = \arg\min_\lambda \text{CV-Score}(\lambda)

where the CV-Score can be based on:
- Prediction error (traditional)
- Information criteria (AIC, BIC)
- Out-of-sample mutual information

Information Criteria
--------------------

Adapt traditional criteria to information-theoretic setting:

.. math::

   \text{AIC}_{\text{info}} = -2 \sum_{j \in \hat{\mathbf{S}}} I_j + 2|\hat{\mathbf{S}}|

.. math::

   \text{BIC}_{\text{info}} = -2 \sum_{j \in \hat{\mathbf{S}}} I_j + |\hat{\mathbf{S}}| \log n

Comparison with Standard oCSE
=============================

.. list-table:: Method Comparison
   :widths: 25 25 25 25
   :header-rows: 1

   * - Aspect
     - Standard oCSE
     - Info-LASSO
     - Hybrid Approach
   * - High Dimensions
     - Limited (p < n)
     - Good (p >> n)
     - Excellent
   * - Nonlinear Relations
     - Excellent
     - Limited
     - Good
   * - Computation Time
     - Slow
     - Fast
     - Medium
   * - Parameter Tuning
     - Minimal
     - Moderate
     - Complex
   * - Theoretical Foundation
     - Strong
     - Developing
     - Emerging

Use Case Guidelines
==================

When to Use Info-LASSO
----------------------

1. **High-Dimensional Data:** :math:`p > n` or :math:`p \approx n`
2. **Mixed Relationships:** Combination of linear and nonlinear dependencies  
3. **Computational Constraints:** Limited time for full oCSE analysis
4. **Regularization Needed:** Risk of overfitting with standard oCSE
5. **LASSO Familiarity:** Team comfortable with regularization approaches

When to Avoid Info-LASSO
------------------------

1. **Purely Nonlinear Systems:** Standard oCSE more appropriate
2. **Low-Dimensional Data:** Overhead not justified for small :math:`p`
3. **Theoretical Requirements:** Need rigorous information-theoretic guarantees
4. **Complex Conditioning:** Requires sophisticated conditioning strategies

Example Application
==================

Consider a high-dimensional time series system:

.. code-block:: python

   import numpy as np
   from causalentropy.core.discovery import discover_network
   
   # Generate high-dimensional data
   T, n = 500, 50  # p >> n scenario
   data = generate_high_dim_system(T, n, sparsity=0.1)
   
   # Apply different methods
   # Standard oCSE (may struggle with high dimensions)
   G_standard = discover_network(data, method='standard', max_lag=2)
   
   # Information LASSO
   G_info_lasso = discover_network(data, method='information_lasso', max_lag=2)
   
   # Pure LASSO (baseline)
   G_lasso = discover_network(data, method='lasso', max_lag=2)
   
   # Compare results
   compare_networks(G_standard, G_info_lasso, G_lasso, true_network)

Future Directions
=================

Research Opportunities
---------------------

1. **Theoretical Analysis:** Develop consistency theory for information-LASSO
2. **Adaptive Methods:** Dynamic weight adjustment during selection
3. **Group Information LASSO:** Extend to grouped variable selection
4. **Nonlinear Extensions:** Kernel or neural network variants
5. **Bayesian Formulation:** Probabilistic interpretation of information weights

Implementation Improvements
--------------------------

1. **Efficient Algorithms:** Specialized solvers for information-weighted problems
2. **Automatic Tuning:** Data-driven :math:`\lambda` selection methods
3. **Parallel Computing:** Distributed computation for large-scale problems
4. **Memory Optimization:** Efficient storage for high-dimensional cases

Conclusion
==========

Information-theoretic LASSO represents a promising direction for combining the strengths 
of information-based causal discovery with the computational and regularization advantages 
of LASSO methods. While still an active area of research, it offers practical solutions 
for high-dimensional causal discovery problems where traditional oCSE methods may struggle.

The approach is particularly valuable in scenarios where:
- Dimensionality exceeds sample size
- Computational efficiency is important
- Mixed linear/nonlinear relationships exist
- Regularization is desired

Future development will likely focus on theoretical foundations, algorithmic improvements, 
and extensions to more complex relationship structures.