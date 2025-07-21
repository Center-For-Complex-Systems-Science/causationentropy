=====================
Pure LASSO Methods
=====================

Pure LASSO methods represent the linear regression baseline for causal discovery in 
the Optimal Causal Entropy framework. While not information-theoretic in nature, these 
methods serve as important benchmarks and provide computationally efficient alternatives 
for linear systems. This section covers the theoretical foundation, implementation, and 
role of LASSO-based approaches in causal network inference.

Mathematical Foundation
=======================

Standard LASSO Formulation
--------------------------

The LASSO (Least Absolute Shrinkage and Selection Operator) solves the optimization problem:

.. math::

   \hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_1

This can be equivalently formulated as a constrained optimization:

.. math::

   \hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 \quad \text{subject to} \quad \|\boldsymbol{\beta}\|_1 \leq t

where :math:`t` corresponds to the constraint level determined by :math:`\lambda`.

Causal Discovery Context
-----------------------

For causal discovery from time series, the LASSO problem becomes:

**Target Variable:** :math:`X_i^{(t)}` for :math:`t = \tau_{\max} + 1, \ldots, T`

**Predictor Matrix:** 
.. math::

   \mathbf{X}_{i,\text{lag}} = \begin{bmatrix}
   X_1^{(\tau_{\max})} & X_1^{(\tau_{\max}-1)} & \cdots & X_1^{(1)} & \cdots & X_n^{(\tau_{\max})} & \cdots & X_n^{(1)} \\
   X_1^{(\tau_{\max}+1)} & X_1^{(\tau_{\max})} & \cdots & X_1^{(2)} & \cdots & X_n^{(\tau_{\max}+1)} & \cdots & X_n^{(2)} \\
   \vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \ddots & \vdots \\
   X_1^{(T-1)} & X_1^{(T-2)} & \cdots & X_1^{(T-\tau_{\max})} & \cdots & X_n^{(T-1)} & \cdots & X_n^{(T-\tau_{\max})}
   \end{bmatrix}

**Response Vector:**
.. math::

   \mathbf{y}_i = \begin{bmatrix} X_i^{(\tau_{\max}+1)} \\ X_i^{(\tau_{\max}+2)} \\ \vdots \\ X_i^{(T)} \end{bmatrix}

The resulting coefficient vector has structure:
.. math::

   \boldsymbol{\beta}_i = [\beta_{i,1}^{(1)}, \beta_{i,1}^{(2)}, \ldots, \beta_{i,1}^{(\tau_{\max})}, \ldots, \beta_{i,n}^{(1)}, \ldots, \beta_{i,n}^{(\tau_{\max})}]^T

where :math:`\beta_{i,j}^{(\tau)}` represents the influence of variable :math:`j` at lag :math:`\tau` on variable :math:`i`.

Causal Interpretation
====================

Edge Detection
--------------

A directed edge from variable :math:`j` to variable :math:`i` at lag :math:`\tau` is inferred if:

.. math::

   |\hat{\beta}_{i,j}^{(\tau)}| > \epsilon

where :math:`\epsilon` is a small threshold (typically machine precision).

The strongest lag for each relationship can be determined by:

.. math::

   \tau_{i,j}^* = \arg\max_{\tau \in \{1,\ldots,\tau_{\max}\}} |\hat{\beta}_{i,j}^{(\tau)}|

Network Construction
-------------------

The inferred adjacency matrix :math:`\mathbf{A}` has entries:

.. math::

   A_{ji} = \begin{cases}
   1 & \text{if } \max_\tau |\hat{\beta}_{i,j}^{(\tau)}| > \epsilon \\
   0 & \text{otherwise}
   \end{cases}

With optional lag information:

.. math::

   L_{ji} = \begin{cases}
   \tau_{i,j}^* & \text{if } A_{ji} = 1 \\
   0 & \text{otherwise}
   \end{cases}

Regularization Parameter Selection
==================================

The choice of :math:`\lambda` critically affects the sparsity-accuracy tradeoff.

Cross-Validation Approach
-------------------------

Standard k-fold cross-validation minimizes prediction error:

.. math::

   \lambda^*_{CV} = \arg\min_\lambda \frac{1}{K} \sum_{k=1}^K \|\mathbf{y}_k^{\text{test}} - \mathbf{X}_k^{\text{test}}\hat{\boldsymbol{\beta}}_k(\lambda)\|_2^2

Information Criteria
--------------------

**Akaike Information Criterion (AIC):**
.. math::

   \text{AIC}(\lambda) = n \log(\text{RSS}(\lambda)/n) + 2|\hat{\mathbf{S}}(\lambda)|

**Bayesian Information Criterion (BIC):**
.. math::

   \text{BIC}(\lambda) = n \log(\text{RSS}(\lambda)/n) + |\hat{\mathbf{S}}(\lambda)| \log n

where :math:`\text{RSS}(\lambda) = \|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}(\lambda)\|_2^2` and 
:math:`|\hat{\mathbf{S}}(\lambda)|` is the number of selected predictors.

Stability Selection
-------------------

For more robust selection, use stability selection across bootstrap samples:

.. math::

   \Pi_j(\lambda) = P(\beta_j(\lambda) \neq 0) = \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\hat{\beta}_j^{(b)}(\lambda) \neq 0)

Select variables with :math:`\Pi_j(\lambda) \geq \pi_{\text{thresh}}$ (typically 0.6-0.8).

Implementation Approaches
=========================

Standard LASSO Implementation
-----------------------------

.. code-block:: python

   from sklearn.linear_model import LassoLarsIC, LassoCV
   import numpy as np
   
   def lasso_causal_discovery(data, max_lag=5, criterion='bic', alpha=None):
       """
       Discover causal network using LASSO regression.
       
       Parameters
       ----------
       data : array (T, n)
           Time series data
       max_lag : int
           Maximum lag to consider
       criterion : str
           Model selection criterion ('aic', 'bic', or 'cv')
       alpha : float or None
           Regularization parameter (if None, automatically selected)
       """
       T, n = data.shape
       
       # Create lagged design matrix
       X_lagged, Y_targets = create_lagged_matrices(data, max_lag)
       
       # Initialize results
       adjacency = np.zeros((n, n))
       coefficients = {}
       
       # Fit LASSO for each target variable
       for i in range(n):
           Y_i = Y_targets[:, i]
           
           if alpha is None:
               if criterion in ['aic', 'bic']:
                   # Use information criterion for model selection
                   lasso = LassoLarsIC(criterion=criterion, 
                                     normalize=True, 
                                     fit_intercept=True)
               else:
                   # Use cross-validation
                   lasso = LassoCV(cv=5, normalize=True, fit_intercept=True)
           else:
               from sklearn.linear_model import Lasso
               lasso = Lasso(alpha=alpha, normalize=True, fit_intercept=True)
           
           # Fit model
           lasso.fit(X_lagged, Y_i)
           
           # Extract causal relationships
           beta_i = lasso.coef_
           coefficients[i] = beta_i
           
           # Determine adjacency (reshape to (n, max_lag) structure)
           beta_reshaped = beta_i.reshape(n, max_lag)
           
           # Check for non-zero coefficients
           for j in range(n):
               if j != i:  # No self-loops
                   if np.any(np.abs(beta_reshaped[j, :]) > 1e-8):
                       adjacency[j, i] = 1  # j -> i
       
       return adjacency, coefficients

Advanced LASSO Variants
======================

Adaptive LASSO
--------------

Uses data-dependent weights to improve selection properties:

.. math::

   \hat{\boldsymbol{\beta}}_{\text{adaptive}} = \arg\min_{\boldsymbol{\beta}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \sum_{j=1}^p \frac{1}{|\hat{\beta}_j^{\text{OLS}}|^\gamma} |\beta_j|

where :math:`\hat{\boldsymbol{\beta}}^{\text{OLS}}` are ordinary least squares estimates and :math:`\gamma > 0`.

Group LASSO for Temporal Structure
----------------------------------

Groups coefficients by variable across all lags:

.. math::

   \hat{\boldsymbol{\beta}}_{\text{group}} = \arg\min_{\boldsymbol{\beta}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \sum_{j=1}^n \|\boldsymbol{\beta}_j\|_2

where :math:`\boldsymbol{\beta}_j = [\beta_{j}^{(1)}, \ldots, \beta_{j}^{(\tau_{\max})}]^T` contains all lag coefficients for variable :math:`j`.

Elastic Net
-----------

Combines L1 and L2 penalties:

.. math::

   \hat{\boldsymbol{\beta}}_{\text{enet}} = \arg\min_{\boldsymbol{\beta}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|_2^2

This addresses multicollinearity issues common in time series data.

Theoretical Properties
=====================

Consistency and Oracle Properties
---------------------------------

Under appropriate conditions, LASSO achieves:

**Selection Consistency:** 
.. math::
   P(\hat{\mathbf{S}} = \mathbf{S}_{\text{true}}) \to 1 \text{ as } n \to \infty

**Parameter Consistency:**
.. math::
   \|\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}_{\text{true}}\|_2 = O_p(\sqrt{s \log p / n})

where :math:`s = |\mathbf{S}_{\text{true}}|` is the true sparsity level.

Conditions for Consistency
--------------------------

Key assumptions for theoretical guarantees:

1. **Restricted Eigenvalue Condition:** 
   .. math::
      \inf_{\boldsymbol{\delta} \in \mathcal{C}_s} \frac{\|\mathbf{X}\boldsymbol{\delta}\|_2^2}{n\|\boldsymbol{\delta}\|_2^2} \geq \phi_{\min} > 0

2. **Sparsity:** :math:`s = o(n / \log p)`

3. **Signal Strength:** :math:`\min_{j \in \mathbf{S}_{\text{true}}} |\beta_j| \geq c\sqrt{\log p / n}`

4. **Regularization Choice:** :math:`\lambda \asymp \sqrt{\log p / n}`

Advantages and Limitations
=========================

Advantages
----------

1. **Computational Efficiency:** Fast algorithms (coordinate descent, LARS)
2. **High-Dimensional Capability:** Handles :math:`p >> n` scenarios
3. **Theoretical Guarantees:** Well-established consistency theory
4. **Interpretability:** Sparse solutions with clear coefficients
5. **Software Maturity:** Robust, well-tested implementations
6. **Automatic Selection:** Built-in variable selection
7. **Scalability:** Efficient for very large datasets

Limitations
-----------

1. **Linearity Assumption:** Cannot detect nonlinear relationships
2. **Correlation Issues:** May select arbitrary variables from correlated groups
3. **Causal Interpretation:** Linear coefficients ≠ causal relationships
4. **Temporal Assumptions:** Assumes stationary, linear dynamics
5. **No Significance Testing:** No built-in statistical testing framework
6. **Parameter Sensitivity:** Results depend heavily on :math:`\lambda` choice

Comparison with Information-Theoretic Methods
=============================================

.. list-table:: Method Comparison
   :widths: 20 25 25 30
   :header-rows: 1

   * - Aspect
     - LASSO
     - Standard oCSE
     - Information LASSO
   * - Relationship Type
     - Linear only
     - Linear + Nonlinear
     - Mixed
   * - Computational Speed
     - Very Fast
     - Slow
     - Moderate
   * - High Dimensions
     - Excellent
     - Limited
     - Good
   * - Statistical Testing
     - Limited
     - Rigorous
     - Developing
   * - Theoretical Foundation
     - Mature
     - Strong (IT)
     - Emerging
   * - Implementation
     - Simple
     - Complex
     - Moderate

When to Use LASSO Methods
========================

Recommended Scenarios
--------------------

1. **Linear Systems:** When relationships are primarily linear
2. **High-Dimensional Data:** :math:`p >> n` scenarios
3. **Computational Constraints:** Limited time/resources
4. **Baseline Analysis:** Initial exploration before sophisticated methods
5. **Benchmarking:** Comparison standard for other methods
6. **Large-Scale Systems:** Very large :math:`n`, :math:`p`, or :math:`T`
7. **Real-Time Applications:** When fast inference is required

Avoid When
----------

1. **Nonlinear Systems:** Complex, nonlinear relationships dominate
2. **Small-Scale Problems:** Information-theoretic methods are feasible
3. **Causal Rigor Required:** Need formal causal guarantees
4. **Heterogeneous Data:** Mixed data types or distributions

Best Practices
==============

Preprocessing
------------

1. **Standardization:** Center and scale variables to unit variance
2. **Stationarity:** Check and ensure stationarity (differencing if needed)
3. **Outlier Detection:** Remove or robust handling of outliers
4. **Missing Data:** Imputation or removal strategies

Model Selection
--------------

1. **Cross-Validation:** Use time series aware CV (e.g., time series split)
2. **Information Criteria:** BIC for conservative selection, AIC for liberal
3. **Stability Selection:** For robust variable selection
4. **Path Analysis:** Examine full regularization path

Post-Processing
--------------

1. **Lag Consolidation:** Combine multiple lags of same variable
2. **Significance Assessment:** Bootstrap or permutation-based confidence intervals
3. **Network Validation:** Compare with known relationships or other methods
4. **Robustness Checks:** Sensitivity analysis across parameter choices

Example Analysis
===============

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.linear_model import LassoLarsIC
   
   def analyze_lasso_path(data, target_var=0, max_lag=5):
       """Analyze LASSO regularization path for causal discovery."""
       
       # Prepare data
       X_lagged, Y_targets = create_lagged_matrices(data, max_lag)
       Y_target = Y_targets[:, target_var]
       
       # Fit LASSO path
       lasso = LassoLarsIC(criterion='bic', fit_intercept=True, normalize=True)
       lasso.fit(X_lagged, Y_target)
       
       # Extract selected variables
       selected_vars = np.where(lasso.coef_ != 0)[0]
       n_vars = data.shape[1]
       
       # Map back to (variable, lag) pairs
       selected_relationships = []
       for idx in selected_vars:
           var_idx = idx // max_lag
           lag_idx = idx % max_lag + 1  # lag starts from 1
           coeff = lasso.coef_[idx]
           selected_relationships.append((var_idx, lag_idx, coeff))
       
       # Print results
       print(f"Target Variable: {target_var}")
       print(f"Selected Relationships:")
       for var_idx, lag, coeff in selected_relationships:
           print(f"  Variable {var_idx} at lag {lag}: {coeff:.4f}")
       
       return selected_relationships, lasso

Integration with oCSE Framework
==============================

LASSO methods are integrated into the oCSE framework as:

1. **Baseline Comparison:** Standard benchmark for evaluation
2. **Initial Screening:** Fast preliminary variable selection  
3. **High-Dimensional Preprocessing:** Dimension reduction before oCSE
4. **Hybrid Approaches:** Combined with information-theoretic methods
5. **Validation Tool:** Cross-validation of oCSE results

Future Directions
================

Research Areas
-------------

1. **Nonlinear Extensions:** Kernel LASSO, neural network regularization
2. **Causal LASSO:** Explicit causal objective functions
3. **Time Series Adaptations:** Specialized methods for temporal data
4. **Robust Variants:** Methods robust to outliers and model misspecification
5. **Bayesian LASSO:** Uncertainty quantification in variable selection

Methodological Improvements
---------------------------

1. **Adaptive Regularization:** Data-driven :math:`\lambda` selection
2. **Group Structures:** Better handling of temporal and cross-sectional grouping
3. **Multi-Task Learning:** Joint learning across multiple target variables
4. **Online Methods:** Streaming/online causal discovery
5. **Distributed Computing:** Scalable implementations for massive datasets

Conclusion
==========

Pure LASSO methods provide a valuable computational and theoretical foundation for 
causal discovery in the oCSE framework. While limited to linear relationships, they 
offer unmatched computational efficiency and theoretical guarantees that make them 
essential tools for:

- High-dimensional problems where information-theoretic methods are infeasible
- Baseline comparisons and method evaluation
- Initial screening in large-scale analyses
- Systems where linear relationships dominate

The integration of LASSO methods with information-theoretic approaches represents a 
promising direction for combining computational efficiency with the ability to detect 
complex, nonlinear relationships. Understanding both approaches and their appropriate 
application domains is crucial for effective causal discovery in practice.