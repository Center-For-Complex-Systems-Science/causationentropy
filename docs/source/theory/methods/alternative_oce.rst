==========================
Alternative Causal Entropy  
==========================

The Alternative optimal Causal Entropy (alternative oCSE) represents a simplified variant 
of the causation entropy framework that begins with an empty conditioning set. This approach 
offers a more exploratory perspective on causal discovery, building relationships purely 
from the data without prior assumptions about autoregressive structure.

Mathematical Foundation
=======================

The alternative oCSE algorithm uses the same conditional mutual information measure as 
standard oCSE, but with a different conditioning strategy:

.. math::

   I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{S}_i^{(t)}) = 
   \sum_{x_j,x_i,\mathbf{s}} p(x_j,x_i,\mathbf{s}) \log \frac{p(x_i,x_j|\mathbf{s})}{p(x_i|\mathbf{s})p(x_j|\mathbf{s})}

The key difference is in the conditioning set evolution:
- **Standard oCSE:** :math:`\mathbf{Z}_i^{(t)} = \mathbf{Z}_{\text{init}} \cup \mathbf{S}_i^{(t)}`
- **Alternative oCSE:** :math:`\mathbf{Z}_i^{(t)} = \mathbf{S}_i^{(t)}` (no initial conditioning)

This means the algorithm starts with marginal mutual information and builds conditional 
dependencies organically through the selection process.

Algorithm Description
====================

The alternative oCSE follows the same two-phase structure as standard oCSE but with 
different initialization and conditioning logic.

Phase 1: Forward Selection without Initial Conditioning
------------------------------------------------------

**Initialization:**
For each target variable :math:`i`:
- Selected predictors: :math:`\mathbf{S}_i = \emptyset`
- Conditioning set: :math:`\mathbf{Z}_i = \emptyset`

**Iteration k=1 (Marginal Selection):**

1. **Marginal Mutual Information:** For each candidate predictor :math:`X_j^{(t-\tau)}`:

   .. math::

      \text{MI}_{j,\tau} = I(X_j^{(t-\tau)}; X_i^{(t)}) = H(X_i^{(t)}) - H(X_i^{(t)} | X_j^{(t-\tau)})

2. **Best Candidate Selection:**

   .. math::

      (j^*, \tau^*) = \arg\max_{j,\tau} \text{MI}_{j,\tau}

3. **Significance Testing:** Test :math:`H_0: I(X_{j^*}^{(t-\tau^*)}; X_i^{(t)}) = 0`

4. **First Selection:** If significant, :math:`\mathbf{S}_i \leftarrow \{X_{j^*}^{(t-\tau^*)}\}`

**Subsequent Iterations (k≥2):**

1. **Conditional Mutual Information:** For remaining candidates:

   .. math::

      \text{CMI}_{j,\tau} = I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{S}_i)

2. **Selection and Conditioning Update:** Following the same logic as standard oCSE

Phase 2: Backward Elimination
-----------------------------

Identical to standard oCSE, but the conditioning set for elimination only includes 
previously selected predictors:

.. math::

   \mathbf{Z}_{-j} = \mathbf{S}_i \setminus \{X_j^{(t-\tau)}\}

Key Algorithmic Differences
===========================

Conditioning Set Evolution
--------------------------

The conditioning set grows incrementally without prior assumptions:

.. math::

   \mathbf{Z}_i^{(0)} &= \emptyset \\
   \mathbf{Z}_i^{(1)} &= \{X_{j_1}^{(t-\tau_1)}\} \\
   \mathbf{Z}_i^{(2)} &= \{X_{j_1}^{(t-\tau_1)}, X_{j_2}^{(t-\tau_2)}\} \\
   &\vdots \\
   \mathbf{Z}_i^{(k)} &= \{X_{j_1}^{(t-\tau_1)}, \ldots, X_{j_k}^{(t-\tau_k)}\}

Information-Theoretic Interpretation
====================================

The alternative approach can be understood through the chain rule of mutual information:

.. math::

   I(X_{j_1}, X_{j_2}, \ldots, X_{j_k}; X_i) = \sum_{m=1}^k I(X_{j_m}; X_i | X_{j_1}, \ldots, X_{j_{m-1}})

The algorithm greedily maximizes each term in this decomposition, building the joint 
information incrementally. This provides a different perspective compared to standard 
oCSE, which conditions on autoregressive structure from the start.

First Selection: Marginal Mutual Information
--------------------------------------------

The first selected predictor maximizes marginal mutual information:

.. math::

   X_{j_1}^{(t-\tau_1)} = \arg\max_{j,\tau} I(X_j^{(t-\tau)}; X_i^{(t)})

This captures the strongest unconditional relationship, which may include autoregressive 
effects if they are the dominant signal.

Subsequent Selections: Conditional Uniqueness
---------------------------------------------

Later selections maximize conditional mutual information:

.. math::

   X_{j_k}^{(t-\tau_k)} = \arg\max_{j,\tau} I(X_j^{(t-\tau)}; X_i^{(t)} | X_{j_1}^{(t-\tau_1)}, \ldots, X_{j_{k-1}}^{(t-\tau_{k-1})})

This ensures each new predictor provides unique information not already captured by 
previously selected variables.

Advantages and Limitations
==========================

Advantages
----------

1. **No Prior Assumptions:** Does not assume autoregressive structure is important
2. **Exploratory Discovery:** May find unexpected relationships not constrained by temporal assumptions
3. **Simpler Implementation:** Fewer parameters and initialization steps
4. **Computational Efficiency:** Slightly faster due to smaller initial conditioning sets
5. **Data-Driven:** Relationships emerge purely from information content in data
6. **Interpretability:** First selection shows strongest marginal relationship

Limitations
-----------

1. **Confounding Risk:** Without autoregressive control, may select spurious relationships
2. **Order Dependence:** Early selections heavily influence later conditioning
3. **Transitivity Issues:** May select indirect relationships as direct causes
4. **Temporal Structure Ignored:** Does not explicitly account for time series nature
5. **Higher False Positive Risk:** Less conservative than standard approach

When to Use Alternative oCSE
============================

Recommended Scenarios
--------------------

1. **Exploratory Analysis:** Initial investigation of unknown systems
2. **Cross-Sectional Data:** When temporal structure is less important
3. **Weak Autoregressive Systems:** Time series without strong temporal dependencies
4. **Comparative Studies:** Baseline for comparison with standard oCSE
5. **High-Dimensional Systems:** When autoregressive conditioning becomes prohibitive
6. **Non-Temporal Networks:** Spatial or other non-temporal relationship discovery

Comparison with Standard oCSE
=============================

Consider a simple three-variable system:

.. math::

   X_1^{(t)} &= 0.5 X_1^{(t-1)} + 0.3 X_2^{(t-1)} + \epsilon_1^{(t)} \\
   X_2^{(t)} &= 0.4 X_2^{(t-1)} + \epsilon_2^{(t)} \\
   X_3^{(t)} &= 0.6 X_3^{(t-1)} + \epsilon_3^{(t)}

**Standard oCSE (Target: :math:`X_1^{(t)}`):**

1. Initial conditioning: :math:`\mathbf{Z}_{\text{init}} = \{X_1^{(t-1)}\}`
2. Evaluate: :math:`I(X_2^{(t-1)}; X_1^{(t)} | X_1^{(t-1)})` and :math:`I(X_3^{(t-1)}; X_1^{(t)} | X_1^{(t-1)})`
3. Likely selects :math:`X_2^{(t-1)}` due to causal relationship

**Alternative oCSE (Target: :math:`X_1^{(t)}`):**

1. No initial conditioning: :math:`\mathbf{Z} = \emptyset`
2. Evaluate: :math:`I(X_1^{(t-1)}; X_1^{(t)})`, :math:`I(X_2^{(t-1)}; X_1^{(t)})`, :math:`I(X_3^{(t-1)}; X_1^{(t)})`
3. Likely selects :math:`X_1^{(t-1)}` first (strongest marginal relationship)
4. Then evaluates :math:`I(X_2^{(t-1)}; X_1^{(t)} | X_1^{(t-1)})` and :math:`I(X_3^{(t-1)}; X_1^{(t)} | X_1^{(t-1)})`
5. Selects :math:`X_2^{(t-1)}` second

Both methods may reach the same final result, but through different paths and with 
different interpretations of the relationships.

Implementation Considerations
============================

Hyperparameter Differences
--------------------------

**Significance Levels:**
- May require more conservative :math:`\alpha` values due to increased multiple testing
- Consider Bonferroni or FDR corrections for the first (marginal) selection phase

**Information Estimator Selection:**
- Same considerations as standard oCSE
- May benefit from robust estimators due to less initial regularization

**Stopping Criteria:**
- Consider earlier stopping due to increased false positive risk
- Monitor conditioning set size to prevent overfitting

Example Implementation
=====================

.. code-block:: python

   def alternative_forward_selection(X, Y, alpha=0.05, n_shuffles=200):
       """
       Alternative oCSE forward selection without initial conditioning.
       
       Parameters
       ----------
       X : array (T, n*tau_max)
           Lagged predictor matrix
       Y : array (T, 1)
           Target variable
       """
       n_predictors = X.shape[1]
       selected = []
       Z_current = None  # Start with empty conditioning set
       
       while True:
           # Evaluate remaining candidates
           remaining = [i for i in range(n_predictors) if i not in selected]
           if not remaining:
               break
               
           cmi_values = []
           for j in remaining:
               X_j = X[:, [j]]
               if Z_current is None:
                   # First iteration: marginal mutual information
                   cmi = mutual_information(X_j, Y)
               else:
                   # Subsequent iterations: conditional mutual information
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
               
           # Accept and update conditioning set
           selected.append(best_idx)
           if Z_current is None:
               Z_current = X_best
           else:
               Z_current = np.hstack([Z_current, X_best])
           
       return selected

Diagnostic Analysis
==================

To understand the differences between standard and alternative oCSE results:

Selection Order Analysis
-----------------------

Compare the order of variable selection:

.. code-block:: python

   def compare_selection_order(X, Y, Z_init):
       """Compare selection order between methods."""
       
       # Standard oCSE
       standard_order = standard_forward_selection(X, Y, Z_init)
       
       # Alternative oCSE  
       alternative_order = alternative_forward_selection(X, Y)
       
       print("Standard oCSE order:", standard_order)
       print("Alternative oCSE order:", alternative_order)
       
       # Check if autoregressive terms selected first in alternative
       auto_vars = identify_autoregressive_variables(X, Y)
       alt_first_auto = any(var in auto_vars for var in alternative_order[:2])
       
       return {
           'standard_order': standard_order,
           'alternative_order': alternative_order,
           'alternative_selects_autoregressive': alt_first_auto
       }

Conditional MI Comparison
------------------------

Analyze how conditioning affects relationship strength:

.. math::

   \Delta\text{CMI}_{j,\tau} = I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{S}_i) - I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_{\text{init}} \cup \mathbf{S}_i)

Positive values indicate relationships that appear stronger without autoregressive conditioning.

Theoretical Implications
=======================

Model Selection Perspective
--------------------------

Alternative oCSE can be viewed as a model selection procedure that builds complexity 
incrementally:

.. math::

   \text{Model}_0: &\quad X_i^{(t)} = \epsilon_i^{(t)} \\
   \text{Model}_1: &\quad X_i^{(t)} = f_1(X_{j_1}^{(t-\tau_1)}) + \epsilon_i^{(t)} \\
   \text{Model}_2: &\quad X_i^{(t)} = f_2(X_{j_1}^{(t-\tau_1)}, X_{j_2}^{(t-\tau_2)}) + \epsilon_i^{(t)} \\
   &\vdots

Each selection represents a model complexity increase justified by information gain.

Connection to Feature Selection
------------------------------

The algorithm is closely related to information-based feature selection methods,
particularly those using mutual information criteria. The key difference is the 
explicit temporal structure and causal interpretation.

Conclusion
==========

Alternative oCSE provides a complementary approach to causal discovery that prioritizes 
data-driven relationship discovery over temporal assumptions. While it may be more 
susceptible to confounding and spurious relationships, it offers valuable insights 
for exploratory analysis and systems where autoregressive structure is not dominant.

The method is particularly useful as:
- A baseline for comparison with standard oCSE
- An exploratory tool for unknown systems  
- A method for cross-sectional or weakly temporal data
- A diagnostic tool to understand the role of autoregressive conditioning

Users should consider both approaches and compare results to gain a comprehensive 
understanding of the causal structure in their data.