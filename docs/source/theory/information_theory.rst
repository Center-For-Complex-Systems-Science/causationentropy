==================
Information Theory
==================

This section provides a comprehensive overview of information theory concepts that 
underpin the Optimal Causal Entropy framework. Understanding these foundational concepts 
is essential for applying and interpreting the methods in this library.

Fundamental Concepts
===================

Entropy
-------

**Shannon Entropy** quantifies the average information content in a random variable.

For a discrete random variable :math:`X` with probability mass function :math:`p(x)`:

.. math::

   H(X) = -\sum_{x} p(x) \log p(x)

For a continuous random variable with probability density function :math:`f(x)`:

.. math::

   H(X) = -\int f(x) \log f(x) \, dx

**Properties:**
- :math:`H(X) \geq 0` with equality if and only if :math:`X` is deterministic
- :math:`H(X)` is maximized when :math:`X` is uniformly distributed
- Units depend on logarithm base: bits (base 2), nats (base e), or dits (base 10)

Joint and Conditional Entropy
-----------------------------

**Joint Entropy** of two variables :math:`X` and :math:`Y`:

.. math::

   H(X,Y) = -\sum_{x,y} p(x,y) \log p(x,y)

**Conditional Entropy** of :math:`X` given :math:`Y`:

.. math::

   H(X|Y) = -\sum_{x,y} p(x,y) \log p(x|y) = H(X,Y) - H(Y)

**Chain Rule of Entropy:**

.. math::

   H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^n H(X_i | X_1, \ldots, X_{i-1})

Mutual Information
================

Definition and Properties
------------------------

**Mutual Information** between :math:`X` and :math:`Y`:

.. math::

   I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)

Alternative formulation using Kullback-Leibler divergence:

.. math::

   I(X;Y) = D_{KL}(p(x,y) \| p(x)p(y)) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}

**Properties:**
- :math:`I(X;Y) \geq 0` with equality if and only if :math:`X` and :math:`Y` are independent
- :math:`I(X;Y) = I(Y;X)` (symmetric)
- :math:`I(X;X) = H(X)` (self-information equals entropy)
- :math:`I(X;Y) \leq \min(H(X), H(Y))` (bounded by marginal entropies)

Conditional Mutual Information
-----------------------------

**Conditional Mutual Information** of :math:`X` and :math:`Y` given :math:`Z`:

.. math::

   I(X;Y|Z) = H(X|Z) - H(X|Y,Z) = H(Y|Z) - H(Y|X,Z)

Equivalently:

.. math::

   I(X;Y|Z) = \sum_{x,y,z} p(x,y,z) \log \frac{p(x,y|z)}{p(x|z)p(y|z)}

**Chain Rule for Mutual Information:**

.. math::

   I(X;Y,Z) = I(X;Y) + I(X;Z|Y)

**Information Decomposition:**

.. math::

   I(X;Y) = I(X;Y|Z) + I(X;Z) - I(X;Z|Y)

This decomposition separates direct relationships from those mediated by :math:`Z`.

Information-Theoretic Measures for Causality
===========================================

Transfer Entropy
---------------

**Transfer Entropy** from :math:`Y` to :math:`X`:

.. math::

   T_{Y \to X} = I(X_{t+1}; Y_t^{(k)} | X_t^{(k)})

where :math:`X_t^{(k)} = (X_t, X_{t-1}, \ldots, X_{t-k+1})` represents the history of :math:`X`.

This measures the information flow from :math:`Y`'s past to :math:`X`'s future, 
beyond what is already contained in :math:`X`'s own past.

Causation Entropy
-----------------

**Causation Entropy** extends transfer entropy by considering multiple potential causes:

.. math::

   CE_{j \to i}(\tau) = I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{H}_i^{(t)}, \mathbf{S}_i^{(t)})

where:
- :math:`\mathbf{H}_i^{(t)}` is the historical information of variable :math:`i`
- :math:`\mathbf{S}_i^{(t)}` is the set of other selected causal variables

The "optimal" aspect comes from systematic selection of :math:`\mathbf{S}_i^{(t)}` 
to maximize information while controlling for statistical significance.

Estimation Methods
==================

The choice of entropy estimator significantly affects the performance of 
information-theoretic causal discovery methods.

Parametric Estimators
--------------------

**Gaussian Entropy:**
For multivariate Gaussian :math:`\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})`:

.. math::

   H(\mathbf{X}) = \frac{1}{2} \log((2\pi e)^k |\boldsymbol{\Sigma}|)

where :math:`k` is the dimensionality.

**Gaussian Mutual Information:**

.. math::

   I(X;Y) = -\frac{1}{2} \log(1 - \rho^2)

where :math:`\rho` is the correlation coefficient.

**Gaussian Conditional Mutual Information:**

.. math::

   I(X;Y|Z) = \frac{1}{2} \log \frac{|\boldsymbol{\Sigma}_Z||\boldsymbol{\Sigma}_{XYZ}|}{|\boldsymbol{\Sigma}_{XZ}||\boldsymbol{\Sigma}_{YZ}|}

Non-Parametric Estimators
-------------------------

**k-Nearest Neighbor (k-NN):**
The Kraskov-Stögbauer-Grassberger (KSG) estimator:

.. math::

   \hat{I}(X;Y) = \psi(k) + \psi(N) - \langle\psi(n_x + 1) + \psi(n_y + 1)\rangle

where:
- :math:`\psi` is the digamma function
- :math:`N` is the sample size
- :math:`n_x, n_y` are neighbor counts in marginal spaces

**Kernel Density Estimation (KDE):**
Estimate density using kernel functions:

.. math::

   \hat{f}(x) = \frac{1}{Nh} \sum_{i=1}^N K\left(\frac{x - x_i}{h}\right)

Then compute entropy as:

.. math::

   \hat{H}(X) = -\int \hat{f}(x) \log \hat{f}(x) \, dx

**Histogram-Based:**
Discretize continuous variables and use discrete entropy formulas:

.. math::

   \hat{H}(X) = -\sum_{i=1}^m \frac{n_i}{N} \log \frac{n_i}{N}

where :math:`n_i` is the count in bin :math:`i`.

Estimator Comparison
===================

.. list-table:: Entropy Estimator Properties
   :widths: 15 20 20 20 25
   :header-rows: 1

   * - Method
     - Bias
     - Variance
     - Complexity
     - Best Use Case
   * - Gaussian
     - Low (if Gaussian)
     - Low
     - O(n³)
     - Linear relationships
   * - k-NN
     - Moderate
     - Moderate
     - O(n² log n)
     - General purpose
   * - KDE
     - Moderate
     - High
     - O(n²)
     - Smooth densities
   * - Histogram
     - High
     - Low
     - O(n)
     - Discrete/mixed data

Advanced Information Measures
============================

Partial Information Decomposition
---------------------------------

Decompose multivariate information into components:

.. math::

   I(Y; X_1, X_2) = \text{Unique}(X_1) + \text{Unique}(X_2) + \text{Redundancy}(X_1, X_2) + \text{Synergy}(X_1, X_2)

This framework helps understand how multiple variables jointly provide 
information about a target.

Information Geometry
-------------------

Information measures have geometric interpretations:

**Fisher Information Metric:**

.. math::

   g_{ij} = \mathbb{E}\left[\frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j}\right]

**Kullback-Leibler Divergence:**

.. math::

   D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)}

This measures the "distance" between probability distributions.

Multivariate Extensions
======================

Multivariate Mutual Information
------------------------------

For :math:`n` variables :math:`X_1, \ldots, X_n`:

.. math::

   I(X_1; \ldots; X_n) = \sum_{i=1}^n H(X_i) - H(X_1, \ldots, X_n)

**Total Correlation:**

.. math::

   C(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i) - H(X_1, \ldots, X_n)

**Dual Total Correlation:**

.. math::

   D(X_1, \ldots, X_n) = H(X_1, \ldots, X_n) - \max_i H(X_1, \ldots, X_n | X_i)

Information Networks
-------------------

Construct networks where edge weights represent information flow:

.. math::

   w_{ij} = I(X_i; X_j | \mathbf{X}_{\setminus \{i,j\}})

This creates a **partial correlation network** in information-theoretic terms.

Practical Considerations
=======================

Sample Size Requirements
-----------------------

Information estimators have different sample size requirements:

**Rule of Thumb:**
- Gaussian: :math:`N \geq 10 \times \text{dimensionality}`
- k-NN: :math:`N \geq 100 \times \text{dimensionality}`  
- KDE: :math:`N \geq 1000 \times \text{dimensionality}`

**High-Dimensional Challenges:**
The "curse of dimensionality" affects all non-parametric estimators. 
Consider dimensionality reduction or parametric assumptions when :math:`d > 10`.

Bias and Variance Tradeoffs
--------------------------

**Bias Sources:**
- Finite sample effects
- Discretization (histogram methods)
- Boundary effects (KDE)
- Model assumptions (Gaussian)

**Variance Sources:**
- Random sampling variation
- Parameter choices (bandwidth, k)
- Outliers and noise

**Bias-Variance Management:**
- Cross-validation for parameter selection
- Bootstrap for uncertainty quantification
- Robust estimators for outlier handling

Statistical Testing
===================

Permutation Tests
----------------

Test :math:`H_0: I(X;Y|Z) = 0` using permutation of :math:`X`:

1. Compute observed :math:`I_{\text{obs}}(X;Y|Z)`
2. Generate :math:`B$ permutations of :math:`X`: :math:`X^{(b)}`
3. Compute null statistics: :math:`I^{(b)} = I(X^{(b)};Y|Z)`
4. P-value: :math:`p = \frac{1 + \sum_{b=1}^B \mathbb{I}(I^{(b)} \geq I_{\text{obs}})}{B + 1}`

Bootstrap Confidence Intervals
-----------------------------

Construct confidence intervals for information measures:

.. math::

   \text{CI}_{1-\alpha}(I) = [Q_{\alpha/2}(\{I^{(b)}\}), Q_{1-\alpha/2}(\{I^{(b)}\})]

where :math:`Q_p$ is the :math:`p$-quantile of bootstrap samples.

Multiple Testing Correction
---------------------------

When testing multiple relationships, correct for multiple comparisons:

**Bonferroni Correction:**
.. math::
   \alpha_{\text{adjusted}} = \frac{\alpha}{m}

**False Discovery Rate (FDR):**
.. math::
   \text{FDR} = \mathbb{E}\left[\frac{\text{False Positives}}{\max(\text{Total Positives}, 1)}\right]

Applications in Causal Discovery
===============================

Variable Selection
-----------------

Use information measures for feature selection:

.. math::

   \mathbf{S}^* = \arg\max_{\mathbf{S}} \left[I(X_{\mathbf{S}}; Y) - \lambda |\mathbf{S}|\right]

This balances information gain with model complexity.

Conditional Independence Testing
------------------------------

Test conditional independence: :math:`X \perp Y | Z$

Equivalent to testing: :math:`I(X;Y|Z) = 0$

**Advantages over linear methods:**
- Detects nonlinear dependencies
- No distributional assumptions
- Robust to outliers (with appropriate estimators)

Network Structure Learning
-------------------------

Learn network structure by testing all pairwise conditional independencies:

For each triple :math:`(X_i, X_j, X_k)`:
- Test :math:`I(X_i; X_j | X_k) = 0`
- Build network based on significant dependencies

Future Directions
================

Emerging Methods
---------------

1. **Deep Learning Estimators:** Neural networks for entropy estimation
2. **Causal Information Theory:** Information measures for causal inference
3. **Quantum Information:** Extensions to quantum systems
4. **Online Estimation:** Streaming entropy estimation
5. **Robust Estimation:** Methods resilient to model misspecification

Open Problems
------------

1. **Optimal Estimation:** Minimax rates for information measure estimation
2. **High-Dimensional Theory:** Consistency in :math:`p >> n` regimes
3. **Causal Identifiability:** When does information identify causation?
4. **Computational Efficiency:** Faster algorithms for large-scale problems

Conclusion
=========

Information theory provides the mathematical foundation for the Optimal Causal Entropy 
framework. Understanding entropy, mutual information, and their estimation is crucial 
for effective application of these methods. Key takeaways:

- **Estimator Choice Matters:** Different methods have different strengths and assumptions
- **Sample Size is Critical:** Information estimators need sufficient data
- **Statistical Testing is Essential:** Always validate relationships with significance tests
- **High Dimensions are Challenging:** Consider regularization or dimensionality reduction

The field continues to evolve, with new estimators and theoretical insights regularly 
emerging. Practitioners should stay informed of developments and validate methods on 
their specific data types and problem domains.