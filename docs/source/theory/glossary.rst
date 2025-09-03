========
Glossary
========

This glossary provides definitions of key terms and concepts used throughout the 
Causation Entropy library and documentation.

.. glossary::

   Causal Discovery
      The process of inferring causal relationships between variables from observational 
      data, without direct experimental intervention. Distinguished from correlation 
      analysis by attempting to identify directional, mechanistic relationships.

   Causation Entropy
      An information-theoretic measure of causal influence based on conditional mutual 
      information. Quantifies how much information a potential cause provides about an 
      effect, beyond what is already known from other variables.

   Conditional Mutual Information (CMI)
      A measure of mutual dependence between two variables given knowledge of a third 
      variable (or set of variables). Mathematically:
      
      .. math::
         I(X; Y | Z) = H(X | Z) - H(X | Y, Z)

   Conditioning Set
      The set of variables :math:`\mathbf{Z}` that are held constant when computing 
      conditional mutual information. In causal discovery, this typically includes 
      confounding variables and previously selected predictors.

   Differential Entropy
      The entropy of a continuous random variable, defined as:
      
      .. math::
         H(X) = -\int f(x) \log f(x) \, dx
         
      where :math:`f(x)` is the probability density function.

   Discrete Entropy
      The entropy of a discrete random variable, defined as:
      
      .. math::
         H(X) = -\sum_i p(x_i) \log p(x_i)
         
      where :math:`p(x_i)` is the probability mass function.

   False Discovery Rate (FDR)
      The expected proportion of false positives among all discoveries (rejected null 
      hypotheses). In causal discovery, this controls the expected fraction of 
      incorrectly identified causal relationships.

   False Positive Rate (FPR)
      The probability of incorrectly identifying a causal relationship when none exists.
      Also known as Type I error rate or :math:`1 - \text{specificity}`.
      
      .. math::
         \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}

   Forward Selection
      A greedy algorithm phase that iteratively selects the predictor variable with 
      the highest conditional mutual information with the target, subject to 
      statistical significance constraints.

   Backward Elimination
      A pruning phase that removes previously selected predictors that no longer 
      maintain statistical significance when conditioned on all other selected variables.

   Granger Causality
      A statistical concept of causality based on predictability: X is said to 
      Granger-cause Y if past values of X contain information that helps predict Y 
      beyond what is contained in past values of Y alone.

   Information Criterion
      A measure used for model selection that balances goodness of fit against model 
      complexity. Common examples include AIC (Akaike Information Criterion) and 
      BIC (Bayesian Information Criterion).

   k-Nearest Neighbor (k-NN) Estimator
      A non-parametric method for estimating probability densities and information 
      measures based on distances to the k-th nearest neighbor in the data space.

   Kernel Density Estimation (KDE)
      A non-parametric method for estimating probability density functions by placing 
      kernel functions (typically Gaussian) at each data point and summing their 
      contributions.

   Lag
      The time delay :math:`\tau` between a potential cause and its effect in time 
      series analysis. A lag of :math:`\tau` means the cause variable at time 
      :math:`t-\tau` potentially influences the effect variable at time :math:`t`.

   LASSO (Least Absolute Shrinkage and Selection Operator)
      A regularization method that performs variable selection by adding an L1 penalty 
      term to the loss function:
      
      .. math::
         \min_\beta \frac{1}{2n}||y - X\beta||_2^2 + \lambda ||\beta||_1

   Maximum Lag
      The maximum time delay :math:`\tau_{\max}` considered in causal discovery. 
      Variables are tested as potential causes at lags :math:`1, 2, \ldots, \tau_{\max}`.

   Mutual Information
      A measure of mutual dependence between two variables, quantifying the amount of 
      information obtained about one variable by observing another:
      
      .. math::
         I(X; Y) = H(X) - H(X | Y) = H(Y) - H(Y | X)

   Network Inference
      The process of reconstructing the structure of a network (graph) from 
      observational data on the nodes. In causal discovery, this involves identifying 
      directed edges representing causal relationships.

   Causation Entropy (oCSE)
      The main algorithmic framework of this library, which combines forward selection 
      and backward elimination of predictors based on conditional mutual information 
      and statistical significance testing.

   Permutation Test
      A non-parametric statistical test that assesses significance by comparing the 
      observed test statistic to a distribution generated by randomly permuting the 
      data under the null hypothesis.

   Spectral Radius
      The largest absolute value among all eigenvalues of a matrix. For stability of 
      dynamic systems, the spectral radius must be less than 1.

   Statistical Significance
      The probability that an observed relationship occurred by chance, typically 
      assessed using p-values and compared to a significance level :math:`\alpha` 
      (commonly 0.05).

   Time Series
      A sequence of data points indexed by time, typically collected at successive, 
      equally-spaced points in time.

   Transfer Entropy
      An information-theoretic measure of directed information transfer between time 
      series, closely related to Granger causality but based on information theory 
      rather than linear prediction.

   True Positive Rate (TPR)
      The probability of correctly identifying a causal relationship when it exists.
      Also known as sensitivity, recall, or statistical power.
      
      .. math::
         \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}

   Vector Autoregression (VAR)
      A multivariate extension of autoregressive models where each variable is 
      regressed on lagged values of itself and all other variables in the system:
      
      .. math::
         \mathbf{x}_t = \mathbf{A}_1 \mathbf{x}_{t-1} + \cdots + \mathbf{A}_p \mathbf{x}_{t-p} + \boldsymbol{\epsilon}_t

Mathematical Notation
====================

Common mathematical symbols used throughout the documentation:

.. list-table:: Mathematical Symbols
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`H(X)`
     - Entropy of random variable X
   * - :math:`I(X; Y)`
     - Mutual information between X and Y
   * - :math:`I(X; Y | Z)`
     - Conditional mutual information between X and Y given Z
   * - :math:`X^{(t)}`
     - Variable X at time t
   * - :math:`X_i^{(t-\tau)}`
     - Variable i at time t-τ (lag τ)
   * - :math:`\mathbf{Z}_i^{(t)}`
     - Conditioning set for variable i at time t
   * - :math:`\tau`
     - Time lag
   * - :math:`\tau_{\max}`
     - Maximum lag considered
   * - :math:`\alpha`
     - Significance level (e.g., 0.05)
   * - :math:`\lambda`
     - Regularization parameter
   * - :math:`\mathbf{A}`
     - Adjacency matrix
   * - :math:`\rho`
     - Spectral radius or correlation coefficient
   * - :math:`\epsilon`
     - Error term or small constant
   * - :math:`\psi(\cdot)`
     - Digamma function
   * - :math:`\Gamma(\cdot)`
     - Gamma function
   * - :math:`|\mathbf{M}|`
     - Determinant of matrix M
   * - :math:`\mathbf{I}_n`
     - n×n identity matrix
   * - :math:`\mathbb{E}[\cdot]`
     - Expected value
   * - :math:`\text{Var}(\cdot)`
     - Variance
   * - :math:`\text{Cov}(\cdot, \cdot)`
     - Covariance

Abbreviations
=============

.. list-table:: Common Abbreviations
   :widths: 20 80
   :header-rows: 1

   * - Abbreviation
     - Full Term
   * - oCSE
     - optimal Causal Entropy
   * - CMI
     - Conditional Mutual Information
   * - MI
     - Mutual Information
   * - KDE
     - Kernel Density Estimation
   * - k-NN
     - k-Nearest Neighbor
   * - KSG
     - Kraskov-Stögbauer-Grassberger (estimator)
   * - LASSO
     - Least Absolute Shrinkage and Selection Operator
   * - VAR
     - Vector Autoregression
   * - AIC
     - Akaike Information Criterion
   * - BIC
     - Bayesian Information Criterion
   * - ROC
     - Receiver Operating Characteristic
   * - AUC
     - Area Under Curve
   * - TPR
     - True Positive Rate
   * - FPR
     - False Positive Rate
   * - FDR
     - False Discovery Rate
   * - TE
     - Transfer Entropy
   * - GC
     - Granger Causality