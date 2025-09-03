================
Theoretical Guide
================

This section provides comprehensive theoretical background for the Causation Entropy library,
including mathematical foundations, algorithmic details, and methodological comparisons.

.. toctree::
   :maxdepth: 2

   glossary
   methods/standard_oce
   methods/alternative_oce
   methods/information_lasso
   methods/lasso_methods
   information_theory
   statistical_foundations

Overview
========

The Causation Entropy (oCSE) framework provides a principled approach to causal network
discovery from time series data using information-theoretic measures. The core philosophy
centers on the idea that causal relationships can be quantified through conditional mutual
information, which measures the information content shared between variables when conditioning
on relevant context.

Mathematical Foundation
=======================

The fundamental quantity in causation entropy is the conditional mutual information:

.. math::

   I(X_j^{(t-\tau)}; X_i^{(t)} | \mathbf{Z}_i^{(t)}) = 
   H(X_i^{(t)} | \mathbf{Z}_i^{(t)}) - H(X_i^{(t)} | X_j^{(t-\tau)}, \mathbf{Z}_i^{(t)})

where:
- :math:`X_j^{(t-\tau)}` is a potential causal variable at lag :math:`\tau`
- :math:`X_i^{(t)}` is the target variable at time :math:`t`
- :math:`\mathbf{Z}_i^{(t)}` is the conditioning set for variable :math:`i`

The "optimal" aspect refers to the systematic selection of the most informative predictors
while controlling for statistical significance through permutation testing.

Key Principles
==============

1. **Information-Theoretic Causation**: Causal relationships are quantified through 
   information measures that capture statistical dependencies beyond linear correlation.

2. **Forward-Backward Selection**: A two-phase algorithm that first selects the most
   informative predictors (forward) and then removes spurious relationships (backward).

3. **Statistical Significance**: All causal relationships are validated through 
   permutation tests to control false positive rates.

4. **Multivariate Conditioning**: The framework properly accounts for confounding
   variables through conditional mutual information.

Next Steps
==========

- :doc:`glossary`: Definitions of key terms and concepts
- :doc:`methods/standard_oce`: Detailed explanation of the standard oCSE algorithm  
- :doc:`methods/alternative_oce`: Alternative oCSE formulation without initial conditioning
- :doc:`methods/information_lasso`: Information-theoretic variant with LASSO regularization
- :doc:`methods/lasso_methods`: Pure LASSO-based approaches for comparison