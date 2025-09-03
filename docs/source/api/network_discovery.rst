Network Discovery
=================

Core algorithms for causal network discovery from time series data.

Main Discovery Function
-----------------------

.. autofunction:: causationentropy.core.discovery.discover_network

Method Implementations
----------------------

.. autofunction:: causationentropy.core.discovery.standard_optimal_causation_entropy

.. autofunction:: causationentropy.core.discovery.alternative_optimal_causation_entropy

.. autofunction:: causationentropy.core.discovery.information_lasso_optimal_causation_entropy

.. autofunction:: causationentropy.core.discovery.lasso_optimal_causation_entropy

Selection Algorithms
--------------------

.. autofunction:: causationentropy.core.discovery.standard_forward

.. autofunction:: causationentropy.core.discovery.alternative_forward

.. autofunction:: causationentropy.core.discovery.backward

Statistical Testing
-------------------

.. autofunction:: causationentropy.core.discovery.shuffle_test

Linear Algebra Utilities
------------------------

.. autofunction:: causationentropy.core.linalg.correlation_log_determinant

Statistical Utilities
---------------------

.. autofunction:: causationentropy.core.stats.auc

.. autofunction:: causationentropy.core.stats.Compute_TPR_FPR

Visualization
-------------

.. autofunction:: causationentropy.core.plotting.roc_curve