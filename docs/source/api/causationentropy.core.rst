causationentropy.core package
==========================

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   causationentropy.core.information
   causationentropy.core.scratch

causationentropy.core.discovery module
-----------------------------------

The core module for causal network discovery algorithms.

.. automodule:: causationentropy.core.discovery
   :members:
   :undoc-members:
   :show-inheritance:

Main Discovery Function
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.discovery.discover_network

Causation Entropy Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.discovery.standard_optimal_causation_entropy
.. autofunction:: causationentropy.core.discovery.alternative_optimal_causation_entropy
.. autofunction:: causationentropy.core.discovery.information_lasso_optimal_causation_entropy
.. autofunction:: causationentropy.core.discovery.lasso_optimal_causation_entropy

Selection Algorithms
~~~~~~~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.discovery.standard_forward
.. autofunction:: causationentropy.core.discovery.alternative_forward
.. autofunction:: causationentropy.core.discovery.backward

Statistical Testing
~~~~~~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.discovery.shuffle_test

causationentropy.core.linalg module
--------------------------------

Linear algebra utilities for information-theoretic computations.

.. automodule:: causationentropy.core.linalg
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: causationentropy.core.linalg.correlation_log_determinant

causationentropy.core.plotting module
----------------------------------

Plotting utilities for visualization.

.. automodule:: causationentropy.core.plotting
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: causationentropy.core.plotting.roc_curve

causationentropy.core.stats module
-------------------------------

Statistical utilities and performance metrics.

.. automodule:: causationentropy.core.stats
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: causationentropy.core.stats.auc
.. autofunction:: causationentropy.core.stats.Compute_TPR_FPR

Module contents
---------------

.. automodule:: causationentropy.core
   :members:
   :undoc-members:
   :show-inheritance:
