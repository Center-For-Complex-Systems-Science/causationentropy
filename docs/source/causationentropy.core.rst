causationentropy.core package
==========================

Core algorithms and mathematical implementations for causal discovery.

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   causationentropy.core.information

causationentropy.core.discovery module
-----------------------------------

The core module for causal network discovery algorithms.

.. automodule:: causationentropy.core.discovery
   :members:
   :show-inheritance:
   :undoc-members:

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
   :show-inheritance:
   :undoc-members:

.. autofunction:: causationentropy.core.linalg.correlation_log_determinant
.. autofunction:: causationentropy.core.linalg.subnetwork
.. autofunction:: causationentropy.core.linalg.companion_matrix

causationentropy.core.plotting module
----------------------------------

Plotting utilities for visualization.

.. automodule:: causationentropy.core.plotting
   :members:
   :show-inheritance:
   :undoc-members:

.. autofunction:: causationentropy.core.plotting.roc_curve

causationentropy.core.stats module
-------------------------------

Statistical utilities and performance metrics.

.. automodule:: causationentropy.core.stats
   :members:
   :show-inheritance:
   :undoc-members:

.. autofunction:: causationentropy.core.stats.auc
.. autofunction:: causationentropy.core.stats.Compute_TPR_FPR

Module contents
---------------

.. automodule:: causationentropy.core
   :members:
   :show-inheritance:
   :undoc-members:
