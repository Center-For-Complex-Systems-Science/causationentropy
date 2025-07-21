causalentropy.core package
==========================

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   causalentropy.core.information
   causalentropy.core.scratch

causalentropy.core.discovery module
-----------------------------------

The core module for causal network discovery algorithms.

.. automodule:: causalentropy.core.discovery
   :members:
   :undoc-members:
   :show-inheritance:

Main Discovery Function
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.discovery.discover_network

Causation Entropy Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.discovery.standard_optimal_causation_entropy
.. autofunction:: causalentropy.core.discovery.alternative_optimal_causation_entropy
.. autofunction:: causalentropy.core.discovery.information_lasso_optimal_causation_entropy
.. autofunction:: causalentropy.core.discovery.lasso_optimal_causation_entropy

Selection Algorithms
~~~~~~~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.discovery.standard_forward
.. autofunction:: causalentropy.core.discovery.alternative_forward
.. autofunction:: causalentropy.core.discovery.backward

Statistical Testing
~~~~~~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.discovery.shuffle_test

causalentropy.core.linalg module
--------------------------------

Linear algebra utilities for information-theoretic computations.

.. automodule:: causalentropy.core.linalg
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: causalentropy.core.linalg.correlation_log_determinant

causalentropy.core.plotting module
----------------------------------

Plotting utilities for visualization.

.. automodule:: causalentropy.core.plotting
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: causalentropy.core.plotting.roc_curve

causalentropy.core.stats module
-------------------------------

Statistical utilities and performance metrics.

.. automodule:: causalentropy.core.stats
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: causalentropy.core.stats.auc
.. autofunction:: causalentropy.core.stats.Compute_TPR_FPR

Module contents
---------------

.. automodule:: causalentropy.core
   :members:
   :undoc-members:
   :show-inheritance:
