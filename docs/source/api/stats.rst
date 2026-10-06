Statistical Utilities
=====================

Network Comparison
------------------

.. autofunction:: causationentropy.core.stats.auc
.. autofunction:: causationentropy.core.stats.Compute_TPR_FPR

Bootstrap Confidence Intervals
------------------------------

.. autofunction:: causationentropy.core.stats.moving_block_bootstrap_indices
.. autofunction:: causationentropy.core.stats.stationary_bootstrap_indices
.. autofunction:: causationentropy.core.stats.bootstrap_confidence_interval
.. autofunction:: causationentropy.core.stats.bootstrap_cmi_confidence_interval

Multiple-Testing Corrections
----------------------------

These functions adjust a set of p-values. To apply a correction to the edges
of a discovered network, use
:func:`causationentropy.graph.utils.apply_test_correction` (see :doc:`graph`).

.. autofunction:: causationentropy.core.stats.bonferroni_correction
.. autofunction:: causationentropy.core.stats.benjamini_hochberg_correction
.. autofunction:: causationentropy.core.stats.benjamini_yekutieli_correction
.. autofunction:: causationentropy.core.stats.adaptive_bh_correction
.. autofunction:: causationentropy.core.stats.estimate_null_proportion
