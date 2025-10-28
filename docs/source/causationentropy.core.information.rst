causationentropy.core.information package
======================================

Information-theoretic measures for causal discovery.

causationentropy.core.information.conditional_mutual_information module
--------------------------------------------------------------------

Conditional mutual information estimators for various distributions.

.. automodule:: causationentropy.core.information.conditional_mutual_information
   :members:
   :show-inheritance:
   :undoc-members:

Main Functions
~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.information.conditional_mutual_information.conditional_mutual_information
.. autofunction:: causationentropy.core.information.conditional_mutual_information.gaussian_conditional_mutual_information

Nonparametric Estimators
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.information.conditional_mutual_information.kde_conditional_mutual_information
.. autofunction:: causationentropy.core.information.conditional_mutual_information.knn_conditional_mutual_information
.. autofunction:: causationentropy.core.information.conditional_mutual_information.geometric_knn_conditional_mutual_information
.. autofunction:: causationentropy.core.information.conditional_mutual_information.histogram_conditional_mutual_information

Distribution-Specific Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.information.conditional_mutual_information.poisson_conditional_mutual_information

causationentropy.core.information.entropy module
---------------------------------------------

Entropy estimators for various distributions and methods.

.. automodule:: causationentropy.core.information.entropy
   :members:
   :show-inheritance:
   :undoc-members:

Utility Functions
~~~~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.information.entropy.l2dist
.. autofunction:: causationentropy.core.information.entropy.hyperellipsoid_check

Nonparametric Estimators
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.information.entropy.kde_entropy
.. autofunction:: causationentropy.core.information.entropy.geometric_knn_entropy

Distribution-Specific Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.information.entropy.poisson_entropy
.. autofunction:: causationentropy.core.information.entropy.poisson_joint_entropy
.. autofunction:: causationentropy.core.information.entropy.negative_binomial_entropy

causationentropy.core.information.mutual_information module
--------------------------------------------------------

Mutual information estimators for various distributions.

.. automodule:: causationentropy.core.information.mutual_information
   :members:
   :show-inheritance:
   :undoc-members:

Parametric Estimators
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.information.mutual_information.gaussian_mutual_information

Nonparametric Estimators
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causationentropy.core.information.mutual_information.kde_mutual_information
.. autofunction:: causationentropy.core.information.mutual_information.knn_mutual_information
.. autofunction:: causationentropy.core.information.mutual_information.geometric_knn_mutual_information

Module contents
---------------

.. automodule:: causationentropy.core.information
   :members:
   :show-inheritance:
   :undoc-members:
