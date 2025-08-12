causalentropy.core.information package
======================================

Information-theoretic measures for causal discovery.

causalentropy.core.information.conditional_mutual_information module
--------------------------------------------------------------------

Conditional mutual information estimators for various distributions.

.. automodule:: causalentropy.core.information.conditional_mutual_information
   :members:
   :undoc-members:
   :show-inheritance:

Main Functions
~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.information.conditional_mutual_information.conditional_mutual_information
.. autofunction:: causalentropy.core.information.conditional_mutual_information.gaussian_conditional_mutual_information

Nonparametric Estimators
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.information.conditional_mutual_information.kde_conditional_mutual_information
.. autofunction:: causalentropy.core.information.conditional_mutual_information.knn_conditional_mutual_information
.. autofunction:: causalentropy.core.information.conditional_mutual_information.geometric_knn_conditional_mutual_information

Distribution-Specific Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.information.conditional_mutual_information.poisson_conditional_mutual_information

causalentropy.core.information.entropy module
---------------------------------------------

Entropy estimators for various distributions and methods.

.. automodule:: causalentropy.core.information.entropy
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
~~~~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.information.entropy.l2dist
.. autofunction:: causalentropy.core.information.entropy.hyperellipsoid_check

Nonparametric Estimators
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.information.entropy.kde_entropy
.. autofunction:: causalentropy.core.information.entropy.geometric_knn_entropy

Distribution-Specific Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.information.entropy.poisson_entropy
.. autofunction:: causalentropy.core.information.entropy.poisson_joint_entropy

causalentropy.core.information.mutual_information module
--------------------------------------------------------

Mutual information estimators for various distributions.

.. automodule:: causalentropy.core.information.mutual_information
   :members:
   :undoc-members:
   :show-inheritance:

Parametric Estimators
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.information.mutual_information.gaussian_mutual_information

Nonparametric Estimators
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: causalentropy.core.information.mutual_information.kde_mutual_information
.. autofunction:: causalentropy.core.information.mutual_information.knn_mutual_information
.. autofunction:: causalentropy.core.information.mutual_information.geometric_knn_mutual_information

Module contents
---------------

.. automodule:: causalentropy.core.information
   :members:
   :undoc-members:
   :show-inheritance:
