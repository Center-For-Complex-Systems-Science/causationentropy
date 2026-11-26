Equation Discovery
==================

Entropic regression (AlMomani, Sun and Bollt, 2020) finds a sparse set of terms
in a library of basis functions for each equation of a dynamical system, using the
same forward selection and backward elimination as optimal causation entropy.

.. autoclass:: causationentropy.core.estimators.EntropicRegression
   :members: fit, predict, equations

.. autofunction:: causationentropy.core.estimators.entropic_regression
.. autofunction:: causationentropy.core.estimators.polynomial_library
.. autofunction:: causationentropy.core.estimators.central_difference
