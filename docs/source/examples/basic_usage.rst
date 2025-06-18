===========
Basic Usage
===========

This example demonstrates the fundamental usage of the library.

Simple Example
==============

.. code-block:: python

   import numpy as np
   from causalentropy import OptimalCausalEntropy
   
   # Generate sample data
   data = np.random.randn(100, 3)
   
   # Initialize the model
   oce = OptimalCausalEntropy()
   
   # Fit the model
   oce.fit(data)
   
   # Get results
   entropy = oce.compute_entropy()
   print(f"Causal entropy: {entropy}")

.. figure:: ../_static/images/diagrams/basic_flow.png
   :alt: Basic workflow diagram
   :width: 600px
   :align: center
   
   Basic workflow of the Optimal Causal Entropy method

Expected Output
===============