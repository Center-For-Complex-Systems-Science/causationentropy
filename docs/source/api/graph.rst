Graph Utilities
===============

Helpers for turning a discovered network into other formats and for
correcting its p-values after discovery. All functions below can also be
imported from ``causationentropy.graph``.

DataFrames
----------

.. autofunction:: causationentropy.graph.utils.network_to_dataframe
.. autofunction:: causationentropy.graph.utils.pcmci_network_to_dataframe

Tigramite (PCMCI) Conversion
----------------------------

.. autofunction:: causationentropy.graph.utils.pcmci_to_networkx
.. autofunction:: causationentropy.graph.utils.networkx_to_pcmci

Multiple-Testing Correction
---------------------------

.. autofunction:: causationentropy.graph.utils.apply_test_correction
