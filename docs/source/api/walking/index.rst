mowl.walking
======================

In this module we provide different methods for generating random walks given a graph.
The algorithms in mOWL are a variation from the original ones. Graphs obtained from ontologies always have labeled edges, therefore the **edge labels are included** in the random walks.

.. important::
   Random walks with size :math:`n` will include :math:`n` nodes with its edges (except in the last node). Therefore a random walk with size :math:`n` will be at most :math:`2n-1` long.

.. automodapi:: mowl.walking
   :no-heading:
   :headings: --

