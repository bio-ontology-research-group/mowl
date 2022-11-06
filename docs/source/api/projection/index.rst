mowl.projection
=============================

In this module we provide different methods for projecting an ontology into a graph: The methods we provide are:


Ontology Projection
------------------------------------

.. automodule:: mowl.projection.base
   :members:
   :show-inheritance:


All the methods will return a list of edges corresponding to the **Edge** class:

Edge
-----

.. automodule:: mowl.projection.edge
   :private-members:
   :members:
   :show-inheritance:


Subclass Hierarchy
---------------------

.. automodule:: mowl.projection.taxonomy.model
   :members:
   :no-inherited-members:
   :show-inheritance:


Subclass Hierarchy With Relations
------------------------------------

.. automodule:: mowl.projection.taxonomy_rels.model
   :members:
   :show-inheritance:


DL2Vec Graph
-------------

.. include:: dl2vec.rst




.. automodule:: mowl.projection.dl2vec.model
   :members:
   :show-inheritance:


OWL2Vec* Graph
----------------

The OWL2Vec* graph follows the rules described in the paper `OWL2Vec*: embedding of OWL ontologies (2021) <https://link.springer.com/article/10.1007%2Fs10994-021-05997-6>`__. The parsing rules are shown in the table below:


.. include:: owl2vec.rst




.. automodule:: mowl.projection.owl2vec_star.model
   :members:
   :show-inheritance:


