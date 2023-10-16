Projecting ontologies into graphs
==================================

Ontologies contain adjacency information that can be projected into a graph. There are different ways of generating such graphs:


.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.projection import CategoricalProjector

   ds = FamilyDataset()
   projector = CategoricalProjector()
   edges = projector.project(ds.ontology)

The ``edges`` generated is a list of :class:`mowl.projection.Edge`


.. tip::

   All the implemented projectors can be found in :doc:`Projectors API docs <../../api/projection/index>`

   




