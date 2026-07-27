Projecting ontologies into graphs
==================================

Ontologies contain adjacency information that can be projected into a graph. There are different ways of generating such graphs:


.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.projection import CategoricalProjector

   ds = FamilyDataset()
   projector = CategoricalProjector("str")
   edges = projector.project(ds.ontology)

The ``edges`` generated is a list of :class:`mowl.projection.Edge`


Nested existential restrictions
---------------------------------

Most projectors only generate an edge for an existential restriction when its
filler is an atomic class. Definitions in phenotype ontologies are usually not
of that shape, and the informative term sits several levels down:

.. code-block:: text

   HP_0000123 SubClassOf has-part some (
       PATO_0000001 and inheres-in some UBERON_0000955)

:class:`mowl.projection.GDAProjector` unfolds such axioms, generating a single
edge whose relation is the composition of the roles traversed:

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.projection import GDAProjector

   ds = FamilyDataset()
   projector = GDAProjector()
   edges = projector.project(ds.ontology)

By default the unfolding runs from phenotype classes (HP, MP) to function or
anatomy classes (GO, UBERON), which is the setting used for gene--disease
association graphs in [zhapa2026]_. Both ends are configurable through the
``source_prefixes`` and ``target_prefixes`` parameters; passing empty tuples
disables the filtering altogether.


.. tip::

   All the implemented projectors can be found in :doc:`Projectors API docs <../../api/projection/index>`

   




