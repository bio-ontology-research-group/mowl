The parts of a method
=====================

A publication fixes many things at once. How a method is trained and how it is evaluated
are described by regimes (:doc:`training_and_evaluation`). What remains, the
*representation*, is itself made of parts, and many publications change exactly one of them:
ELBE replaces the balls of ELEmbeddings by boxes, TransBox replaces ELBE's relation model,
RegD replaces the way inclusion is scored, MultiELBE replaces the class model again.
:mod:`mowl.representation` names these parts, and every method in the catalogue states its
own.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Part
     - Vocabulary
     - What it decides
   * - View
     - :class:`~mowl.representation.View`
     - What the method is given: normal-form axioms, class expressions, a graph, a corpus,
       text, queries, a canonical model.
   * - Space
     - :class:`~mowl.representation.Space`
     - Where embeddings live: a flat space, the unit cube, a hyperbolic space, a finite
       domain, ...
   * - Class model
     - :class:`~mowl.representation.ClassModel`
     - What a class is: a ball, a box, a union of boxes, a cone, a fuzzy set, ...
   * - Relation model
     - :class:`~mowl.representation.RelationModel`
     - What a relation is: a shift, a box of shifts, a head and a tail box, a matrix, a
       network, ...
   * - Individual model
     - :class:`~mowl.representation.IndividualModel`
     - What an individual is, and with it what the domain is: a point, a small region, a
       one-element class, an element of a finite domain.
   * - Constructors
     - :class:`~mowl.representation.Construction`
     - How the region of a compound class (and, exists, not, or) is obtained: exactly,
       approximately, by a trained network, by a fuzzy operator.
   * - Measure
     - :class:`~mowl.representation.Measure`
     - How "inside" is scored: distance of centres, bounds per coordinate, a volume ratio,
       an angle, ...
   * - Parametrisation
     - :class:`~mowl.representation.Parametrisation`
     - How the embedding of a name is produced: looked up in a table, or computed from
       text or from a graph. Only the latter give embeddings to names not seen in
       training.

The parts of a method are on its card:

.. testcode::

   from mowl.catalog import get_method

   for label, value in get_method("transbox").representation.rows():
       print(f"{label}: {value}")

.. testoutput::

   View of the ontology: normal-form axioms
   Space: euclidean
   A class is: box
   A relation is: box of translations
   An individual is: point
   "Inside" is scored by: bounds per coordinate
   Compound classes: and: exact; exists: exact; not: none; or: none
   Embeddings come from: lookup table
   Combination of independent parts: yes
   Remarks on the parts: Role chains are sums of the relation boxes, role inclusions are inclusions of them.

Methods can be listed by their parts. Which methods have boxes as classes, and what do
they use as relations?

.. testcode::

   from mowl.catalog import list_methods
   from mowl.representation import ClassModel, RelationModel, View

   for card in list_methods(parts=(ClassModel.BOX, View.NORMAL_FORMS)):
       print(card.key, "-", ", ".join(r.value for r in card.representation.relations))

.. testoutput::

   box2el - head and tail boxes
   boxel - affine map
   boxsel - affine map
   elbe - translation
   elkem - affine map with range and error box
   transbox - box of translations
   regd - translation

The same relation model appears in methods that are trained on different views:

.. testcode::

   print([c.key for c in list_methods(parts=RelationModel.TRANSLATION_BOX)])

.. testoutput::

   ['query2box', 'transbox']

Benchmarks whose test data name things that training did not see state so
(``Benchmark.requires``), and only methods whose embeddings are computed, not looked up,
are offered for them (OnT computes its embeddings from text, RESHUFFLE from the graph):

.. testcode::

   from mowl.benchmarks import methods_for

   print([c.key for c in list_methods(inductive=True)])
   print([c.key for c in methods_for("kg-inductive-fb15k237-v1")])

.. testoutput::

   ['eike', 'reshuffle', 'hit', 'ont']
   ['reshuffle', 'ont']

The last column of the table in :doc:`catalogue` says whether a method is a combination of
independent parts. Methods whose parts were designed together (GroundedEL), or share a
parameter (the bump vectors of Box\ :sup:`2`\ EL belong to the classes but are used only by
the relations), say so.
