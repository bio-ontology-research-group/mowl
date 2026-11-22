Use a trained model
===================

Every model in mOWL answers the same questions, whatever it looks like inside. You get the
answers through its *geometry*: ``model.as_geometry``, or ``geometry_of(model)`` for a model
or a module. The geometry shares the parameters of the model, so nothing is copied.

Train a model first. Here it is ELBE on the family ontology:

.. testcode::

   import logging
   import torch as th
   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import ELBE

   logging.disable(logging.CRITICAL)   # keep the training log out of this page
   th.manual_seed(0)
   model = ELBE(FamilyDataset(), embed_dim=16, margin=0.05, learning_rate=0.01)
   model.train(epochs=400)

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...

Regions of classes
------------------

A class is a region. For ELBE it is a box, given by its centre and its half-width in every
dimension. Other methods have balls, unions of boxes, cones and so on. You ask for the
regions of classes by their row numbers (``model.class_index_dict`` maps IRIs to rows):

.. testcode::

   geometry = model.as_geometry
   to_row = model.class_index_dict

   def ids(*names):
       return th.tensor([to_row["http://" + name] for name in names])

   mother = geometry.classes(ids("Mother"))
   print(type(mother).__name__, tuple(mother.centre.shape), tuple(mother.offset.shape))

.. testoutput::

   Box (1, 16) (1, 16)

Is one class inside another?
----------------------------

``geometry.inside(a, b)`` is the cost of "region a lies inside region b". The cost is zero
when the method's measure says it holds exactly, and grows the further the regions are from
it. A low cost means "more true". Costs of different methods are not comparable, but
the costs of one model are:

.. testcode::

   def cost(sub, sup):
       return float(geometry.inside(geometry.classes(ids(sub)), geometry.classes(ids(sup))))

   print(cost("Mother", "Parent") < cost("Parent", "Mother"))
   print(cost("Mother", "Parent") < cost("Mother", "Male"))

.. testoutput::

   True
   True

Mother is inside Parent, and Parent is not inside Mother. Mother is not inside Male
(Female and Male are disjoint in the ontology).

Compound class expressions
--------------------------

A class expression can be more than a name. ``expression_region`` turns any OWL class
expression into a region of the geometry. It uses only the interface, so it works for every
method that can build the compound class. Then ``subsumption_cost`` and ``rank_classes``
answer questions about it. ``rank_classes`` lists the named classes that lie inside an
expression, best first. For individuals, ``rank_individuals`` lists them by how well they
are members of an expression; the family ontology has no individuals, so there is no example
here.

.. testcode::

   from mowl.owlapi import OWLAPIAdapter
   from mowl.representation.library.logic_expressions import (
       expression_region, rank_classes, subsumption_cost, id_maps)

   adapter = OWLAPIAdapter()
   cls = lambda name: adapter.create_class("http://" + name)
   has_child = adapter.create_object_property("http://hasChild")
   maps = id_maps(model)                 # class, property and individual rows

   # a parent who is female: Female and Parent
   mothers = adapter.create_object_intersection_of(cls("Female"), cls("Parent"))
   best = rank_classes(model, mothers, *maps, top_k=1)
   print(best[0][0])

   # is a Mother something that has a child who is a Person?
   some_child = adapter.create_object_some_values_from(has_child, cls("Person"))
   print(subsumption_cost(model, cls("Mother"), some_child, *maps) <
         subsumption_cost(model, cls("Male"), some_child, *maps))

.. testoutput::

   http://Mother
   True

The result of ``expression_region`` is a region again, so it can be compared with ``inside``
like any other. With ``with_gap=True`` it also returns how far the conjunctions inside the
expression are from being empty.

What a model cannot answer
--------------------------

A region family is not closed under every constructor. Two boxes intersect in a box, but a
box has no complement, and two balls do not intersect in a ball. When you ask for something
the geometry cannot do, you get a ``ConstructorUnavailable`` error that names the constructor
and the geometry; you do not get a wrong number:

.. testcode::

   from mowl.representation import ConstructorUnavailable

   not_male = adapter.create_complement_of(cls("Male"))
   try:
       expression_region(model, not_male, *maps)
   except ConstructorUnavailable as error:
       print(error)

.. testoutput::

   Box has no complement. A class expression with a complement cannot be evaluated by ComposedGeometry.

Methods that have the constructor (fuzzy methods such as FALCON and DF-ALC, cone methods such
as Al-cone, the subspace method E2R) answer the same call. The catalogue says which
constructors each method has: the row "Compound classes" on its card, or the table in
:doc:`combinations`.

Which part says what
--------------------

The geometry names its own parts, using the vocabulary of :doc:`index`:

.. testcode::

   for part, value in geometry.parts.items():
       print(f"{part}: {value}")

.. testoutput::

   classes: box
   relations: translation
   individuals: region
   measure: bounds per coordinate
   conjunction: exact

The same call works for methods that were not built from parts, such as an ELEmbeddings model
trained with its own module (``geometry_of`` finds the adapter for it).
