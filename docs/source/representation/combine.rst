Combine parts
=============

:class:`~mowl.models.ComposedEL` is a model that you assemble. It trains on the normal-form
axioms of an ontology, like ELEmbeddings or ELBE do, and takes its class model, relation
model and measure of "inside" from the registries of :mod:`mowl.representation`. You can
do this in two ways.

Build a published method by name
--------------------------------

Most methods of the catalogue are combinations of parts. Name one and you get the same
losses as the method's own module (the tests check this number by number). The options of
the method are passed on:

.. testcode::

   import logging
   import torch as th
   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import ComposedEL

   logging.disable(logging.CRITICAL)
   th.manual_seed(0)
   dataset = FamilyDataset()

   model = ComposedEL(dataset, method="transbox", embed_dim=16, learning_rate=0.01)
   model.train(epochs=20)
   print(model.as_geometry.parts["classes"], "with", model.as_geometry.parts["relations"])

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...

``method`` takes any key of :data:`mowl.representation.METHODS`. The methods that
``ComposedEL`` can train are the ones that learn from axioms. Query answering methods
(Query2Box, BetaE, ConE, AConE) learn from queries, and ``ComposedEL`` refuses them with
an error that says so.

Choose the parts yourself
-------------------------

You can also name a class model, a relation model and a measure. This makes combinations
that no paper has published. Here are balls as classes, a relation that is a shift with a
learned tolerance (from EmEL(var)), and the way RegD measures inclusion, by distance to the
boundary and depth:

.. testcode::

   model = ComposedEL(dataset, classes="ball", relations="translation with tolerance",
                      measure="boundary and depth", embed_dim=16, margin=0.05,
                      learning_rate=0.01)
   model.train(epochs=100)
   print(model.as_geometry.parts["classes"], "/", model.as_geometry.parts["relations"],
         "/", model.as_geometry.parts["measure"])

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...

If you leave out ``measure``, the class model's own measure is used. The default objective
(``Objective(margin)``) is the semantic one: with a strict margin, a loss of zero means the
axiom holds. The objectives of the publications are options of the same
:class:`~mowl.representation.Objective`, and a named ``method`` sets the ones its authors
used (see :doc:`defaults`).

Replace one part of a published method
--------------------------------------

The main use of the parts: take a published method and change one thing. Name the method
and give the part you want instead. A part is a name from its registry (the lists below), a
function ``(size, embed_dim)`` or a part you built yourself, which is used as it is.

.. testcode::

   from mowl.representation import Objective

   elbe = ComposedEL(dataset, method="elbe", relations="affine map", embed_dim=16)
   print(elbe.module.parts["classes"], "/", elbe.module.parts["relations"])

   # the objective of the method with one option changed
   inclusion = ComposedEL(dataset, method="elbe", objective_options={"gci3": "inclusion"},
                          embed_dim=16)
   print(inclusion.module.objective.gci3_reading)

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...

The arguments are the same for every method and for every general model
(:class:`~mowl.models.ComposedEL`, :class:`~mowl.models.ComposedALC`):

``classes``, ``relations``, ``individuals``, ``measure``
    Another class model, relation model, individual model or measure. ``relations="none"``
    removes the relation model. A new class model gets individuals of its own kind unless you
    give ``individuals`` too.
``objective``, ``objective_options``
    Another :class:`~mowl.representation.Objective`, or options of the method's own
    objective that are replaced (:meth:`~mowl.representation.Objective.replace`).
``regularizers``
    Penalties that are added to those of the method.

Two things can go wrong, and both are errors, never silence. A pair of parts that cannot
exist (balls under a measure that compares boxes) raises
:class:`~mowl.representation.ConstructorUnavailable` and names both sides. An argument the
method cannot take raises ``TypeError`` and lists the accepted ones.

.. testcode::

   from mowl.representation import ConstructorUnavailable

   try:
       ComposedEL(dataset, method="elbe", classes="ball", embed_dim=16)
   except ConstructorUnavailable as error:
       print("refused:", str(error)[:60])

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...

The other general models do the same with the parts of their kind of method:
:class:`~mowl.models.ComposedSequenceModel` takes ``view``, ``walker``, ``source``,
``encoder``, ``reading``, ``measure`` and ``scorer``; the query model
(:class:`~mowl.models.composed.query.QueryEmbeddingModel`) takes ``intersection``,
``distance``, ``negation`` and, for ConE, ``relations``;
:class:`~mowl.models.constructed.ConstructedModel` takes ``source`` and ``reading``. A
method that has no such part (a method that reads sentences has no walker) refuses the
argument. :doc:`swaps` lists, for every method and every part, what works and what is
refused and why.

The named classes follow one rule
---------------------------------

``ELEmbeddings``, ``ELBE``, ``BoxEL`` and all the other named classes are one combination
each. They take their own options (the ones in their signature) and nothing else: a part
argument they do not list is refused with an error that says where to go. To replace a
part, use the general model with the key of the method.

.. testcode::

   from mowl.models import ELEmbeddings

   try:
       ELEmbeddings(dataset, relations="affine map")
   except TypeError as error:
       print("refused:", str(error)[:60])

   # the same swap, through the general model
   model = ComposedEL(dataset, method="elem", relations="translation with tolerance",
                      embed_dim=16)

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...

What parts exist
----------------

The registries are plain dictionaries from a name to the part. Their content grows as parts
are added, so ask the code:

.. testcode::

   from mowl.representation import METHODS, CLASS_MODELS, RELATION_MODELS, MEASURES

   print("methods:", sorted(METHODS))
   print("class models:", sorted(CLASS_MODELS))
   print("relation models:", sorted(RELATION_MODELS))
   print("measures:", sorted(MEASURES))

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   methods: [...]
   class models: [...]
   relation models: [...]
   measures: [...]

:doc:`combinations` shows the same lists as tables, with the combination behind every method.
Not every combination makes sense. A measure that compares boxes needs boxes, and some
measures need a particular relation model. A pair that does not fit fails with a message that names the
part or the constructor that is missing.

When a part cannot do something
-------------------------------

A part may lack a constructor or an axiom form. The error is
:class:`~mowl.representation.ConstructorUnavailable`, and it means one of these:

* The region family is not closed under the operation. Two balls do not intersect in a ball,
  and a box has no complement.
* The relation model has no reading of it. A relation model without ``role_inclusion``
  cannot read role inclusions, and one without ``image`` cannot give the region of the
  successors of a set.
* The geometry does not define the constructor at all (a hierarchy method has no
  existential restriction).

The error is never silent. A geometry that cannot read an axiom form is not trained on it:
``ComposedEL`` leaves that form out of the training data, so a hierarchy method trains on the
subclass axioms only. The forms a geometry reads are in ``geometry.supported_forms``.

.. testcode::

   from mowl.representation import ConstructorUnavailable
   from mowl.representation.library.logic_expressions import expression_region, id_maps
   from mowl.owlapi import OWLAPIAdapter

   adapter = OWLAPIAdapter()
   cls = lambda name: adapter.create_class("http://" + name)
   both = adapter.create_object_intersection_of(cls("Female"), cls("Parent"))
   try:
       expression_region(model, both, *id_maps(model))   # `model` has balls as classes
   except ConstructorUnavailable as error:
       print(error)

.. testoutput::

   The intersection of two Ball regions is not a Ball. A class expression with an intersection cannot be evaluated by ComposedGeometry.

Regularisation is a part too
----------------------------

How the embeddings are kept in a range (a penalty on the norm, a projection onto the unit
sphere, the unit cube of box lattice embeddings) is a component with its own page in the
training section, :class:`~mowl.training.Regularization`. It works with composed models:

.. testcode::

   from mowl.training import Regularization, NormPenalty

   model = ComposedEL(dataset, classes="box", relations="translation", embed_dim=16)
   model.training_regime = Regularization(NormPenalty(parameters="class_embed"))
   model.train(epochs=3)

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...

A constraint that the geometry needs to be defined (the unit cube of box lattice
embeddings) is applied by the geometry itself after every step, and no regime can switch it
off.
