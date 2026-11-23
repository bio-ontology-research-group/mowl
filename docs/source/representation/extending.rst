Add a part or a method
======================

A new part works with every other part, and a new method that is a combination of parts is a
function of a few lines. This page shows how, and what the tests must show before it is
merged. The general checklist for adding a method (catalogue card, model, documentation) is in
:doc:`/methods/adding_a_method`; this page is about the representation.

Register a part
---------------

There is one decorator for each kind of part. It makes the part available by name to
:class:`~mowl.models.ComposedEL` and to :func:`mowl.representation.compose`:

============================================  ==============================================
Decorator                                     What it registers
============================================  ==============================================
``register_class_model(name)``                a class model (a subclass of ``Classes``)
``register_relation_model(name)``             a relation model (a subclass of ``Relations``)
``register_measure(name, default_for=...)``   a measure (a subclass of ``RegionMeasure``)
``register_method(key, trained_on=...)``      a function that builds a published method
``register_adapter(*class_names)``            the way to describe an existing module
============================================  ==============================================

Names are unique. Registering a second part under a taken name is an error, because it
would silently change what the name means for everyone.

A part has a ``part`` attribute with its value in the vocabulary
(:class:`~mowl.representation.ClassModel`, :class:`~mowl.representation.RelationModel`,
:class:`~mowl.representation.Measure`). It is how a geometry reports its parts. If your part
is a new kind of thing, add a value to the vocabulary in ``mowl/representation/vocabulary.py``.

The example is a relation model that is a shift, but twice as long as its vector. It reuses
:class:`~mowl.representation.Translation`, so role axioms work too:

.. testcode::

   import logging
   import torch as th
   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import ComposedEL
   from mowl.representation import Translation, register_relation_model, RELATION_MODELS

   logging.disable(logging.CRITICAL)
   th.manual_seed(0)

   @register_relation_model("double translation")
   class DoubleTranslation(Translation):
       part = "translation"          # the vocabulary value this part implements

       def image(self, ids, region):
           return region.translate(2 * self.vectors(ids))

       def preimage(self, ids, region):
           return region.translate(-2 * self.vectors(ids))

   model = ComposedEL(FamilyDataset(), classes="box", relations="double translation",
                      embed_dim=16, learning_rate=0.01)
   model.train(epochs=30)
   print(model.as_geometry.parts["relations"])

   del RELATION_MODELS["double translation"]    # leave the registry as it was

.. testoutput::
   :hide:
   :options: +ELLIPSIS

   ...

Register a method
-----------------

A published method that is a combination of parts is a function that returns the geometry,
registered under the key of its catalogue card. The function takes the numbers of classes,
relations and individuals, the embedding size and the options of the method. If the method
also has a module of its own, the function takes ``tables`` as well. With ``tables=None`` it
builds new parameters. With ``tables=module`` it shares the parameters of that module, which
is how ``geometry_of(module)`` describes the module without copying it. See ``elbe`` in
``mowl/representation/methods.py`` for a function that does both.

The example registers boxes with shifts, scored the way RegD scores inclusion:

.. testcode::

   from mowl.representation import (ComposedGeometry, BoxClasses, Objective, MEASURES,
                                    METHODS, METHOD_KINDS, register_method)

   @register_method("box_boundary_depth")
   def box_boundary_depth(nb_classes, nb_relations, nb_individuals=0, embed_dim=50,
                          margin=0.1):
       classes = BoxClasses(nb_classes, embed_dim)
       relations = Translation(nb_relations, embed_dim)
       return ComposedGeometry(classes, relations, MEASURES["boundary and depth"](),
                               Objective(margin))

   model = ComposedEL(FamilyDataset(), method="box_boundary_depth", embed_dim=16,
                      margin=0.05)
   print(model.as_geometry.parts["measure"])

   del METHODS["box_boundary_depth"], METHOD_KINDS["box_boundary_depth"]

.. testoutput::

   boundary and depth

A method that learns from queries and not from axioms is registered with
``trained_on="queries"``; ``ComposedEL`` then refuses it with a clear error.

Three more things let a general model build a method that is not made from sizes alone:

* a part *constructed from the ontology* (FaithEL's strongly faithful TBox, the ids of top
  and bottom of a grounded domain) is made by a function registered with
  :func:`~mowl.representation.dataset_inputs.register_dataset_inputs`; ``ComposedEL`` passes
  what it returns to the builder (``embed_dim`` among it is the dimension the construction
  fixes);
* a *joint objective* (:mod:`mowl.representation.joint_objective`) reads every axiom form of
  a step at once, at one draw of elements; a geometry that has one
  (``geometry.joint_objective``) is trained by ``ComposedEL`` with the regime
  :class:`~mowl.training.joint.JointObjectiveSteps`;
* a builder whose geometry is not a :class:`~mowl.representation.ComposedGeometry` names the
  options its parts are (``builder.part_options``, for example ``{"classes": "class_model"}``),
  so that ``classes=``, ``relations=``, ``measure=`` and ``objective=`` reach it.

.. testcode::

   from mowl.representation.library.grounded_parts import UnionDepth

   model = ComposedEL(FamilyDataset(), method="grounded", embed_dim=4, relations="translation",
                      measure=UnionDepth(agg="lse"))
   print(model.module.parts["relations"], model.module.joint_objective.part)

.. testoutput::

   box of translations grounded constraints

Then, in the repository:

#. Add the method's entry to ``mowl/representation/catalog.py`` (its parts, as the catalogue
   states them). A test compares the entry with the parts of the geometry the function builds
   (class model, relation model, measure) and fails if they disagree.
#. Add the card of the method to ``mowl/catalog/entries/``, as for any method.
#. Add the adapter (``register_adapter("MyModule")``) if the method also has a module of its
   own, so that ``geometry_of`` can describe trained models of it.
#. Describe it in :doc:`combinations`; that page is generated, so nothing is to be written.

What the tests must show
------------------------

A part is right when two things hold. Both are tests in ``tests/representation/``:

**Same numbers as the module.** The method rebuilt from parts gives the same losses as
the existing module, for every axiom form the module implements, positive and negative, and
for every option of the module that changes a loss. The test builds the module, takes
``geometry_of(module)`` (which shares its parameters), and compares with ``th.allclose``. If
the method has no module, compare with the formulas of the paper, written out by hand for a
few rows.

**One combination outside the paper.** The part is combined with a part of another method
(a new relation model with balls and with boxes, a new measure with a translation), and a
``ComposedEL`` trains for a few epochs with a loss that goes down. A part that only works in
the one paper it came from is not yet a part.

Quirks of a paper or of its code (a constant added to a loss, a signed radius, an overlap
where an inclusion is meant) are not the default of a part. They are options, and the
function that builds the published method sets them: see :doc:`defaults`. If you find that
the existing module of a method has a defect, fix it in the module, make the correct
behaviour the default, give the old behaviour an option that the published recipes select,
add a test for both, and add a line to the changelog.
