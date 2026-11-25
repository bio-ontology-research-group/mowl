Constructed models
==================

Some methods are not trained. They *construct* a model of the ontology, by completion, by a
solver or in closed form, and read it as regions: the canonical model of an
:math:`\mathcal{EL}^{++}` ontology read as unions of intervals (Lacerda et al. 2024) or as
faces of a simplex (Gutiérrez-Basulto and Schockaert 2018), any finite model read as
axis-aligned cones (Özçep et al. 2023), a box model found by a mixed-integer programme
(Leemhuis and Kutz 2025). In mOWL each of them is a combination of three parts
(:mod:`mowl.representation.construction`):

* a **source** builds the model: the canonical model by completion, the least model of
  indexed tables, the atoms of a Boolean TBox, an ultimately periodic temporal model, the
  grounding of an attributed ontology, or a box model from a solver. The source says which
  logic it covers and which reasoning it needs;
* a **reading** places the elements of the model in space and turns every class into a
  region: element sets, cones, unions of intervals, faces of a simplex, or boxes;
* the **exact measure** decides inclusion, disjointness, emptiness and membership of these
  regions. A cost is 0.0 where the statement holds and 1.0 where it does not.

The parts are put together into a geometry with the same interface as a trained model, so
that the evaluation regimes and the class expressions of :doc:`use` work with it.

Build a model and ask it questions
----------------------------------

A finite model is a domain ``0 .. n - 1`` with the extensions of the concepts and roles.
Read it as cones and ask whether one class lies inside another:

.. testcode::

   import torch as th
   from mowl.representation.construction import FiniteModel, construct

   model = FiniteModel(3, concepts={"A": {0}, "B": {0, 1}, "C": {2}},
                       roles={"r": {(0, 2)}}, individuals={"a": 0})
   g = construct(model, "finite model", "half-axes")
   A, B, C = (g.classes(th.tensor([g.names.index(n)])) for n in "ABC")

   print(g.inside(A, B).item(), g.inside(B, A).item())          # A ⊑ B holds, B ⊑ A not
   print(g.inside(A, g.exists(th.tensor([0]), C)).item())       # A ⊑ ∃r.C holds
   print(g.apart(A, g.negation(A)).item())                      # cones have a negation
   print(g.member(th.tensor([0]), B).item())                    # a is a B

.. testoutput::

   0.0 1.0
   0.0
   0.0
   0.0

``construct(data, source, reading)`` is the one entry point. The names of the sources and
readings are in :data:`~mowl.representation.construction.SOURCES` and
:data:`~mowl.representation.construction.READINGS`. The geometry says what it is made of:

.. testcode::

   print(g.parts["source"], "/", g.parts["reading"], "/", g.parts["classes"], "/",
         g.parts["measure"])

.. testoutput::

   finite model / half-axes / axis-aligned cone / exact

Swap the reading
----------------

Every reading of a finite model satisfies a normal-form axiom exactly when the finite model
does. The readings differ in what else they can express. Element sets, cones and unions of
intervals are closed under complement and union. Faces of a simplex are convex, so they are
not closed under union:

.. testcode::

   from mowl.representation import ConstructorUnavailable

   for reading in ["element sets", "half-axes", "intervals", "simplex"]:
       g = construct(model, "finite model", reading)
       A, B = (g.classes(th.tensor([g.names.index(n)])) for n in "AB")
       try:
           union = g.disjunction(A, B)
           print(reading, g.inside(A, B).item(), "union:", g.inside(B, union).item())
       except ConstructorUnavailable:
           print(reading, g.inside(A, B).item(), "union: not a region of the family")

.. testoutput::

   element sets 0.0 union: 0.0
   half-axes 0.0 union: 0.0
   intervals 0.0 union: 0.0
   simplex 0.0 union: not a region of the family

A combination that cannot work is refused with the reason. A finite model cannot in general
be read as single boxes, because boxes have the Helly property and finite models do not:

.. testcode::

   try:
       construct(model, "finite model", "boxes")
   except ConstructorUnavailable as error:
       print("Helly" in str(error))

.. testoutput::

   True

The named methods
-----------------

The constructed methods of the catalogue are named combinations
(:class:`~mowl.representation.construction.Recipe`), registered as methods that are
``trained_on="construction"``:

=================  ================  ==============
catalogue key      source            reading
=================  ================  ==============
canonical_model    canonical model   element sets
least_model        least model       element sets
interval_union     canonical model   intervals
convex_polyhedral  canonical model   simplex
convex_regions     finite model      simplex
alcone-exact       canonical model   half-axes
temporal_cones     lasso model       half-axes
attributed_dl      attributed        simplex
box_milp           box milp          boxes
=================  ================  ==============

A recipe builds its method from data and can swap a part. Here the Boolean TBox
:math:`A \sqsubseteq B` is read once as cones (Özçep et al. 2023, Prop. 7) and once as
intervals:

.. testcode::

   from mowl.geometric.cones import Atomic, ConceptAssertion, GCI, Or
   from mowl.representation.construction import RECIPES

   tbox = [GCI(Atomic("A"), Atomic("B")),
           ConceptAssertion(Or((Atomic("A"), Atomic("C"))), "a")]
   recipe = RECIPES["alcone-exact"]
   print(recipe.parts["source"], "+", recipe.parts["reading"])
   for reading in ("half-axes", "intervals"):
       g = recipe.build(tbox, source="boolean atoms", reading=reading)
       A, B = (g.classes(th.tensor([g.names.index(n)])) for n in "AB")
       print(reading, g.inside(A, B).item(), g.inside(B, A).item())

.. testoutput::

   canonical model + half-axes
   half-axes 0.0 1.0
   intervals 0.0 1.0

Temporal models have a time line. ``next`` and ``until`` act on it, and an inclusion holds
when it holds at every time point:

.. testcode::

   from mowl.geometric.temporal_cones import TemporalOntology

   onto = TemporalOntology.parse(tbox=["A <= X B", "B <= X A", "A <= ~B"], abox=["A(a)"])
   g = RECIPES["temporal_cones"].build(onto)
   A, B = (g.classes(th.tensor([g.names.index(n)])) for n in "AB")
   print(g.inside(A, g.next(B)).item(), g.inside(A, B).item())

.. testoutput::

   0.0 1.0

For a dataset, :class:`~mowl.models.constructed.ConstructedModel` builds the method with the ids of the
dataset's classes, relations and individuals. Its ``module`` scores axioms like the module of
a trained model, so the evaluation regimes rank with it. ``train`` does nothing: the model
is constructed when it is created.

.. testcode::

   import logging
   from mowl.datasets.builtin import FamilyDataset
   from mowl.models.constructed import ConstructedModel

   logging.disable(logging.CRITICAL)
   dataset = FamilyDataset()
   model = ConstructedModel(dataset, method="interval_union")
   c = model.class_index_dict
   rows = th.tensor([[c["http://Father"], c["http://Male"]],
                     [c["http://Male"], c["http://Father"]]])
   print(model.module(rows, "gci0").flatten().tolist())
   cones = ConstructedModel(dataset, method="interval_union", reading="half-axes")
   print(cones.parts["reading"], cones.module(rows, "gci0").flatten().tolist())

.. testoutput::

   [0.0, 1.0]
   half-axes [0.0, 1.0]

The classes of :mod:`mowl.geometric` that build these models
(:class:`~mowl.geometric.intervals.IntervalUnionModel`,
:class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`,
:class:`~mowl.geometric.convex_regions.ConvexRegionModel`,
:class:`~mowl.geometric.cones.ALConeInterpretation`,
:class:`~mowl.geometric.temporal_cones.TemporalALConeInterpretation`,
:class:`~mowl.geometric.attributed.AttributedConvexModel`) are built from these parts; their
``module`` is the geometry. The direct constructions they replaced are kept in
:mod:`mowl.baselines.geometric`, and ``tests/baselines/pairs_construct.py`` checks that both
give the same regions and the same truth values.

A constructed part of a trained method
--------------------------------------

The canonical mixture (``canonical_mixture``) constructs the canonical model and trains only
its individuals (:mod:`mowl.representation.library.mixture_methods`). Its constructed part
comes from a source and a reading of this interface: the source ``canonical completion``
builds the canonical model as the completion leaves it, and the reading ``membership``
restricts it to the elements of the named classes and the individuals, as sparse sets of
elements, without a dense matrix over the elements. With the probability-mass measure of the
method, the constructed part alone decides subsumptions exactly:

.. testcode::

   from mowl.representation.library.mixture_measures import ProbabilityMass

   c = model.class_index_dict
   names = [n for n, _ in sorted(c.items(), key=lambda kv: kv[1])]
   g = construct(dataset.ontology, "canonical completion", "membership", names=names,
                 roles=[], individuals=[], measure=ProbabilityMass())
   print(g.parts["source"], "/", g.parts["reading"])
   print(g.cost("gci0", rows).flatten().tolist())

.. testoutput::

   canonical completion / membership
   [0.0, 1.0]

Guarantees as data
------------------

Sources and readings carry their guarantees
(:class:`~mowl.representation.construction.Guarantee`): what holds, for which logic and
under which condition. The geometry lists those of its parts:

.. testcode::

   g = construct(model.module.built, "finite model", "intervals")
   for guarantee in g.guarantees[:1]:
       print(guarantee.kind, "-", guarantee.logic)

.. testoutput::

   soundness - EL++ normal forms

Add a source or a reading
-------------------------

A source subclasses :class:`~mowl.representation.construction.Source` (``build(data)``
returns a :class:`~mowl.representation.construction.FiniteModel`, a
:class:`~mowl.representation.construction.TemporalModel` or a model of its own, with
``produces`` saying which). A reading subclasses
:class:`~mowl.representation.construction.Reading`. A reading of finite models is easiest as
a subclass of :class:`~mowl.representation.library.construct_readings.CellReading` with a
region class derived from :class:`~mowl.representation.construction.CellRegion`, which gives
the family's own ``contains`` and says which operations it is closed under. Register both
with :func:`~mowl.representation.construction.register_source` and
:func:`~mowl.representation.construction.register_reading`. A new named method is a
:class:`~mowl.representation.construction.Recipe` passed to
:func:`~mowl.representation.construction.register_recipe`.
