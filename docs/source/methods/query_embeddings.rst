Query embeddings: Query2Box, BetaE, ConE, AConE
===============================================

Four methods answer logical queries over a knowledge graph by embedding the *query* into a
region, and the answers as points in or near it: Query2Box [ren2020q2b]_ (boxes), BetaE
[ren2020betae]_ (products of Beta distributions), ConE [zhang2021cone]_ (sector-cones)
and AConE [he2024acone]_ (cones with relations as rotations). The catalogue keys are
``query2box``, ``betae``, ``cone_qe`` and ``acone``; their guarantees, with sources, are on
the cards in the :doc:`method catalogue <catalogue>`.

Class expressions are queries
-----------------------------

A query is a tree of four operators over anchor entities, which is what an
:math:`\mathcal{ALC}` class expression is (AConE, Section 4, makes the correspondence
explicit):

.. list-table::
   :header-rows: 1

   * - Class expression
     - Query operator
     - Query2Box
     - BetaE
     - ConE
     - AConE
   * - :math:`\exists r.C`
     - projection
     - translation of the box
     - MLP
     - MLP
     - rotation
   * - :math:`C \sqcap D`
     - intersection
     - attention, min offset
     - attention
     - attention, min aperture
     - as ConE
   * - :math:`C \sqcup D`
     - union
     - disjunctive normal form, best branch (all four)
     -
     -
     -
   * - :math:`\neg C`
     - negation
     - not defined
     - inverts the parameters
     - complement cone
     - complement cone

All four share one translation, :mod:`mowl.nn.query.expression`: a class expression (or a
query in the nested-tuple format of the query datasets) becomes a tree
(:class:`~mowl.nn.query.expression.Node`), unions are moved to the top
(:func:`~mowl.nn.query.expression.to_dnf`), and queries of equal shape are embedded as one
batch. A geometry (:mod:`mowl.nn.query.geometry`) only defines the operators. A model
therefore answers any class expression built from named classes, nominals, ``∃``, ``⊓``,
``⊔`` and (BetaE, ConE, AConE) ``¬`` and ``∀`` (as :math:`\neg\exists r.\neg C`).

From an ontology to a graph (an adaptation)
-------------------------------------------

The methods were published for knowledge graphs. mOWL reads an ontology as one
(:func:`mowl.projection.query_graph.extract_graph`): the entities are the named classes and
the individuals, and

* ``A ⊑ B``, ``A ≡ B`` and ``a : B`` give the relation ``isa``;
* ``A ⊑ ∃r.B`` (conjunctions on the right are split) and ``r(a, b)`` give the relation
  ``r``;
* every edge has an inverse, and ``isa`` is closed under transitivity (``hierarchy_closure``).

Axioms with a complex left-hand side, ``⊥``, universal restrictions, unions, negations and
role axioms are not read (they are counted in ``model.skipped_axioms``). Role edges are not
propagated down the hierarchy. A named class in a query is *everything below it*, so the
answers of ``A`` are its subclasses and instances, and ``A ⊑ B`` is "``A`` is an answer of
``B``"; ``C ⊑ ∃r.D`` is "``C`` is an answer of ``∃r.D``". Answers are entities of the graph
(closed world over them), not consequences of the ontology.

Training samples queries of the structures 1p, 2p, 3p, 2i, 3i, ip, pi (and, with negation,
2in, 3in, inp, pin, pni) from the graph and minimises the negative sampling loss of the
reference codes. This loop is shared by the four methods
(:class:`~mowl.models.composed.query.QueryEmbeddingModel`). Default sizes are small.

The four methods are combinations of parts
------------------------------------------

Each method is a named combination of an entity model, a relation model, an intersection
operator, a negation operator and a distance (the builders
:func:`~mowl.representation.library.query_methods.query2box`,
:func:`~mowl.representation.library.query_methods.betae`,
:func:`~mowl.representation.library.query_methods.cone_qe` and
:func:`~mowl.representation.library.query_methods.acone`, registered in
:data:`mowl.representation.METHODS` as trained on queries), plus the other parts of the
training and evaluation regimes: the objective
(:class:`~mowl.representation.library.query_objective.QueryObjective`), the query sampler
(:mod:`mowl.training.query_sampler`) and the evaluation regime. ``model.module`` is the
geometry built from the parts; the keyword arguments of a model class go to the builder, so
any part can be swapped. Parts of different region families do not fit and are refused with
an error that says which.

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import BetaE, Query2Box
   from mowl.representation.library.query_objective import QueryObjective

   dataset = FamilyDataset()
   model = Query2Box(dataset, embed_dim=8, intersection="exact", distance="l2",
                     objective=QueryObjective(kind="hinge"))
   print(model.parts["method"], model.parts["conjunction"], model.parts["objective"])
   beta = BetaE(dataset, embed_dim=8, intersection="mean", negation=None)
   print(beta.module.supports_negation)
   try:
       Query2Box(dataset, embed_dim=8, negation="complement")
   except TypeError as error:
       print(type(error).__name__)

.. testoutput::

   query2box exact hinge negative sampling
   False
   ConstructorUnavailable

The objective can also be the margin loss of answers that GroundedEL's queries use
(:class:`~mowl.representation.library.query_objective.AnswerMarginObjective`, the part
:class:`~mowl.representation.library.query_objective.AnswerMarginLoss`), and the answers of a
model whose entities are points and whose relations are boxes of translations (Query2Box) can
be *searched* over its entities, as GroundedEL searches its elements:
``score_query(structure, queries, search="beam")`` (the max-min beam over witnesses,
:class:`~mowl.evaluation.query_search.MaxMinBeam`) or ``search="compile"`` (the query
compiled to a union of boxes, :class:`~mowl.evaluation.query_search.RegionCompiler`), through
:class:`~mowl.representation.library.translation_box_domain.TranslationBoxDomain`.

.. testcode::

   from mowl.representation.library.query_objective import AnswerMarginObjective

   model = Query2Box(dataset, embed_dim=8, objective=AnswerMarginObjective(margin=1.0))
   scores = model.score_query(("e", ("r", "r")), [(0, (0, 1)), (1, (1, 0))], search="beam", k=4)
   print(model.parts["objective"], tuple(scores.shape) == (2, model.module.num_entities))

.. testoutput::

   answer margin negative sampling True

The earlier hand-written implementations, one geometry and one model class per paper, are in
:mod:`mowl.baselines` (``mowl.baselines.models.query2box`` and so on). They are the reference
that the combinations are tested against (same losses, same scores, same training run).

Tasks and evaluation
--------------------

* ``gci0``, ``gci2`` and ``class_assertion`` are scored for the rank-based regimes, so the
  models take part in the joint ``common-*`` benchmarks;
* ``score_query(structure, queries)`` lets
  :class:`mowl.evaluation.LogicalQueryAnswering` rank the answers of logical queries (the
  protocol of the three query embedding papers). ``model.triples`` is the training graph,
  with the ids the queries use;
* :meth:`~mowl.models.composed.query.QueryEmbeddingModel.answer` ranks the entities for a
  class expression.

Differences from the papers: the graph (above), small default networks (the papers use
hidden layers of 1600 units and dimension 400 to 800), a single optimiser and learning
rate, no easy/hard answer split in training. ConE: the angle of the intersection is
computed with ``atan2`` (the code clamps ``|x| < 1e-3``). AConE: only the aperture-additive
rotation of the paper is implemented (the variant of the code that trains), not the
aperture-multiplicative one or the multicone semantics of the union, and the extraction of
SROI⁻ axioms from a learned model is not implemented.

Example
-------

.. testcode::

   import mowl
   mowl.init_jvm("1g")
   from java.util import HashSet
   from mowl.datasets import Dataset
   from mowl.models import ConE
   from mowl.owlapi import OWLAPIAdapter

   ad = OWLAPIAdapter()
   ns = "http://example.org/query#"
   onto = ad.create_ontology(ns + "o")
   cls = {n: ad.create_class(ns + n) for n in ["Animal", "Mammal", "Dog", "Cat", "Meat"]}
   eats = ad.create_object_property(ns + "eats")
   axioms = HashSet()
   for sub, sup in [("Mammal", "Animal"), ("Dog", "Mammal"), ("Cat", "Mammal")]:
       axioms.add(ad.create_subclass_of(cls[sub], cls[sup]))
   for sub in ["Dog", "Cat"]:
       axioms.add(ad.create_subclass_of(
           cls[sub], ad.create_object_some_values_from(eats, cls["Meat"])))
   ad.owl_manager.addAxioms(onto, axioms)

   model = ConE(Dataset(onto), embed_dim=8, batch_size=16, negatives=4)
   model.train(epochs=2, steps_per_epoch=2)
   # animals that eat meat: Animal ⊓ ∃eats.Meat (a class expression is a query)
   query = ad.create_object_intersection_of(
       cls["Animal"], ad.create_object_some_values_from(eats, cls["Meat"]))
   top = model.answer(query, top_k=3)
   print(len(top), model.skipped_axioms)

.. testoutput::

   3 0
