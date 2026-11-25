Expressivity, inference tasks and reasoners
===========================================

mOWL takes embedding methods apart into pieces that can be put together in new ways: a
*method* (the representation), a *training regime*, an *evaluation regime* and a
*benchmark*. Publications are recorded as particular combinations of these pieces (the
recipes of :doc:`../benchmarks/index`), but the pieces are not tied to the publication they
come from. Which pieces fit together is decided by three things, which this page describes:

* the **expressivity** a method supports, in description logic notation;
* the **inference tasks** a method can score and an evaluation measures;
* the **reasoner** that provides the logical ground truth.

Expressivity
------------

Every catalogued method states the logic it supports as a :class:`mowl.logic.Expressivity`:
a set of constructs (:class:`mowl.logic.Construct`) with a name in the usual notation.

.. testcode::

   from mowl.logic import Expressivity, Construct

   el = Expressivity.parse("ELH(∘)⊥ with ABox")
   print(el, "|", Construct.ROLE_CHAIN in el, Construct.NOMINAL in el)
   print(Expressivity.parse("EL") <= Expressivity.parse("ALC"))
   print([c.name for c in Expressivity.parse("EL⊥").missing(Expressivity.parse("ALC"))])

.. testoutput::

   ELH(∘)⊥ with ABox | True False
   True
   ['DISJUNCTION', 'NEGATION', 'UNIVERSAL']

The notation: a base (``EL``: ⊤, ⊓, ∃; ``ALC``: all Boolean constructors, ∃ and ∀; ``S``:
``ALC`` with transitive roles; ``SR``: with complex role inclusions) followed by ``H`` (role
hierarchy), ``O`` (nominals), ``I`` (inverse roles), ``F`` / ``N`` / ``Q`` (functionality,
number restrictions, qualified number restrictions), and for the EL family ``(∘)`` (role
chains) and ``⊥``; ``(D)`` stands for datatypes. ``with ABox`` is added when class and role
assertions are supported. Named logics such as ``EL++`` and ``DL-Lite_H`` are in
:data:`mowl.logic.NAMED`.

For an implemented method the stated expressivity is what the implementation *uses* of an
ontology, which can be less than the paper covers. The catalogue can be queried by it:

.. testcode::

   from mowl.catalog import get_method, list_methods

   print(get_method("elem").supported_expressivity)
   print(get_method("box2el").supported_expressivity)
   print(sorted(c.key for c in list_methods(implemented=True, covers="ALC",
                                            task="subsumption")))

.. testoutput::

   EL⊥ with ABox
   ELH(∘)⊥ with ABox
   ['alcone', 'alcone-exact', 'cate', 'composed-alc', 'dfalc', 'falcon', 'graph_kge']

Checking an ontology before embedding it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`mowl.logic.ontology_expressivity` computes what an ontology uses, and
:func:`mowl.logic.check_ontology` reports the axioms that go beyond what a method supports.
Such axioms are ignored, or only partly used, when the ontology is embedded.

.. testcode::

   from mowl.owlapi import OWLAPIAdapter
   from mowl.logic import ontology_expressivity, check_ontology

   adapter = OWLAPIAdapter()
   ns = "http://mowl.borg/docs/logic#"
   ontology = adapter.create_ontology("http://mowl.borg/docs/logic")
   a, b = adapter.create_class(ns + "A"), adapter.create_class(ns + "B")
   r = adapter.create_object_property(ns + "r")
   for axiom in (adapter.create_subclass_of(a, adapter.create_object_some_values_from(r, b)),
                 adapter.create_subclass_of(b, adapter.create_object_all_values_from(r, a))):
       adapter.owl_manager.addAxiom(ontology, axiom)

   print(ontology_expressivity(ontology))
   report = check_ontology(ontology, "EL⊥", warn=False)
   print(report.ok, report.unsupported_axioms, report.by_axiom_type)
   print(report)

.. testoutput::

   ALC
   False 1 {'SubClassOf': 1}
   1 of 2 logical axioms go beyond EL⊥ (the ontology uses ALC) and will be ignored or only partly used: 1 SubClassOf. Constructs outside the supported logic: ∀r.C (1).

Every model runs this check on its training ontology when it is created and emits a
:class:`UserWarning` if axioms are outside its expressivity; the result is kept in
``model.expressivity_report``. ``model.check_expressivity()`` runs it again, and
``mowl.base_models.model.Model.check_expressivity_on_init = False`` switches the automatic
check off. The check uses the OWL API, which mOWL packages.

Inference tasks
---------------

:mod:`mowl.logic.tasks` names what a method is asked to decide, in the vocabulary of
description logic reasoning, and relates it to the tasks of knowledge graph and knowledge
base completion.

.. testcode::

   from mowl.logic import TASKS, COMPLETION_TASKS

   for key in ("subsumption", "satisfiability", "instance-checking", "role-assertion"):
       task = TASKS[key]
       print(f"{task.name} ({task.box.value}): {task.question}  ->  {task.as_subsumption}")
   link = COMPLETION_TASKS["link-prediction"]
   print(link.setting, "|", link.candidates, "|", link.inference_task)

.. testoutput::

   Subsumption checking (TBox): O ⊨ C ⊑ D ?  ->  C ⊑ D
   Concept satisfiability (TBox): O ⊭ C ⊑ ⊥ ?  ->  not (C ⊑ ⊥)
   Instance checking (ABox): O ⊨ C(a) ?  ->  {a} ⊑ C
   Role assertion checking (ABox): O ⊨ r(a, b) ?  ->  {a} ⊑ ∃r.{b}
   knowledge graph completion | r(a, ?) or r(?, b) | role-assertion

.. list-table::
   :header-rows: 1
   :widths: 24 10 30 36

   * - Reasoning task
     - About
     - Question
     - As a subsumption test
   * - Subsumption checking
     - TBox
     - O ⊨ C ⊑ D ?
     - C ⊑ D
   * - Classification
     - TBox
     - all concept names A, B with O ⊨ A ⊑ B
     - A ⊑ B for all pairs
   * - Concept satisfiability
     - TBox
     - can C have an instance?
     - not (C ⊑ ⊥)
   * - Disjointness
     - TBox
     - O ⊨ C ⊓ D ⊑ ⊥ ?
     - C ⊓ D ⊑ ⊥
   * - Consistency
     - KB
     - does O have a model?
     - not (⊤ ⊑ ⊥)
   * - Instance checking
     - ABox
     - O ⊨ C(a) ?
     - {a} ⊑ C
   * - Realization
     - ABox
     - most specific concept names of each individual
     - {a} ⊑ A for all concept names A
   * - Instance retrieval
     - ABox
     - all a with O ⊨ C(a)
     - {a} ⊑ C for all individuals a
   * - Role assertion checking
     - ABox
     - O ⊨ r(a, b) ?
     - {a} ⊑ ∃r.{b}
   * - Logical query answering (one-hop, multi-hop)
     - KB
     - all a with O ⊨ q(a)
     - {a} ⊑ C_q, the query rolled up into a class expression

Almost every task is a subsumption test between two class expressions. The class
expressions can be obtained in two ways: *explicitly*, by adding a definition ``X ≡ C``
before training or reasoning, which works for every method and reasoner; or *in the latent
space*, by composing the region of ``C`` from the learned parts, which needs the method to
have the corresponding operation (boxes can be intersected, balls cannot).

Logical query answering, as studied for knowledge graphs (one-hop queries, multi-hop paths,
intersections, unions, negation), fits the same scheme: a tree-shaped query with one answer
variable is a class expression, and answering it is instance retrieval. The TBox then acts
as constraints on the answers and provides classes that queries can use.

Completion tasks rank candidate axioms. Knowledge graph completion ranks role assertions
only (link prediction); knowledge base completion also ranks axioms with class expressions.
What the correct answers are makes the same task a different problem
(:class:`mowl.logic.Target`): if the test axioms are *entailed* by the training ontology,
the evaluation measures whether the embedding has learned the logic (deductive inference);
if they are held out and *undetermined*, it measures the prediction of plausible new
knowledge. A candidate axiom has exactly one :class:`mowl.logic.LogicalStatus`: entailed,
contradictory or undetermined. The faithfulness criteria of Mashkova and Hoehndorf (2026),
:class:`mowl.logic.FaithfulnessCriterion`, say how a score should order the three.

Method cards list the inference tasks a method gives a score for (``tasks``), and
evaluation regime cards name the task they measure (``task``) and their target.

Open and closed world
---------------------

OWL has open world semantics. An axiom that the ontology does not entail is not false: it
is *undetermined* unless adding it makes the ontology inconsistent
(:class:`~mowl.logic.LogicalStatus`). Training and evaluation of embeddings mostly need
false examples, and take them from what is absent. That is a closed world assumption, and
it is a property of the *regime* (where negatives come from, which candidates count as
wrong, how a negated query atom is answered), not of the embedding method. mOWL records it
as a :class:`~mowl.logic.WorldAssumption`:

.. list-table::
   :header-rows: 1
   :widths: 26 16 22 18

   * - Assumption
     - Contradictory
     - Undetermined
     - Entailed, not stated
   * - ``OPEN`` (open world)
     - false
     - neither
     - true
   * - ``CLOSED_ENTAILED`` (closed world over entailed axioms)
     - false
     - false
     - true
   * - ``CLOSED_STATED`` (closed world over stated axioms)
     - false
     - false
     - false

* **Training.** Uniform corruption (``either-side-corruption`` and its relatives) is
  ``CLOSED_STATED``: a corrupted axiom is a negative even when the ontology entails it.
  ``dele-closure-filtered-negatives`` removes the entailed ones (``CLOSED_ENTAILED``).
  ``contradiction-negatives`` uses provably false axioms (``OPEN``). Methods that train
  without negatives, and the exact constructions, make no closed world assumption.
* **Evaluation.** A ranking in which every candidate outside the known axioms counts as
  wrong is ``CLOSED_STATED`` (the joint ``common-*`` benchmarks, the Box²EL rankings, link
  prediction). ``dele-evaluation`` removes entailed candidates (``CLOSED_ENTAILED``). The
  faithfulness hierarchy, the KB embedding properties and the exact model check compare
  against entailment (``OPEN``).
* **Query answering.** The task :data:`~mowl.logic.TASKS` ``"query-answering"`` asks for
  certain answers: individuals ``a`` with ``O ⊨ q(a)``. The benchmarks of the query
  embedding literature (``query-answering`` regime; Query2Box, BetaE, ConE, AConE) instead
  rank the answers in a hidden complete graph, and a negated atom holds when the edge is
  absent. A query with negation therefore has many answers under the closed world reading
  and almost none under the open world one, and a model trained for the first is not a
  model of the ontology in the second sense. Negation in those methods is complement in
  the graph, not the ``¬`` of the description logic.

The same completion task is a different problem under each assumption, as it is under each
:class:`~mowl.logic.Target`. A regime card states its assumption, a benchmark takes it from
its evaluation regime, and a method card states the assumption of the method's own
objective (how it trains when no training regime is attached).

.. testcode::

   from mowl.benchmarks import get_benchmark, list_benchmarks
   from mowl.catalog import get_method, get_regime
   from mowl.logic import LogicalStatus, WorldAssumption

   print(get_regime("either-side-corruption").assumption)
   print(get_regime("dele-closure-filtered-negatives").assumption)
   print(get_benchmark("common-go-subsumption-prediction").assumption.name)
   print(get_method("query2box").assumption)
   print(WorldAssumption.OPEN.is_false(LogicalStatus.UNDETERMINED),
         WorldAssumption.CLOSED_ENTAILED.is_false(LogicalStatus.UNDETERMINED))
   print(len(list_benchmarks(assumption=WorldAssumption.OPEN)) > 0)

.. testoutput::

   closed world over stated axioms
   closed world over entailed axioms
   CLOSED_STATED
   closed world over stated axioms
   False True
   True

Which training regimes a model takes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A model declares the hooks of :class:`~mowl.training.TrainingRegime` its training loop
consults (``training_hooks``). All models train in the same loop
(:class:`mowl.training.loop.TrainingLoop`), and most consult every hook and take any regime.
A model whose objective is fixed by its method declares fewer: the ALC models take schedule,
stopping, regularisation and after-step regimes but not a loss or negative sampler; the
query embedding models take the optimiser, schedule and stopping hooks; the canonical
mixture takes schedule, stopping and projection regimes. Assigning a regime that changes something the model does not consult
raises ``TypeError``; a regime is never silently ignored.

.. testcode::

   from mowl.models import CanonicalMixture, DiskEmbeddings, Query2Box
   from mowl.training import LinearDecay, PairwiseMarginRankingLoss, unsupported_hooks

   print(sorted(unsupported_hooks(PairwiseMarginRankingLoss(), DiskEmbeddings)))
   print(sorted(unsupported_hooks(PairwiseMarginRankingLoss(), Query2Box)))
   print(sorted(unsupported_hooks(LinearDecay(), Query2Box)))
   print(len(unsupported_hooks(PairwiseMarginRankingLoss(), CanonicalMixture)) > 0)

.. testoutput::

   []
   ['filter_negatives', 'loss']
   []
   True

Reasoners
---------

Reasoners are backends chosen by expressivity (:mod:`mowl.reasoning`). mOWL packages ELK
(polynomial, EL with role hierarchies and chains, ⊥ and assertions), HermiT (complete
for OWL 2 DL) and KM. :func:`mowl.reasoning.create_reasoner` picks the cheapest backend that is
complete for an ontology, and warns when a backend is asked to reason over constructs it
does not cover. Other reasoners are added with :func:`mowl.reasoning.register_reasoner`.

mOWL also packages the Kobayashi-MaRust reasoner (``km``), a classifier for OWL 2 DL with
verified certificates. It is optional: HermiT is the default beyond what ELK covers, and KM
is used when asked for by name (``create_reasoner(ontology, "km")``,
``SubsumptionOracle(ontology, reasoner="km")``). Its OWL API binding classifies; instance
checks are made as the subsumption ``{a} ⊑ C``.
Further JAR files can be put on the class path with
``mowl.init_jvm(memory, extra_classpath=...)`` or ``MOWL_EXTRA_CLASSPATH``.

.. testcode::

   from mowl.reasoning import select_reasoner, list_reasoners

   print([b.name for b in list_reasoners()])
   print(select_reasoner("ELH(∘)⊥ with ABox").name, select_reasoner("ALC").name)

.. testoutput::

   ['structural', 'elk', 'hermit', 'km']
   elk hermit

:class:`mowl.reasoning.SubsumptionOracle` answers the tasks above through subsumption
between class expressions, using definitions that it adds and removes again, and gives the
logical status of a candidate axiom.

.. testcode::

   from mowl.reasoning import SubsumptionOracle

   oracle = SubsumptionOracle(ontology)      # the ALC ontology from above
   some_r_b = adapter.create_object_some_values_from(r, b)
   print(oracle.backend.name)
   print(oracle.entails_subsumption(a, some_r_b), oracle.entails_subsumption(b, a))
   print(oracle.status(a, some_r_b).value, oracle.status(b, a).value)

.. testoutput::

   hermit
   True False
   entailed undetermined
