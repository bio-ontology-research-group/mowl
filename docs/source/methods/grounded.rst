GroundedEL
==========

:class:`mowl.models.GroundedEL` is an original method of mOWL. Concepts are unions of
:math:`K` axis-parallel boxes, as in :class:`~mowl.models.MultiBoxEL`, and the domain is a
finite set of *elements*: one per class of the normalised ontology and one per individual.
Each element is a point with a "dead" score, which is 1 for elements of unsatisfiable
classes. Axioms are trained in two ways at once:

* on the elements, with Łukasiewicz hinges on their memberships in the classes and roles,
  so that the elements form an interpretation of the ontology; and
* on points sampled from the regions, so that the boxes satisfy the axioms as regions.

Training uses no reasoner. Initialised from the labels of a least model
(``init="elements"``, see :func:`mowl.geometric.least_model.least_model`), the model
satisfies the ontology exactly before any training step.

.. list-table::
   :widths: 25 75

   * - Expressivity
     - ELH(∘)⊥ with ABox; inverse roles with ``role_mode="translation"``
   * - Inference tasks
     - subsumption, satisfiability, disjointness, instance checking, role assertions,
       logical query answering
   * - Roles
     - unions of product boxes (head × tail), or translation boxes
   * - Training
     - the regime ``grounded-el-training``: a composition of shared regimes (see below)
   * - Testing
     - ``crisp_report`` / ``evaluate_model``; the regime ``query-answering``

.. testcode::

   import os
   import tempfile

   from org.semanticweb.owlapi.model import IRI
   from mowl.datasets import PathDataset
   from mowl.models import GroundedEL
   from mowl.owlapi import OWLAPIAdapter

   adapter = OWLAPIAdapter()
   ns = "http://mowl.borg/docs/grounded#"
   ontology = adapter.create_ontology("http://mowl.borg/docs/grounded")
   a, b, c, d = (adapter.create_class(ns + x) for x in "ABCD")
   r = adapter.create_object_property(ns + "r")
   for axiom in (adapter.create_subclass_of(a, b),
                 adapter.create_subclass_of(b, adapter.create_object_some_values_from(r, c)),
                 adapter.create_subclass_of(adapter.create_object_intersection_of(c, d),
                                            adapter.data_factory.getOWLNothing())):
       adapter.owl_manager.addAxiom(ontology, axiom)
   path = os.path.join(tempfile.mkdtemp(prefix="mowl_grounded_"), "ontology.owl")
   adapter.owl_manager.saveOntology(ontology, IRI.create("file:" + path))

   model = GroundedEL(PathDataset(path), embed_dim=2, n_components=2, epochs=60, seed=0,
                      log_every=0)
   history = model.train()
   print(history[-1]["total"] < history[0]["total"])
   report = model.crisp_report()
   print(sorted(report))
   print(model.card.key, "|", model.expressivity)

.. testoutput::

   True
   ['elem_viol', 'n_axioms_violated_by_depth', 'n_dead_elements', 'n_empty_classes', 'region_viol', 'worst_depth']
   grounded | ELH(∘)⊥ with ABox

``crisp_report`` gives the fraction of element and of region violations of the trained
model without a reasoner; ``evaluate_model(use_reasoner=True)`` adds the comparison with
the canonical model (existence of witnesses, dead elements, faithfulness on named classes,
and whether the model is exact).

Made of parts
-------------

GroundedEL is a combination of the parts of :mod:`mowl.representation`
(:mod:`~mowl.representation.library.grounded_parts`); each part is an object with its own
tables, combined by :class:`mowl.nn.GroundedELModule`:

* the **element domain** (``ElementDomain``), the part the others share: the positions of the
  elements, their deadness and the temperature of the soft memberships;
* the **class model** (``BoxUnionClasses``): a union of boxes per class;
* the **relation model**: a union of products of boxes (``ProductBoxRelations``,
  ``role_mode="product"``) or of boxes of translations (``TranslationBoxRelations``,
  ``role_mode="translation"``);
* the **measure** (``UnionDepth``): the signed depth of a point in a union of boxes, its soft
  reading :math:`\sigma(\text{depth}/\tau)` and its crisp reading;
* the **individual model**: the point of the individual's element.

The **objective** is :class:`~mowl.representation.library.grounded_objective.GroundedObjective`:
for every axiom form an element term, a rule in a fuzzy logic at sampled elements
(:mod:`mowl.nn.el.grounded.constraints`,
:class:`~mowl.representation.library.grounded_logic.LukasiewiczRules`: the premises, the life
of the element and the conclusion), and a region term, a depth hinge at points sampled in the
regions of the left-hand side
(:class:`~mowl.representation.library.grounded_logic.DepthHinges`), plus the minimality
regulariser. Which elements and which points a step reads are two parts of their own,
:class:`~mowl.representation.library.grounded_sampling.ElementSampler` and
:class:`~mowl.representation.library.grounded_sampling.RegionSampler`. The training regime is
the composition of the section below, the evaluation regime of ``crisp_report``
:class:`~mowl.representation.library.grounded_evaluation.CrispViolations`. Each is
swappable: the logic by ``logic=`` (``"godel"``, ``"product"``), the class model, relation
model and measure by ``class_model=``, ``relation_model=``, ``measure=`` (or ``role_mode``,
``agg``), the objective, the samplers and the evaluation regime by ``objective=``,
``element_sampler=``, ``region_sampler=``, ``crisp_regime=`` (or their options such as
``n_region_samples``, ``neg_weight``), the training regime by a
:class:`~mowl.training.ComposedRegime`. The query algebra (:mod:`mowl.nn.el.grounded.queries`)
reads the geometry through the methods of the parts.

.. testcode::

   from mowl.models import GroundedEL
   from mowl.representation.library.grounded_sampling import RegionSampler

   swapped = GroundedEL.from_tables(
       {"A": 0, "B": 1}, {"r": 0}, gci0=[[0, 1]], gci2=[[0, 0, 1]], embed_dim=2,
       relation_model="box of translations", region_sampler=RegionSampler(n_samples=1),
       log_every=0)
   print(type(swapped.module.relation_model).__name__, swapped.role_mode)

.. testoutput::

   TranslationBoxRelations translation

.. testcode::

   from mowl.representation import METHODS

   module = METHODS["grounded"](4, 2, 6, 4, role_mode="translation")
   print(module.parts["classes"], "|", module.parts["relations"], "|", module.parts["shared"])
   print(type(module.class_model).__name__, type(module.relation_model).__name__)

.. testoutput::

   union of boxes | box of translations | element domain
   BoxUnionClasses TranslationBoxRelations

The element constraints are linear in the degrees (Łukasiewicz), so over a table of
memberships they are a linear program
(:class:`~mowl.representation.library.grounded_program.GroundedProgram`) that a solver fits
on small problems; with deadness derived by the bottom rules (the Horn reading) its solution
is the least model. The boxes themselves are not convex in their parameters; the solved table
is loaded into the geometry by the exact-limit construction
(:func:`~mowl.representation.library.grounded_program.load_into`).

.. testcode::

   import numpy as np
   from mowl.representation.library.grounded_program import GroundedProgram, crisp_tables
   from mowl.representation.library.solved_grounding import AxiomTable, Domain
   from mowl.representation.library.solved_solver import ScipySolver

   # A ⊑ B, B ⊑ ∃r.C over the classes A, B, C (elements 0, 1, 2)
   table = AxiomTable(Domain(3, 0, 1), {"gci0": np.array([[0, 1]]),
                                        "gci2": np.array([[1, 0, 2]])})
   program = GroundedProgram(table)
   result = ScipySolver().solve(program, {"eps": 1.0, "eps_r": 1.0, "eps_dead": 1.0})
   labels, edges, dead = crisp_tables(program, result.values)
   print(labels.astype(int).tolist(), edges.tolist())

.. testoutput::

   [[1, 1, 0], [0, 1, 0], [0, 0, 1]] [[0, 2, 0], [1, 2, 0]]

The model class only wires the parts; what it computes besides the objective is done by
general parts that other geometries of elements and unions of boxes can use:

* the exact-limit initialisation (``init="elements"``) is
  :class:`~mowl.representation.library.finite_model_placement.FiniteModelPlacement`, which
  realises labels and edges of a finite interpretation as separated points and boxes; the
  labels and the syntactic edges are read by
  :func:`~mowl.representation.library.finite_model_placement.interpretation_labels`,
  :func:`~mowl.representation.library.finite_model_placement.close_labels` and
  :func:`~mowl.representation.library.finite_model_placement.syntactic_edges`;
* idle-component re-seeding is
  :class:`~mowl.representation.library.union_reseeding.OwnPointReseeding` (with the exact
  counts of :func:`~mowl.representation.library.union_reseeding.slot_point_counts`);
* the inductive scores and loss of ``compute_data_loss`` are
  :class:`~mowl.representation.library.inductive_membership.SoftClassScores` and
  :class:`~mowl.representation.library.inductive_membership.MembershipCrossEntropy`;
* ``evaluate_model`` is the evaluation regime
  :class:`~mowl.representation.library.grounded_evaluation.ExactDomainCheck` over the crisp
  report (``exact_regime=`` replaces it);
* ``train`` is the loop :class:`~mowl.training.sampled_objective.SampledObjectiveTraining`
  of any model whose step is one sampled objective.

.. testcode::

   import torch as th
   from mowl.representation.library import grounded_parts as gp
   from mowl.representation.library.finite_model_placement import FiniteModelPlacement

   measure = gp.UnionDepth()
   domain = gp.ElementDomain(3, 3, 2)
   classes = gp.BoxUnionClasses(3, 3, 2, measure, domain)
   roles = gp.TranslationBoxRelations(1, 3, 2, measure, domain)
   labels = th.eye(3, dtype=th.bool)
   labels[0, 1] = True                       # the element of A is also in B
   FiniteModelPlacement(delta=0.1).place(domain, classes, roles, labels, edges=[(0, 2, 0)])
   points = domain.positions()
   inside = th.stack([classes.depth(points, th.full((3,), c), crisp=True) > 0
                      for c in range(3)], 1)
   print(inside.int().tolist())

.. testoutput::

   [[1, 1, 0], [0, 1, 0], [0, 0, 1]]

Logical query answering
-----------------------

:mod:`mowl.nn.el.grounded.queries` answers the query types of the query embedding
literature (``1p``, ``2p``, ``3p``, ``2i``, ``3i``, ``2u``, ``ip``, ``pi``, ``up``) over a
trained model in translation mode. A query is compiled to a union of boxes: a constant is a
point, a role step is a Minkowski sum with the role's boxes, a conjunction is the pairwise
intersection of boxes and a disjunction their union. Entities and concepts are scored by
their depth in the query region; alternatively a beam search scores a query over witness
elements with max-min combination. In the terms of :doc:`../logic/index`, this is the
inference task *logical query answering*: the query is a class expression, and the TBox
constrains the answers and provides classes that queries can use.

Training as shared regimes
--------------------------

The loss of GroundedEL (memberships of elements, sampled points of regions, minimality)
belongs to the method. How the model is trained does not: each step of the training loop
is a :class:`~mowl.training.TrainingRegime` that other models can use too.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Regime
     - What it does
     - Also usable with
   * - :class:`~mowl.training.TemperatureAnnealing`
     - anneals the temperature of soft memberships
     - any module with ``set_tau`` or ``tau`` (MultiBoxEL)
   * - :class:`~mowl.training.ConstantThenCosine`, :class:`~mowl.training.LinearDecay`
     - learning-rate decay (``lr_final``)
     - every EL model
   * - :class:`~mowl.training.IdleComponentReseeding`
     - relocates components that cover nothing
     - models with ``reseed_components`` (MultiBoxEL)
   * - :class:`~mowl.training.LagrangianDualWeights`
     - weights loss terms by dual variables
     - models with ``dual_weights`` and per-term losses

Without a regime assigned, ``train`` uses
:meth:`GroundedELTraining.for_model <mowl.training.GroundedELTraining.for_model>`, the
composition given by the model's hyperparameters. To change the design, assign a regime.
The example draws the corrupted facts with a shared negative sampler, removes those that
are stated facts (which the model's own sampler does not do) and keeps the rest of the
default design:

.. testcode::

   import torch as th
   from mowl.training import (ComposedRegime, FixedColumnCorruption, GroundedELTraining,
                              KnownPositiveFilter)

   facts = th.tensor([[0, 0, 1], [1, 0, 2], [2, 1, 3], [0, 0, 3]])
   kb = GroundedEL.from_tables(
       {"A": 0, "B": 1}, {"r": 0, "s": 1}, gci0=th.tensor([[0, 1]]),
       object_property_assertion=facts,
       individual_index_dict={f"a{i}": i for i in range(6)},
       role_mode="translation", embed_dim=4, epochs=200, seed=0, neg_weight=0.1,
       learning_rate=0.05, log_every=0)
   kb.training_regime = ComposedRegime(
       FixedColumnCorruption(columns={"object_property_assertion": 2},
                             require_negative_loss=False, seed=0),
       KnownPositiveFilter(seed=0),
       GroundedELTraining.for_model(kb))
   history = kb.train()
   print(kb.training_regime.key)

.. testoutput::

   fixed-column-corruption+known-positive-filtering+grounded-el-training

Query answering as an evaluation regime
---------------------------------------

:class:`mowl.evaluation.LogicalQueryAnswering` is the testing protocol of the query
embedding literature as a regime: the answers of a query that are found only on a
*complete* set of facts are ranked among all individuals, after removing the query's other
answers. The complete facts are given by the caller: held-out facts (prediction) or the
facts entailed by the ontology (inference with the TBox as constraint). Any model with
``score_query(structure, queries)`` can be tested; GroundedEL provides it.

.. testcode::

   from mowl.evaluation import LogicalQueryAnswering, sample_queries

   held_out = [[3, 1, 4], [2, 1, 5]]
   queries = sample_queries(facts.numpy(), held_out, structures=("1p", "2p"),
                            num_queries=5, seed=0)
   result = LogicalQueryAnswering(queries).evaluate(kb)
   print(sorted(result))
   print(sorted(result["average"]))

.. testoutput::

   ['1p', '2p', 'average']
   ['hits@1', 'hits@10', 'hits@3', 'mrr']

Guarantees and limitations are listed on the card ``grounded`` of the
:doc:`method catalogue <catalogue>`; the least-model solver has the card ``least_model``.
