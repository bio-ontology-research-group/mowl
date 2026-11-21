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
