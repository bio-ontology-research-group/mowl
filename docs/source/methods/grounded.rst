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
     - own loop: temperature annealing, element and region sampling, optional
       corrupted-fact negatives, optional Lagrangian dual weights, idle-component
       re-seeding. These are not yet available as training regimes for other methods.

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

Guarantees and limitations are listed on the card ``grounded`` of the
:doc:`method catalogue <catalogue>`; the least-model solver has the card ``least_model``.
