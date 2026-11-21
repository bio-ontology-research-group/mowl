Region-based knowledge graph embeddings (BoxE, ExpressivE, RESHUFFLE, Concept2Box)
==================================================================================

These four methods were published for knowledge graphs, not for description logic
ontologies: BoxE, ExpressivE and RESHUFFLE embed the *triples* of a graph, Concept2Box
embeds a two-view graph of concepts and entities. mOWL implements their geometry and
scoring as published and connects them to an ontology in the most natural supported way.
What that connection is, and what it is not, is stated here and on the cards ``boxe``,
``expressive``, ``reshuffle`` and ``concept2box`` of the :doc:`method catalogue
<catalogue>`. Nothing below claims expressivity beyond it.

.. list-table::
   :header-rows: 1
   :widths: 14 30 28 28

   * - Method
     - Geometry
     - Triples / axioms used
     - Scored axiom forms
   * - :class:`~mowl.models.boxe.model.BoxE`
     - relation: one box per argument position; entity: point and bump
       (PyKEEN's ``BoxE``)
     - edges of a projection
     - ``gci0``, ``gci2``
   * - :class:`~mowl.models.expressive.model.ExpressivE`
     - relation: hyper-parallelogram in the virtual triple space
     - edges of a projection
     - ``gci0``, ``gci2``
   * - :class:`~mowl.models.reshuffle.model.Reshuffle`
     - entity: random binary matrix, relation: soft coordinate permutation, GNN
     - edges of a projection
     - ``gci0``, ``gci2``
   * - :class:`~mowl.models.concept2box.model.Concept2Box`
     - concept: Gumbel box, individual: point
     - ``gci0``, ``gci2``, class and role assertions
     - ``gci0``, ``gci2``, ``class_assertion``, ``object_property_assertion``

BoxE, ExpressivE and RESHUFFLE on a projection
----------------------------------------------

The three models share one path. A projector of :mod:`mowl.projection` turns the ontology
into triples (the default is :class:`~mowl.projection.OWL2VecStarProjector`: ``C subclassof D``,
``C r D`` for :math:`C \sqsubseteq \exists r.D`, ``a type C`` for :math:`C(a)`, and ``a r b``
for :math:`r(a, b)`), a PyKEEN model is trained on them by
:class:`~mowl.models.GraphPlusPyKEENModel` and an axiom :math:`C \sqsubseteq D` is scored by
the score of the edge ``C subclassof D``. The class that holds the shared choices is
:class:`~mowl.models.graph_kge.region_kge.RegionKGEModel`; any other projector can be passed
with ``projector=``.

* **BoxE** is PyKEEN's implementation (there is no code of ours). The paper's unary
  relations (types as boxes) become binary ``type`` edges.
* **ExpressivE** is written as a PyKEEN model (:mod:`mowl.nn.graph.expressive`) after the
  authors' code: ``tanh`` on entities, centres and widths, the minimal denominator
  ``min_denom`` and the distance of Equation 3. Only the base model is provided. The
  pattern theorems (symmetry, hierarchy, mutual exclusion) are checked on hand-built
  parallelograms in ``tests/nn/test_expressive_module.py``.
* **RESHUFFLE** is written after the authors' code (:mod:`mowl.nn.graph.reshuffle`); the
  paper text itself was not used. Differences: the random features are drawn once, the graph
  is the projection (transductive), and
  :meth:`~mowl.nn.graph.reshuffle.Reshuffle.score_on_graph` encodes another graph with fresh
  features for the inductive setting.

In all three the guarantees of the papers (rule patterns, expressiveness, bounded inference)
refer to the *projected triples*. They are not statements about the description logic
semantics, and axioms the projector does not read (conjunctions, role chains, ...) are
ignored.

.. testcode::

    import os, tempfile
    import torch as th
    from mowl.datasets import PathDataset
    from mowl.models import BoxE, ExpressivE, Reshuffle

    ns = "http://example.org/kg#"
    axioms = [
        "SubClassOf(<{ns}A> <{ns}B>)", "SubClassOf(<{ns}B> <{ns}C>)",
        "SubClassOf(<{ns}A> ObjectSomeValuesFrom(<{ns}r> <{ns}D>))",
        "ClassAssertion(<{ns}A> <{ns}a1>)", "ClassAssertion(<{ns}D> <{ns}d1>)",
        "ObjectPropertyAssertion(<{ns}r> <{ns}a1> <{ns}d1>)",
    ]
    path = os.path.join(tempfile.mkdtemp(), "onto.owl")
    with open(path, "w") as f:
        f.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\nOntology(<http://example.org/kg>\n")
        f.write("\n".join(a.format(ns=ns) for a in axioms) + "\n)\n")
    dataset = PathDataset(path)

    th.manual_seed(0)
    model = ExpressivE(dataset, embed_dim=8, batch_size=16)   # or BoxE(...), Reshuffle(...)
    model.train(epochs=2)
    classes = model.class_index_dict
    rows = th.tensor([[classes[ns + "A"], classes[ns + "B"]]])
    print(model.axiom_costs("gci0", rows).shape)

.. testoutput::
    :options: +ELLIPSIS

    torch.Size([1])

Concept2Box on an ontology
--------------------------

Concept2Box embeds two graphs and the links between them: concepts as Gumbel boxes and
entities as points. An ontology gives the three parts directly.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Part of the paper
     - In the ontology
     - Loss
   * - ontology view (concept graph)
     - :math:`C \sqsubseteq D` (``gci0``), :math:`C \sqsubseteq \exists r.D` (``gci2``)
     - :math:`|\phi - 1|^2` for a true, :math:`|\phi|^2` for a corrupted triple, with the
       conditional probability :math:`\phi` of Equation 4
   * - instance view (entity graph)
     - role assertions :math:`r(a, b)`
     - translational distance, hinge on corruptions
   * - cross-view links
     - class assertions :math:`C(a)`
     - vector-to-box distance (Equations 8 and 9), margin on corruptions

:class:`~mowl.models.concept2box.model.Concept2Box` uses the generic EL training loop and
negative sampling of :class:`~mowl.base_models.EmbeddingELModel`. Conjunctions and
existentials on the left, bottom and role axioms are not used, and the model says so in a
warning. Adaptations of the paper: the BERT encoder of the texts of concept and relation is
replaced by learned embeddings joined by the same kind of MLP (no text); the pairwise hinge
of the instance and cross views is split into a positive and a negative part so that the
shared loop applies; the Gumbel box formulas were written from memory of Dasgupta et al.
(2020), and no reference code was consulted.

.. testcode::

    from mowl.models import Concept2Box

    th.manual_seed(0)
    model = Concept2Box(dataset, embed_dim=8, batch_size=16)
    model.train(epochs=2)
    print(sorted(model.scored_forms))

.. testoutput::
    :options: +ELLIPSIS

    ...
    ['class_assertion', 'gci0', 'gci2', 'object_property_assertion']
