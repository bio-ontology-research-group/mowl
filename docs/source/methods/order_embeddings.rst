Order and box embeddings of the class taxonomy
==============================================

Three methods embed a **partial order** rather than a description logic ontology:

* :class:`OrderEmbeddings <mowl.models.order_embeddings.model.OrderEmbeddings>`
  (catalogue key ``order_embeddings``; Vendrov et al., ICLR 2016,
  `arXiv:1511.06361 <https://arxiv.org/abs/1511.06361>`_): a class is a vector in
  :math:`\mathbb{R}^d_+`, and :math:`C \sqsubseteq D` holds when the vector of :math:`C` is
  coordinatewise at least that of :math:`D` (reversed product order). The violation is
  :math:`E(x, y) = \lVert \max(0, y - x) \rVert_2` and the loss is the max-margin loss of
  Eq. 4.
* :class:`BoxLattice <mowl.models.box_lattice.model.BoxLattice>` (``box_lattice``; Vilnis et
  al., ACL 2018, `arXiv:1805.06627 <https://arxiv.org/abs/1805.06627>`_): a class is a box
  in :math:`[0, 1]^d` (minimum and :math:`\Delta`), volume is probability and
  :math:`C \sqsubseteq D` is the target :math:`P(D \mid C) = \mathrm{vol}(C \cap D) /
  \mathrm{vol}(C) = 1`. Disjoint boxes get a gradient from the lower bound of the joint.
* :class:`GumbelBox <mowl.models.gumbel_box.model.GumbelBox>` (``gumbel_box``; Dasgupta et
  al., NeurIPS 2020, `arXiv:2010.04831 <https://arxiv.org/abs/2010.04831>`_): the box corners
  are Gumbel random variables; intersections stay Gumbel boxes (Lemma 1), the expected
  volume is a softplus (Section 4.2) and every pair has a gradient.

What is supported
-----------------

The connection to an ontology is the **subclass axioms between named classes**
(:math:`C \sqsubseteq D`, normal form ``gci0``). These are the order pairs used in training
and the only axioms that are scored (``scored_forms = ("gci0",)``): the tasks are
subsumption prediction and subsumption inference between named classes, so the models take
part in the joint ``common-*-subsumption-*`` benchmarks. Conjunction, existential
restrictions, bottom, assertions and role axioms have no meaning in these geometries; they
are left out of training (the model reports them as beyond its expressivity, ``taxonomy``),
and so are subclass axioms that involve classes introduced by normalisation. The box meet
is not the conjunction of classes.

All three are **adaptations**: the papers embed WordNet hypernymy, textual entailment,
caption order and MovieLens co-occurrences. The geometry and the losses follow the papers;
the data (stated subclass pairs, with corrupted pairs as negatives, as the order-embedding
paper does for hypernymy) and the full-batch generic training loop are mOWL's. A difference
from the paper (for example the lower-bound surrogate of the box lattice, the signs of
Algorithm 1 of the Gumbel paper) is stated in the docstring of the module and on the
catalogue card. The authors' reference code was not consulted.

Example
-------

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import BoxLattice, GumbelBox, OrderEmbeddings

   for method in (OrderEmbeddings, BoxLattice, GumbelBox):
       model = method(FamilyDataset(), embed_dim=4)
       model.train(epochs=2, validate_every=10)
       print(method.__name__, sorted(model.training_datasets))

.. testoutput::
   :options: +ELLIPSIS

   ...OrderEmbeddings ['gci0']
   ...BoxLattice ['gci0']
   ...GumbelBox ['gci0']

Built from parts
----------------

Each class is the combination of parts that its builder in
:mod:`mowl.representation.library.hierarchy_methods` names: a class model, a measure and an
objective (``order_embeddings``: vectors of the reversed product order, the order violation,
the max-margin hinge; ``box_lattice``: boxes in the unit cube, the volume ratio, the cross
entropy with frozen sizes in negative pairs and the ``project()`` onto the cube;
``gumbel_box``: the same boxes with the smoothed volume ratio), with the regulariser
of the options (``max_regularization``). ``model.module`` is a
:class:`~mowl.representation.ComposedGeometry`, ``model.module.parts`` says what it is made of,
and every part can be replaced by another one, here the measure of the Gumbel boxes on the
boxes of the box lattice:

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import BoxLattice
   from mowl.representation.library.hierarchy_measures import LogVolumeRatio

   model = BoxLattice(FamilyDataset(), embed_dim=4)
   print(model.module.parts["classes"], "/", model.module.parts["measure"])
   model.module.measure = LogVolumeRatio(0.05, 0.05, gumbel_shift=True, tail="linear")
   model.train(epochs=1, validate_every=10)
   print(model.module.parts["measure"])

.. testoutput::
   :options: +ELLIPSIS

   ...box / volume ratio
   smoothed volume ratio

The hand-written modules and models that these combinations replaced are kept in
:mod:`mowl.baselines` as the reference that the combinations are tested against (same
losses, scores and training run, ``tests/baselines``).

References
----------

* Vendrov, I., Kiros, R., Fidler, S., Urtasun, R. (2016). Order-Embeddings of Images and
  Language. ICLR 2016. https://arxiv.org/abs/1511.06361
* Vilnis, L., Li, X., Murty, S., McCallum, A. (2018). Probabilistic Embedding of Knowledge
  Graphs with Box Lattice Measures. ACL 2018. https://arxiv.org/abs/1805.06627
* Dasgupta, S. S., Boratko, M., Zhang, D., Vilnis, L., Li, X. L., McCallum, A. (2020).
  Improving Local Identifiability in Probabilistic Box Embeddings. NeurIPS 2020.
  https://arxiv.org/abs/2010.04831
