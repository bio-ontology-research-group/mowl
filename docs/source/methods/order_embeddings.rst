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

References
----------

* Vendrov, I., Kiros, R., Fidler, S., Urtasun, R. (2016). Order-Embeddings of Images and
  Language. ICLR 2016. https://arxiv.org/abs/1511.06361
* Vilnis, L., Li, X., Murty, S., McCallum, A. (2018). Probabilistic Embedding of Knowledge
  Graphs with Box Lattice Measures. ACL 2018. https://arxiv.org/abs/1805.06627
* Dasgupta, S. S., Boratko, M., Zhang, D., Vilnis, L., Li, X. L., McCallum, A. (2020).
  Improving Local Identifiability in Probabilistic Box Embeddings. NeurIPS 2020.
  https://arxiv.org/abs/2010.04831
