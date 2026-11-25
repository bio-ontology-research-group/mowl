Binary Code Box embeddings (GBC-Box, VBC-Box)
=============================================

:class:`~mowl.models.bcbox.model.BinaryCodeBox` reimplements the binary code box embeddings of
Zhang, Boratko, Musco and McCallum, *Modeling Transitivity and Cyclicity in Directed Graphs via
Binary Code Box Embeddings* (NeurIPS 2022). The catalogue key is ``bcbox``. The method embeds a
**directed graph**, not an ontology: in mOWL that graph is the atomic subsumption graph or any
other graph produced by a :mod:`mowl.projection` projector.

Geometry
--------

Every vertex :math:`u` is a box :math:`\prod_{i=1}^d [x^-_i(u), x^+_i(u)]`. The energy of the
directed edge :math:`(u, v)` is :math:`E(u, v) = -\log P((u, v) \in E)`, a sum over dimensions of
:math:`-\log F_i` with the Gumbel-box score of Eq. (3)

.. math::

   F_i = \frac{\mathrm{LSE}_\nu\big(-\mathrm{LSE}_\tau(-x^+_i, -y^+_i)
         - \mathrm{LSE}_\tau(x^-_i, y^-_i), 0\big)}{\mathrm{LSE}_\nu(y^+_i - y^-_i, 0)},

where :math:`x` is the box of :math:`u`, :math:`y` the box of :math:`v`,
:math:`\mathrm{LSE}_t(a, b) = t \log(e^{a/t} + e^{b/t})` and :math:`\mathrm{LSE}_0 = \max`. The
energy is low when the box of :math:`v` lies inside the box of :math:`u`. An edge is predicted
when :math:`E(u, v) < T` for a global threshold :math:`T`. Three variants (``variant=``):

* ``"gbox"`` (G-Box): :math:`E = \sum_i -\log F_i`, Eq. (2). The paper proves (Theorem 3,
  Corollary 4) that no threshold and parameters reconstruct a graph with a chordless directed
  cycle on more than two vertices, because :math:`E(u,v) - E(v,u) = g(u) - g(v)` with
  :math:`g(u) = -\log \mathrm{vol}(u)`.
* ``"gbc"`` (GBC-Box, global codes, Eq. 8): :math:`k` code vectors
  :math:`B_{\cdot, j} \in [0, 1]^d` and :math:`E = \min_j \sum_i B_{i,j} (-\log F_i)`.
* ``"vbc"`` (VBC-Box, per-vertex codes, Eq. 9): every vertex has a code
  :math:`u^\diamond \in [0,1]^d` and :math:`E = \sum_i u^\diamond_i v^\diamond_i (-\log F_i)`.

``code_dims`` is :math:`d_{bin}`: only the last ``code_dims`` dimensions carry trainable codes,
the others have code 1 (Section 4.5). With all codes equal to 1 both binary code models are
G-Box. GBC-Box with :math:`k \ge 2` (Theorem 5) and VBC-Box (Theorems 6-8) can model every
directed graph.

Training
--------

As in Section 4.4 of the paper: the binary cross-entropy of Eq. (10),
:math:`\sum_{(u,v)} E(u,v) - \sum_{(u',v')} \log(1 - e^{-E(u',v')})`, over minibatches of positive
edges, with ``num_negs`` (32 in the paper) negatives per positive obtained by replacing the
source or the target with a random vertex, self-adversarial negative weights, and the lasso
:math:`w_r \lVert \mathbf{1} - B \rVert_{\ell_1}` with :math:`w_r` = ``code_reg`` (Section
4.5), which penalises dropping dimensions and so keeps more of the transitivity bias.

The training is not code of the model but a combination of regimes (:mod:`mowl.training`) that
the method names; read it, or replace one piece:

.. testcode::

   from mowl.datasets.builtin import FamilyDataset
   from mowl.models import BinaryCodeBox
   from mowl.training import EdgeCorruption, EpochBatches, SelfAdversarialBCE

   model = BinaryCodeBox(FamilyDataset(), embed_dim=4, num_negs=8, batch_size=4)
   print([r.key for r in model.training_regime.regimes])
   # the same method with the plain sum of Eq. (10) instead of the self-adversarial weights
   parts = [r if not isinstance(r, SelfAdversarialBCE) else SelfAdversarialBCE(False)
            for r in model.training_regime.regimes]
   from mowl.training import ComposedRegime
   model.training_regime = ComposedRegime(*parts)
   print(len(model.train(epochs=2)))

.. testoutput::

   ['epoch-batches', 'edge-corruption', 'known-row-mask', 'self-adversarial-bce']
   2

What the paper leaves open and mOWL chooses:

* codes are the sigmoid of free logits (the paper only says :math:`B \in [0, 1]`);
* the self-adversarial weights are those of RotatE (softmax over the negatives, no gradient),
  which makes the negative term a weighted average instead of the plain sum of Eq. (10);
  ``self_adversarial=False`` gives the plain sum;
* corruptions that are training edges or self-loops are masked out;
* for VBC-Box the lasso sums over the codes of all vertices.

The public repository of the authors contains G-Box and T-Box but not the binary code models.

From ontology to graph
----------------------

The model is a :class:`~mowl.base_models.GraphModel`, not an :class:`~mowl.base_models.EmbeddingELModel`:
the paper has no conjunction, no existential restriction and no roles, so the only part of an
EL normal form it could use is ``gci0``, and the method was designed for arbitrary (possibly
cyclic) graphs. The default projector is :class:`~mowl.projection.TaxonomyProjector` (asserted
:math:`C \sqsubseteq D` between named classes). Any projector can be passed; edge labels are
dropped (``relations=`` keeps only edges with the given labels), and duplicates and self-loops
are removed. mOWL edges point from the subclass to the superclass, the paper's from the larger
box to the smaller one, so a projected edge ``src -> dst`` is trained as the paper's edge
``(dst, src)``: the box of ``src`` should lie inside the box of ``dst``.
:meth:`~mowl.models.bcbox.model.BinaryCodeBox.edge_energy` and
:meth:`~mowl.models.bcbox.model.BinaryCodeBox.score_edges` take mOWL direction, and
``model.evaluation_model`` can be passed to :class:`~mowl.evaluation.SubsumptionEvaluator`.

A caveat for subsumption graphs: entailment is transitive, so a cycle of subsumptions makes all
its classes equivalent and the deductive closure contains every reverse edge. The closure of a
subsumption graph therefore has no chordless cycles, and the extra capacity of binary codes is
not needed to represent it; codes pay off for projected relations that are not transitive.

Relation to unions of boxes and to EL++
---------------------------------------

The paper describes GBC-Box as "a union over intersections". This union and these
intersections are of **edge sets**, not of regions: with 0/1 codes, :math:`S_j` the dimensions
selected by code :math:`j` and :math:`E_i` the interval-containment graph of dimension
:math:`i`, the modelled edge set is :math:`\bigcup_j \bigcap_{i \in S_j} E_i` (Eq. 6). In terms
of regions, :math:`(u, v)` is an edge iff *there is a code word* :math:`j` such that the
projection of the box of :math:`v` onto :math:`S_j` lies in the projection of the box of
:math:`u` (equivalently, the cylinder over the first projection lies in the cylinder over the
second).

* **GBC-Box with** :math:`k` **codes is not "a union of** :math:`k` **boxes".** In
  :class:`~mowl.models.MultiBoxEL` (and other multibox methods) a concept is the set union of its
  :math:`k` boxes and :math:`C \sqsubseteq D` means
  :math:`\bigcup_j b_j(C) \subseteq \bigcup_l b_l(D)`. Both have :math:`k` boxes per vertex when
  the code supports are disjoint, but the relations differ: GBC asks for one index-aligned
  containment (:math:`\exists j`), the union semantics for all components of :math:`C` to be
  covered (:math:`\forall j`, :math:`\exists l` in the sufficient form). Inclusion of unions is
  reflexive and transitive, so a union-of-boxes model cannot represent a chordless cycle any more
  than a single box can; GBC-Box can, precisely because its relation is a union of :math:`k`
  preorders and is not transitive.
* **Per-vertex codes** compare two vertices only in the dimensions both select
  (:math:`D_u \cap D_v`). Reading a vertex as the cylinder over its selected dimensions does not
  make the edge relation cylinder inclusion (that would also require :math:`D_u \subseteq D_v`);
  this is again not a set inclusion and not transitive, which is what allows any directed graph
  (Theorem 6) at the price of transitivity bias (paper, Section 4.5 and Table 2).
* **Conjunction and disjointness** (``gci1``, ``gci1_bot``) would need a region for
  :math:`C \sqcap D` and a notion of emptiness. Neither exists in the paper, since a vertex is
  not a set under the binary-code relation. Reading each vertex as the union of its code-word
  cylinders restores a set semantics, but then subsumption becomes inclusion of unions: this is
  MultiBoxEL with unbounded components, and it loses the cycle capacity that the codes were
  introduced for. Any other reading (for example, intersecting the boxes of :math:`C` and
  :math:`D` per code word and testing "some code word") would be a new method, not something
  that follows from the paper. mOWL therefore implements only the directed-graph model.
* For the Helly-number argument: the single-box obstruction to
  :math:`T_3 = \{A \equiv X_1 \sqcap X_2, B \equiv X_2 \sqcap X_3, C \equiv X_1 \sqcap X_3, A, B, C`
  pairwise disjoint :math:`\}` is about intersections and emptiness of concept regions. Binary
  codes change which pairs of vertices count as edges, not the intersection behaviour of
  regions, so they do not address it; unions of boxes (MultiBoxEL) do.

Built from parts
----------------

The model is a combination of parts, defined once in
:func:`~mowl.representation.library.hierarchy_methods.bcbox`: boxes with codes as the class
model (:class:`~mowl.representation.library.hierarchy_classes.CodedBoxes`), the smoothed
volume ratio that reads the codes as the measure
(:class:`~mowl.representation.library.hierarchy_measures.LogVolumeRatio`), the energy
:math:`E` as the cost of an edge and :math:`-\log(1 - e^{-E})` as that of a corruption,
the self-adversarial binary cross entropy of Eq. (10)
(:func:`~mowl.representation.library.hierarchy_objectives.bce_edge_loss`) and the code lasso as
the regulariser (``code_reg``). The earlier hand-written module is kept for comparison in
:mod:`mowl.baselines`.

.. testcode::

    from mowl.datasets.builtin import FamilyDataset
    from mowl.models.bcbox.model import BinaryCodeBox

    model = BinaryCodeBox(FamilyDataset(), embed_dim=4, variant="gbc", code_reg=0.1)
    print(model.module.parts["classes"], "/", model.module.parts["measure"])
    print(len(model.module.regularizers))

.. testoutput::
    :options: +ELLIPSIS

    ...soft box / smoothed volume ratio
    1

Example
-------

.. testcode::

    import torch as th
    from mowl.datasets.builtin import FamilyDataset
    from mowl.models.bcbox.model import BinaryCodeBox

    th.manual_seed(0)
    dataset = FamilyDataset()
    model = BinaryCodeBox(dataset, embed_dim=8, variant="vbc", num_negs=8,
                          learning_rate=0.05, code_reg=1e-4)
    losses = model.train(epochs=50)
    print(losses[-1] < losses[0])
    p_down = model.score_edges(["http://Father"], ["http://Parent"]).item()
    p_up = model.score_edges(["http://Parent"], ["http://Father"]).item()
    print(p_down > p_up)

.. testoutput::

    True
    True

Guarantees and limitations are listed on the card ``bcbox`` of the :doc:`method catalogue
<catalogue>`.

Parts
-----

The class model :class:`~mowl.representation.library.hierarchy_classes.CodedBoxes` gives the
trainable codes (``trainable_codes``) and the vector of a class (``vectors``: the two corners
and, for ``vbc``, the code), and the measure
:class:`~mowl.representation.library.hierarchy_measures.LogVolumeRatio` turns an energy into
the probability of an edge (``probability``); the model classes only call them.
