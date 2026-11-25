"""Binary code box embeddings of projected ontology graphs [zhang2022bcbox]_."""

from functools import partial

import torch as th

from mowl.models.bcbox.common import BinaryCodeBoxBase, best_f1, _SubsumptionScorer  # noqa: F401
from mowl.nn.graph.bcbox.module import BinaryCodeBoxModule
from mowl.training.graph import EdgeCorruption, EpochBatches, KnownRowMask, SelfAdversarialBCE
from mowl.training.loop import RegimeTrained, Rows


class BinaryCodeBox(RegimeTrained, BinaryCodeBoxBase):
    """Binary code box embeddings (GBC-Box / VBC-Box) of a projected ontology graph.

    Reimplementation of Zhang, Boratko, Musco and McCallum, *Modeling Transitivity and
    Cyclicity in Directed Graphs via Binary Code Box Embeddings*, NeurIPS 2022
    [zhang2022bcbox]_. The geometry, energies and regulariser are in
    :class:`mowl.nn.graph.bcbox.BinaryCodeBoxModule`, a combination of parts
    (``model.module.parts``) defined once in
    :func:`mowl.representation.library.hierarchy_methods.bcbox`: boxes with codes as the
    class model, the smoothed volume ratio that reads the codes as the measure, the energy
    and :math:`-\\log(1 - e^{-E})` as the costs of an edge and of a corruption, and the
    code lasso as the regulariser.

    **Why a** :class:`~mowl.base_models.GraphModel` **and not an**
    :class:`~mowl.base_models.EmbeddingELModel`. The paper models one simple directed graph
    :math:`G = (V, E)` and nothing else: there is no conjunction, no existential
    restriction and no notion of a role. An ``EmbeddingELModel`` restricted to ``gci0``
    would carry seven unused normal forms, the EL normaliser and its auxiliary classes, and
    would tie the method to the atomic subsumption graph. As a ``GraphModel`` it embeds the
    edges produced by any :class:`~mowl.projection.base.ProjectionModel`: by default the
    :class:`~mowl.projection.TaxonomyProjector` (the asserted atomic subsumption graph,
    i.e. the ``gci0`` axioms between named classes), but equally a relation projection
    whose edges are not transitive and may form cycles, which is the case the paper was
    designed for.

    **From projected edges to the paper's graph.** All projected edges (optionally only
    those whose label is in ``relations``) are merged into one simple directed graph:
    labels are dropped, duplicates and self-loops are removed (paper, footnote 2). In the
    paper an edge :math:`(u, v)` has low energy when the box of :math:`v` lies inside the
    box of :math:`u` (Eq. 3 normalises by the volume of :math:`v`), so edges point from the
    general to the specific vertex. mOWL edges point the other way (``C -> D`` for
    :math:`C \\sqsubseteq D`), so a projected edge ``src -> dst`` is trained as the paper's
    edge ``(dst, src)``: the box of ``src`` should lie in the box of ``dst``.
    :meth:`score_edges` and :meth:`edge_energy` take edges in mOWL direction.

    **Training (Section 4.4)** is the combination of regimes in :attr:`default_regimes`
    (read ``model.training_regime``; assign another to change one piece):
    :class:`~mowl.training.EpochBatches` (minibatches of positive edges),
    :class:`~mowl.training.EdgeCorruption` (``num_negs`` negatives per positive, 32 in the
    paper, replacing its source or its target by a uniformly drawn vertex),
    :class:`~mowl.training.KnownRowMask` (corruptions that hit a training edge or a self-loop
    are masked) and :class:`~mowl.training.SelfAdversarialBCE` (the binary cross-entropy of
    Eq. (10), negatives weighted self-adversarially, Sun et al., 2019, as the paper states).
    The loss adds :math:`w_r \\lVert \\mathbf{1} - B \\rVert_{\\ell_1}` (Section 4.5) with
    :math:`w_r` = ``code_reg``, the regularisation of the module. Adam is used.

    **Deviations / unspecified details.** (i) The paper does not say how codes are kept in
    :math:`[0, 1]`; mOWL uses a sigmoid of a free logit (see the module). (ii) The paper does
    not give the form of the self-adversarial weights; mOWL uses RotatE's softmax over the
    negatives' logits with inverse temperature ``adversarial_temperature`` and no gradient
    through the weights, which makes the negative term a weighted average rather than the
    plain sum of Eq. (10) (set ``self_adversarial=False`` for the plain sum). (iii) For
    VBC-Box the lasso is summed over the codes of all vertices at every step. (iv) The
    paper's G-Box parameters are the corner means directly (Eq. 3, Appendix A); no Euler
    constant offset is applied.

    :param dataset: mOWL dataset; its training ontology is projected.
    :type dataset: :class:`mowl.datasets.Dataset`
    :param embed_dim: Box dimension :math:`d`.
    :type embed_dim: int
    :param variant: ``"vbc"`` (per-vertex codes, Eq. 9), ``"gbc"`` (global codes, Eq. 8) or
        ``"gbox"`` (plain Gumbel boxes, Eq. 3, the baseline without codes).
    :type variant: str
    :param num_codes: Number :math:`k` of global codes (``"gbc"`` only).
    :type num_codes: int
    :param code_dims: :math:`d_{bin}`: number of dimensions with trainable codes (the last
        ones); ``None`` means all.
    :type code_dims: int, optional
    :param intersection_temp: :math:`\\tau`.
    :type intersection_temp: float
    :param volume_temp: :math:`\\nu`.
    :type volume_temp: float
    :param code_reg: Lasso weight :math:`w_r`.
    :type code_reg: float
    :param code_init: Mean of the initial code logits.
    :type code_init: float
    :param num_negs: Negatives per positive edge.
    :type num_negs: int
    :param self_adversarial: Use self-adversarial negative weights.
    :type self_adversarial: bool
    :param adversarial_temperature: Inverse temperature of the self-adversarial weights.
    :type adversarial_temperature: float
    :param learning_rate: Adam learning rate.
    :type learning_rate: float
    :param batch_size: Number of positive edges per minibatch.
    :type batch_size: int
    :param relations: If given, keep only projected edges with these labels.
    :type relations: list of str, optional
    :param projector: Projection of the ontology to a graph. Defaults to
        :class:`~mowl.projection.TaxonomyProjector`.
    :type projector: :class:`~mowl.projection.base.ProjectionModel`, optional
    :param model_filepath: Where :meth:`train` saves the module's state dict.
    :type model_filepath: str, optional
    :param device: Torch device.
    :type device: str
    """

    def init_module(self):
        """Create the :class:`~mowl.nn.graph.bcbox.BinaryCodeBoxModule`."""
        self._module = BinaryCodeBoxModule(
            self.num_nodes, self.embed_dim, variant=self.variant, num_codes=self.num_codes,
            code_dims=self.code_dims, intersection_temp=self.intersection_temp,
            volume_temp=self.volume_temp, code_init=self.code_init,
            code_weight=self.code_reg).to(self.device)

    # ------------------------------------------------------------- training
    save_always = True

    @property
    def default_regimes(self):
        """The regimes that make the training of the method (Section 4.4)."""
        forms = ("gci0",)
        return (partial(EpochBatches, self.batch_size, forms),
                partial(EdgeCorruption, self.num_negs, (1, 0), self.num_nodes, forms),
                partial(KnownRowMask, forms),
                partial(SelfAdversarialBCE, self.self_adversarial,
                        self.adversarial_temperature, forms))

    @property
    def training_datasets(self):
        """The edges as the axioms ``src ⊑ dst`` (rows ``(src, dst)``, the order of the paper
        reversed) of the form ``gci0``."""
        rows = getattr(self, "_rows", None)
        if rows is None or rows.data.device != th.device(self.device):
            rows = self._rows = Rows(self.edge_index.flip(1).to(self.device))
        return {"gci0": rows}

    def compute_loss(self, pos):
        """The loss of a minibatch ``pos`` of edges in the direction of the paper (shape
        ``(b, 2)``) as training computes it: Eq. (10) and the code lasso."""
        return self.batch_loss({"gci0": pos.flip(1)})
