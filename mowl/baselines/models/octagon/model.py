r"""Octagon-product :math:`\mathcal{EL}^{++}` embedding model."""
from mowl.base_models.elmodel import EmbeddingELModel
from mowl.baselines.nn.el.octagon.module import OctagonELModule


class OctagonEL(EmbeddingELModel):
    r"""Concepts as products of 2-D octagons, roles as translations.

    An **adaptation** of the octagon regions of Charpenay & Schockaert, "Capturing
    Knowledge Graphs and Rules with Octagon Embeddings" (IJCAI 2024). The paper uses
    octagons as *relation* regions of a knowledge graph embedding; mOWL uses them as
    *concept* regions: a concept is a product of 2-D octagons over consecutive coordinate
    pairs of :math:`\mathbb{R}^d` (``embed_dim`` must be even). The octagon geometry
    (normalisation, emptiness, intersection) is the paper's; the concept/role design and
    the losses are ours. See :class:`~mowl.baselines.nn.el.octagon.module.OctagonELModule` for the
    loss of each normal form and :doc:`/methods/octagon` for the Helly analysis: products
    of 2-D octagons have Helly number exactly 3, so unlike boxes they can model
    :math:`T_3` (three pairwise intersecting concepts without a common point) in
    :math:`d = 2`, but no family of four concepts with all triples intersecting can have an
    empty common intersection.

    Training uses the generic loop of :class:`~mowl.base_models.EmbeddingELModel` with the
    default :meth:`compute_loss` (mean positive loss plus mean negative loss). Negative
    sampling is used for ``gci2`` only, with the exact non-inclusion loss
    :func:`~mowl.baselines.nn.el.octagon.losses.non_inclusion_loss`. Role axioms
    (``load_role_axioms=True``) are not supported.

    :param dataset: mOWL dataset.
    :param embed_dim: Even embedding dimension. Defaults to ``50``.
    :param margin: Inclusion margin. Defaults to ``0.0``.
    :param disjoint_margin: Margin of disjointness axioms; ``None`` uses ``margin``.
        Defaults to ``0.1`` so that a zero loss implies true disjointness.
    :param neg_margin: Margin of the negative ``gci2`` loss. Defaults to ``0.1``.
    :param init_radius: Radii are initialised uniformly in ``[0, init_radius]``.
    :param min_width: Width floor regulariser of the module (see
        :class:`~mowl.baselines.nn.el.octagon.module.OctagonELModule`). Defaults to ``0.0``.
    :param learning_rate: Adam learning rate. Defaults to ``0.001``.
    :param lr_decay: Exponential learning-rate decay per epoch (the learning rate of epoch
        ``e`` is ``learning_rate * lr_decay ** e``). Hinge losses that must reach exactly
        zero (equalities such as :math:`A \equiv X_1 \sqcap X_2`) need it: with a constant
        learning rate Adam oscillates around the boundary. Defaults to ``1.0`` (none).
    :param batch_size: Batch size (kept for interface parity; the generic loop uses full
        batches). Defaults to ``4096``.
    :param model_filepath: Checkpoint path.
    :param device: Torch device. Defaults to ``"cpu"``.
    :param neg_sampling_gcis: GCIs to sample negatives for; ``None`` uses the module's
        ``neg_capable_gcis``.
    """

    def __init__(self, dataset, embed_dim=50, margin=0.0, disjoint_margin=0.1,
                 neg_margin=0.1, init_radius=0.5, min_width=0.0, learning_rate=0.001,
                 lr_decay=1.0,
                 batch_size=4096, model_filepath=None, device="cpu",
                 neg_sampling_gcis=None):
        if isinstance(embed_dim, int) and embed_dim % 2:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate, neg_sampling_gcis=neg_sampling_gcis)
        self.margin = margin
        self.disjoint_margin = disjoint_margin
        self.neg_margin = neg_margin
        self.init_radius = init_radius
        self.min_width = min_width
        self.lr_decay = float(lr_decay)
        self._optimizer = None
        self.init_module()

    def init_module(self):
        self.module = OctagonELModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            nb_inds=len(self.individual_index_dict) or None,
            embed_dim=self.embed_dim,
            margin=self.margin,
            disjoint_margin=self.disjoint_margin,
            neg_margin=self.neg_margin,
            init_radius=self.init_radius,
            min_width=self.min_width,
        ).to(self.device)

    def get_optimizer(self):
        self._optimizer = super().get_optimizer()
        return self._optimizer

    def on_epoch_start(self, epoch, epochs):
        if self._optimizer is not None and self.lr_decay != 1.0:
            for group in self._optimizer.param_groups:
                group["lr"] = self.learning_rate * self.lr_decay ** epoch

    def count_class_parameters(self):
        """Number of scalar parameters per concept: :math:`5d` (centre and 8 radii per pair)."""
        widths = [self.module.class_embed.weight.shape[1],
                  self.module.class_radius.weight.shape[1]]
        return int(sum(widths))
