r"""Octagon-product :math:`\mathcal{EL}^{++}` embedding model."""
from mowl.models.composed.model import ComposedEL
from mowl.representation.library.shapes_training import ExponentialDecay


class OctagonEL(ComposedEL):
    r"""Concepts as products of 2-D octagons, roles as translations.

    An **adaptation** of the octagon regions of Charpenay & Schockaert, "Capturing
    Knowledge Graphs and Rules with Octagon Embeddings" (IJCAI 2024). The paper uses
    octagons as *relation* regions of a knowledge graph embedding; mOWL uses them as
    *concept* regions: a concept is a product of 2-D octagons over consecutive coordinate
    pairs of :math:`\mathbb{R}^d` (``embed_dim`` must be even). The octagon geometry
    (normalisation, emptiness, intersection) is the paper's; the concept/role design and
    the losses are ours. See :class:`~mowl.nn.OctagonELModule` for the
    parts, and :doc:`/methods/octagon` for the Helly analysis: products
    of 2-D octagons have Helly number exactly 3, so unlike boxes they can model
    :math:`T_3` (three pairwise intersecting concepts without a common point) in
    :math:`d = 2`, but no family of four concepts with all triples intersecting can have an
    empty common intersection.

    The model is the combination of parts called ``"octagon"``
    (:func:`~mowl.representation.library.shapes_methods.octagon`): products of octagons as
    classes, translations as relations, individuals as points and the exact octagon tests
    (:class:`~mowl.representation.library.shapes_measures.OctagonBounds`) as measure. It is a
    :class:`~mowl.models.ComposedEL`, so any of the parts can be swapped
    (``ComposedEL(dataset, method="octagon", ...)``) and any training or evaluation regime can
    be attached. The hand-written loss module of the earlier implementation is
    :class:`mowl.baselines.models.octagon.model.OctagonEL`.

    Training uses the generic loop of :class:`~mowl.base_models.EmbeddingELModel` with the
    default :meth:`compute_loss` (mean positive loss plus mean negative loss). Negative
    sampling is used for ``gci2`` only, with the exact non-inclusion test of the measure.
    Role axioms are not supported.

    :param dataset: mOWL dataset.
    :param embed_dim: Even embedding dimension. Defaults to ``50``.
    :param margin: Inclusion margin. Defaults to ``0.0``.
    :param disjoint_margin: Margin of disjointness axioms; ``None`` uses ``margin``.
        Defaults to ``0.1`` so that a zero loss implies true disjointness.
    :param neg_margin: Margin of the negative ``gci2`` loss. Defaults to ``0.1``.
    :param init_radius: Radii are initialised uniformly in ``[0, init_radius]``.
    :param min_width: Width floor regulariser of the module (see
        :class:`~mowl.representation.library.shapes_methods.MinWidthPenalty`). Defaults to ``0.0``.
    :param learning_rate: Adam learning rate. Defaults to ``0.001``.
    :param lr_decay: Exponential learning-rate decay per epoch (the learning rate of epoch
        ``e`` is ``learning_rate * lr_decay ** e``; the regime
        :class:`~mowl.representation.library.shapes_training.ExponentialDecay`). Hinge losses
        that must reach exactly zero (equalities such as :math:`A \equiv X_1 \sqcap X_2`)
        need it: with a constant learning rate Adam oscillates around the boundary. Defaults
        to ``1.0`` (none).
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
        self.lr_decay = float(lr_decay)
        super().__init__(dataset, method="octagon", embed_dim=embed_dim, margin=margin,
                         learning_rate=learning_rate, batch_size=batch_size,
                         model_filepath=model_filepath, device=device,
                         neg_sampling_gcis=neg_sampling_gcis, disjoint_margin=disjoint_margin,
                         neg_margin=neg_margin, init_radius=init_radius, min_width=min_width)
        if self.lr_decay != 1.0:
            self.training_regime = ExponentialDecay(self.lr_decay)

    disjoint_margin = property(lambda self: self.method_options["disjoint_margin"])
    neg_margin = property(lambda self: self.method_options["neg_margin"])
    init_radius = property(lambda self: self.method_options["init_radius"])
    min_width = property(lambda self: self.method_options["min_width"])

    def count_class_parameters(self):
        """Number of scalar parameters per concept: :math:`5d` (centre and 8 radii per pair)."""
        widths = [self.module.class_embed.weight.shape[1],
                  self.module.class_radius.weight.shape[1]]
        return int(sum(widths))
