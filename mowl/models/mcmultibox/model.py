import logging

from mowl.models.mcmultibox.common import VolumeMultiBoxELOptions
from mowl.nn.el.mcmultibox.module import VolumeMultiBoxELModule

logger = logging.getLogger(__name__)


def _steps_per_epoch(regime):
    """Steps per epoch of the regime (the ``steps_per_epoch`` of one of its parts)."""
    for part in getattr(regime, "regimes", (regime,)):
        steps = getattr(part, "steps_per_epoch", None)
        if isinstance(steps, int) and steps >= 1:
            return steps
    return 1


class VolumeMultiBoxEL(VolumeMultiBoxELOptions):
    r"""Volume-based MultiBoxEL, after [aharoni2024]_ (Chapter 7).

    Concepts are unions of ``num_boxes`` boxes in :math:`\mathbb{R}^d`, roles are unions of boxes
    in :math:`\mathbb{R}^{2d}`, and every normal form is trained with the loss
    :math:`1 - \mathrm{Vol}(A \cap B)/\mathrm{Vol}(A)` whose volumes are estimated on a fixed
    quasi-Monte-Carlo point set (see :class:`~mowl.nn.el.mcmultibox.module.VolumeMultiBoxELModule`
    for the exact losses, including the simplified role chain loss).

    This is **not** :class:`mowl.models.MultiBoxEL`, which is a different method (Federico,
    Leemhuis and Kutz, 2026: unions of boxes with Box\ :sup:`2`\ EL-style distance losses).

    **Training.** As in the thesis, the training set of each normal form is split into
    ``num_batches`` batches per epoch (thesis: 5) and one optimiser step is taken per batch;
    this is the regime :class:`~mowl.training.splits.ShuffledSplits`, with the checkpoint
    selection :class:`~mowl.training.elkem.ValidationCheckpoint`, that the method names as its
    defaults, and any regime can replace them.
    The thesis gives no optimiser or learning rate; we use Adam with ``learning_rate``.
    Per-batch memory is bounded by ``max_chunk_elements``: the axioms of a batch are processed
    in chunks and the points in checkpointed chunks of ``point_chunk_size``, so neither the
    batch size nor the point count changes the peak memory (about 2 GiB with the defaults,
    measured on the CPU; see
    :class:`~mowl.nn.el.mcmultibox.module.VolumeMultiBoxELModule`).

    **What is trained.** ``gci0`` to ``gci3`` and, with ``load_role_axioms=True``, role inclusions
    and role chains. The thesis defines nothing for the bottom normal forms, assertions or
    negatives: bottom datasets and assertion datasets are skipped with a warning, so ontologies
    with disjointness axioms are trained without them (unsatisfiable concepts are not pushed to
    the empty set).

    **Evaluation (thesis Section 7.1.2).** The distance of a pair is the fraction of points
    sampled from the boxes of the left concept that fall outside the right concept
    (:meth:`~mowl.nn.el.mcmultibox.module.VolumeMultiBoxELModule.distance`); lower is better.
    The thesis writes the *count* of contained points; the normalisation by the number of points
    and the sign flip are ours. Set :attr:`eval_gci_name` before evaluating.

    **Thesis hyper-parameters:** ``embed_dim=5``, ``num_boxes=5``, ``num_points=20000``,
    5 epochs, ``num_batches=5``, on GPU, about 8 hours per run. The defaults below use these
    except for ``bound`` and ``learning_rate`` which the thesis does not state.

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension :math:`d` of the concept space.
    :param num_boxes: Boxes per multibox.
    :param bound: Half-width :math:`b_o` of the space :math:`S` (not in the thesis).
    :param num_points: Number of quasi-Monte-Carlo points.
    :param temperature: Sigmoid temperature (thesis: 1).
    :param aggregate: ``"min"`` (default) or ``"mean"`` (literal thesis formula); see
        :func:`~mowl.representation.library.multibox_parts.soft_indicator`.
    :param learning_rate: Adam learning rate (not in the thesis).
    :param num_batches: Batches per epoch.
    :param seed: Seed for the point set, the initialisation and batch shuffling.
    :param point_chunk_size: Points processed at a time.
    :param max_chunk_elements: Largest intermediate tensor of a loss (axioms x points x
        boxes x coordinates); ``None`` processes a whole batch at once (memory then grows with
        the number of axioms).
    :param eval_points_per_box: Points per box in the evaluation distance.
    :param load_role_axioms: Train on role inclusions and role chains.
    """

    module_class = VolumeMultiBoxELModule

    def init_module(self):
        super().init_module()
        # the training of the method: shuffled splits and checkpoint selection per epoch
        self.module.default_regimes = (self._default_splits, self._default_checkpoint)
        self._attach_default_regimes()

    def _default_splits(self):
        from mowl.training.splits import ShuffledSplits
        return ShuffledSplits(self.num_batches, self.seed, forms=self.TRAINABLE_GCIS,
                              what="the thesis")

    def _default_checkpoint(self):
        from mowl.training.elkem import ValidationCheckpoint
        return ValidationCheckpoint("loss", steps_per_epoch=self.num_batches)

    def train(self, epochs: int, validate_every=1, epoch_callback=None):
        """Train with ``num_batches`` optimiser steps per epoch (the regime
        :class:`~mowl.training.splits.ShuffledSplits`; a model whose regime was removed gets
        it again). With another regime attached an epoch is one optimiser step, unless that
        regime has a ``steps_per_epoch``.

        :param epochs: Number of epochs (thesis: 5).
        :param validate_every: Validate every this many epochs on ``eval_gci_name`` of the
            validation set if one exists; the best state is kept in ``model_filepath``.
        :param epoch_callback: Optional ``callable(epoch, model)`` called after each epoch.
        """
        if self.training_regime is None:
            self._attach_default_regimes()
        steps = int(_steps_per_epoch(self.training_regime))
        callback = epoch_callback
        if epoch_callback is not None and steps > 1:
            def every_epoch(step: int, model):
                if (step + 1) % steps == 0:
                    epoch_callback(step // steps, model)
            callback = every_epoch
        self.training_losses = []
        return super().train(epochs * steps, validate_every=validate_every,
                             epoch_callback=callback)
