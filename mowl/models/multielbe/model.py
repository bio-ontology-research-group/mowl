"""MultiELBE model class."""
from __future__ import annotations


from mowl.models.multielbe.common import MultiELBEOptions
from mowl.nn import MultiELBEModule
from mowl.training.structural import MethodRegimes


class MultiELBE(MethodRegimes, MultiELBEOptions):
    """Multi-component ELBE embedding model.

    Represents every ontology class as a union of ``n_components`` (:math:`k`)
    axis-parallel boxes in :math:`\\mathbb{R}^d` under ELBE inclusion, gated
    pairwise intersection, and translation role semantics.

    Guarantees:

    - **Exact k = 1 reduction:** With ``n_components=1``,
      ``component_aggregation="hard"``, and ``component_reg=0.0``, the
      underlying :class:`~mowl.nn.MultiELBEModule` computes the exact same
      per-GCI losses and MSE training loss (to within ``float32`` tolerance
      ``1e-6``) as :class:`~mowl.models.ELBE`.
    - **Exact initialisation and anchor hook:** When ``init_boxes`` is provided,
      :meth:`init_module` applies :meth:`~mowl.geometric.init.ExactBoxInit.apply_to_module`
      after module construction; when ``anchor_weight > 0``,
      :meth:`get_regularization_loss` adds
      ``anchor_weight * init_boxes.anchor_loss(self.module)``.

    Non-guarantees:

    - Gradient-based training is non-convex and is not guaranteed to reach a
      zero-loss global optimum from arbitrary random initialisations.
    - At finite :math:`\\tau > 0`, the soft-aggregated loss is not zero at a
      hard geometric model because :func:`~mowl.representation.library.multibox_parts.soft_minimum` over
      target components is strictly below the unweighted mean unless all target
      components coincide (pulling target components toward one another);
      ``diversity_reg`` exists to counteract this target-collapse pressure.
    - Role axiom training (``load_role_axioms=True``) is not supported by
      :class:`~mowl.nn.MultiELBEModule`.

    Parameters
    ----------
    dataset : mowl.datasets.Dataset
        mOWL dataset to use for training and evaluation.
    embed_dim : int, default 50
        The embedding dimension :math:`d`.
    margin : float, default 0.1
        ELBE inclusion margin (positive margin tightens).
    n_components : int, default 1
        Number of box components :math:`k \\ge 1` per ontology class.
    component_reg : float, default 0.0
        Weight for the anti-collapse pairwise component centre regulariser.
    component_aggregation : str, default "hard"
        Component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, optional
        Optional alias for ``tau_start`` when setting a fixed temperature.
    tau_start : float, default 1.0
        Initial soft-assignment temperature :math:`\\tau_{\\text{start}} > 0`.
    tau_end : float, default 0.05
        Final soft-assignment temperature :math:`\\tau_{\\text{end}} > 0` when
        ``tau_anneal=True``. Defaults to ``0.05`` because ``float32`` softmax
        weights underflow to exact ``0.0`` once the score gap exceeds
        :math:`\\sim 100\\tau`, so smaller ``tau_end`` reintroduces dead
        components late in training.
    tau_anneal : bool, default False
        Whether to anneal ``module.tau`` geometrically from ``tau_start`` to
        ``tau_end`` over the training epochs in :meth:`on_epoch_start`.
        ``module.tau`` is stored as a registered buffer in ``state_dict()``,
        and the annealing schedule restarts from ``tau_start`` at every
        :meth:`train` call.
    diversity_reg : float, default 0.0
        Diversity regularisation weight stored on the module.
    reg_norm : int, default 1
        Regularisation norm order (retained for interface parity with
        :class:`~mowl.models.ELBE`).
    learning_rate : float, default 0.001
        Adam learning rate.
    batch_size : int, default 32768
        Training batch size.
    model_filepath : str, optional
        Path for saving and loading model checkpoints.
    device : str, default 'cpu'
        PyTorch device string.
    neg_sampling_gcis : list of str, optional
        Explicit list of GCI names for negative sampling, or ``None`` to use
        the module's ``neg_capable_gcis``.
    init_boxes : mowl.geometric.init.ExactBoxInit, optional
        Optional exact-model box initialisation applied in :meth:`init_module`.
    anchor_weight : float, default 0.0
        Non-negative weight multiplying ``init_boxes.anchor_loss(self.module)``
        in :meth:`get_regularization_loss` when ``init_boxes`` is provided.
    objective : {"axiom", "point", "mixed", "hybrid"}, default "axiom"
        Training objective mode (``"axiom"``, ``"point"``, ``"mixed"``, or
        ``"hybrid"``). Under ``objective="point"`` and ``"mixed"``, ABox
        ``class_assertion`` is still trained in the main loop via
        :math:`-\\log m_C(a)` while ``object_property_assertion`` is zeroed
        (handled via ``point_role_loss``).
    beta : float, default 0.1
        Temperature :math:`\\beta > 0` for the sigmoid boundary of soft box
        membership under ``objective="point"``, ``"mixed"``, or ``"hybrid"``.
    neg_weight : float, default 0.1
        Non-negative weight :math:`\\lambda_{\\text{neg}}` on the negative point
        membership term under ``objective="point"``, ``"mixed"``, or ``"hybrid"``.
    volume_reg : float, default 1e-3
        Non-negative weight :math:`\\lambda_{\\text{vol}}` on the named-class
        volume penalty (excluding ``TOP`` and ``BOTTOM``) under
        ``objective="point"``, ``"mixed"``, or ``"hybrid"``.
    points_per_epoch : int, default 100000
        Total pair budget sampled per epoch from
        :class:`~mowl.datasets.el.points.PointSupervision` under
        ``objective="point"``, ``"mixed"``, or ``"hybrid"``.
    points_seed : int, default 0
        Base random seed for :class:`~mowl.datasets.el.points.PointSupervision`
        construction and per-epoch sampling under ``objective="point"``,
        ``"mixed"``, or ``"hybrid"``.

    References
    ----------
    G. Federico, M. Leemhuis and O. Kutz, "Introducing Multiboxes: Increasing
    the Trustworthiness of Ontology Embeddings", JOWO 2026.
    """

    module_class = MultiELBEModule

    def init_module(self):
        """Make the module (:class:`~mowl.nn.MultiELBEModule`) and attach the regimes of the
        reading (:meth:`method_regimes`) unless a regime is attached."""
        super().init_module()
        self._attach_method_regimes()

    def _annealing(self):
        """The cooling schedule of ``tau_anneal=True``, as a regime."""
        from mowl.training.schedules import TemperatureAnnealing
        return [TemperatureAnnealing(self.tau_start, self.tau_end)] if self.tau_anneal else []

    def method_regimes(self):
        """The training regimes of the reading, which a regime assigned to
        :attr:`training_regime` is added to (one of the same class replaces one of them):
        under ``"point"``, ``"mixed"`` and ``"hybrid"``
        :class:`~mowl.training.structural.SampledPointLoss` (the point terms on a sample of
        ``points_per_epoch`` pairs, seed ``points_seed + epoch``); none under ``"axiom"``; with
        ``tau_anneal=True`` in every reading :class:`~mowl.training.TemperatureAnnealing` from
        ``tau_start`` to ``tau_end``."""
        from mowl.training.structural import SampledPointLoss
        regimes = self._annealing()
        if self.objective == "axiom":
            return regimes
        return regimes + [SampledPointLoss(self.points_per_epoch, self.points_seed)]

    def compute_loss(self, pos_scores, neg_scores=None):
        """Compute MSE loss for MultiELBE (matching ELBE).

        Positive samples should have scores close to 0.
        Negative samples should have scores close to 1.
        """
        return self._loss_part.combine(None, pos_scores, neg_scores)
