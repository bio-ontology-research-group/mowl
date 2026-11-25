"""MultiBoxEL model class."""
from __future__ import annotations

from mowl.models.multiboxel.common import MultiBoxELOptions
from mowl.nn import MultiBoxELModule
from mowl.training.structural import MethodRegimes


class MultiBoxEL(MethodRegimes, MultiBoxELOptions):
    """Multi-component Box\\ :math:`^2`\\ EL embedding model.

    Represents every ontology class as a union of ``n_components`` (:math:`k`)
    axis-parallel boxes in :math:`\\mathbb{R}^d` under Box\\ :math:`^2`\\ EL
    inclusion, gated pairwise intersection, and role semantics.

    Guarantees:

    - **Exact k = 1 reduction:** With ``n_components=1`` and
      ``component_reg=0.0``, the underlying :class:`~mowl.nn.MultiBoxELModule`
      computes the exact same per-GCI losses and regularisation loss (to within
      ``float32`` tolerance ``1e-6``) as :class:`~mowl.models.BoxSquaredEL`.
    - **Exact initialisation and anchor hook:** When ``init_boxes`` is provided,
      :meth:`init_module` applies :meth:`~mowl.geometric.init.ExactBoxInit.apply_to_module`
      after module construction; when ``anchor_weight > 0``,
      :meth:`get_regularization_loss` adds
      ``anchor_weight * init_boxes.anchor_loss(self.module)``.
    - **Negative sampling under ``objective="structural"``:** Negative sampling
      stays enabled by default under ``objective="structural"`` for ``gci2`` and
      ``object_property_assertion`` (``neg_capable_gcis``) exactly as under
      ``objective="axiom"``; positive role and assertion scores pass
      ``gamma = -self.margin`` (tightening inclusion), whereas negative scores
      (``gci2`` with masked active non-empty slots and
      ``object_property_assertion``) keep Box\\ :math:`^2`\\ EL's convention
      (``gamma = +margin``, which relaxes them by the margin), so the
      negative-sampling term is unchanged between the axiom and structural
      objectives.

    Non-guarantees:

    - Gradient-based training is non-convex and is not guaranteed to reach a
      zero-loss global optimum from arbitrary random initialisations.
    - Role axiom training (``load_role_axioms=True``) is not supported by
      :class:`~mowl.nn.MultiBoxELModule`.
    - Under ``component_aggregation="soft"`` at finite ``tau > 0``, the soft
      loss is not zero at a hard model (:func:`~mowl.representation.library.multibox_parts.soft_minimum`
      :math:`\\le \\text{mean}` pulls all target components toward source
      components), which is why ``diversity_reg`` exists.
    - Although ``module.tau`` is stored as a registered buffer in
      ``module.state_dict()``, the annealing schedule in :meth:`on_epoch_start`
      is indexed by the local ``epoch`` counter of each :meth:`train` call and
      therefore restarts from ``tau_start`` at every :meth:`train` call.

    Parameters
    ----------
    dataset : mowl.datasets.Dataset
        mOWL dataset to use for training and evaluation.
    embed_dim : int, default 50
        The embedding dimension :math:`d`.
    n_components : int, default 2
        Number of box components :math:`k \\ge 1` per ontology class.
    margin : float, default 0.02
        Inclusion margin :math:`\\gamma`.
    reg_norm : int, default 1
        Regularisation norm order (retained for interface parity with
        :class:`~mowl.models.BoxSquaredEL`).
    learning_rate : float, default 0.001
        Adam learning rate.
    batch_size : int, default 32768
        Training batch size.
    delta : float, default 2.5
        Negative-sampling distance target :math:`\\delta`.
    reg_factor : float, default 0.2
        Weight for the Box²EL bump regularisation term.
    component_reg : float, default 0.0
        Weight for the anti-collapse pairwise component centre regulariser.
    num_negs : int, default 4
        Number of negative samples per positive (retained for interface parity
        with :class:`~mowl.models.BoxSquaredEL`).
    model_filepath : str, optional
        Path for saving and loading model checkpoints.
    device : str, default 'cpu'
        PyTorch device string.
    neg_sampling_gcis : list of str, optional
        Explicit list of GCI names for negative sampling, or ``None`` to use
        the module's ``neg_capable_gcis``.
    init_boxes : mowl.geometric.init.ExactBoxInit, optional
        Optional exact-model box initialisation applied in :meth:`init_module`.
        Rows are matched by IRI against :attr:`class_index_dict`; rows of the
        auxiliary classes introduced by normalisation keep their random
        initialisation.
    anchor_weight : float, default 0.0
        Non-negative weight multiplying ``init_boxes.anchor_loss(self.module)``
        in :meth:`get_regularization_loss` when ``init_boxes`` is provided.
    component_aggregation : str, default "hard"
        Component aggregation mode: ``"hard"`` or ``"soft"``.
    tau_start : float, default 1.0
        Initial soft-assignment temperature :math:`\\tau_{\\text{start}} > 0`.
    tau_end : float, default 0.05
        Final soft-assignment temperature :math:`\\tau_{\\text{end}} > 0` when
        ``tau_anneal=True``. Defaults to ``0.05`` because ``float32`` softmax
        weights :math:`\\exp(-\\Delta / \\tau)` underflow to ``0.0`` once the
        score gap :math:`\\Delta` exceeds :math:`\\approx 100 \\tau`; smaller
        ``tau_end`` values therefore reintroduce dead components with exact
        zero gradient late in training.
    tau_anneal : bool, default False
        Whether to anneal ``module.tau`` geometrically from ``tau_start`` to
        ``tau_end`` over the training epochs in :meth:`on_epoch_start`. The
        schedule restarts from ``tau_start`` on every call to :meth:`train`.
    diversity_reg : float, default 0.0
        Weight for the pairwise component volume overlap regulariser.
    tau : float, optional
        Optional alias for ``tau_start`` when setting a fixed temperature.
    objective : {"axiom", "point", "mixed", "hybrid", "structural"}, default "axiom"
        Training objective mode (``"axiom"``, ``"point"``, ``"mixed"``,
        ``"hybrid"``, or ``"structural"``). Under ``objective="point"`` and
        ``"mixed"``, ABox ``class_assertion`` is still trained in the main loop
        via :math:`-\\log m_C(a)` while ``object_property_assertion`` is zeroed
        (handled via ``point_role_loss``). Under ``objective="structural"``,
        negative sampling stays enabled exactly as under ``"axiom"`` (scoring
        ``gci2`` and ``object_property_assertion`` negatives with ``margin`` as
        ``gamma``).
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
        ``"mixed"``, or ``"hybrid"``, and for per-epoch re-seeding under
        ``objective="structural"``.
    max_components : int, optional, default None
        Maximum component slots :math:`K \\ge k` per class under
        ``objective="structural"``; defaults to ``n_components ** 2`` when ``None``.
    structural_definitions : bool, default True
        Whether ``objective="structural"`` compiles acyclic binary conjunction
        definitions :math:`A \\equiv X \\sqcap Y` (including through role
        auxiliaries :math:`A \\sqsubseteq \\exists r.C \\sqsubseteq \\text{Aux}`)
        via :class:`~mowl.nn.el.structural.DefinitionTable`.
    existence_weight : float, default 1.0
        Non-negative weight multiplying :func:`~mowl.baselines.nn.el.multiboxel.structural_losses.existence_loss`
        under ``objective="structural"``.
    size_reg : float, default 1e-3
        Non-negative weight multiplying the primitive box width regulariser
        ``mean(2 * |offset|)`` under ``objective="structural"``.
    reseed_every : int, default 0
        Epoch interval for heuristic discrete re-seeding
        (:meth:`~mowl.nn.MultiBoxELModule.reseed`) under
        ``objective="structural"`` (``0`` disables re-seeding).
    canonical_model : mowl.geometric.completion.CanonicalModel, optional, default None
        Optional pre-computed :class:`~mowl.geometric.completion.CanonicalModel`
        used to determine semantic satisfiability in
        :meth:`~mowl.nn.el.structural.ExistenceTargets.from_normal_forms` under
        ``objective="structural"``.
    bot_agg : {"mean", "sum"}, default "mean"
        Aggregation over active component pairs in ``gci1_bot`` under
        ``objective="structural"`` (passed as ``agg`` to
        :func:`~mowl.baselines.nn.el.multiboxel.structural_losses.gci1_bot_structural`).
    witness_weight : float, default 0.0
        Non-negative weight multiplying :func:`~mowl.baselines.nn.el.multiboxel.structural_losses.witness_loss`
        under ``objective="structural"``.
    reseed_mode : {"dead", "idle"}, default "dead"
        Discrete re-seeding mode for :meth:`~mowl.nn.MultiBoxELModule.reseed`
        under ``objective="structural"``: ``"dead"`` relocates nested or thin
        primitive components to violated conjunction pair partners; ``"idle"``
        requires ``witness_weight > 0`` and relocates the least-used primitive
        leaf components of dead defined classes to their witness points.
    reseed_size : float, default 0.15
        Positive box half-width :math:`> 0` assigned in every coordinate to
        re-seeded components under ``reseed_mode="idle"``.
    reseed_batch : int, default 1
        Maximum number of dead defined classes :math:`\\ge 1` processed per
        :meth:`~mowl.nn.MultiBoxELModule.reseed` call under
        ``reseed_mode="idle"``.

    References
    ----------
    G. Federico, M. Leemhuis and O. Kutz, "Introducing Multiboxes: Increasing
    the Trustworthiness of Ontology Embeddings", JOWO 2026.
    """

    module_class = MultiBoxELModule

    def init_module(self):
        """Make the module (:class:`~mowl.nn.MultiBoxELModule`) and attach the regimes of the
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

        - ``objective="structural"``: :class:`~mowl.training.structural.StructuralTables`
          (definition table and targets compiled from the training data when training
          starts) and :class:`~mowl.training.structural.ExtraPenalties` (the existence,
          size and witness penalties at every step);
        - ``"point"``, ``"mixed"``, ``"hybrid"``:
          :class:`~mowl.training.structural.SampledPointLoss` (the point terms on a sample
          of ``points_per_epoch`` pairs, seed ``points_seed + epoch``);
        - ``"axiom"``: none;
        - with ``tau_anneal=True`` in every reading: :class:`~mowl.training.TemperatureAnnealing`
          from ``tau_start`` to ``tau_end``.
        """
        from mowl.training.structural import ExtraPenalties, SampledPointLoss, StructuralTables
        regimes = self._annealing()
        if self.objective == "structural":
            return regimes + [StructuralTables(lambda model: dict(
                k=model.n_components, max_components=model.max_components,
                enabled=model.structural_definitions, cm=model.canonical_model)),
                ExtraPenalties()]
        if self.objective in ("point", "mixed", "hybrid"):
            return regimes + [SampledPointLoss(self.points_per_epoch, self.points_seed)]
        return regimes
