"""MultiELBE model class."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.datasets.el.points import (
    PointSupervision,
    build_or_load_point_supervision,
)
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.nn import MultiELBEModule
from mowl.nn.el.pointloss import (
    _VALID_OBJECTIVES,
    _validate_objective,
    volume_penalty,
)


class MultiELBE(EmbeddingELModel):
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
      hard geometric model because :func:`~mowl.nn.el.softagg.softmin` over
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
        Training objective mode (O1 ``"axiom"``, O3 ``"point"``, O2
        ``"mixed"``, or ``"hybrid"``).
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
    """

    def __init__(
        self,
        dataset,
        embed_dim=50,
        margin=0.1,
        n_components=1,
        component_reg=0.0,
        component_aggregation="hard",
        tau=None,
        tau_start=1.0,
        tau_end=0.05,
        tau_anneal=False,
        diversity_reg=0.0,
        init_boxes=None,
        anchor_weight=0.0,
        reg_norm=1,
        learning_rate=0.001,
        batch_size=4096 * 8,
        model_filepath=None,
        device="cpu",
        neg_sampling_gcis=None,
        objective="axiom",
        beta=0.1,
        neg_weight=0.1,
        volume_reg=1e-3,
        points_per_epoch=100_000,
        points_seed=0,
    ):
        if float(anchor_weight) < 0:
            raise ValueError(f"anchor_weight must be >= 0, got {anchor_weight!r}")
        if component_aggregation not in ("hard", "soft"):
            raise ValueError(
                f"component_aggregation must be 'hard' or 'soft', got {component_aggregation!r}"
            )
        eff_tau_start = float(tau) if tau is not None else float(tau_start)
        eff_tau_end = float(tau_end)
        if eff_tau_start <= 0.0 or eff_tau_end <= 0.0:
            raise ValueError(
                f"tau_start and tau_end must be > 0, got tau_start={eff_tau_start!r}, tau_end={eff_tau_end!r}"
            )
        if float(diversity_reg) < 0.0:
            raise ValueError(f"diversity_reg must be >= 0, got {diversity_reg!r}")
        _validate_objective(objective)
        if isinstance(beta, bool) or not isinstance(beta, (int, float)) or float(beta) <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta!r}")
        if (
            isinstance(neg_weight, bool)
            or not isinstance(neg_weight, (int, float))
            or float(neg_weight) < 0.0
        ):
            raise ValueError(f"neg_weight must be >= 0, got {neg_weight!r}")
        if (
            isinstance(volume_reg, bool)
            or not isinstance(volume_reg, (int, float))
            or float(volume_reg) < 0.0
        ):
            raise ValueError(f"volume_reg must be >= 0, got {volume_reg!r}")
        if (
            isinstance(points_per_epoch, bool)
            or not isinstance(points_per_epoch, int)
            or points_per_epoch < 1
        ):
            raise ValueError(
                f"points_per_epoch must be an int >= 1, got {points_per_epoch!r}"
            )
        if isinstance(points_seed, bool) or not isinstance(points_seed, int):
            raise ValueError(f"points_seed must be an int, got {points_seed!r}")
        super().__init__(
            dataset,
            embed_dim,
            batch_size,
            extended=True,
            model_filepath=model_filepath,
            device=device,
            learning_rate=learning_rate,
            neg_sampling_gcis=neg_sampling_gcis,
        )

        self.margin = margin
        self.n_components = n_components
        self.component_reg = component_reg
        self.component_aggregation = component_aggregation
        self.tau_start = eff_tau_start
        self.tau_end = eff_tau_end
        self.tau_anneal = bool(tau_anneal)
        self.tau = eff_tau_start
        self.diversity_reg = float(diversity_reg)
        self.reg_norm = reg_norm
        self.init_boxes = init_boxes
        self.anchor_weight = float(anchor_weight)
        self.objective = objective
        self.beta = float(beta)
        self.neg_weight = float(neg_weight)
        self.volume_reg = float(volume_reg)
        self.points_per_epoch = int(points_per_epoch)
        self.points_seed = int(points_seed)
        self._point_supervision: Optional[PointSupervision] = None
        self._loaded = False
        self.extended = False
        self._mse_criterion = th.nn.MSELoss()
        self.init_module()

    def init_module(self):
        """Initialise :class:`~mowl.nn.MultiELBEModule` and apply ``init_boxes`` if present."""
        self.module = MultiELBEModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict),
            embed_dim=self.embed_dim,
            margin=self.margin,
            n_components=self.n_components,
            component_reg=self.component_reg,
            component_aggregation=self.component_aggregation,
            tau=self.tau,
            diversity_reg=self.diversity_reg,
            objective=self.objective,
            beta=self.beta,
            neg_weight=self.neg_weight,
            volume_reg=self.volume_reg,
        ).to(self.device)
        if self.init_boxes is not None:
            self.init_boxes.apply_to_module(
                self.module, class_index_dict=self.class_index_dict
            )
        if self._point_supervision is not None:
            self.module.set_point_supervision(self._point_supervision)

    @property
    def point_supervision(self) -> Optional[PointSupervision]:
        """Return the cached :class:`~mowl.datasets.el.points.PointSupervision` if built.

        Guarantees:
        - Returns the :class:`~mowl.datasets.el.points.PointSupervision`
          instance attached to this model once :meth:`_ensure_point_supervision`
          has been called, or ``None`` beforehand.

        Non-guarantees:
        - Does not trigger lazy construction of point supervision.

        Returns
        -------
        PointSupervision or None
            Cached point supervision bundle or ``None``.
        """
        return self._point_supervision

    def _ensure_point_supervision(self) -> PointSupervision:
        """Lazily build or load :class:`~mowl.datasets.el.points.PointSupervision` and attach to ``module``.

        Guarantees:
        - Builds or loads the :class:`~mowl.datasets.el.points.PointSupervision`
          once and registers ``elem_embed`` on ``self.module``.

        Non-guarantees:
        - Does not re-run canonical model completion on subsequent calls unless
          ``self._point_supervision`` is reset.

        Returns
        -------
        PointSupervision
            The active point supervision bundle.
        """
        if self._point_supervision is None:
            self._point_supervision = build_or_load_point_supervision(
                self.dataset,
                self.named_class_index_dict,
                self.individual_index_dict,
                self.object_property_index_dict,
                rng_seed=self.points_seed,
            )
        if self.module.point_supervision is not self._point_supervision:
            self.module.set_point_supervision(self._point_supervision)
        return self._point_supervision

    def get_optimizer(self):
        """Ensure ``elem_embed`` is registered before constructing the Adam optimiser.

        Guarantees:
        - Under ``objective="point"``, ``"mixed"``, or ``"hybrid"``, calls
          :meth:`_ensure_point_supervision` before ``super().get_optimizer()``
          so ``self.module.elem_embed`` parameters are included in the
          optimiser parameter group.

        Non-guarantees:
        - Does not construct ``elem_embed`` when ``objective="axiom"``.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimiser over ``self.module.parameters()``.
        """
        if self.objective != "axiom":
            self._ensure_point_supervision()
        return super().get_optimizer()

    def get_negative_sampling_config(self):
        """Disable GCI negative sampling by default when GCI losses are zeroed in point/mixed mode.

        Guarantees:
        - Returns an empty dict ``{}`` when ``objective in ("point", "mixed")`` and
          ``neg_sampling_gcis is None``.
        - Delegates to ``super().get_negative_sampling_config()`` otherwise (including
          ``"axiom"`` and ``"hybrid"``).

        Non-guarantees:
        - Does not disable GCI negative sampling if the caller explicitly passed
          a non-``None`` ``neg_sampling_gcis`` list.

        Returns
        -------
        dict
            Mapping from GCI name to negative-sampling configuration.
        """
        if self.objective in ("point", "mixed") and self.neg_sampling_gcis is None:
            return {}
        return super().get_negative_sampling_config()

    def on_epoch_start(self, epoch: int, epochs: int) -> None:
        """Update ``self.module.tau`` according to the geometric annealing schedule when ``tau_anneal=True``.

        Guarantees:
        - When ``tau_anneal=True`` and ``epochs > 1``, sets
          ``module.tau = tau_start * (tau_end / tau_start) ** (epoch / (epochs - 1))``
          so epoch ``0`` uses ``tau_start`` and epoch ``epochs - 1`` uses ``tau_end``.
        - ``module.tau`` is updated in-place as a registered buffer and saved in
          ``module.state_dict()``.
        - When ``tau_anneal=False``, leaves ``module.tau`` unchanged.

        Non-guarantees:
        - When ``epochs <= 1`` and ``tau_anneal=True``, ``module.tau`` remains ``tau_start``.
        - The annealing schedule is indexed by the local ``epoch`` counter of each
          :meth:`train` call and therefore restarts from ``tau_start`` at every
          :meth:`train` invocation.
        """
        if self.tau_anneal:
            if epochs > 1:
                tau = self.tau_start * (
                    (self.tau_end / self.tau_start) ** (epoch / (epochs - 1))
                )
            else:
                tau = self.tau_start
            self.tau = float(tau)
            self.module.tau = float(tau)

    def additional_training_loss(self, epoch: int) -> Optional[th.Tensor]:
        """Compute sampled canonical-element point and role losses plus volume regulariser.

        Guarantees:
        - Returns ``None`` when ``objective == "axiom"``.
        - Under ``objective in ("point", "mixed", "hybrid")``, samples at most
          ``points_per_epoch`` pairs using seed ``points_seed + epoch`` and
          returns ``point_membership_loss + point_role_loss + volume_reg * volume_penalty(offsets)``
          where ``TOP`` (``owl:Thing``) and ``BOTTOM`` (``owl:Nothing``) are
          excluded from the volume penalty.

        Non-guarantees:
        - Does not include region-level GCI losses (handled by the main epoch
          loop).

        Parameters
        -------
        epoch : int
            Zero-based training epoch index used to seed per-epoch sampling.

        Returns
        -------
        torch.Tensor or None
            Scalar additional training loss tensor, or ``None`` under
            ``objective="axiom"``.
        """
        if self.objective == "axiom":
            return None
        ps = self._ensure_point_supervision()
        epoch_rng = np.random.default_rng(self.points_seed + int(epoch))
        batch = ps.sample(self.points_per_epoch, rng=epoch_rng)
        loss = self.module.point_membership_loss(batch)
        loss = loss + self.module.point_role_loss(batch)
        if self.volume_reg > 0.0 and len(self.named_class_index_dict) > 0:
            non_top_indices = [
                int(idx)
                for iri, idx in self.named_class_index_dict.items()
                if iri not in (TOP, BOTTOM)
            ]
            if non_top_indices:
                cls_idx = th.tensor(
                    non_top_indices, device=self.device, dtype=th.long
                )
                _, offsets = self.module.class_boxes(cls_idx)
                loss = loss + self.volume_reg * volume_penalty(offsets)
        return loss

    def get_regularization_loss(self):
        """Return module regularisation loss plus weighted anchor loss when active."""
        loss = super().get_regularization_loss()
        if self.init_boxes is not None and self.anchor_weight > 0:
            loss = loss + self.anchor_weight * self.init_boxes.anchor_loss(self.module)
        return loss

    def compute_loss(self, pos_scores, neg_scores=None):
        """Compute MSE loss for MultiELBE (matching ELBE).

        Positive samples should have scores close to 0.
        Negative samples should have scores close to 1.
        """
        pos_mean = th.mean(pos_scores)
        loss = self._mse_criterion(
            pos_mean, th.zeros_like(pos_mean, requires_grad=False)
        )

        if neg_scores is not None:
            neg_mean = th.mean(neg_scores)
            loss += self._mse_criterion(
                neg_mean, th.ones_like(neg_mean, requires_grad=False)
            )

        return loss
