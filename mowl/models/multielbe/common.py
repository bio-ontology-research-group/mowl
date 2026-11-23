"""What the MultiELBE model class and its direct implementation share: the options and their
checks, the point supervision the module reads, the schedule of the temperature.

Nothing here computes a loss: the class users get (:class:`mowl.models.MultiELBE`) builds its
module from parts and reads its losses there; the direct implementation in
:mod:`mowl.baselines` has its own module and loss code.
"""
from __future__ import annotations

from typing import Optional


from mowl.base_models.elmodel import EmbeddingELModel
from mowl.datasets.el.points import (
    PointSupervision,
    build_or_load_point_supervision,
)
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.training.batching import epoch_rng
from mowl.training.losses import MSETargetLoss
from mowl.representation.library.structural_points import POINT_READINGS

__all__ = ["MultiELBEOptions"]


class MultiELBEOptions(EmbeddingELModel):
    """The model of MultiELBE without its module and its losses: the constructor (the options
    of :class:`mowl.models.MultiELBE`, documented there) and the point supervision. A
    subclass names the module class (:attr:`module_class`) and computes the extra training
    loss."""

    #: The module class: built in :meth:`init_module` with the options of the model.
    module_class = None

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
                f"tau_start and tau_end must be > 0, got tau_start={eff_tau_start!r}, "
                f"tau_end={eff_tau_end!r}"
            )
        if float(diversity_reg) < 0.0:
            raise ValueError(f"diversity_reg must be >= 0, got {diversity_reg!r}")
        if objective not in POINT_READINGS:
            raise ValueError(f"objective must be one of {POINT_READINGS}, got {objective!r}")
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
        self._loss_part = MSETargetLoss(0.0, 1.0, reduce="mean_first")
        self.init_module()

    def init_module(self):
        """Initialise the module (:attr:`module_class`) and apply ``init_boxes`` if present."""
        self.module = self.module_class(
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
        """Lazily build or load :class:`~mowl.datasets.el.points.PointSupervision` and attach it
        to ``module``.

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

    # -- data the extra training loss reads -------------------------------------------------
    def _point_batch(self, epoch: int):
        """The sample of point pairs and role triples of an epoch (seed
        ``points_seed + epoch``)."""
        ps = self._ensure_point_supervision()
        return ps.sample(self.points_per_epoch,
                         rng=epoch_rng(self.points_seed, epoch))

    def _volume_classes(self):
        """The named classes the volume penalty covers: all but ``owl:Thing`` and
        ``owl:Nothing``."""
        return [int(idx) for iri, idx in self.named_class_index_dict.items()
                if iri not in (TOP, BOTTOM)]
