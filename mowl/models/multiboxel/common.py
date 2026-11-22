"""What the MultiBoxEL model class and its direct implementation share: the options and their
checks, the data the module reads (point supervision, structural tables, saved witness
points), the schedules of the temperature and of re-seeding, and loading.

Nothing here computes a loss: the class users get (:class:`mowl.models.MultiBoxEL`) builds its
module from parts and reads its losses there; the direct implementation in
:mod:`mowl.baselines` has its own module and loss code.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch as th
import torch.nn as nn

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.datasets.el.points import (
    PointSupervision,
    build_or_load_point_supervision,
)
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.training.batching import epoch_rng
from mowl.nn.el.structural import (
    DefinitionTable,
    ExistenceTargets,
    build_structural_tables,
)

__all__ = ["MultiBoxELOptions"]


class MultiBoxELOptions(EmbeddingELModel):
    """The model of MultiBoxEL without its module and its losses: the constructor (the
    options of :class:`mowl.models.MultiBoxEL`, documented there), the data the module reads
    and loading. A subclass names the module class (:attr:`module_class`) and computes the
    extra training loss."""

    #: The module class: built in :meth:`init_module` with the options of the model.
    module_class = None

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 n_components=2,
                 margin=0.02,
                 reg_norm=1,
                 learning_rate=0.001,
                 batch_size=4096 * 8,
                 delta=2.5,
                 reg_factor=0.2,
                 component_reg=0.0,
                 num_negs=4,
                 model_filepath=None,
                 device='cpu',
                 neg_sampling_gcis=None,
                 init_boxes=None,
                 anchor_weight=0.0,
                 component_aggregation="hard",
                 tau_start=1.0,
                 tau_end=0.05,
                 tau_anneal=False,
                 diversity_reg=0.0,
                 tau=None,
                 objective="axiom",
                 beta=0.1,
                 neg_weight=0.1,
                 volume_reg=1e-3,
                 points_per_epoch=100_000,
                 points_seed=0,
                 max_components=None,
                 structural_definitions=True,
                 existence_weight=1.0,
                 size_reg=1e-3,
                 reseed_every=0,
                 canonical_model=None,
                 bot_agg="mean",
                 witness_weight=0.0,
                 reseed_mode="dead",
                 reseed_size=0.15,
                 reseed_batch=1,
                 ):
        if float(anchor_weight) < 0:
            raise ValueError(f"anchor_weight must be >= 0, got {anchor_weight!r}")
        if not isinstance(bot_agg, str) or bot_agg not in ("mean", "sum"):
            raise ValueError(f"bot_agg must be 'mean' or 'sum', got {bot_agg!r}")
        if (
            isinstance(witness_weight, bool)
            or not isinstance(witness_weight, (int, float))
            or float(witness_weight) < 0.0
        ):
            raise ValueError(f"witness_weight must be >= 0, got {witness_weight!r}")
        if not isinstance(reseed_mode, str) or reseed_mode not in ("dead", "idle"):
            raise ValueError(
                f"reseed_mode must be 'dead' or 'idle', got {reseed_mode!r}"
            )
        if reseed_mode == "idle" and float(witness_weight) <= 0.0:
            raise ValueError(
                "reseed_mode='idle' requires witness_weight > 0, "
                f"got witness_weight={witness_weight!r}"
            )
        if (
            isinstance(reseed_size, bool)
            or not isinstance(reseed_size, (int, float))
            or float(reseed_size) <= 0.0
        ):
            raise ValueError(f"reseed_size must be > 0, got {reseed_size!r}")
        if (
            isinstance(reseed_batch, bool)
            or not isinstance(reseed_batch, int)
            or reseed_batch < 1
        ):
            raise ValueError(
                f"reseed_batch must be an int >= 1, got {reseed_batch!r}"
            )
        eff_tau_start = float(tau) if tau is not None else float(tau_start)
        eff_tau_end = float(tau_end)
        if eff_tau_start <= 0.0 or eff_tau_end <= 0.0:
            raise ValueError(
                f"tau_start and tau_end must be > 0, got tau_start={eff_tau_start!r}, tau_end={eff_tau_end!r}"
            )
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
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis)

        self.n_components = n_components
        self.margin = margin
        self.reg_norm = reg_norm
        self.delta = delta
        self.reg_factor = reg_factor
        self.component_reg = component_reg
        self.num_negs = num_negs
        self.init_boxes = init_boxes
        self.anchor_weight = float(anchor_weight)
        self.component_aggregation = component_aggregation
        self.tau_start = eff_tau_start
        self.tau_end = eff_tau_end
        self.tau_anneal = bool(tau_anneal)
        self.tau = eff_tau_start
        self.diversity_reg = diversity_reg
        self.objective = objective
        self.beta = beta
        self.neg_weight = neg_weight
        self.volume_reg = volume_reg
        self.points_per_epoch = int(points_per_epoch)
        self.points_seed = int(points_seed)
        self.max_components = max_components
        self.structural_definitions = structural_definitions
        self.existence_weight = existence_weight
        self.size_reg = size_reg
        self.reseed_every = reseed_every
        self.canonical_model = canonical_model
        self.bot_agg = bot_agg
        self.witness_weight = float(witness_weight)
        self.reseed_mode = reseed_mode
        self.reseed_size = float(reseed_size)
        self.reseed_batch = int(reseed_batch)
        self._point_supervision: Optional[PointSupervision] = None
        self._structural_tables: Optional[
            Tuple[DefinitionTable, ExistenceTargets, Optional[int]]
        ] = None
        self._checkpoint_witness_rows: Optional[int] = None
        self._loaded = False
        self.extended = False
        self.init_module()

    def init_module(self):
        """Initialise the module (:attr:`module_class`), apply ``init_boxes`` if present, and
        re-attach cached supervision/tables."""
        self.module = self.module_class(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict),
            embed_dim=self.embed_dim,
            n_components=self.n_components,
            gamma=self.margin,
            delta=self.delta,
            reg_factor=self.reg_factor,
            component_reg=self.component_reg,
            component_aggregation=self.component_aggregation,
            tau=self.tau,
            diversity_reg=self.diversity_reg,
            objective=self.objective,
            beta=self.beta,
            neg_weight=self.neg_weight,
            volume_reg=self.volume_reg,
            margin=self.margin,
            max_components=self.max_components,
            structural_definitions=self.structural_definitions,
            existence_weight=self.existence_weight,
            size_reg=self.size_reg,
            reseed_every=self.reseed_every,
            bot_agg=self.bot_agg,
            witness_weight=self.witness_weight,
            reseed_mode=self.reseed_mode,
            reseed_size=self.reseed_size,
            reseed_batch=self.reseed_batch,
        ).to(self.device)
        self.max_components = self.module.max_components
        self.structural_definitions = self.module.structural_definitions
        self.diversity_reg = self.module.diversity_reg
        self.beta = self.module.beta
        self.neg_weight = self.module.neg_weight
        self.volume_reg = self.module.volume_reg
        self.existence_weight = self.module.existence_weight
        self.size_reg = self.module.size_reg
        self.reseed_every = self.module.reseed_every
        self.bot_agg = self.module.bot_agg
        self.witness_weight = self.module.witness_weight
        self.reseed_mode = self.module.reseed_mode
        self.reseed_size = self.module.reseed_size
        self.reseed_batch = self.module.reseed_batch
        if self.init_boxes is not None:
            self.init_boxes.apply_to_module(
                self.module, class_index_dict=self.class_index_dict)
        if self._point_supervision is not None:
            self.module.set_point_supervision(self._point_supervision)
        if self._structural_tables is not None:
            table, targets, top_idx = self._structural_tables
            self.module.set_structural_tables(table, targets, top_idx=top_idx)
        if (
            self._checkpoint_witness_rows is not None
            and self.module.witness_embed is None
        ):
            self.module.witness_embed = nn.Embedding(
                self._checkpoint_witness_rows, self.embed_dim
            ).to(
                device=self.module.class_center.weight.device,
                dtype=self.module.class_center.weight.dtype,
            )

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

        - Does not construct ``elem_embed`` when ``objective`` is ``"axiom"``
          or ``"structural"``.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimiser over ``self.module.parameters()``.
        """
        if self.objective in ("point", "mixed", "hybrid"):
            self._ensure_point_supervision()
        return super().get_optimizer()

    def get_negative_sampling_config(self):
        """Disable GCI negative sampling by default when GCI losses are zeroed in point/mixed mode.

        Guarantees:

        - Returns an empty dict ``{}`` when ``objective in ("point", "mixed")`` and
          ``neg_sampling_gcis is None``.
        - Delegates to ``super().get_negative_sampling_config()`` otherwise (including
          ``"axiom"``, ``"hybrid"``, and ``"structural"``).

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
        """Trigger structural re-seeding.

        Guarantees:

        - Under ``objective="structural"``, calls
          ``module.reseed(epoch_rng(points_seed, epoch))`` when
          ``reseed_every > 0 and epoch > 0 and epoch % reseed_every == 0``.

        The temperature schedule of ``tau_anneal=True`` is not here: it is the training regime
        :class:`~mowl.training.TemperatureAnnealing` (``method_regimes`` of the model), which
        sets ``module.tau`` to ``tau_start * (tau_end / tau_start) ** (epoch / (epochs - 1))``
        (``tau_start`` when ``epochs <= 1``; every :meth:`train` call starts at epoch 0).
        """
        if (
            self.objective == "structural"
            and self.reseed_every > 0
            and epoch > 0
            and epoch % self.reseed_every == 0
        ):
            self.module.reseed(epoch_rng(self.points_seed, epoch))

    def reseed_components(self, epoch: int = 0) -> int:
        """Re-seed dead or idle components (:meth:`~mowl.nn.MultiBoxELModule.reseed`):
        the protocol of :class:`mowl.training.IdleComponentReseeding`, for use instead
        of the built-in ``reseed_every`` (leave that at 0 when the regime is used)."""
        return self.module.reseed(epoch_rng(self.points_seed, epoch))

    def _peek_checkpoint_witness_rows(self, path: Optional[str]) -> Optional[int]:
        """Return ``num_embeddings`` if ``witness_embed.weight`` is present in the checkpoint at ``path``."""
        if not isinstance(path, str) or not os.path.exists(path):
            return None
        try:
            state = th.load(path, weights_only=True, map_location="cpu")
        except Exception:
            return None
        if isinstance(state, dict) and "witness_embed.weight" in state:
            w = state["witness_embed.weight"]
            if isinstance(w, th.Tensor) and w.ndim == 2:
                return int(w.shape[0])
        return None

    def load_best_model(self):
        """Reload the best checkpoint and re-attach structural tables when ``objective='structural'``."""
        if self.objective == "structural" and self._structural_tables is None:
            self._structural_tables = build_structural_tables(
                self,
                k=self.n_components,
                max_components=self.max_components,
                enabled=self.structural_definitions,
                cm=self.canonical_model,
            )
        self._checkpoint_witness_rows = self._peek_checkpoint_witness_rows(
            self.model_filepath
        )
        super().load_best_model()

    def from_pretrained(self, model):
        """Load pretrained weights and ensure structural tables are attached before ``load_state_dict``."""
        if not isinstance(model, str):
            raise TypeError(
                "Parameter model must be a string pointing to the model file."
            )
        if not os.path.exists(model):
            raise FileNotFoundError("Pretrained model path does not exist")

        if self.objective == "structural":
            if self._structural_tables is None:
                self._structural_tables = build_structural_tables(
                    self,
                    k=self.n_components,
                    max_components=self.max_components,
                    enabled=self.structural_definitions,
                    cm=self.canonical_model,
                )
            table, targets, top_idx = self._structural_tables
            self.module.set_structural_tables(table, targets, top_idx=top_idx)

        wit_rows = self._peek_checkpoint_witness_rows(model)
        if wit_rows is not None and (
            self.module.witness_embed is None
            or self.module.witness_embed.num_embeddings != wit_rows
        ):
            self._checkpoint_witness_rows = wit_rows
            self.module.witness_embed = nn.Embedding(
                wit_rows, self.embed_dim
            ).to(
                device=self.module.class_center.weight.device,
                dtype=self.module.class_center.weight.dtype,
            )

        super().from_pretrained(model)

    def get_regularization_loss(self):
        """Return module regularisation loss plus weighted anchor loss when active."""
        loss = super().get_regularization_loss()
        if self.init_boxes is not None and self.anchor_weight > 0:
            loss = loss + self.anchor_weight * self.init_boxes.anchor_loss(self.module)
        return loss

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

