"""MultiBoxEL neural module."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn

from mowl.nn import ELModule
import mowl.nn.el.boxsquaredel.losses as B2L
from mowl.nn.el.boxsquaredel.module import BoxSquaredELModule
from mowl.nn.el.multibox_utils import effective_components as _effective_components
import mowl.nn.el.multiboxel.losses as L
from mowl.nn.el.pointloss import (
    _VALID_OBJECTIVES,
    _validate_beta,
    _validate_ind_halfwidth,
    compute_class_assertion_score_point,
    compute_point_membership_loss,
    compute_point_modelness,
    compute_target_membership,
    gather_element_embeddings,
    membership_bce,
    soft_box_membership,
)
from mowl.nn.el.softagg import _validate_tau, diversity_penalty
from mowl.nn.el.structural import (
    DefinitionTable,
    ExistenceTargets,
    Regions,
    WitnessTargets,
    existence_loss,
    gci0_bot_structural,
    gci0_structural,
    gci1_bot_structural,
    gci1_structural,
    inside_measure,
    nonempty_measure,
    pair_size,
    pair_slots,
    region_size,
    slot_components,
    witness_loss,
)

if TYPE_CHECKING:
    from mowl.datasets.el.points import PointSupervision


_MULTIBOXEL_OBJECTIVES = _VALID_OBJECTIVES + ("structural",)


def _validate_multiboxel_objective(objective: str) -> str:
    """Validate that ``objective`` is a supported MultiBoxEL objective."""
    if not isinstance(objective, str) or objective not in _MULTIBOXEL_OBJECTIVES:
        raise ValueError(
            f"objective must be one of {_MULTIBOXEL_OBJECTIVES}, got {objective!r}"
        )
    return objective


class MultiBoxELModule(ELModule):
    """Multi-component Box\\ :math:`^2`\\ EL neural module.

    Every ontology class :math:`C` is represented as a union of
    :math:`k = \\text{n\\_components}` axis-parallel boxes in :math:`\\mathbb{R}^d`,
    stored in ``class_center`` and ``class_offset`` of shape
    ``(nb_ont_classes, k * d)`` and viewed as ``(B, k, d)``. Each class has a
    single translation bump vector in ``bump_classes`` of shape
    ``(nb_ont_classes, d)`` shared by its components. Roles and optional
    individuals are parameterised identically to
    :class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule`.

    Guarantees:
    - **Exact k = 1 reduction:** When ``n_components == 1`` and ``component_reg == 0.0``
      with weights copied from a :class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule`,
      every GCI loss (positive and negative) and :meth:`regularization_loss`
      agree with :class:`~mowl.nn.el.boxsquaredel.module.BoxSquaredELModule` to
      within floating-point tolerance ``1e-6``.
    - **Effective component count:** :meth:`effective_components` counts, for
      each class on CPU via corner arithmetic, the components whose half-widths
      exceed ``eps`` in every coordinate and whose box is not contained in
      another component of the same class (with exact ties broken by keeping
      the lowest-index component).
    - **Checkpointed temperature buffer:** ``tau`` is stored as a registered
      ``float32`` buffer so it is saved and restored in ``state_dict()`` while
      still accepting ``module.tau = float_value`` assignments.

    Non-guarantees:
    - Role axiom losses (``role_inclusion``, ``role_chain``) are not implemented
      (``role_axiom_capable = False``).
    - Negative sampling is only supported for ``gci2`` and
      ``object_property_assertion`` (``neg_capable_gcis``).
    - Under ``component_aggregation="soft"`` at finite ``tau > 0``, the soft
      loss is not zero at a hard model because :func:`~mowl.nn.el.softagg.softmin`
      (:math:`\\le \\text{mean}`) pulls all target components toward each source
      component in proportion to their softmax weights; ``diversity_reg``
      (:func:`~mowl.nn.el.softagg.diversity_penalty`) exists to counteract this
      collapse pressure.

    Parameters
    ----------
    nb_ont_classes : int
        Number of ontology classes.
    nb_rels : int
        Number of object properties (roles).
    nb_inds : Optional[int], default None
        Number of individuals (ABox).
    embed_dim : int, default 50
        Ambient Euclidean dimension :math:`d`.
    n_components : int, default 2
        Number of box components :math:`k \\ge 1` per class.
    gamma : float, default 0.0
        Inclusion margin :math:`\\gamma`.
    delta : float, default 2.0
        Negative-sampling distance target :math:`\\delta` for ``gci2`` and
        ``object_property_assertion``.
    reg_factor : float, default 0.05
        Weight for the Box²EL bump :math:`L_2` regulariser.
    component_reg : float, default 0.0
        Weight for the anti-collapse pairwise component centre regulariser
        :math:`\\text{mean}_C \\text{mean}_{i<j} \\text{relu}(\\tau - \\|c_{C,i} - c_{C,j}\\|_2)`
        with :math:`\\tau = 0.1`.
    component_aggregation : str, default "hard"
        Component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-assignment temperature stored as a registered buffer on the module.
    diversity_reg : float, default 0.0
        Weight for the pairwise component volume overlap regulariser
        :func:`~mowl.nn.el.softagg.diversity_penalty`.
    objective : str, default "axiom"
        Training objective mode: ``"axiom"``, ``"point"``, ``"mixed"``,
        ``"hybrid"``, or ``"structural"``. Under ``objective="point"`` and
        ``"mixed"``, ABox ``class_assertion`` is still trained in the main loop
        via :math:`-\\log m_C(a)` while ``object_property_assertion`` is zeroed
        (handled via :meth:`point_role_loss`).
    beta : float, default 0.1
        Sigmoid temperature :math:`\\beta > 0` for soft box membership under
        ``objective="point"``, ``"mixed"``, or ``"hybrid"``.
    neg_weight : float, default 0.1
        Weight multiplying random-negative BCE terms under ``objective="point"``,
        ``"mixed"``, or ``"hybrid"``.
    volume_reg : float, default 0.0
        Volume regularisation weight stored on the module for ``objective="point"``,
        ``"mixed"``, or ``"hybrid"``.
    ind_halfwidth : float, optional
        Virtual box half-width around individual targets in role pairs under
        ``objective="point"``, ``"mixed"``, or ``"hybrid"``; defaults to
        ``5.0 * beta``.
    margin : float, default 0.02
        Tightening margin :math:`m` used by ``objective="structural"`` for GCI
        inclusions, ``gci1_bot`` separation, and :func:`~mowl.nn.el.structural.existence_loss`.
        For Box\\ :math:`^2`\\ EL role and assertion scores (where ``gamma`` loosens
        inclusions via :math:`\\text{relu}(\\text{dist} + 2 o - \\gamma)`), positive
        scores pass ``gamma = -margin`` so :math:`m > 0` tightens inclusion, while
        negative scores (:func:`~mowl.nn.el.multiboxel.losses.gci2_score_neg` and
        negative ``object_property_assertion``) keep Box\\ :math:`^2`\\ EL's
        convention (``gamma = +margin``, which relaxes them by the margin), so
        the negative-sampling term is unchanged between the axiom and structural
        objectives.
    max_components : int, optional, default None
        Maximum component slots :math:`K \\ge k` per class under
        ``objective="structural"``; defaults to ``n_components ** 2`` when ``None``.
    structural_definitions : bool, default True
        Whether ``objective="structural"`` compiles acyclic conjunction
        definitions :math:`A \\equiv X \\sqcap Y` via
        :class:`~mowl.nn.el.structural.DefinitionTable`.
    existence_weight : float, default 1.0
        Non-negative weight on :func:`~mowl.nn.el.structural.existence_loss` in
        :meth:`structural_extra_loss`.
    size_reg : float, default 1e-3
        Non-negative weight on the primitive side-length regulariser
        ``mean(2 * |offset|)`` in :meth:`structural_extra_loss`.
    reseed_every : int, default 0
        Epoch interval for heuristic discrete re-seeding (:meth:`reseed`) under
        ``objective="structural"`` (``0`` disables re-seeding).
    bot_agg : {"mean", "sum"}, default "mean"
        Aggregation over active component pairs in :meth:`gci1_bot_loss` under
        ``objective="structural"`` (passed as ``agg`` to
        :func:`~mowl.nn.el.structural.gci1_bot_structural`).
    witness_weight : float, default 0.0
        Non-negative weight on :func:`~mowl.nn.el.structural.witness_loss` in
        :meth:`structural_extra_loss` under ``objective="structural"``. When
        ``witness_weight > 0.0`` and ``targets.witness`` is attached via
        :meth:`set_structural_tables`, allocates ``self.witness_embed`` with
        one learnable point per satisfiable target class.
    reseed_mode : {"dead", "idle"}, default "dead"
        Discrete re-seeding mode for :meth:`reseed` under
        ``objective="structural"``: ``"dead"`` relocates nested or thin
        primitive components to violated conjunction pair partners; ``"idle"``
        requires ``witness_weight > 0`` and relocates the least-used primitive
        leaf components of dead defined classes to their witness points.
    reseed_size : float, default 0.15
        Positive box half-width :math:`> 0` assigned in every coordinate to
        re-seeded components under ``reseed_mode="idle"``.
    reseed_batch : int, default 1
        Maximum number of dead defined classes :math:`\\ge 1` processed per
        :meth:`reseed` call under ``reseed_mode="idle"``.

    References
    ----------
    G. Federico, M. Leemhuis and O. Kutz, "Introducing Multiboxes: Increasing
    the Trustworthiness of Ontology Embeddings", JOWO 2026.
    """

    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})
    role_axiom_capable = False

    def __init__(
        self,
        nb_ont_classes: int,
        nb_rels: int,
        nb_inds: Optional[int] = None,
        embed_dim: int = 50,
        n_components: int = 2,
        gamma: float = 0.0,
        delta: float = 2.0,
        reg_factor: float = 0.05,
        component_reg: float = 0.0,
        component_aggregation: str = "hard",
        tau: float = 1.0,
        diversity_reg: float = 0.0,
        objective: str = "axiom",
        beta: float = 0.1,
        neg_weight: float = 0.1,
        volume_reg: float = 0.0,
        ind_halfwidth: Optional[float] = None,
        margin: float = 0.02,
        max_components: Optional[int] = None,
        structural_definitions: bool = True,
        existence_weight: float = 1.0,
        size_reg: float = 1e-3,
        reseed_every: int = 0,
        bot_agg: str = "mean",
        witness_weight: float = 0.0,
        reseed_mode: str = "dead",
        reseed_size: float = 0.15,
        reseed_batch: int = 1,
    ) -> None:
        if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
            raise ValueError(f"n_components must be an int >= 1, got {n_components!r}")
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim < 1:
            raise ValueError(f"embed_dim must be an int >= 1, got {embed_dim!r}")
        if component_aggregation not in ("hard", "soft"):
            raise ValueError(
                f"component_aggregation must be 'hard' or 'soft', got {component_aggregation!r}"
            )
        tau_f = _validate_tau(tau)
        if (
            isinstance(diversity_reg, bool)
            or not isinstance(diversity_reg, (int, float))
            or float(diversity_reg) < 0.0
        ):
            raise ValueError(f"diversity_reg must be >= 0, got {diversity_reg!r}")
        _validate_multiboxel_objective(objective)
        if objective == "structural" and component_aggregation != "hard":
            raise ValueError(
                "component_aggregation must be 'hard' when objective='structural', "
                f"got {component_aggregation!r}"
            )
        if max_components is None:
            eff_max_components = n_components * n_components
        else:
            if (
                isinstance(max_components, bool)
                or not isinstance(max_components, int)
                or max_components < n_components
            ):
                raise ValueError(
                    f"max_components must be an int >= n_components ({n_components}), got {max_components!r}"
                )
            eff_max_components = int(max_components)
        if not isinstance(structural_definitions, bool):
            raise ValueError(
                f"structural_definitions must be a bool, got {structural_definitions!r}"
            )
        if (
            isinstance(existence_weight, bool)
            or not isinstance(existence_weight, (int, float))
            or float(existence_weight) < 0.0
        ):
            raise ValueError(f"existence_weight must be >= 0, got {existence_weight!r}")
        if (
            isinstance(size_reg, bool)
            or not isinstance(size_reg, (int, float))
            or float(size_reg) < 0.0
        ):
            raise ValueError(f"size_reg must be >= 0, got {size_reg!r}")
        if (
            isinstance(reseed_every, bool)
            or not isinstance(reseed_every, int)
            or reseed_every < 0
        ):
            raise ValueError(f"reseed_every must be an int >= 0, got {reseed_every!r}")
        if not isinstance(bot_agg, str) or bot_agg not in ("mean", "sum"):
            raise ValueError(
                f"bot_agg must be 'mean' or 'sum', got {bot_agg!r}"
            )
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
        beta_f = _validate_beta(beta)
        ind_hw_f = (
            5.0 * beta_f
            if ind_halfwidth is None
            else _validate_ind_halfwidth(ind_halfwidth)
        )
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
        super().__init__()

        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds

        self.embed_dim = embed_dim
        self.n_components = n_components

        self.class_center = self.init_class_embeddings(
            nb_ont_classes, n_components, embed_dim
        )
        self.class_offset = self.init_class_embeddings(
            nb_ont_classes, n_components, embed_dim
        )

        self.head_center = self.init_embeddings(nb_rels, embed_dim)
        self.head_offset = self.init_embeddings(nb_rels, embed_dim)
        self.tail_center = self.init_embeddings(nb_rels, embed_dim)
        self.tail_offset = self.init_embeddings(nb_rels, embed_dim)

        self.bump_classes = self.init_embeddings(nb_ont_classes, embed_dim)

        if self.nb_inds is not None and self.nb_inds > 0:
            self.bump_individuals = self.init_embeddings(nb_inds, embed_dim)
            self.ind_center = self.init_embeddings(nb_inds, embed_dim)
            self.ind_offset = self.init_embeddings(nb_inds, embed_dim)
        else:
            self.bump_individuals = None
            self.ind_center = None
            self.ind_offset = None

        self.gamma = gamma
        self.margin = float(margin)
        self.max_components = eff_max_components
        self.structural_definitions = bool(structural_definitions)
        self.existence_weight = float(existence_weight)
        self.size_reg = float(size_reg)
        self.reseed_every = int(reseed_every)
        self.bot_agg = bot_agg
        self.witness_weight = float(witness_weight)
        self.reseed_mode = reseed_mode
        self.reseed_size = float(reseed_size)
        self.reseed_batch = int(reseed_batch)
        self.witness_embed: Optional[nn.Embedding] = None
        self.definition_table: Optional[DefinitionTable] = None
        self.existence_targets: Optional[ExistenceTargets] = None
        self.top_idx: Optional[int] = None
        self._structural_device: Optional[th.device] = None
        self._def_mask_dev: Optional[th.Tensor] = None
        self._def_x_dev: Optional[th.Tensor] = None
        self._def_y_dev: Optional[th.Tensor] = None
        self._target_classes_dev: Optional[th.Tensor] = None
        self._target_pairs_dev: Optional[th.Tensor] = None
        self._witness_pos_dev: Optional[th.Tensor] = None
        self._witness_neg_dev: Optional[th.Tensor] = None
        self._prim_mask_dev: Optional[th.Tensor] = None
        self._has_prim: bool = False
        self.last_reseed_count: int = 0
        self.delta = delta
        self.reg_factor = reg_factor
        self.component_reg = component_reg
        self.component_aggregation = component_aggregation
        self.register_buffer("tau", th.tensor(tau_f, dtype=th.float32))
        self.diversity_reg = float(diversity_reg)
        self.objective = objective
        self._custom_ind_halfwidth = ind_halfwidth is not None
        self.beta = beta_f
        self.ind_halfwidth = ind_hw_f
        self.neg_weight = float(neg_weight)
        self.volume_reg = float(volume_reg)
        self.point_supervision: Optional[PointSupervision] = None
        self.elem_embed: Optional[nn.Embedding] = None
        self.__dict__["_ind_embed_readonly"] = True

    def __setattr__(self, name: str, value: object) -> None:
        if name == "ind_embed":
            if not self.__dict__.get("_ind_embed_readonly", False):
                return
            raise AttributeError(
                "ind_embed is a read-only property aliasing ind_center."
            )
        if name == "tau" and "_buffers" in self.__dict__ and "tau" in self._buffers:
            tau_f = _validate_tau(value)  # type: ignore[arg-type]
            self._buffers["tau"].fill_(tau_f)
            return
        if name == "objective":
            _validate_multiboxel_objective(value)  # type: ignore[arg-type]
            if (
                value == "structural"
                and self.__dict__.get("component_aggregation", "hard") != "hard"
            ):
                raise ValueError(
                    "component_aggregation must be 'hard' when objective='structural'."
                )
        elif name == "component_aggregation":
            if value not in ("hard", "soft"):
                raise ValueError(
                    f"component_aggregation must be 'hard' or 'soft', got {value!r}"
                )
            if value != "hard" and self.__dict__.get("objective") == "structural":
                raise ValueError(
                    "component_aggregation must be 'hard' when objective='structural'."
                )
        elif name == "bot_agg":
            if not isinstance(value, str) or value not in ("mean", "sum"):
                raise ValueError(
                    f"bot_agg must be 'mean' or 'sum', got {value!r}"
                )
        elif name == "witness_weight":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) < 0.0
            ):
                raise ValueError(f"witness_weight must be >= 0, got {value!r}")
            if float(value) <= 0.0 and self.__dict__.get("reseed_mode") == "idle":
                raise ValueError(
                    "reseed_mode='idle' requires witness_weight > 0, "
                    f"got witness_weight={value!r}"
                )
            value = float(value)
        elif name == "reseed_mode":
            if not isinstance(value, str) or value not in ("dead", "idle"):
                raise ValueError(
                    f"reseed_mode must be 'dead' or 'idle', got {value!r}"
                )
            if (
                value == "idle"
                and "witness_weight" in self.__dict__
                and float(self.__dict__["witness_weight"]) <= 0.0
            ):
                raise ValueError(
                    "reseed_mode='idle' requires witness_weight > 0, "
                    f"got witness_weight={self.__dict__['witness_weight']!r}"
                )
        elif name == "reseed_size":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) <= 0.0
            ):
                raise ValueError(f"reseed_size must be > 0, got {value!r}")
            value = float(value)
        elif name == "reseed_batch":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"reseed_batch must be an int >= 1, got {value!r}"
                )
            value = int(value)
        elif name == "beta":
            beta_f = _validate_beta(value)  # type: ignore[arg-type]
            super().__setattr__(name, beta_f)
            if not self.__dict__.get("_custom_ind_halfwidth", False):
                super().__setattr__("ind_halfwidth", 5.0 * beta_f)
            return
        elif name == "ind_halfwidth":
            hw_f = _validate_ind_halfwidth(value)  # type: ignore[arg-type]
            if "beta" in self.__dict__:
                super().__setattr__("_custom_ind_halfwidth", True)
            super().__setattr__(name, hw_f)
            return
        super().__setattr__(name, value)

    @property
    def ind_embed(self) -> Optional[nn.Embedding]:
        """Read-only alias for ``ind_center`` so individual point lookups use a uniform attribute name.

        Guarantees:
        - Returns the exact same ``nn.Embedding`` instance as ``self.ind_center``
          (or ``None`` when ``nb_inds`` is ``None`` or ``0``) without registering
          a duplicate submodule in ``state_dict()``.

        Non-guarantees:
        - Read-only property; assigning to ``module.ind_embed`` raises :class:`AttributeError`.
        """
        return self.__dict__.get("_modules", {}).get("ind_center", None)

    # Single-component tables (roles, bumps, individuals) use the Box²EL initialiser.
    init_embeddings = BoxSquaredELModule.init_embeddings

    def init_class_embeddings(
        self,
        num_classes: int,
        n_components: int,
        embed_dim: int,
        min: float = -1.0,
        max: float = 1.0,
    ) -> nn.Embedding:
        """Initialise a multi-component class embedding table normalised per component."""
        embeddings = nn.Embedding(num_classes, n_components * embed_dim)
        nn.init.uniform_(embeddings.weight, a=min, b=max)
        if num_classes > 0:
            viewed = embeddings.weight.data.view(num_classes, n_components, embed_dim)
            norms = th.linalg.norm(viewed, dim=-1, keepdim=True)
            viewed.div_(norms)
        embeddings.n_components = n_components
        return embeddings

    def class_boxes(self, idx: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Return ``(centers, offsets)`` of shape ``(B, k, d)`` for class indices ``idx``.

        Guarantees:
        - ``offsets`` is non-negative (computed via ``torch.abs``).

        Non-guarantees:
        - Does not sort or deduplicate components.
        """
        b_size = idx.shape[0]
        cen = self.class_center(idx).view(b_size, self.n_components, self.embed_dim)
        off = th.abs(self.class_offset(idx)).view(
            b_size, self.n_components, self.embed_dim
        )
        return cen, off

    def _sync_structural_device(self, device: th.device) -> None:
        """Stage structural table tensors onto ``device`` once when attached or moved."""
        if self._structural_device == device:
            return
        self._structural_device = device
        table = self.definition_table
        if table is not None:
            self._def_mask_dev = table.def_mask.to(device=device)
            self._def_x_dev = table.def_x.to(device=device)
            self._def_y_dev = table.def_y.to(device=device)
            if table._masked_gci0_keys.device != device:
                table._masked_gci0_keys = table._masked_gci0_keys.to(device=device)
            if table._masked_gci1_keys.device != device:
                table._masked_gci1_keys = table._masked_gci1_keys.to(device=device)
        else:
            self._def_mask_dev = None
            self._def_x_dev = None
            self._def_y_dev = None

        targets = self.existence_targets
        if targets is not None:
            self._target_classes_dev = targets.classes.to(device=device, dtype=th.long)
            self._target_pairs_dev = targets.pairs.to(device=device, dtype=th.long)
            if targets.witness is not None:
                self._witness_pos_dev = targets.witness.pos.to(
                    device=device, dtype=th.long
                )
                self._witness_neg_dev = targets.witness.neg.to(
                    device=device, dtype=th.long
                )
            else:
                self._witness_pos_dev = None
                self._witness_neg_dev = None
        else:
            self._target_classes_dev = None
            self._target_pairs_dev = None
            self._witness_pos_dev = None
            self._witness_neg_dev = None

        if self.witness_embed is not None and self.witness_embed.weight.device != device:
            self.witness_embed = self.witness_embed.to(device=device)

        if self.nb_ont_classes > 0:
            prim_mask = th.ones(self.nb_ont_classes, dtype=th.bool, device=device)
            if self.top_idx is not None and 0 <= self.top_idx < self.nb_ont_classes:
                prim_mask[self.top_idx] = False
            if self._def_mask_dev is not None:
                prim_mask = prim_mask & (~self._def_mask_dev)
            self._prim_mask_dev = prim_mask
            self._has_prim = bool(prim_mask.any().item())
        else:
            self._prim_mask_dev = None
            self._has_prim = False

    def set_structural_tables(
        self,
        table: DefinitionTable,
        targets: ExistenceTargets,
        top_idx: Optional[int] = None,
    ) -> None:
        """Attach :class:`~mowl.nn.el.structural.DefinitionTable` and :class:`~mowl.nn.el.structural.ExistenceTargets`.

        Guarantees:
        - Sets ``self.definition_table``, ``self.existence_targets``, and
          ``self.top_idx``, and stages device copies of the table and target
          index tensors on ``class_center.weight.device``.
        - When ``targets.witness is not None`` and ``self.witness_weight > 0.0``,
          allocates ``self.witness_embed`` of shape ``(W, embed_dim)`` with the
          same dtype, device, and initialiser as ``class_center``, re-allocating
          only if ``self.witness_embed is None`` or the number of witnesses
          ``W`` changes.

        Non-guarantees:
        - Does not re-validate table dimensions against ``nb_ont_classes``.

        Parameters
        ----------
        table : DefinitionTable
            Compiled conjunction definition table.
        targets : ExistenceTargets
            Satisfiable classes, conjunction pairs, and optional witness targets.
        top_idx : Optional[int], default None
            Class index of ``owl:Thing`` to exclude from size regularisation.
        """
        self.definition_table = table
        self.existence_targets = targets
        self.top_idx = top_idx
        if (
            targets is not None
            and targets.witness is not None
            and self.witness_weight > 0.0
        ):
            n_wit = int(targets.witness.classes.numel())
            if (
                self.witness_embed is None
                or self.witness_embed.num_embeddings != n_wit
            ):
                emb = self.init_class_embeddings(n_wit, 1, self.embed_dim)
                self.witness_embed = emb.to(
                    device=self.class_center.weight.device,
                    dtype=self.class_center.weight.dtype,
                )
        self._structural_device = None
        self._sync_structural_device(self.class_center.weight.device)

    def witness_points(self) -> Optional[th.Tensor]:
        """Return detached CPU witness point coordinates of shape ``(W, embed_dim)``, or ``None`` when disabled.

        Guarantees:
        - Returns ``None`` when ``self.witness_embed is None``.
        - Otherwise returns a detached CPU ``torch.Tensor`` of shape
          ``(W, embed_dim)`` with ``requires_grad == False``.

        Non-guarantees:
        - Row order follows ``self.existence_targets.witness.classes``, not
          ``0 .. nb_ont_classes - 1``.
        """
        if self.witness_embed is None:
            return None
        with th.no_grad():
            return self.witness_embed.weight.detach().cpu().clone()

    def class_regions(self, idx: th.Tensor) -> Regions:
        """Materialise :class:`~mowl.nn.el.structural.Regions` with ``K = max_components`` slots for class indices ``idx``.

        Primitive classes occupy slots ``0 .. k - 1`` with ``w = |offset| >= 0``
        (active) and pad slots ``k .. K - 1`` with ``cen = 0, w = -1``
        (inactive). Defined classes in ``self.definition_table`` are computed
        recursively via :func:`~mowl.nn.el.structural.pair_slots` from their
        definers ``(def_x, def_y)``.

        Guarantees:
        - Returns :class:`~mowl.nn.el.structural.Regions` of shapes
          ``(B, K, d)`` for ``cen`` and ``w`` and ``(B, K)`` for ``active``.
        - Inactive slots always have ``cen = 0`` and ``w = -1``.

        Non-guarantees:
        - Active computed slots may be geometrically empty (``w[..., m] < 0``).

        Parameters
        ----------
        idx : torch.Tensor
            1-D integer tensor of class indices of length ``B``.

        Returns
        -------
        Regions
            Materialised class regions.
        """
        device = self.class_center.weight.device
        dtype = self.class_center.weight.dtype
        idx_dev = (
            idx
            if (idx.device == device and idx.dtype == th.long)
            else idx.to(device=device, dtype=th.long)
        )
        b_size = int(idx_dev.shape[0])
        k = self.n_components
        max_k = self.max_components
        dim = self.embed_dim

        if b_size == 0:
            return Regions(
                cen=th.zeros((0, max_k, dim), dtype=dtype, device=device),
                w=-th.ones((0, max_k, dim), dtype=dtype, device=device),
                active=th.zeros((0, max_k), dtype=th.bool, device=device),
            )

        raw_cen, raw_off = self.class_boxes(idx_dev)
        if max_k > k:
            pad_n = max_k - k
            pad_cen = th.zeros((b_size, pad_n, dim), dtype=dtype, device=device)
            pad_w = -th.ones((b_size, pad_n, dim), dtype=dtype, device=device)
            cen = th.cat([raw_cen, pad_cen], dim=1)
            w = th.cat([raw_off, pad_w], dim=1)
            active = th.cat(
                [
                    th.ones((b_size, k), dtype=th.bool, device=device),
                    th.zeros((b_size, pad_n), dtype=th.bool, device=device),
                ],
                dim=1,
            )
        else:
            cen = raw_cen
            w = raw_off
            active = th.ones((b_size, k), dtype=th.bool, device=device)

        table = self.definition_table
        if table is not None and table.n_defined > 0:
            self._sync_structural_device(device)
            assert self._def_mask_dev is not None
            assert self._def_x_dev is not None
            assert self._def_y_dev is not None
            is_def = self._def_mask_dev[idx_dev]
            if bool(is_def.any().item()):
                def_idx = idx_dev[is_def]
                dx = self._def_x_dev[def_idx]
                dy = self._def_y_dev[def_idx]
                rx = self.class_regions(dx)
                ry = self.class_regions(dy)
                r_def = pair_slots(rx, ry, max_k)
                cen = cen.clone()
                w = w.clone()
                active = active.clone()
                cen[is_def] = r_def.cen
                w[is_def] = r_def.w
                active[is_def] = r_def.active

        return Regions(cen=cen, w=w, active=active)

    def all_class_regions(self) -> Regions:
        """Return materialised :class:`~mowl.nn.el.structural.Regions` for all classes on CPU without gradients.

        Guarantees:
        - Returns :class:`~mowl.nn.el.structural.Regions` of shape
          ``(nb_ont_classes, max_components, embed_dim)`` on CPU with
          ``requires_grad == False``.

        Non-guarantees:
        - Evaluates all ``nb_ont_classes`` in a single batch.
        """
        with th.no_grad():
            idx = th.arange(
                self.nb_ont_classes,
                device=self.class_center.weight.device,
                dtype=th.long,
            )
            r = self.class_regions(idx)
            return Regions(
                cen=r.cen.detach().cpu(),
                w=r.w.detach().cpu(),
                active=r.active.detach().cpu(),
            )

    def structural_extra_loss(self) -> th.Tensor:
        """Compute structural existence loss, primitive width regularisation, and optional witness loss.

        Evaluates ``existence_weight * existence_loss(cls_sizes, pair_sizes, margin)``
        plus ``size_reg * mean(2 * |offset|)`` over primitive (raw) slots of all
        classes except ``owl:Thing`` (``self.top_idx``), plus
        ``witness_weight * witness_loss(inside_pos, inside_neg, margin)`` when
        ``witness_weight > 0.0`` and witness targets/embeddings are present.

        Guarantees:
        - Returns a scalar ``torch.Tensor`` on the module's current device.
        - Excludes ``self.top_idx`` and accepted defined classes from the size
          regularisation term.
        - Evaluates witness membership on materialised :meth:`class_regions`,
          so defined classes propagate gradients to their primitive definers.

        Non-guarantees:
        - When ``self.existence_targets is None``, the existence and witness
          terms are ``0.0``.
        """
        device = self.class_center.weight.device
        dtype = self.class_center.weight.dtype
        self._sync_structural_device(device)

        targets = self.existence_targets
        if targets is not None and self.existence_weight > 0.0:
            assert self._target_classes_dev is not None
            assert self._target_pairs_dev is not None
            if self._target_classes_dev.numel() > 0:
                r_cls = self.class_regions(self._target_classes_dev)
                cls_sizes = region_size(r_cls)
            else:
                cls_sizes = th.empty(0, dtype=dtype, device=device)

            if self._target_pairs_dev.numel() > 0:
                r_a = self.class_regions(self._target_pairs_dev[:, 0])
                r_b = self.class_regions(self._target_pairs_dev[:, 1])
                p_sizes = pair_size(r_a, r_b)
            else:
                p_sizes = th.empty(0, dtype=dtype, device=device)

            ex_term = self.existence_weight * existence_loss(
                cls_sizes, p_sizes, self.margin
            )
        else:
            ex_term = th.zeros((), dtype=dtype, device=device)

        if self.size_reg > 0.0 and self._has_prim:
            assert self._prim_mask_dev is not None
            raw_off = th.abs(self.class_offset.weight[self._prim_mask_dev])
            size_term = self.size_reg * (2.0 * raw_off).mean()
        else:
            size_term = th.zeros((), dtype=dtype, device=device)

        total = ex_term + size_term
        if (
            targets is not None
            and targets.witness is not None
            and self.witness_weight > 0.0
            and self.witness_embed is not None
        ):
            assert self._witness_pos_dev is not None
            assert self._witness_neg_dev is not None
            if self._witness_pos_dev.numel() > 0:
                w_pos = self.witness_embed(self._witness_pos_dev[:, 0])
                r_pos = self.class_regions(self._witness_pos_dev[:, 1])
                in_pos = inside_measure(w_pos, r_pos)
            else:
                in_pos = th.empty(0, dtype=dtype, device=device)

            if self._witness_neg_dev.numel() > 0:
                w_neg = self.witness_embed(self._witness_neg_dev[:, 0])
                r_neg = self.class_regions(self._witness_neg_dev[:, 1])
                in_neg = inside_measure(w_neg, r_neg)
            else:
                in_neg = th.empty(0, dtype=dtype, device=device)

            total = total + self.witness_weight * witness_loss(
                in_pos, in_neg, self.margin
            )

        return total

    def reseed(self, rng: np.random.Generator) -> int:
        """Relocate dead or idle primitive components under ``no_grad``.

        Dispatches on ``self.reseed_mode``:

        * ``"dead"`` (round-4 §5):
          1. Identifies dead primitive components (nested inside another active
             component of the same class, with exact ties broken by lower index,
             or ``min_m |offset| < 1e-3``).
          2. Processes violated pair targets ``(C, D)`` (``NE(C cap D) < margin``)
             in ``rng.permutation`` order, using each dead component at most once
             and placing it at the partner's largest active non-empty slot with
             half its width.
          3. Clamps the largest component of any primitive class target still
             satisfying ``NE(C) < margin`` to ``max(|offset|, margin)``.
        * ``"idle"`` (round-5 §4, requires ``witness_weight > 0``):
          1. Finds dead accepted defined classes :math:`A` with
             :math:`\\text{NE}(A) = \\max_{s \\text{ active}} \\text{ne}(A_s) < \\text{margin}`.
          2. Counts component usage ``u[X, c]`` over primitive :math:`X` and
             component :math:`c` across accepted defined classes :math:`A'` with
             :math:`\\text{NE}(A') \\ge \\text{margin}` whose best slot
             (:math:`\\arg\\max \\text{ne}`) uses component :math:`c` of :math:`X`
             via :func:`~mowl.nn.el.structural.slot_components`.
          3. Processes dead defined classes in ``rng.permutation`` order, at most
             ``self.reseed_batch`` per call: for each primitive leaf :math:`X`
             of :math:`A`'s definition tree (deduplicated), if some active
             component of :math:`X` already contains :math:`A`'s witness point
             :math:`p` by at least ``margin`` (:math:`\\text{inside}(p, X_c) \\ge \\text{margin}`),
             skips :math:`X`; otherwise selects :math:`c^* = \\arg\\min_c u[X, c]`
             (lowest index on ties), sets
             ``center[X, c*] := p + 0.01 * N(0, I)`` (drawn from ``rng``) and
             ``offset[X, c*] := reseed_size`` in every coordinate, and increments
             ``u[X, c*] += 1`` and the move counter.
          4. Runs the primitive class target clamp step (clamping the largest
             component of any primitive target with ``NE(C) < margin`` to
             ``max(|offset|, margin)``).

        Guarantees:
        - Returns the total number of re-seeding moves and stores it in
          ``self.last_reseed_count``.
        - Never modifies defined classes' raw rows.
        - Under ``reseed_mode="idle"``, raises :class:`ValueError` if witness
          targets or ``witness_embed`` are not attached.

        Non-guarantees:
        - Does not reset optimiser moments for moved rows.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded NumPy random generator for permuting targets and sampling
            idle re-seed noise.

        Returns
        -------
        int
            Number of re-seeded components.
        """
        targets = self.existence_targets
        if self.reseed_mode == "idle":
            if (
                self.witness_weight <= 0.0
                or self.witness_embed is None
                or targets is None
                or targets.witness is None
            ):
                raise ValueError(
                    "reseed_mode='idle' requires witness_weight > 0 and attached witness targets."
                )
            if self.nb_ont_classes == 0:
                self.last_reseed_count = 0
                return 0

            device = self.class_center.weight.device
            dtype = self.class_center.weight.dtype
            n = self.nb_ont_classes
            k = self.n_components
            dim = self.embed_dim

            with th.no_grad():
                cen_view = self.class_center.weight.data.view(n, k, dim)
                off_view = self.class_offset.weight.data.view(n, k, dim)

                table = self.definition_table
                if table is not None:
                    def_mask_np = table.def_mask.detach().cpu().numpy()
                else:
                    def_mask_np = np.zeros(n, dtype=bool)

                moves = 0
                if table is not None and table.n_defined > 0:
                    wit_classes_list = targets.witness.classes.detach().cpu().tolist()
                    cls_to_wit_row: Dict[int, int] = {
                        int(c): idx for idx, c in enumerate(wit_classes_list)
                    }

                    def_indices = [int(a) for a in np.flatnonzero(def_mask_np)]
                    def_idx_dev = th.tensor(def_indices, device=device, dtype=th.long)
                    r_def = self.class_regions(def_idx_dev)
                    ne_all = nonempty_measure(r_def.w)
                    ne_masked = ne_all.masked_fill(~r_def.active, float("-inf"))
                    best_slots = th.argmax(ne_masked, dim=1).detach().cpu().tolist()
                    best_nes = th.amax(ne_masked, dim=1).detach().cpu().tolist()

                    u = np.zeros((n, k), dtype=np.int64)
                    dead_defined: List[int] = []
                    for a_cls, best_s, best_ne in zip(
                        def_indices, best_slots, best_nes
                    ):
                        if float(best_ne) >= self.margin:
                            comp_map = slot_components(
                                table, int(a_cls), int(best_s), k
                            )
                            for prim_x, comp_c in comp_map.items():
                                if 0 <= prim_x < n and 0 <= comp_c < k:
                                    u[prim_x, comp_c] += 1
                        elif int(a_cls) in cls_to_wit_row:
                            dead_defined.append(int(a_cls))

                    if dead_defined:
                        perm = rng.permutation(len(dead_defined))
                        batch_count = min(len(dead_defined), self.reseed_batch)
                        for p_idx in perm[:batch_count]:
                            a_cls = dead_defined[int(p_idx)]
                            w_row = cls_to_wit_row[a_cls]
                            p = self.witness_embed.weight.data[w_row]
                            leaves = list(slot_components(table, a_cls, 0, k).keys())
                            for x_cls in leaves:
                                w_x = th.abs(off_view[x_cls])
                                ins_x = th.amin(
                                    w_x - th.abs(p.unsqueeze(0) - cen_view[x_cls]),
                                    dim=-1,
                                )
                                if bool((ins_x >= self.margin).any().item()):
                                    continue
                                c_star = int(np.argmin(u[x_cls]))
                                noise = (
                                    th.as_tensor(
                                        rng.standard_normal(dim),
                                        dtype=dtype,
                                        device=device,
                                    )
                                    * 0.01
                                )
                                cen_view[x_cls, c_star].copy_(p + noise)
                                off_view[x_cls, c_star].fill_(float(self.reseed_size))
                                u[x_cls, c_star] += 1
                                moves += 1

                if targets.classes.numel() > 0:
                    for c_raw in targets.classes.detach().cpu().tolist():
                        c_idx = int(c_raw)
                        if def_mask_np[c_idx]:
                            continue
                        w_c = th.abs(off_view[c_idx])
                        ne_c = nonempty_measure(w_c)
                        if float(ne_c.max().item()) < self.margin:
                            best_j = int(th.argmax(ne_c).item())
                            off_view[c_idx, best_j].copy_(
                                th.clamp(w_c[best_j], min=float(self.margin))
                            )
                            moves += 1

                self.last_reseed_count = moves
                return moves

        if targets is None or self.nb_ont_classes == 0:
            self.last_reseed_count = 0
            return 0

        eps = 1e-3
        device = self.class_center.weight.device
        n = self.nb_ont_classes
        k = self.n_components
        dim = self.embed_dim

        with th.no_grad():
            cen_view = self.class_center.weight.data.view(n, k, dim)
            off_view = self.class_offset.weight.data.view(n, k, dim)

            cen_np = cen_view.detach().cpu().numpy()
            off_np = th.abs(off_view.detach()).cpu().numpy()
            lo_np = cen_np - off_np
            hi_np = cen_np + off_np
            valid_np = np.all(off_np >= eps, axis=-1)

            if self.definition_table is not None:
                def_mask_np = self.definition_table.def_mask.detach().cpu().numpy()
            else:
                def_mask_np = np.zeros(n, dtype=bool)

            dead_queues: Dict[int, List[int]] = {}
            for c in range(n):
                if def_mask_np[c]:
                    dead_queues[c] = []
                    continue
                v = valid_np[c]
                lo_c = lo_np[c]
                hi_c = hi_np[c]
                dead_c: List[int] = []
                for i in range(k):
                    if not v[i]:
                        dead_c.append(i)
                        continue
                    contained = False
                    for j in range(k):
                        if i == j or not v[j]:
                            continue
                        if np.all(lo_c[j] <= lo_c[i]) and np.all(hi_c[i] <= hi_c[j]):
                            equal_boxes = np.all(lo_c[i] <= lo_c[j]) and np.all(
                                hi_c[j] <= hi_c[i]
                            )
                            if not equal_boxes or j < i:
                                contained = True
                                break
                    if contained:
                        dead_c.append(i)
                dead_queues[c] = dead_c

            moves = 0
            if targets.pairs.numel() > 0:
                pairs_dev = targets.pairs.to(device=device, dtype=th.long)
                r_a = self.class_regions(pairs_dev[:, 0])
                r_b = self.class_regions(pairs_dev[:, 1])
                ps = pair_size(r_a, r_b)
                violated_mask = (ps < self.margin).detach().cpu().numpy()
                violated_indices = np.flatnonzero(violated_mask)

                if violated_indices.size > 0:
                    perm = rng.permutation(violated_indices.size)
                    pairs_list = targets.pairs.detach().cpu().tolist()
                    for p_idx in violated_indices[perm]:
                        c_idx, d_idx = int(pairs_list[p_idx][0]), int(pairs_list[p_idx][1])
                        if dead_queues.get(c_idx):
                            mover_cls = c_idx
                            partner_cls = d_idx
                        elif dead_queues.get(d_idx):
                            mover_cls = d_idx
                            partner_cls = c_idx
                        else:
                            continue

                        r_p = self.class_regions(
                            th.tensor([partner_cls], device=device, dtype=th.long)
                        )
                        ne_p = nonempty_measure(r_p.w[0])
                        ne_p_masked = ne_p.masked_fill(~r_p.active[0], float("-inf"))
                        best_j = int(th.argmax(ne_p_masked).item())
                        best_ne = float(ne_p_masked[best_j].item())
                        if best_ne <= 0.0:
                            continue

                        dead_slot = dead_queues[mover_cls].pop(0)
                        cen_view[mover_cls, dead_slot].copy_(r_p.cen[0, best_j])
                        off_view[mover_cls, dead_slot].copy_(
                            0.5 * th.abs(r_p.w[0, best_j])
                        )
                        moves += 1

            if targets.classes.numel() > 0:
                for c_raw in targets.classes.detach().cpu().tolist():
                    c_idx = int(c_raw)
                    if def_mask_np[c_idx]:
                        continue
                    w_c = th.abs(off_view[c_idx])
                    ne_c = nonempty_measure(w_c)
                    if float(ne_c.max().item()) < self.margin:
                        best_j = int(th.argmax(ne_c).item())
                        off_view[c_idx, best_j].copy_(
                            th.clamp(w_c[best_j], min=float(self.margin))
                        )
                        moves += 1

            self.last_reseed_count = moves
            return moves

    def set_point_supervision(self, ps: PointSupervision) -> None:
        """Attach :class:`~mowl.datasets.el.points.PointSupervision` and lazily create ``elem_embed``.

        Guarantees:
        - Sets ``self.point_supervision = ps``.
        - Allocates ``self.elem_embed`` of shape ``(ps.n_elements, embed_dim)`` on
          the same device and dtype as ``class_center.weight``, with rows for
          individual-backed elements never indexed from ``elem_embed``.

        Non-guarantees:
        - Calling again with a different ``ps.n_elements`` re-allocates ``elem_embed``.

        Parameters
        ----------
        ps : PointSupervision
            Canonical-element supervision table.
        """
        self.point_supervision = ps
        device = self.class_center.weight.device
        dtype = self.class_center.weight.dtype
        if self.elem_embed is None or self.elem_embed.num_embeddings != ps.n_elements:
            self.elem_embed = self.init_embeddings(
                ps.n_elements, self.embed_dim
            ).to(device=device, dtype=dtype)

    def element_embeddings(self, elem_idx: th.Tensor) -> th.Tensor:
        """Return point coordinates ``(B, d)`` for ``elem_idx``, aliasing individual elements to ``ind_embed``.

        Guarantees:
        - Class-element rows are gathered from ``self.elem_embed`` and
          individual-element rows from ``self.ind_embed``, preserving separate
          gradient flows while keeping ``ind_offset`` untouched.

        Non-guarantees:
        - Raises :class:`RuntimeError` if :meth:`set_point_supervision` has not
          been called.

        Parameters
        ----------
        elem_idx : torch.Tensor
            1-D integer tensor of canonical element indices of length ``B``.

        Returns
        -------
        torch.Tensor
            Element coordinates of shape ``(B, embed_dim)``.
        """
        if self.point_supervision is None or self.elem_embed is None:
            raise RuntimeError(
                "PointSupervision has not been set on this module; call set_point_supervision(ps) first."
            )
        return gather_element_embeddings(
            elem_idx, self.point_supervision, self.elem_embed, self.ind_embed
        )

    def element_bumps(self, elem_idx: th.Tensor) -> th.Tensor:
        """Return Box²EL bump vectors ``(B, d)`` of each element's own class or individual.

        Guarantees:
        - Looks up ``self.bump_classes`` for class elements and
          ``self.bump_individuals`` for individual elements according to
          ``self.point_supervision``.

        Non-guarantees:
        - Raises :class:`RuntimeError` if :meth:`set_point_supervision` has not
          been called, or :class:`ValueError` if an individual element is indexed
          when ``bump_individuals is None``.

        Parameters
        ----------
        elem_idx : torch.Tensor
            1-D integer tensor of canonical element indices of length ``B``.

        Returns
        -------
        torch.Tensor
            Bump vectors of shape ``(B, embed_dim)``.
        """
        if self.point_supervision is None:
            raise RuntimeError(
                "PointSupervision has not been set on this module; call set_point_supervision(ps) first."
            )
        ps = self.point_supervision
        device = self.bump_classes.weight.device
        dtype = self.bump_classes.weight.dtype
        idx = elem_idx.to(device=device, dtype=th.long)
        if idx.numel() == 0:
            return th.empty((0, self.embed_dim), device=device, dtype=dtype)

        is_ind = ps.element_is_individual.to(device=device)[idx]
        cls_or_ind = ps.element_class_index.to(device=device)[idx]
        if not bool(is_ind.any().item()):
            return self.bump_classes(cls_or_ind)
        if self.bump_individuals is None:
            raise ValueError(
                "PointSupervision contains individual elements, but bump_individuals is None."
            )
        if bool(is_ind.all().item()):
            return self.bump_individuals(cls_or_ind)

        out = th.empty((idx.shape[0], self.embed_dim), device=device, dtype=dtype)
        out[~is_ind] = self.bump_classes(cls_or_ind[~is_ind])
        out[is_ind] = self.bump_individuals(cls_or_ind[is_ind])
        return out

    def point_membership_loss(
        self,
        batch: Union[Mapping[str, th.Tensor], th.Tensor],
        neg: bool = False,
        weight: Optional[float] = None,
    ) -> th.Tensor:
        """Compute point-membership BCE loss.

        Guarantees:
        - Accepts either a ``(B, 2)`` integer tensor ``[elem_idx, class_idx]`` or
          a sample dictionary from :meth:`~mowl.datasets.el.points.PointSupervision.sample`.
        - Differentiable with respect to ``elem_embed``, ``ind_center``,
          ``class_center``, and ``class_offset``.

        Non-guarantees:
        - Raises :class:`RuntimeError` if :meth:`set_point_supervision` has not
          been called.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor] or torch.Tensor
            Sample dictionary or ``(B, 2)`` pair tensor.
        neg : bool, default False
            When ``batch`` is a tensor, whether targets are ``0.0`` (``True``) or ``1.0`` (``False``).
        weight : float, optional
            Optional sample weight override.

        Returns
        -------
        torch.Tensor
            Scalar BCE membership loss.
        """
        return compute_point_membership_loss(self, batch, neg=neg, weight=weight)

    def target_membership(self, points: th.Tensor, elem_y: th.Tensor) -> th.Tensor:
        """Compute soft membership of ``points`` ``(B, d)`` in target element ``elem_y`` ``(B,)``.

        Uses ``self.class_boxes`` for class targets and a virtual box of
        half-width ``self.ind_halfwidth`` around ``self.ind_embed`` for
        individual targets.
        """
        return compute_target_membership(self, points, elem_y)

    def point_role_loss(
        self,
        batch: Union[Mapping[str, th.Tensor], th.Tensor],
        neg: bool = False,
        weight: Optional[float] = None,
    ) -> th.Tensor:
        """Compute canonical-element role BCE loss via Box²EL head/tail boxes and element bumps.

        For each role triple ``(x_C, r, y_D)``:
        - Head term: :math:`m_{H_r}(x + b_D)`.
        - Tail term: :math:`m_{T_r}(y + b_C)` when ``y_D`` is a class element, or
          soft membership of the translated tail centre :math:`c_{T_r} - b_C` in
          the virtual box of half-width ``self.ind_halfwidth`` around
          ``self.ind_embed[b]`` when ``y_D`` is an individual target.
        - Combined role membership: :math:`m_{\\text{head}} \\cdot m_{\\text{tail}}`.

        Guarantees:
        - Accepts either a ``(B, 3)`` integer tensor ``[elem_x, r_idx, elem_y]``
          or a sample dictionary with key ``"roles"``.
        - Differentiable with respect to element points, element bumps, and
          role head/tail box parameters, without accessing ``ind_offset``.
        - Uses ``self.ind_halfwidth`` for individual targets exactly as
          :class:`~mowl.nn.el.multielbe.module.MultiELBEModule` does.

        Non-guarantees:
        - Raises :class:`RuntimeError` if :meth:`set_point_supervision` has not
          been called.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor] or torch.Tensor
            Role triple tensor of shape ``(B, 3)`` or sample dictionary.
        neg : bool, default False
            Whether targets are ``0.0`` (``True``) or ``1.0`` (``False``).
        weight : float, optional
            Optional sample weight override.

        Returns
        -------
        torch.Tensor
            Scalar BCE role loss.
        """
        if self.point_supervision is None or self.elem_embed is None:
            raise RuntimeError(
                "PointSupervision has not been set on this module; call set_point_supervision(ps) first."
            )
        device = self.class_center.weight.device
        dtype = self.class_center.weight.dtype
        roles = batch.get("roles") if isinstance(batch, Mapping) else batch
        if roles is None or roles.numel() == 0:
            return th.zeros((), device=device, dtype=dtype)

        r_tensor = roles.to(device=device, dtype=th.long)
        elem_x = r_tensor[:, 0]
        rel_idx = r_tensor[:, 1]
        elem_y = r_tensor[:, 2]

        x = self.element_embeddings(elem_x)
        b_c = self.element_bumps(elem_x)
        b_d = self.element_bumps(elem_y)

        cen_h = self.head_center(rel_idx).unsqueeze(1)
        off_h = th.abs(self.head_offset(rel_idx)).unsqueeze(1)
        cen_t = self.tail_center(rel_idx)
        off_t = th.abs(self.tail_offset(rel_idx)).unsqueeze(1)

        m_head = soft_box_membership(x + b_d, cen_h, off_h, beta=self.beta).squeeze(1)

        ps = self.point_supervision
        is_ind_y = ps.element_is_individual.to(device=device)[elem_y]
        if not bool(is_ind_y.any().item()):
            y = self.element_embeddings(elem_y)
            m_tail = soft_box_membership(
                y + b_c, cen_t.unsqueeze(1), off_t, beta=self.beta
            ).squeeze(1)
        elif bool(is_ind_y.all().item()):
            m_tail = self.target_membership(cen_t - b_c, elem_y)
        else:
            b_size = int(elem_y.shape[0])
            m_tail = th.empty((b_size,), device=device, dtype=dtype)
            cls_mask = ~is_ind_y
            y_cls = self.element_embeddings(elem_y[cls_mask])
            m_tail[cls_mask] = soft_box_membership(
                y_cls + b_c[cls_mask],
                cen_t[cls_mask].unsqueeze(1),
                off_t[cls_mask],
                beta=self.beta,
            ).squeeze(1)
            m_tail[is_ind_y] = self.target_membership(
                cen_t[is_ind_y] - b_c[is_ind_y], elem_y[is_ind_y]
            )

        m_role = m_head * m_tail

        target = 0.0 if neg else 1.0
        eff_w = (self.neg_weight if neg else 1.0) if weight is None else float(weight)
        return membership_bce(m_role, target, weight=eff_w)

    def class_assertion_score_point(self, data: th.Tensor) -> th.Tensor:
        """Compute point-mode class assertion score :math:`-\\log m_C(a)` for ``data`` ``(B, 2)``.

        Guarantees:
        - Returns a 1-D tensor of shape ``(B,)`` using ``self.ind_center``
          without touching ``self.ind_offset``.

        Non-guarantees:
        - Raises :class:`ValueError` if ``nb_inds`` was ``None`` or ``0``.

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, 2)`` with columns ``[ind_idx, class_idx]``.

        Returns
        -------
        torch.Tensor
            Score tensor of shape ``(B,)`` where lower values indicate higher membership.
        """
        return compute_class_assertion_score_point(self, data)

    def point_modelness(
        self,
        ps: Optional[PointSupervision] = None,
        chunk_size: int = 8192,
    ) -> Dict[str, float]:
        """Return ``dict(pos_frac=..., hardneg_frac=...)`` evaluated in chunks on the current device.

        Guarantees:
        - Evaluates all positive and hard-negative pairs under ``torch.no_grad()``
          in chunks of size ``chunk_size`` on the module's current device.

        Non-guarantees:
        - Does not evaluate role pairs or random negatives.

        Parameters
        ----------
        ps : PointSupervision, optional
            Supervision object (defaults to ``self.point_supervision``).
        chunk_size : int, default 8192
            Maximum number of pairs evaluated per forward chunk.

        Returns
        -------
        Dict[str, float]
            Dictionary with keys ``"pos_frac"`` and ``"hardneg_frac"``.
        """
        return compute_point_modelness(self, ps=ps, chunk_size=chunk_size)

    def _zero_loss(self, data: th.Tensor) -> th.Tensor:
        return th.zeros(
            (data.shape[0],),
            device=self.class_center.weight.device,
            dtype=self.class_center.weight.dtype,
        )

    def gci0_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI0 loss (:math:`C \\sqsubseteq D`), dispatching to :func:`~mowl.nn.el.structural.gci0_structural` under ``objective="structural"``.

        Guarantees:
        - Under ``objective="structural"``, masks out defining rows from
          ``self.definition_table`` to exact ``0.0``.

        Non-guarantees:
        - Under ``objective="structural"``, vacuously ``0.0`` when :math:`C` has
          no active non-empty slots.
        """
        if self.objective == "structural":
            r_c = self.class_regions(data[:, 0])
            r_d = self.class_regions(data[:, 1])
            out = gci0_structural(r_c, r_d, self.margin)
            if self.definition_table is not None:
                masked = self.definition_table.masked_gci0(data)
                out = th.where(masked, th.zeros_like(out), out)
            return out
        if self.training and self.objective in ("point", "mixed"):
            return self._zero_loss(data)
        return L.gci0_loss(
            data,
            self.class_center,
            self.class_offset,
            self.gamma,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci0_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI0-bot loss (:math:`C \\sqsubseteq \\bot`), dispatching to :func:`~mowl.nn.el.structural.gci0_bot_structural` under ``objective="structural"``."""
        if self.objective == "structural":
            r_c = self.class_regions(data[:, 0])
            return gci0_bot_structural(r_c)
        if self.training and self.objective == "point":
            return self._zero_loss(data)
        return L.gci0_bot_loss(
            data,
            self.class_offset,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci1_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI1 loss (:math:`C \\sqcap D \\sqsubseteq E`), dispatching to :func:`~mowl.nn.el.structural.gci1_structural` under ``objective="structural"``."""
        if self.objective == "structural":
            r_c = self.class_regions(data[:, 0])
            r_d = self.class_regions(data[:, 1])
            r_e = self.class_regions(data[:, 2])
            masked = (
                self.definition_table.masked_gci1(data)
                if self.definition_table is not None
                else None
            )
            return gci1_structural(r_c, r_d, r_e, self.margin, masked=masked)
        if self.training and self.objective in ("point", "mixed"):
            return self._zero_loss(data)
        return L.gci1_loss(
            data,
            self.class_center,
            self.class_offset,
            self.gamma,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci1_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI1-bot loss (:math:`C \\sqcap D \\sqsubseteq \\bot`).

        Under ``objective="structural"``, gates active slots of :math:`C` and
        :math:`D` by detached non-emptiness
        (``r.active & (nonempty_measure(r.w.detach()) > 0.0)``) before calling
        :func:`~mowl.nn.el.structural.gci1_bot_structural` with ``agg=self.bot_agg``.

        Guarantees:
        - At ``margin = 0``, the ``ne > 0`` slot gate has the identical zero set
          to gating on ``active`` alone (every pair of non-empty active slots is
          disjoint up to boundary contact).

        Non-guarantees:
        - Deviates from design §4 (which gates on ``active`` alone) at
          ``margin > 0``: empty computed slots (e.g. from defined classes) with
          :math:`-\\text{margin} < \\text{ne} \\le 0` are gated out rather than
          pushed below :math:`-\\text{margin}`.
        """
        if self.objective == "structural":
            r_c = self.class_regions(data[:, 0])
            r_d = self.class_regions(data[:, 1])
            act_c = r_c.active & (nonempty_measure(r_c.w.detach()) > 0.0)
            act_d = r_d.active & (nonempty_measure(r_d.w.detach()) > 0.0)
            return gci1_bot_structural(
                Regions(cen=r_c.cen, w=r_c.w, active=act_c),
                Regions(cen=r_d.cen, w=r_d.w, active=act_d),
                self.margin,
                agg=self.bot_agg,
            )
        if self.training and self.objective == "point":
            return self._zero_loss(data)
        return L.gci1_bot_loss(
            data,
            self.class_center,
            self.class_offset,
            self.gamma,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci2_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI2 loss (:math:`C \\sqsubseteq \\exists r.D`), masking inactive or empty slots under ``objective="structural"``.

        Guarantees:
        - Under ``objective="structural"``, evaluates :func:`~mowl.nn.el.multiboxel.losses.gci2_score`
          with tightening ``gamma = -self.margin`` when ``neg=False``, or
          :func:`~mowl.nn.el.multiboxel.losses.gci2_score_neg` with
          ``gamma = +self.margin`` when ``neg=True``, using ``aggregation="hard"``
          and slot masks ``active & (ne(w) > 0)``.

        Non-guarantees:
        - Under ``objective="structural"``, when every target slot of :math:`D`
          is masked out (an all-empty, i.e. unsatisfiable, target :math:`D`),
          the target minimum term in :func:`~mowl.nn.el.multiboxel.losses.gci2_score`
          returns ``0.0`` (and when every source slot of :math:`C` is also
          masked out, the positive score is vacuously ``0.0``).
        - Under ``objective="structural"`` with ``neg=True``, when every source
          slot of :math:`C` is masked out (an all-empty source), the source half
          of :func:`~mowl.nn.el.multiboxel.losses.gci2_score_neg` returns ``0.0``
          (and ``0.0`` overall when every target slot of :math:`D` is also masked out).
        """
        if self.objective == "structural":
            r_c = self.class_regions(data[:, 0])
            r_d = self.class_regions(data[:, 2])
            act_c = r_c.active & (nonempty_measure(r_c.w) > 0.0)
            act_d = r_d.active & (nonempty_measure(r_d.w) > 0.0)
            bump_c = self.bump_classes(data[:, 0])
            bump_d = self.bump_classes(data[:, 2])
            cen_head = self.head_center(data[:, 1])
            off_head = th.abs(self.head_offset(data[:, 1]))
            cen_tail = self.tail_center(data[:, 1])
            off_tail = th.abs(self.tail_offset(data[:, 1]))
            fn = L.gci2_score_neg if neg else L.gci2_score
            role_gamma = self.margin if neg else -self.margin
            return fn(
                r_c.cen,
                th.abs(r_c.w),
                bump_c,
                r_d.cen,
                th.abs(r_d.w),
                bump_d,
                cen_head,
                off_head,
                cen_tail,
                off_tail,
                role_gamma,
                self.delta,
                aggregation="hard",
                tau=self.tau,
                active_c=act_c,
                active_d=act_d,
            )
        if self.training and self.objective in ("point", "mixed"):
            return self._zero_loss(data)
        return L.gci2_loss(
            data,
            self.class_center,
            self.class_offset,
            self.head_center,
            self.head_offset,
            self.tail_center,
            self.tail_offset,
            self.bump_classes,
            self.gamma,
            self.delta,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci3_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI3 loss (:math:`\\exists r.C \\sqsubseteq D`), masking inactive or empty target slots under ``objective="structural"``.

        Guarantees:
        - Under ``objective="structural"``, evaluates the squared
          :func:`~mowl.nn.el.multiboxel.losses.gci3_score_boxes` with tightening
          ``gamma = -self.margin``, ``aggregation="hard"``, and target mask
          ``r_d.active & (ne(r_d.w) > 0)``.

        Non-guarantees:
        - Under ``objective="structural"``, returns ``0.0`` when every target
          slot of :math:`D` is masked out (an all-empty, i.e. unsatisfiable,
          target makes :math:`\\exists r.C \\sqsubseteq D` vacuously ``0`` unless
          :math:`D` is forced non-empty by ``existence_loss``).
        """
        if self.objective == "structural":
            r_d = self.class_regions(data[:, 2])
            act_d = r_d.active & (nonempty_measure(r_d.w) > 0.0)
            cen_head = self.head_center(data[:, 0])
            off_head = th.abs(self.head_offset(data[:, 0]))
            bump_c = self.bump_classes(data[:, 1])
            return L.gci3_score_boxes(
                r_d.cen,
                th.abs(r_d.w),
                act_d,
                cen_head,
                off_head,
                bump_c,
                gamma=-self.margin,
                aggregation="hard",
                tau=self.tau,
            ).square()
        if self.training and self.objective in ("point", "mixed"):
            return self._zero_loss(data)
        return L.gci3_loss(
            data,
            self.class_center,
            self.class_offset,
            self.head_center,
            self.head_offset,
            self.tail_center,
            self.tail_offset,
            self.bump_classes,
            self.gamma,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci3_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI3-bot loss (:math:`\\exists r.C \\sqsubseteq \\bot`) via :func:`~mowl.nn.el.multiboxel.losses.gci3_bot_score_boxes`."""
        if self.objective == "structural":
            off_head = th.abs(self.head_offset(data[:, 0]))
            return L.gci3_bot_score_boxes(
                off_head,
                aggregation="hard",
                tau=self.tau,
            ).square()
        if self.training and self.objective == "point":
            return self._zero_loss(data)
        return L.gci3_bot_loss(
            data,
            self.head_offset,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def class_assertion_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute class assertion loss (:math:`C(a)`), masking inactive or empty class slots under ``objective="structural"``.

        Guarantees:
        - Under ``objective="structural"``, evaluates
          :func:`~mowl.nn.el.multiboxel.losses.class_assertion_score_boxes` with
          tightening ``gamma = -self.margin``, ``aggregation="hard"``, and class
          mask ``r_c.active & (ne(r_c.w) > 0)``.

        Non-guarantees:
        - Under ``objective="structural"``, returns ``0.0`` when every slot of
          class :math:`C` is masked out (an all-empty, i.e. unsatisfiable, class
          makes :math:`C(a)` vacuously ``0`` unless :math:`C` is forced
          non-empty by ``existence_loss``).
        """
        if self.objective == "structural":
            if self.ind_center is None or self.ind_offset is None:
                raise ValueError(
                    "The number of individuals must be specified to use this loss function."
                )
            r_c = self.class_regions(data[:, 1])
            act_c = r_c.active & (nonempty_measure(r_c.w) > 0.0)
            cen_i = self.ind_center(data[:, 0])
            off_i = th.abs(self.ind_offset(data[:, 0]))
            return L.class_assertion_score_boxes(
                r_c.cen,
                th.abs(r_c.w),
                act_c,
                cen_i,
                off_i,
                gamma=-self.margin,
                aggregation="hard",
                tau=self.tau,
            )
        if self.objective in ("point", "mixed", "hybrid"):
            return self.class_assertion_score_point(data)
        if self.ind_center is None or self.ind_offset is None:
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        return L.class_assertion_loss(
            data,
            self.ind_center,
            self.ind_offset,
            self.class_center,
            self.class_offset,
            self.gamma,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def object_property_assertion_loss(
        self, data: th.Tensor, neg: bool = False
    ) -> th.Tensor:
        """Compute object property assertion loss (:math:`r(a, b)`), passing ``gamma = -self.margin`` when ``neg=False`` and ``gamma = +self.margin`` when ``neg=True`` under ``objective="structural"``."""
        if self.objective == "structural":
            if (
                self.ind_center is None
                or self.ind_offset is None
                or self.bump_individuals is None
            ):
                raise ValueError(
                    "The number of individuals must be specified to use this loss function."
                )
            role_gamma = self.margin if neg else -self.margin
            return L.object_property_assertion_loss(
                data,
                self.ind_center,
                self.ind_offset,
                self.head_center,
                self.head_offset,
                self.tail_center,
                self.tail_offset,
                self.bump_individuals,
                role_gamma,
                self.delta,
                neg=neg,
                aggregation="hard",
                tau=self.tau,
            )
        if self.training and self.objective in ("point", "mixed"):
            return self._zero_loss(data)
        if (
            self.ind_center is None
            or self.ind_offset is None
            or self.bump_individuals is None
        ):
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        return L.object_property_assertion_loss(
            data,
            self.ind_center,
            self.ind_offset,
            self.head_center,
            self.head_offset,
            self.tail_center,
            self.tail_offset,
            self.bump_individuals,
            self.gamma,
            self.delta,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def regularization_loss(self) -> th.Tensor:
        """Compute Box²EL bump regularisation plus optional component and diversity penalties.

        Guarantees:
        - When ``component_reg == 0.0`` and ``diversity_reg == 0.0`` (or
          ``n_components == 1``), returns the exact same tensor value as
          :meth:`BoxSquaredELModule.regularization_loss`.

        Non-guarantees:
        - When ``component_reg == 0.0`` and ``diversity_reg == 0.0``, unused
          components are permitted to collapse or nest inside other components.
        """
        loss = B2L.reg_loss(self.bump_classes, self.reg_factor)
        if self.bump_individuals is not None:
            loss = loss + B2L.reg_loss(self.bump_individuals, self.reg_factor)
        if self.component_reg > 0.0 and self.n_components > 1:
            tau = 0.1
            centers = self.class_center.weight.view(
                self.nb_ont_classes, self.n_components, self.embed_dim
            )
            idx_i, idx_j = th.triu_indices(
                self.n_components,
                self.n_components,
                offset=1,
                device=centers.device,
            )
            diffs = centers[:, idx_i, :] - centers[:, idx_j, :]
            dists = th.linalg.norm(diffs, dim=-1)
            comp_penalty = th.relu(tau - dists).mean(dim=1).mean(dim=0)
            loss = loss + self.component_reg * comp_penalty
        if self.diversity_reg > 0.0 and self.n_components > 1 and self.nb_ont_classes > 0:
            centers = self.class_center.weight.view(
                self.nb_ont_classes, self.n_components, self.embed_dim
            )
            offsets = th.abs(self.class_offset.weight).view(
                self.nb_ont_classes, self.n_components, self.embed_dim
            )
            loss = loss + self.diversity_reg * diversity_penalty(centers, offsets)
        return loss

    def effective_components(self, eps: float = 1e-3) -> th.Tensor:
        """Count non-degenerate, non-nested components per class on CPU using corner arithmetic.

        A component :math:`i` of class :math:`C` is counted if all of its
        half-widths satisfy :math:`|o_{C,i,m}| > \\text{eps}` and its box is not
        contained in any other valid component :math:`j \\neq i` of :math:`C`
        (checked on corners :math:`\\text{lo}_j \\le \\text{lo}_i` and
        :math:`\\text{hi}_i \\le \\text{hi}_j`; if two components have identical
        corners, only the lowest-index component is kept).

        Guarantees:
        - Returns a CPU ``torch.Tensor`` of shape ``(nb_ont_classes,)`` with
          integer counts in ``0 .. n_components``.

        Non-guarantees:
        - Does not check whether a component is covered by the *union* of two or
          more other components when it is not contained in any single one.

        Parameters
        ----------
        eps : float, default 1e-3
            Minimum half-width threshold for a component to be considered
            non-degenerate.

        Returns
        -------
        torch.Tensor
            1-D integer tensor of length ``nb_ont_classes`` on CPU.
        """
        centers = self.class_center.weight.view(
            self.nb_ont_classes, self.n_components, self.embed_dim
        )
        offsets = self.class_offset.weight.view(
            self.nb_ont_classes, self.n_components, self.embed_dim
        )
        return _effective_components(centers, offsets, eps=eps)
