"""MultiELBE neural module."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch as th
import torch.nn as nn

from mowl.nn import ELModule
from mowl.nn.el.boxsquaredel.module import BoxSquaredELModule
from mowl.nn.el.multibox_utils import effective_components as _effective_components
from mowl.nn.el.multiboxel.module import MultiBoxELModule
import mowl.nn.el.multielbe.losses as L
from mowl.nn.el.pointloss import (
    _VALID_OBJECTIVES,
    compute_class_assertion_score_point,
    compute_point_membership_loss,
    compute_point_modelness,
    gather_element_embeddings,
    membership_bce,
    soft_box_membership,
    union_membership,
)
from mowl.nn.el.softagg import _validate_tau, diversity_penalty

if TYPE_CHECKING:
    from mowl.datasets.el.points import PointSupervision


class MultiELBEModule(ELModule):
    """Multi-component ELBE neural module.

    Every ontology class :math:`C` is represented as a union of
    :math:`k = \\text{n\\_components}` axis-parallel boxes in :math:`\\mathbb{R}^d`,
    stored in ``class_embed`` and ``class_offset`` of shape
    ``(nb_ont_classes, k * d)`` and viewed as ``(B, k, d)``. Roles are single
    translation vectors in ``rel_embed`` of shape ``(nb_rels, d)``, and optional
    individuals are single boxes in ``ind_embed`` and ``ind_offset`` of shape
    ``(nb_inds, d)``, parameterised identically to
    :class:`~mowl.nn.el.elbe.module.ELBEModule`.

    Guarantees:
    - **Exact k = 1 reduction (identical to ELBEModule at equal margin):** When
      ``n_components == 1``, ``component_aggregation == "hard"``,
      ``component_reg == 0.0``, and ``diversity_reg == 0.0`` with weights copied
      from an :class:`~mowl.nn.el.elbe.module.ELBEModule`, every GCI loss
      (positive and negative) is identical to
      :class:`~mowl.nn.el.elbe.module.ELBEModule` at equal ``margin`` to within
      floating-point tolerance ``1e-6``. Both ``MultiELBEModule`` and
      ``ELBEModule`` keep the default ``margin=0.1`` as specified in the brief
      (note that the high-level :class:`~mowl.models.ELBE` model wrapper defaults
      to ``margin=0``, so comparing ``MultiELBE`` and ``ELBE`` models requires
      passing an equal ``margin`` explicitly).
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
    margin : float, default 0.1
        ELBE inclusion margin (positive margin tightens; kept at ``0.1`` as
        specified in the brief, matching :class:`~mowl.nn.el.elbe.module.ELBEModule`).
    n_components : int, default 1
        Number of box components :math:`k \\ge 1` per class.
    component_reg : float, default 0.0
        Weight for the anti-collapse pairwise component centre regulariser
        :math:`\\text{mean}_C \\text{mean}_{i<j} \\text{relu}(\\tau_c - \\|c_{C,i} - c_{C,j}\\|_2)`
        with :math:`\\tau_c = 0.1`.
    component_aggregation : str, default "hard"
        Component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-assignment temperature stored as a registered buffer on the module.
    diversity_reg : float, default 0.0
        Diversity regularisation weight stored on the module.
    """

    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})
    role_axiom_capable = False

    def __init__(
        self,
        nb_ont_classes: int,
        nb_rels: int,
        nb_inds: Optional[int] = None,
        embed_dim: int = 50,
        margin: float = 0.1,
        n_components: int = 1,
        component_reg: float = 0.0,
        component_aggregation: str = "hard",
        tau: float = 1.0,
        diversity_reg: float = 0.0,
        objective: str = "axiom",
        beta: float = 0.1,
        neg_weight: float = 0.1,
        volume_reg: float = 0.0,
    ) -> None:
        super().__init__()
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
        if objective not in _VALID_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {_VALID_OBJECTIVES}, got {objective!r}"
            )
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

        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds

        self.embed_dim = embed_dim
        self.margin = margin
        self.n_components = n_components
        self.component_reg = component_reg
        self.component_aggregation = component_aggregation
        self.register_buffer("tau", th.tensor(tau_f, dtype=th.float32))
        self.diversity_reg = float(diversity_reg)
        self.objective = objective
        self.beta = float(beta)
        self.neg_weight = float(neg_weight)
        self.volume_reg = float(volume_reg)
        self.point_supervision: Optional["PointSupervision"] = None
        self.elem_embed: Optional[nn.Embedding] = None

        self.class_embed = self.init_class_embeddings(
            nb_ont_classes, n_components, embed_dim
        )
        self.class_offset = self.init_class_embeddings(
            nb_ont_classes, n_components, embed_dim
        )

        self.rel_embed = self.init_embeddings(nb_rels, embed_dim)

        if self.nb_inds is not None:
            self.ind_embed = self.init_embeddings(nb_inds, embed_dim)
            self.ind_offset = self.init_embeddings(nb_inds, embed_dim)
        else:
            self.ind_embed = None
            self.ind_offset = None

    def __setattr__(self, name: str, value: object) -> None:
        if name == "tau" and "_buffers" in self.__dict__ and "tau" in self._buffers:
            tau_f = _validate_tau(value)  # type: ignore[arg-type]
            self._buffers["tau"].fill_(tau_f)
            return
        if name == "objective":
            if value not in _VALID_OBJECTIVES:
                raise ValueError(
                    f"objective must be one of {_VALID_OBJECTIVES}, got {value!r}"
                )
        elif name == "beta":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
                raise ValueError(f"beta must be > 0, got {value!r}")
            value = float(value)
        elif name == "neg_weight":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
                raise ValueError(f"neg_weight must be >= 0, got {value!r}")
            value = float(value)
        elif name == "volume_reg":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
                raise ValueError(f"volume_reg must be >= 0, got {value!r}")
            value = float(value)
        super().__setattr__(name, value)

    init_embeddings = BoxSquaredELModule.init_embeddings
    init_class_embeddings = MultiBoxELModule.init_class_embeddings

    def set_point_supervision(self, ps: "PointSupervision") -> None:
        """Attach :class:`~mowl.datasets.el.points.PointSupervision` and create ``elem_embed``.

        Guarantees:
        - Sets ``self.point_supervision = ps``.
        - Allocates ``self.elem_embed`` of shape ``(ps.n_elements, embed_dim)`` on
          the same device and dtype as ``class_embed.weight``, with rows for
          individual-backed elements never indexed from ``elem_embed``.

        Non-guarantees:
        - Re-calling with a different ``n_elements`` replaces ``self.elem_embed``.

        Parameters
        ----------
        ps : PointSupervision
            Canonical-element supervision table.
        """
        self.point_supervision = ps
        device = self.class_embed.weight.device
        dtype = self.class_embed.weight.dtype
        if self.elem_embed is None or self.elem_embed.num_embeddings != ps.n_elements:
            self.elem_embed = self.init_embeddings(
                ps.n_elements, self.embed_dim
            ).to(device=device, dtype=dtype)

    def element_embeddings(self, elem_idx: th.Tensor) -> th.Tensor:
        """Look up element point embeddings of shape ``(B, d)`` with individual aliasing.

        Guarantees:
        - Class-element rows are gathered from ``self.elem_embed`` and individual-element
          rows are gathered from ``self.ind_embed``, preserving separate gradient flows.

        Non-guarantees:
        - Raises ``RuntimeError`` if :meth:`set_point_supervision` has not been called.

        Parameters
        ----------
        elem_idx : torch.Tensor
            1-D integer tensor of element indices of length ``B``.

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

    def point_membership_loss(
        self,
        batch: Union[th.Tensor, Dict[str, th.Tensor]],
        neg: bool = False,
        weight: Optional[float] = None,
    ) -> th.Tensor:
        """Compute point-in-box BCE membership loss for a pair tensor or sampled dict.

        Guarantees:
        - Accepts either a ``(B, 2)`` integer tensor ``(elem_idx, class_idx)`` (with
          target ``1 - float(neg)``) or a dict from :meth:`PointSupervision.sample`
          containing ``"pos"``, ``"hard"``, and ``"rand"`` keys.
        - Differentiable with respect to ``elem_embed``, ``ind_embed``, ``class_embed``,
          and ``class_offset``.

        Non-guarantees:
        - Raises ``RuntimeError`` if :meth:`set_point_supervision` has not been called.

        Parameters
        ----------
        batch : torch.Tensor or dict of str to torch.Tensor
            Pair tensor of shape ``(B, 2)`` or sample dictionary.
        neg : bool, default False
            If ``batch`` is a tensor, whether targets are ``0.0`` (``True``) or ``1.0`` (``False``).
        weight : float, optional
            Optional sample weight when ``batch`` is a tensor.

        Returns
        -------
        torch.Tensor
            Scalar BCE membership loss.
        """
        return compute_point_membership_loss(self, batch, neg=neg, weight=weight)

    def point_role_loss(
        self,
        batch: Union[th.Tensor, Dict[str, th.Tensor]],
        neg: bool = False,
        weight: Optional[float] = None,
    ) -> th.Tensor:
        """Compute MultiELBE translation role loss for canonical role edges :math:`(x_C, r, x_D)`.

        Uses soft membership of the translated source element :math:`x_C + r` in
        the target concept :math:`D`'s union of boxes:
        :math:`m(x_C, r, x_D) = m_D(x_C + r)`.

        Guarantees:
        - Accepts either a ``(B, 3)`` integer tensor ``(elem_x, r_idx, elem_y)`` or a
          sample dict with key ``"roles"``.
        - Differentiable with respect to ``elem_embed``/``ind_embed``, ``rel_embed``,
          ``class_embed``, and ``class_offset``.

        Non-guarantees:
        - Does not use ``ind_offset`` when target elements are individuals.

        Parameters
        ----------
        batch : torch.Tensor or dict of str to torch.Tensor
            Role triple tensor of shape ``(B, 3)`` or sample dictionary.
        neg : bool, default False
            Whether targets are ``0.0`` (``True``) or ``1.0`` (``False``).
        weight : float, optional
            Optional sample weight.

        Returns
        -------
        torch.Tensor
            Scalar BCE role loss.
        """
        if self.point_supervision is None or self.elem_embed is None:
            raise RuntimeError(
                "point_supervision is not set; call set_point_supervision(ps) first."
            )
        device = self.class_embed.weight.device
        dtype = self.class_embed.weight.dtype
        if isinstance(batch, dict):
            triples = batch.get("roles")
            if triples is None or triples.numel() == 0:
                return th.zeros((), dtype=dtype, device=device)
        else:
            triples = batch
            if triples.numel() == 0:
                return th.zeros((), dtype=dtype, device=device)

        triples_dev = triples.to(device=device, dtype=th.long)
        elem_x = triples_dev[:, 0]
        r_idx = triples_dev[:, 1]
        elem_y = triples_dev[:, 2]

        x = self.element_embeddings(elem_x)
        r = self.rel_embed(r_idx)
        x_shifted = x + r

        ps = self.point_supervision
        is_ind_y = ps.element_is_individual.to(device=device)[elem_y]
        cls_or_ind_y = ps.element_class_index.to(device=device)[elem_y]
        if not bool(is_ind_y.any().item()):
            cen_d, off_d = self.class_boxes(cls_or_ind_y)
        else:
            b_size = elem_y.shape[0]
            cen_d = th.empty(
                (b_size, self.n_components, self.embed_dim),
                dtype=x_shifted.dtype,
                device=device,
            )
            off_d = th.zeros(
                (b_size, self.n_components, self.embed_dim),
                dtype=x_shifted.dtype,
                device=device,
            )
            cls_mask = ~is_ind_y
            if bool(cls_mask.any().item()):
                c_cls, o_cls = self.class_boxes(cls_or_ind_y[cls_mask])
                cen_d[cls_mask] = c_cls
                off_d[cls_mask] = o_cls
            if bool(is_ind_y.any().item()):
                if self.ind_embed is None:
                    raise ValueError(
                        "Individual elements present in PointSupervision but module has no ind_embed."
                    )
                c_ind = (
                    self.ind_embed(cls_or_ind_y[is_ind_y])
                    .unsqueeze(1)
                    .expand(-1, self.n_components, -1)
                )
                cen_d[is_ind_y] = c_ind

        comp_m = soft_box_membership(x_shifted, cen_d, off_d, beta=self.beta)
        m = union_membership(comp_m)
        target_val = 0.0 if neg else 1.0
        eff_w = (self.neg_weight if neg else 1.0) if weight is None else float(weight)
        return membership_bce(m, target_val, weight=eff_w)

    def class_assertion_score_point(self, data: th.Tensor) -> th.Tensor:
        """Compute point-based ABox class assertion score :math:`-\\log(m_C(x_a) + \\varepsilon)`.

        Guarantees:
        - Returns shape ``(B, 1)`` using ``self.ind_embed`` without touching ``self.ind_offset``.

        Non-guarantees:
        - Raises ``ValueError`` if ``nb_inds`` was ``None`` or ``0``.

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, 2)`` with columns ``(a, C)``.

        Returns
        -------
        torch.Tensor
            Score tensor of shape ``(B, 1)`` where lower means higher membership.
        """
        return compute_class_assertion_score_point(self, data)

    def point_modelness(
        self,
        ps: Optional["PointSupervision"] = None,
        chunk_size: int = 8192,
    ) -> Dict[str, float]:
        """Compute ``pos_frac`` (:math:`m \\ge 0.5`) and ``hardneg_frac`` (:math:`m < 0.5`).

        Guarantees:
        - Evaluates in chunks on the module's current device under ``torch.no_grad()``.

        Non-guarantees:
        - Does not evaluate role pairs.

        Parameters
        ----------
        ps : PointSupervision, optional
            Supervision object (defaults to ``self.point_supervision``).
        chunk_size : int, default 8192
            Number of pairs per evaluation chunk.

        Returns
        -------
        dict of str to float
            Dictionary with keys ``"pos_frac"`` and ``"hardneg_frac"``.
        """
        return compute_point_modelness(self, ps=ps, chunk_size=chunk_size)

    @property
    def class_center(self) -> nn.Embedding:
        """Alias for ``class_embed`` used by :class:`~mowl.geometric.init.ExactBoxInit`.

        Guarantees:
        - Returns the exact same ``nn.Embedding`` instance as ``self.class_embed``
          without registering a duplicate parameter in ``state_dict()``.

        Non-guarantees:
        - Read-only property; reassigning ``class_center`` is not supported.
        """
        return self.class_embed

    def class_boxes(self, idx: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Return ``(centers, offsets)`` of shape ``(B, k, d)`` for class indices ``idx``.

        Guarantees:
        - ``offsets`` is non-negative (computed via ``torch.abs``).

        Non-guarantees:
        - Does not sort or deduplicate components.

        Parameters
        ----------
        idx : torch.Tensor
            1-D integer tensor of class indices of length ``B``.

        Returns
        -------
        centers : torch.Tensor
            Component centres of shape ``(B, n_components, embed_dim)``.
        offsets : torch.Tensor
            Non-negative component half-widths of shape ``(B, n_components, embed_dim)``.
        """
        b_size = idx.shape[0]
        cen = self.class_embed(idx).view(b_size, self.n_components, self.embed_dim)
        off = th.abs(self.class_offset(idx)).view(
            b_size, self.n_components, self.embed_dim
        )
        return cen, off

    def gci0_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI0 loss for :math:`C \\sqsubseteq D` via :func:`~mowl.nn.el.multielbe.losses.gci0_loss`.

        Guarantees:
        - Returns shape ``(B, 1)``, identical to ``ELBEModule.gci0_loss`` at ``k = 1``
          and equal ``margin``.

        Non-guarantees:
        - ``neg=True`` is not distinct from ``neg=False``.

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, 2)`` with columns ``(C, D)``.
        neg : bool, default False
            Unused negative-sampling flag.

        Returns
        -------
        torch.Tensor
            Per-row loss tensor of shape ``(B, 1)``.
        """
        if self.objective in ("point", "mixed"):
            return th.zeros(
                (len(data), 1),
                dtype=self.class_embed.weight.dtype,
                device=data.device,
            )
        return L.gci0_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci0_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI0-bot loss for :math:`C \\sqsubseteq \\bot` via :func:`~mowl.nn.el.multielbe.losses.gci0_bot_loss`.

        Guarantees:
        - Returns shape ``(B,)``, identical to ``ELBEModule.gci0_bot_loss`` at ``k = 1``.

        Non-guarantees:
        - ``neg=True`` is not distinct from ``neg=False``.

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, >=1)`` with column ``0`` holding ``C``.
        neg : bool, default False
            Unused negative-sampling flag.

        Returns
        -------
        torch.Tensor
            Per-row loss tensor of shape ``(B,)``.
        """
        if self.objective == "point":
            return th.zeros(
                (len(data),),
                dtype=self.class_embed.weight.dtype,
                device=data.device,
            )
        return L.gci0_bot_loss(
            data,
            self.class_offset,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci1_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute gated GCI1 loss for :math:`C \\sqcap D \\sqsubseteq E` via :func:`~mowl.nn.el.multielbe.losses.gci1_loss`.

        Guarantees:
        - Returns shape ``(B, 1)``, identical to ``ELBEModule.gci1_loss`` at ``k = 1``
          and equal ``margin`` (both empty and non-empty intersections).

        Non-guarantees:
        - ``neg=True`` is not distinct from ``neg=False``.

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, 3)`` with columns ``(C, D, E)``.
        neg : bool, default False
            Unused negative-sampling flag.

        Returns
        -------
        torch.Tensor
            Per-row loss tensor of shape ``(B, 1)``.
        """
        if self.objective in ("point", "mixed"):
            return th.zeros(
                (len(data), 1),
                dtype=self.class_embed.weight.dtype,
                device=data.device,
            )
        return L.gci1_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci1_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI1-bot loss for :math:`C \\sqcap D \\sqsubseteq \\bot` via :func:`~mowl.nn.el.multielbe.losses.gci1_bot_loss`.

        Guarantees:
        - Returns shape ``(B, 1)``, identical to ``ELBEModule.gci1_bot_loss`` at ``k = 1``
          and equal ``margin``.
        - Zero if and only if **all** :math:`k^2` component pairs have zero ELBE
          disjointness score (mean of non-negatives).

        Non-guarantees:
        - ``neg=True`` is not distinct from ``neg=False``.

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, >=2)`` with columns ``(C, D, ...)``.
        neg : bool, default False
            Unused negative-sampling flag.

        Returns
        -------
        torch.Tensor
            Per-row loss tensor of shape ``(B, 1)``.
        """
        if self.objective == "point":
            return th.zeros(
                (len(data), 1),
                dtype=self.class_embed.weight.dtype,
                device=data.device,
            )
        return L.gci1_bot_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci2_score(self, data: th.Tensor) -> th.Tensor:
        """Compute positive GCI2 score for :math:`C \\sqsubseteq \\exists r.D` via :func:`~mowl.nn.el.multielbe.losses.gci2_score`.

        Guarantees:
        - Returns shape ``(B, 1)``, identical to ``ELBEModule.gci2_score`` at ``k = 1``
          and equal ``margin``.

        Non-guarantees:
        - Evaluates only the positive inclusion direction (``neg=False``).

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, 3)`` with columns ``(C, r, D)``.

        Returns
        -------
        torch.Tensor
            Per-row score tensor of shape ``(B, 1)``.
        """
        return L.gci2_score(
            data,
            self.class_embed,
            self.class_offset,
            self.rel_embed,
            self.margin,
            n_components=self.n_components,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci2_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI2 loss for :math:`C \\sqsubseteq \\exists r.D` (positive or negative) via :func:`~mowl.nn.el.multielbe.losses.gci2_loss`.

        Guarantees:
        - Returns shape ``(B, 1)``, identical to ``ELBEModule.gci2_loss`` at ``k = 1``
          and equal ``margin`` for both ``neg=False`` and ``neg=True``.
        - For :math:`k \\ge 2`, aggregates ``neg=False`` as
          :math:`\\text{mean}_i \\text{agg}_j s(C_i, D_j, r)` (where :math:`\\text{agg}_j = \\min_j`
          under ``"hard"`` and :math:`\\text{softmin}_{\\tau, j}` under ``"soft"``), and
          aggregates ``neg=True`` as the mean over all :math:`k^2` component pairs
          :math:`\\text{mean}_{i,j} s_{\\text{neg}}(C_i, D_j, r)` identically under both
          ``"hard"`` and ``"soft"`` (no ``softmin``).

        Non-guarantees:
        - Roles are single translation vectors rather than multi-box relations.

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, 3)`` with columns ``(C, r, D)``.
        neg : bool, default False
            If ``True``, evaluate the negative separation score.

        Returns
        -------
        torch.Tensor
            Per-row score tensor of shape ``(B, 1)``.
        """
        if self.objective in ("point", "mixed"):
            return th.zeros(
                (len(data), 1),
                dtype=self.class_embed.weight.dtype,
                device=data.device,
            )
        return L.gci2_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.rel_embed,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci3_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI3 overlap loss for :math:`\\exists r.C \\sqsubseteq D` via :func:`~mowl.nn.el.multielbe.losses.gci3_loss`.

        Guarantees:
        - Returns shape ``(B, 1)``, identical to ``ELBEModule.gci3_loss`` at ``k = 1``
          and equal ``margin``.

        Non-guarantees:
        - Uses ELBE's overlap formula rather than full box inclusion.
        - ``neg=True`` is not distinct from ``neg=False``.

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, 3)`` with columns ``(r, C, D)``.
        neg : bool, default False
            Unused negative-sampling flag.

        Returns
        -------
        torch.Tensor
            Per-row loss tensor of shape ``(B, 1)``.
        """
        if self.objective in ("point", "mixed"):
            return th.zeros(
                (len(data), 1),
                dtype=self.class_embed.weight.dtype,
                device=data.device,
            )
        return L.gci3_loss(
            data,
            self.class_embed,
            self.class_offset,
            self.rel_embed,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def gci3_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute GCI3-bot loss for :math:`\\exists r.C \\sqsubseteq \\bot` via :func:`~mowl.nn.el.multielbe.losses.gci3_bot_loss`.

        Guarantees:
        - Indexes ``data[:, 1]`` for :math:`C` (rows are ``(r, C, bot)``) and
          returns shape ``(B,)``, identical to ``ELBEModule.gci3_bot_loss`` at ``k = 1``.

        Non-guarantees:
        - ``neg=True`` is not distinct from ``neg=False``.

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, >=2)`` with columns ``(r, C, ...)``.
        neg : bool, default False
            Unused negative-sampling flag.

        Returns
        -------
        torch.Tensor
            Per-row loss tensor of shape ``(B,)``.
        """
        if self.objective == "point":
            return th.zeros(
                (len(data),),
                dtype=self.class_embed.weight.dtype,
                device=data.device,
            )
        return L.gci3_bot_loss(
            data,
            self.class_offset,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def class_assertion_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        """Compute ABox class assertion loss for :math:`C(a)` via :func:`~mowl.nn.el.multielbe.losses.class_assertion_loss`.

        Guarantees:
        - Returns shape ``(B, 1)``, identical to ``ELBEModule.class_assertion_loss``
          at ``k = 1`` and equal ``margin``.

        Non-guarantees:
        - ``neg=True`` is not distinct from ``neg=False``.

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, 2)`` with columns ``(a, C)``.
        neg : bool, default False
            Unused negative-sampling flag.

        Returns
        -------
        torch.Tensor
            Per-row loss tensor of shape ``(B, 1)``.

        Raises
        ------
        ValueError
            If ``nb_inds`` was ``None`` or ``0`` at module construction.
        """
        if self.ind_embed is None or self.ind_offset is None or self.nb_inds == 0:
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        if self.objective in ("point", "mixed"):
            return self.class_assertion_score_point(data)
        return L.class_assertion_loss(
            data,
            self.ind_embed,
            self.ind_offset,
            self.class_embed,
            self.class_offset,
            self.margin,
            n_components=self.n_components,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def object_property_assertion_loss(
        self, data: th.Tensor, neg: bool = False
    ) -> th.Tensor:
        """Compute ABox object property assertion loss for :math:`r(a, b)` via :func:`~mowl.nn.el.multielbe.losses.object_property_assertion_loss`.

        Guarantees:
        - Returns shape ``(B, 1)``, identical to ``ELBEModule.object_property_assertion_loss``
          at equal ``margin`` for both ``neg=False`` and ``neg=True``.

        Non-guarantees:
        - Individuals and roles are single-component embeddings.

        Parameters
        ----------
        data : torch.Tensor
            Integer batch of shape ``(B, 3)`` with columns ``(a, r, b)``.
        neg : bool, default False
            If ``True``, evaluate the negative separation score.

        Returns
        -------
        torch.Tensor
            Per-row score tensor of shape ``(B, 1)``.

        Raises
        ------
        ValueError
            If ``nb_inds`` was ``None`` or ``0`` at module construction.
        """
        if self.ind_embed is None or self.ind_offset is None or self.nb_inds == 0:
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        if self.objective in ("point", "mixed"):
            return th.zeros(
                (len(data), 1),
                dtype=self.class_embed.weight.dtype,
                device=data.device,
            )
        return L.object_property_assertion_loss(
            data,
            self.ind_embed,
            self.ind_offset,
            self.rel_embed,
            self.margin,
            neg=neg,
            aggregation=self.component_aggregation,
            tau=self.tau,
        )

    def regularization_loss(self) -> th.Tensor:
        """Compute optional component centre and volume-overlap diversity penalties.

        Guarantees:
        - Returns a scalar ``0.0`` tensor on ``class_embed.weight.device`` when
          ``n_components == 1`` or ``component_reg == 0.0 and diversity_reg == 0.0``.

        Non-guarantees:
        - When both ``component_reg == 0.0`` and ``diversity_reg == 0.0``, unused
          components are permitted to collapse or nest inside other components.

        Returns
        -------
        torch.Tensor
            Scalar regularisation loss tensor.
        """
        loss = th.zeros(
            (),
            device=self.class_embed.weight.device,
            dtype=self.class_embed.weight.dtype,
        )
        if self.component_reg > 0.0 and self.n_components > 1:
            tau_c = 0.1
            centers = self.class_embed.weight.view(
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
            comp_penalty = th.relu(tau_c - dists).mean(dim=1).mean(dim=0)
            loss = loss + self.component_reg * comp_penalty
        if self.diversity_reg > 0.0 and self.n_components > 1 and self.nb_ont_classes > 0:
            centers = self.class_embed.weight.view(
                self.nb_ont_classes, self.n_components, self.embed_dim
            )
            offsets = th.abs(self.class_offset.weight).view(
                self.nb_ont_classes, self.n_components, self.embed_dim
            )
            loss = loss + self.diversity_reg * diversity_penalty(centers, offsets)
        return loss

    def effective_components(self, eps: float = 1e-3) -> th.Tensor:
        """Count non-degenerate, non-nested components per class on CPU using corner arithmetic.

        Delegates to :func:`mowl.nn.el.multibox_utils.effective_components`.

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
        centers = self.class_embed.weight.view(
            self.nb_ont_classes, self.n_components, self.embed_dim
        )
        offsets = self.class_offset.weight.view(
            self.nb_ont_classes, self.n_components, self.embed_dim
        )
        return _effective_components(centers, offsets, eps=eps)
