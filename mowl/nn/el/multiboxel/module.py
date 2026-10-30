"""MultiBoxEL neural module."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple, Union

import torch as th
import torch.nn as nn

from mowl.nn import ELModule
import mowl.nn.el.boxsquaredel.losses as B2L
from mowl.nn.el.boxsquaredel.module import BoxSquaredELModule
from mowl.nn.el.multibox_utils import effective_components as _effective_components
import mowl.nn.el.multiboxel.losses as L
from mowl.nn.el.pointloss import (
    _validate_beta,
    _validate_ind_halfwidth,
    _validate_objective,
    compute_class_assertion_score_point,
    compute_point_membership_loss,
    compute_point_modelness,
    compute_target_membership,
    gather_element_embeddings,
    membership_bce,
    soft_box_membership,
)
from mowl.nn.el.softagg import _validate_tau, diversity_penalty

if TYPE_CHECKING:
    from mowl.datasets.el.points import PointSupervision


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
        Training objective mode: ``"axiom"``, ``"point"``, ``"mixed"``, or
        ``"hybrid"``. Under ``objective="point"`` and ``"mixed"``, ABox
        ``class_assertion`` is still trained in the main loop via
        :math:`-\\log m_C(a)` while ``object_property_assertion`` is zeroed
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

    References
    ----------
    .. [federico2026] Federico, P., Leemhuis, M., & Kutz, O. (2026). Multi-box
       concept embeddings for description logics (JOWO 2026).
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
        _validate_objective(objective)
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
            _validate_objective(value)  # type: ignore[arg-type]
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
