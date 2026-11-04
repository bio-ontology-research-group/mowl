"""GroundedELModule: element-grounded multi-box and product-role neural module.

Implements §2 and §5 of the GroundedEL design:

1. **Syntactic domain elements (:math:`\\Delta`):** ``nb_elements`` elements where
   indices ``0 .. nb_classes - 1`` correspond to class elements :math:`x_A` and
   indices ``nb_classes .. nb_elements - 1`` correspond to ABox individuals. Each
   element has a learnable position :math:`p_x \\in \\mathbb{R}^d` (with an
   optional feature encoder hook for data elements) and a learnable deadness
   logit :math:`\\ell_x` with :math:`\\delta_x = \\sigma(\\ell_x) \\in (0, 1)`.
2. **Class multi-boxes:** Each class :math:`A` has :math:`K = \\text{n\\_components}`
   axis-aligned boxes with centres :math:`c_{A,k} \\in \\mathbb{R}^d`, half-widths
   :math:`w_{A,k} = \\operatorname{softplus}(\\tilde{w}_{A,k}) \\ge 0`, and an
   active mask ``class_active``. Signed depth reuses
   :func:`~mowl.nn.el.structural.inside_measure` semantics under ``agg="max"``
   or smooth ``logsumexp`` under ``agg="lse"``.
3. **Product-of-box roles:** Each role :math:`r` has
   :math:`K_r = \\text{n\\_role\\_components}` head/tail box pairs
   :math:`(H_{r,k}, T_{r,k})` in :math:`\\mathbb{R}^d \\times \\mathbb{R}^d`.
4. **Monotone capacity (R3) and exact-limit construction (R2, §5):**
   :meth:`expand_components` appends zero-width inactive components without
   changing any depth or membership output; :meth:`exact_limit_init` realises any
   given least-model label matrix and edge set with disjoint :math:`\\delta`-boxes
   around element positions.

Guarantees:
- **Reasoner-free (R1):** Contains no imports or calls to reasoners or semantic
  completion routines.
- **Exact limit realisation (R2):** After :meth:`exact_limit_init` with labels
  and edges of a consistent least model, every crisp grounded element hinge and
  every crisp region hinge inside active boxes evaluates to exact ``0.0``.
- **Bitwise monotone capacity expansion (R3):** Calling :meth:`expand_components`
  with ``new_K >= n_components`` leaves :meth:`depth`, :meth:`membership`,
  :meth:`role_depth`, and :meth:`role_membership` bitwise identical.

Non-guarantees:
- Under ``agg="max"``, gradients with respect to box parameters flow only through
  the winning active component and bottleneck coordinate.
- Under ``agg="lse"`` at finite ``agg_tau > 0``, smooth depth exceeds crisp
  maximum depth by up to :math:`\\tau_{\\text{agg}} \\log K` when multiple active
  components have equal depth.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from mowl.nn.el.structural import Regions, inside_measure

__all__ = ["GroundedELModule"]

_INACTIVE_RAW_WIDTH: float = -100.0
_LIVE_LOGIT: float = -100.0
_DEAD_LOGIT: float = 100.0
_EMPTY_DEPTH_FLOOR: float = -1e6
_TOP_DEPTH_CEIL: float = 1e6


def _validate_positive_float(name: str, val: Union[float, int, th.Tensor]) -> float:
    """Validate that ``val`` is a strictly positive finite scalar float."""
    if isinstance(val, th.Tensor):
        if val.dtype == th.bool or val.numel() != 1:
            raise ValueError(f"{name} must be a positive scalar, got {val!r}")
        v = float(val.detach().item())
    elif isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a positive number, got {val!r}")
    else:
        v = float(val)
    if v <= 0.0 or math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be > 0 and finite, got {val!r}")
    return v


def _validate_finite_float(name: str, val: Union[float, int]) -> float:
    """Validate that ``val`` is a finite scalar float."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {val!r}")
    v = float(val)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be finite, got {val!r}")
    return v


def _validate_int_ge(name: str, val: int, min_val: int) -> int:
    """Validate that ``val`` is an integer >= ``min_val`` (rejecting booleans)."""
    if isinstance(val, bool) or not isinstance(val, int) or val < min_val:
        raise ValueError(f"{name} must be an int >= {min_val}, got {val!r}")
    return int(val)


def _inv_softplus(y: float) -> float:
    """Compute the inverse of softplus :math:`\\log(\\exp(y) - 1)` for :math:`y > 0`."""
    if y <= 0.0:
        raise ValueError(f"delta must be > 0, got {y!r}")
    if y > 20.0:
        return y
    return y + math.log(-math.expm1(-y))


class GroundedELModule(nn.Module):
    """Element-grounded multi-box and product-role neural module for :math:`\\mathcal{EL}^{++}`.

    Guarantees:
    - **Index convention:** Elements ``0 .. nb_classes - 1`` correspond 1-to-1
      with classes ``0 .. nb_classes - 1``; elements ``nb_classes .. nb_elements - 1``
      are individuals.
    - **Top and bottom semantics:** When ``top_id`` is set, :meth:`membership`
      returns exact ``1.0`` and :meth:`depth` returns ``+1e6`` for ``top_id``.
      When ``bot_id`` is set, :meth:`membership` returns exact ``0.0`` and
      :meth:`depth` returns ``-1e6`` for ``bot_id``.
    - **Monotone capacity (R3):** :meth:`expand_components` appends inactive
      zero-width components while keeping :meth:`depth`, :meth:`membership`,
      :meth:`role_depth`, and :meth:`role_membership` bitwise identical.
    - **Exact-limit realisation (R2):** :meth:`exact_limit_init` places elements
      at generic positions rescaled so that :math:`\\delta`-boxes of distinct
      elements are strictly disjoint, sets active class boxes according to the
      given ``least_model_labels``, and sets active role product boxes according
      to the given ``edges``.

    Non-guarantees:
    - Does not compute semantic completion or least-model labels itself; labels
      and edges passed to :meth:`exact_limit_init` must be supplied by the caller.
    - Under ``agg="max"``, gradients flow only through the argmax active
      component and argmin coordinate.

    Parameters
    ----------
    nb_elements : int
        Total number of domain elements :math:`|\\Delta| \\ge \\text{nb\\_classes}`.
    nb_classes : int
        Number of ontology classes :math:`\\ge 1`.
    nb_rels : int, default 0
        Number of object properties (roles) :math:`\\ge 0`.
    embed_dim : int, default 50
        Ambient Euclidean dimension :math:`d \\ge 1`.
    n_components : int, default 2
        Number of box components :math:`K \\ge 1` per class.
    n_role_components : int, default 2
        Number of product-box components :math:`K_r \\ge 1` per role.
    tau : float, default 1.0
        Sigmoid temperature :math:`\\tau > 0` converting signed depth to soft
        membership.
    agg : {"max", "lse"}, default "max"
        Component aggregation mode over active components: ``"max"`` uses hard
        maximum (:func:`~mowl.nn.el.structural.inside_measure`); ``"lse"`` uses
        ``agg_tau * logsumexp(depth / agg_tau)``.
    agg_tau : float, default 0.1
        Temperature :math:`\\tau_{\\text{agg}} > 0` for ``agg="lse"``.
    dead_init_logit : float, default -4.0
        Initial value for per-element deadness logits :math:`\\ell_x` so initial
        deadness is :math:`\\sigma(-4) \\approx 0.018`.
    element_features : Optional[torch.Tensor], default None
        Optional feature matrix of shape ``(nb_elements, F)`` or
        ``(nb_elements - nb_classes, F)`` for data elements.
    encoder : Optional[torch.nn.Module], default None
        Optional encoder mapping rows of ``element_features`` to ``embed_dim``.
    top_id : Optional[int], default None
        Optional class index for :math:`\\top` (``owl:Thing``).
    bot_id : Optional[int], default None
        Optional class index for :math:`\\bot` (``owl:Nothing``).
    """

    def __init__(
        self,
        nb_elements: int,
        nb_classes: int,
        nb_rels: int = 0,
        embed_dim: int = 50,
        n_components: int = 2,
        n_role_components: int = 2,
        tau: float = 1.0,
        agg: str = "max",
        agg_tau: float = 0.1,
        dead_init_logit: float = -4.0,
        element_features: Optional[th.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        top_id: Optional[int] = None,
        bot_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.nb_classes = _validate_int_ge("nb_classes", nb_classes, 1)
        self.nb_elements = _validate_int_ge("nb_elements", nb_elements, self.nb_classes)
        self.nb_rels = _validate_int_ge("nb_rels", nb_rels, 0)
        self.embed_dim = _validate_int_ge("embed_dim", embed_dim, 1)
        self.n_components = _validate_int_ge("n_components", n_components, 1)
        self.n_role_components = _validate_int_ge(
            "n_role_components", n_role_components, 1
        )
        tau_f = _validate_positive_float("tau", tau)
        if not isinstance(agg, str) or agg not in ("max", "lse"):
            raise ValueError(f"agg must be 'max' or 'lse', got {agg!r}")
        self.agg = agg
        self.agg_tau = _validate_positive_float("agg_tau", agg_tau)
        self.dead_init_logit = _validate_finite_float("dead_init_logit", dead_init_logit)

        if top_id is not None:
            if isinstance(top_id, bool) or not isinstance(top_id, int) or not (0 <= top_id < self.nb_classes):
                raise ValueError(
                    f"top_id must be an int in [0, {self.nb_classes}), got {top_id!r}"
                )
        if bot_id is not None:
            if isinstance(bot_id, bool) or not isinstance(bot_id, int) or not (0 <= bot_id < self.nb_classes):
                raise ValueError(
                    f"bot_id must be an int in [0, {self.nb_classes}), got {bot_id!r}"
                )
        if top_id is not None and bot_id is not None and top_id == bot_id:
            raise ValueError(f"top_id and bot_id must be distinct, got {top_id}")
        self.top_id: Optional[int] = top_id
        self.bot_id: Optional[int] = bot_id

        if element_features is not None:
            if not isinstance(element_features, th.Tensor) or element_features.ndim != 2:
                raise ValueError(
                    "element_features must be a 2-D torch.Tensor of shape "
                    "(nb_elements, F) or (nb_elements - nb_classes, F)."
                )
            n_inds = self.nb_elements - self.nb_classes
            if element_features.shape[0] not in (self.nb_elements, n_inds):
                raise ValueError(
                    f"element_features.shape[0] must be nb_elements ({self.nb_elements}) "
                    f"or nb_elements - nb_classes ({n_inds}), got {element_features.shape[0]}"
                )
            self.register_buffer(
                "element_features",
                element_features.detach().clone().to(dtype=th.float32),
            )
        else:
            self.element_features = None

        self.encoder: Optional[nn.Module] = encoder

        self.elem_pos = nn.Embedding(self.nb_elements, self.embed_dim)
        nn.init.uniform_(self.elem_pos.weight, a=-0.5, b=0.5)

        self.dead_logits = nn.Parameter(
            th.full((self.nb_elements,), self.dead_init_logit, dtype=th.float32)
        )

        self.class_centers = nn.Parameter(
            th.empty((self.nb_classes, self.n_components, self.embed_dim), dtype=th.float32)
        )
        self.class_raw_widths = nn.Parameter(
            th.empty((self.nb_classes, self.n_components, self.embed_dim), dtype=th.float32)
        )
        nn.init.uniform_(self.class_centers, a=-0.5, b=0.5)
        nn.init.constant_(self.class_raw_widths, _inv_softplus(0.2))
        self.register_buffer(
            "class_active",
            th.ones((self.nb_classes, self.n_components), dtype=th.bool),
        )

        self.role_head_centers = nn.Parameter(
            th.empty(
                (self.nb_rels, self.n_role_components, self.embed_dim), dtype=th.float32
            )
        )
        self.role_head_raw_widths = nn.Parameter(
            th.empty(
                (self.nb_rels, self.n_role_components, self.embed_dim), dtype=th.float32
            )
        )
        self.role_tail_centers = nn.Parameter(
            th.empty(
                (self.nb_rels, self.n_role_components, self.embed_dim), dtype=th.float32
            )
        )
        self.role_tail_raw_widths = nn.Parameter(
            th.empty(
                (self.nb_rels, self.n_role_components, self.embed_dim), dtype=th.float32
            )
        )
        if self.nb_rels > 0:
            nn.init.uniform_(self.role_head_centers, a=-0.5, b=0.5)
            nn.init.constant_(self.role_head_raw_widths, _inv_softplus(0.2))
            nn.init.uniform_(self.role_tail_centers, a=-0.5, b=0.5)
            nn.init.constant_(self.role_tail_raw_widths, _inv_softplus(0.2))
        self.register_buffer(
            "role_active",
            th.ones((self.nb_rels, self.n_role_components), dtype=th.bool),
        )

        self.register_buffer("tau", th.tensor(tau_f, dtype=th.float32))

    def __setattr__(self, name: str, value: object) -> None:
        if name == "tau" and "_buffers" in self.__dict__ and "tau" in self._buffers:
            tau_f = _validate_positive_float("tau", value)  # type: ignore[arg-type]
            self._buffers["tau"].fill_(tau_f)
            return
        if name == "agg":
            if not isinstance(value, str) or value not in ("max", "lse"):
                raise ValueError(f"agg must be 'max' or 'lse', got {value!r}")
        elif name == "agg_tau":
            value = _validate_positive_float("agg_tau", value)  # type: ignore[arg-type]
        super().__setattr__(name, value)

    def set_tau(self, t: float) -> None:
        """Set the sigmoid membership temperature ``tau``.

        Guarantees:
        - Updates the registered ``tau`` buffer in-place to ``float(t)``.
        - Raises :class:`ValueError` if ``t <= 0`` or is non-finite.

        Non-guarantees:
        - Does not alter ``agg_tau``.

        Parameters
        ----------
        t : float
            Strictly positive finite temperature :math:`\\tau > 0`.
        """
        tau_f = _validate_positive_float("tau", t)
        self._buffers["tau"].fill_(tau_f)

    def _to_index_tensor(
        self,
        ids: Union[int, Sequence[int], th.Tensor],
        max_bound: int,
        name: str,
    ) -> th.Tensor:
        """Convert ``ids`` to a ``torch.long`` tensor on module device and validate bounds."""
        device = self.elem_pos.weight.device
        if isinstance(ids, int) and not isinstance(ids, bool):
            idx = th.tensor([ids], dtype=th.long, device=device)
        elif isinstance(ids, th.Tensor):
            if ids.dtype == th.bool:
                raise ValueError(f"{name} must be integer indices, got bool tensor")
            idx = ids.to(device=device, dtype=th.long)
            if idx.ndim == 0:
                idx = idx.unsqueeze(0)
        elif isinstance(ids, (list, tuple, np.ndarray)):
            idx = th.as_tensor(ids, dtype=th.long, device=device)
            if idx.ndim == 0:
                idx = idx.unsqueeze(0)
        else:
            raise TypeError(f"{name} must be an int, Sequence, or Tensor, got {type(ids).__name__}")

        if idx.numel() > 0:
            min_i = int(idx.min().item())
            max_i = int(idx.max().item())
            if min_i < 0 or max_i >= max_bound:
                raise ValueError(
                    f"{name} indices out of range [0, {max_bound}): min={min_i}, max={max_i}"
                )
        return idx

    def positions(
        self,
        element_ids: Optional[Union[int, Sequence[int], th.Tensor]] = None,
    ) -> th.Tensor:
        """Return element positions :math:`p_x \\in \\mathbb{R}^d` for ``element_ids``.

        Uses ``encoder(element_features)`` when ``element_features`` is attached
        for the corresponding elements (either all elements when
        ``element_features.shape[0] == nb_elements`` or individual elements
        ``id >= nb_classes`` when ``element_features.shape[0] == nb_elements - nb_classes``).

        Guarantees:
        - Returns a tensor of shape ``(*element_ids.shape, embed_dim)`` (or
          ``(nb_elements, embed_dim)`` when ``element_ids is None``).
        - Raises :class:`ValueError` if any index is outside ``[0, nb_elements)``.

        Non-guarantees:
        - Does not normalise or clamp coordinates.

        Parameters
        ----------
        element_ids : Optional[Union[int, Sequence[int], torch.Tensor]], default None
            Element indices in ``[0, nb_elements)``, or ``None`` for all elements.

        Returns
        -------
        torch.Tensor
            Element position coordinates of shape ``(..., embed_dim)``.
        """
        device = self.elem_pos.weight.device
        scalar_input = isinstance(element_ids, int) and not isinstance(element_ids, bool)
        if element_ids is None:
            idx = th.arange(self.nb_elements, dtype=th.long, device=device)
        else:
            idx = self._to_index_tensor(element_ids, self.nb_elements, "element_ids")

        orig_shape = idx.shape
        flat_idx = idx.reshape(-1)
        pos = self.elem_pos(flat_idx)

        if self.element_features is not None:
            feats = self.element_features
            if feats.shape[0] == self.nb_elements:
                sel_feats = feats[flat_idx]
                pos = self.encoder(sel_feats) if self.encoder is not None else sel_feats
            else:
                is_ind = flat_idx >= self.nb_classes
                if bool(is_ind.any().item()):
                    ind_local = flat_idx[is_ind] - self.nb_classes
                    sel_feats = feats[ind_local]
                    enc_pos = (
                        self.encoder(sel_feats) if self.encoder is not None else sel_feats
                    )
                    pos = pos.clone()
                    pos[is_ind] = enc_pos.to(dtype=pos.dtype)

        out = pos.reshape(*orig_shape, self.embed_dim)
        return out.squeeze(0) if scalar_input else out

    def dead(
        self,
        element_ids: Optional[Union[int, Sequence[int], th.Tensor]] = None,
    ) -> th.Tensor:
        """Return element dead scores :math:`\\delta_x = \\sigma(\\ell_x) \\in [0, 1]` for ``element_ids``.

        Guarantees:
        - Returns a tensor of shape ``element_ids.shape`` (or ``(nb_elements,)``
          when ``element_ids is None``) with values in ``[0, 1]``.
        - Raises :class:`ValueError` if any index is outside ``[0, nb_elements)``.

        Non-guarantees:
        - At extreme logits (``<= -100`` or ``>= 100`` in ``float32``), saturates
          to exact ``0.0`` or ``1.0`` with zero gradient.

        Parameters
        ----------
        element_ids : Optional[Union[int, Sequence[int], torch.Tensor]], default None
            Element indices in ``[0, nb_elements)``, or ``None`` for all elements.

        Returns
        -------
        torch.Tensor
            Element dead scores :math:`\\delta_x`.
        """
        scalar_input = isinstance(element_ids, int) and not isinstance(element_ids, bool)
        if element_ids is None:
            return th.sigmoid(self.dead_logits)
        idx = self._to_index_tensor(element_ids, self.nb_elements, "element_ids")
        out = th.sigmoid(self.dead_logits[idx])
        return out.squeeze(0) if scalar_input else out

    def class_regions(
        self,
        class_ids: Optional[Union[int, Sequence[int], th.Tensor]] = None,
    ) -> Regions:
        """Materialise :class:`~mowl.nn.el.structural.Regions` for ``class_ids``.

        Half-widths are :math:`w = \\operatorname{softplus}(\\tilde{w})` on active
        components and ``0.0`` on inactive components; a component is active iff
        ``class_active`` is ``True`` and all coordinate widths are strictly positive.

        Guarantees:
        - Returns :class:`~mowl.nn.el.structural.Regions` with shapes
          ``(B, n_components, embed_dim)`` for ``cen`` and ``w`` and
          ``(B, n_components)`` for ``active``.
        - Inactive components have ``w == 0.0`` and ``active == False``.

        Non-guarantees:
        - Does not override ``top_id`` or ``bot_id`` (handled in :meth:`depth`
          and :meth:`membership`).

        Parameters
        ----------
        class_ids : Optional[Union[int, Sequence[int], torch.Tensor]], default None
            1-D class indices in ``[0, nb_classes)``, or ``None`` for all classes.

        Returns
        -------
        Regions
            Materialised class box regions.
        """
        device = self.class_centers.device
        if class_ids is None:
            idx = th.arange(self.nb_classes, dtype=th.long, device=device)
        else:
            idx = self._to_index_tensor(class_ids, self.nb_classes, "class_ids")
            if idx.ndim != 1:
                raise ValueError(
                    f"class_ids must be 1-D (or scalar), got shape {tuple(idx.shape)}"
                )

        cen = self.class_centers[idx]
        raw_w = self.class_raw_widths[idx]
        act_buf = self.class_active[idx]
        w_pos = F.softplus(raw_w)
        w = th.where(act_buf.unsqueeze(-1), w_pos, th.zeros_like(w_pos))
        eff_active = act_buf & (th.amin(w, dim=-1) > 0.0)
        return Regions(cen=cen, w=w, active=eff_active)

    def role_regions(
        self,
        rel_ids: Optional[Union[int, Sequence[int], th.Tensor]] = None,
    ) -> Tuple[Regions, Regions]:
        """Materialise head and tail :class:`~mowl.nn.el.structural.Regions` for ``rel_ids``.

        Guarantees:
        - Returns ``(head_regions, tail_regions)`` each of shape
          ``(B, n_role_components, embed_dim)`` with a shared ``active`` mask
          of shape ``(B, n_role_components)``.
        - Inactive role components have ``w == 0.0`` and ``active == False``.

        Non-guarantees:
        - Raises :class:`ValueError` if ``rel_ids`` is non-empty when ``nb_rels == 0``.

        Parameters
        ----------
        rel_ids : Optional[Union[int, Sequence[int], torch.Tensor]], default None
            1-D role indices in ``[0, nb_rels)``, or ``None`` for all roles.

        Returns
        -------
        Tuple[Regions, Regions]
            ``(head_regions, tail_regions)`` for the requested roles.
        """
        device = self.role_head_centers.device
        if rel_ids is None:
            idx = th.arange(self.nb_rels, dtype=th.long, device=device)
        else:
            idx = self._to_index_tensor(rel_ids, self.nb_rels, "rel_ids")
            if idx.ndim != 1:
                raise ValueError(
                    f"rel_ids must be 1-D (or scalar), got shape {tuple(idx.shape)}"
                )

        h_cen = self.role_head_centers[idx]
        h_raw = self.role_head_raw_widths[idx]
        t_cen = self.role_tail_centers[idx]
        t_raw = self.role_tail_raw_widths[idx]
        act_buf = self.role_active[idx]

        h_w_pos = F.softplus(h_raw)
        t_w_pos = F.softplus(t_raw)
        h_w = th.where(act_buf.unsqueeze(-1), h_w_pos, th.zeros_like(h_w_pos))
        t_w = th.where(act_buf.unsqueeze(-1), t_w_pos, th.zeros_like(t_w_pos))
        eff_active = (
            act_buf & (th.amin(h_w, dim=-1) > 0.0) & (th.amin(t_w, dim=-1) > 0.0)
        )
        return (
            Regions(cen=h_cen, w=h_w, active=eff_active),
            Regions(cen=t_cen, w=t_w, active=eff_active),
        )

    def _align_points_and_ids(
        self,
        points: th.Tensor,
        ids: Union[int, Sequence[int], th.Tensor],
        max_bound: int,
        id_name: str,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Validate and broadcast ``points`` ``(B, d)`` and ``ids`` ``(B,)``."""
        if not isinstance(points, th.Tensor):
            raise TypeError(f"points must be a torch.Tensor, got {type(points).__name__}")
        if points.ndim == 1:
            if points.shape[0] != self.embed_dim:
                raise ValueError(
                    f"1-D points must have length embed_dim={self.embed_dim}, got {tuple(points.shape)}"
                )
            pts = points.unsqueeze(0)
        elif points.ndim == 2:
            if points.shape[1] != self.embed_dim:
                raise ValueError(
                    f"points must have second dimension embed_dim={self.embed_dim}, got {tuple(points.shape)}"
                )
            pts = points
        else:
            raise ValueError(f"points must be 2-D (B, d), got shape {tuple(points.shape)}")

        idx = self._to_index_tensor(ids, max_bound, id_name)
        if idx.ndim != 1:
            raise ValueError(f"{id_name} must be 1-D (or scalar), got shape {tuple(idx.shape)}")

        b_pts = pts.shape[0]
        b_idx = idx.shape[0]
        if b_pts == b_idx:
            return pts, idx
        if b_pts == 1 and b_idx > 1:
            return pts.expand(b_idx, self.embed_dim), idx
        if b_idx == 1 and b_pts > 1:
            return pts, idx.expand(b_pts)
        raise ValueError(
            f"Incompatible batch sizes between points ({b_pts}) and {id_name} ({b_idx})"
        )

    def _aggregate_slot_depth(
        self,
        slot_depth: th.Tensor,
        active: th.Tensor,
        crisp: bool = False,
    ) -> th.Tensor:
        """Aggregate per-component signed depth ``(B, K)`` over ``active`` slots."""
        has_active = active.any(dim=1, keepdim=True)  # (B, 1)
        inactive_fill = th.where(
            has_active,
            th.full_like(slot_depth, float("-inf")),
            th.full_like(slot_depth, _EMPTY_DEPTH_FLOOR),
        )
        masked = th.where(active, slot_depth, inactive_fill)
        if self.agg == "max" or crisp:
            return th.amax(masked, dim=1)
        lse = self.agg_tau * th.logsumexp(masked / self.agg_tau, dim=1)
        return th.where(
            has_active.squeeze(1),
            lse,
            th.full_like(lse, _EMPTY_DEPTH_FLOOR),
        )

    def depth(
        self,
        points: th.Tensor,
        class_ids: Union[int, Sequence[int], th.Tensor],
        top_id: Optional[int] = None,
        bot_id: Optional[int] = None,
        crisp: bool = False,
    ) -> th.Tensor:
        """Compute signed box depth :math:`\\operatorname{dep}(z, A)` of ``points`` in ``class_ids``.

        Reuses :func:`~mowl.nn.el.structural.inside_measure` when ``agg == "max"``
        (or ``crisp=True``) and uses ``agg_tau * logsumexp(depth / agg_tau)`` over
        active components when ``agg == "lse"`` and ``crisp=False``.

        Guarantees:
        - Returns a 1-D tensor of shape ``(B,)``.
        - Under ``agg="max"`` (or ``crisp=True``), entry ``b`` is strictly
          positive if and only if ``points[b]`` lies strictly inside the interior
          of at least one active component box of ``class_ids[b]`` (or
          ``class_ids[b] == top_id``).
        - Classes with zero active components (or ``class_ids[b] == bot_id``)
          return the finite floor ``-1e6``.

        Non-guarantees:
        - Under ``agg="max"``, gradients flow only to the single active component
          attaining the maximum depth.

        Parameters
        ----------
        points : torch.Tensor
            Point coordinates of shape ``(B, d)`` (or ``(1, d)`` / ``(d,)``).
        class_ids : Union[int, Sequence[int], torch.Tensor]
            Class indices of shape ``(B,)`` (or scalar).
        top_id : Optional[int], default None
            Optional override for ``self.top_id``.
        bot_id : Optional[int], default None
            Optional override for ``self.bot_id``.
        crisp : bool, default False
            If ``True``, forces hard ``max`` aggregation even when ``self.agg == "lse"``.

        Returns
        -------
        torch.Tensor
            Signed depths of shape ``(B,)``.
        """
        pts, idx = self._align_points_and_ids(
            points, class_ids, self.nb_classes, "class_ids"
        )
        bsz = int(pts.shape[0])
        if bsz == 0:
            return th.empty((0,), dtype=pts.dtype, device=pts.device)

        regions = self.class_regions(idx)
        if pts.dtype != regions.cen.dtype:
            regions = Regions(
                cen=regions.cen.to(dtype=pts.dtype),
                w=regions.w.to(dtype=pts.dtype),
                active=regions.active,
            )

        if self.agg == "max" or crisp:
            dep = inside_measure(pts, regions)
        else:
            coord_depth = regions.w - th.abs(pts.unsqueeze(1) - regions.cen)
            slot_depth = th.amin(coord_depth, dim=-1)
            dep = self._aggregate_slot_depth(slot_depth, regions.active, crisp=False)

        eff_top = self.top_id if top_id is None else top_id
        eff_bot = self.bot_id if bot_id is None else bot_id
        if eff_top is not None:
            dep = th.where(
                idx == eff_top,
                th.full_like(dep, _TOP_DEPTH_CEIL),
                dep,
            )
        if eff_bot is not None:
            dep = th.where(
                idx == eff_bot,
                th.full_like(dep, _EMPTY_DEPTH_FLOOR),
                dep,
            )
        return dep

    def membership(
        self,
        points: th.Tensor,
        class_ids: Union[int, Sequence[int], th.Tensor],
        top_id: Optional[int] = None,
        bot_id: Optional[int] = None,
    ) -> th.Tensor:
        """Compute soft class membership :math:`m(z, A) = \\sigma(\\operatorname{dep}(z, A) / \\tau)`.

        Guarantees:
        - Returns a 1-D tensor of shape ``(B,)`` with values in ``[0, 1]``.
        - Entries with ``class_ids == top_id`` evaluate to exact ``1.0`` and
          entries with ``class_ids == bot_id`` evaluate to exact ``0.0``.

        Non-guarantees:
        - For ordinary classes at finite ``tau > 0``, values lie strictly in
          ``(0, 1)`` unless ``depth / tau`` saturates floating-point precision.

        Parameters
        ----------
        points : torch.Tensor
            Point coordinates of shape ``(B, d)`` (or ``(1, d)`` / ``(d,)``).
        class_ids : Union[int, Sequence[int], torch.Tensor]
            Class indices of shape ``(B,)`` (or scalar).
        top_id : Optional[int], default None
            Optional override for ``self.top_id``.
        bot_id : Optional[int], default None
            Optional override for ``self.bot_id``.

        Returns
        -------
        torch.Tensor
            Soft memberships of shape ``(B,)``.
        """
        pts, idx = self._align_points_and_ids(
            points, class_ids, self.nb_classes, "class_ids"
        )
        dep = self.depth(pts, idx, top_id=top_id, bot_id=bot_id)
        m = th.sigmoid(dep / self.tau.to(dtype=dep.dtype))

        eff_top = self.top_id if top_id is None else top_id
        eff_bot = self.bot_id if bot_id is None else bot_id
        if eff_top is not None:
            m = th.where(idx == eff_top, th.ones_like(m), m)
        if eff_bot is not None:
            m = th.where(idx == eff_bot, th.zeros_like(m), m)
        return m

    def _align_role_inputs(
        self,
        x_points: th.Tensor,
        y_points: th.Tensor,
        rel_ids: Union[int, Sequence[int], th.Tensor],
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Validate and broadcast ``x_points``, ``y_points``, and ``rel_ids`` to ``(B, d)`` and ``(B,)``."""
        for name, p in (("x_points", x_points), ("y_points", y_points)):
            if not isinstance(p, th.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(p).__name__}")
        xp = x_points.unsqueeze(0) if x_points.ndim == 1 else x_points
        yp = y_points.unsqueeze(0) if y_points.ndim == 1 else y_points
        if xp.ndim != 2 or xp.shape[1] != self.embed_dim:
            raise ValueError(
                f"x_points must have shape (B, {self.embed_dim}), got {tuple(x_points.shape)}"
            )
        if yp.ndim != 2 or yp.shape[1] != self.embed_dim:
            raise ValueError(
                f"y_points must have shape (B, {self.embed_dim}), got {tuple(y_points.shape)}"
            )

        idx = self._to_index_tensor(rel_ids, self.nb_rels, "rel_ids")
        if idx.ndim != 1:
            raise ValueError(f"rel_ids must be 1-D (or scalar), got shape {tuple(idx.shape)}")

        sizes = {xp.shape[0], yp.shape[0], idx.shape[0]}
        non_unit = {s for s in sizes if s != 1}
        if len(non_unit) > 1:
            raise ValueError(
                f"Incompatible batch sizes for x_points ({xp.shape[0]}), "
                f"y_points ({yp.shape[0]}), and rel_ids ({idx.shape[0]})"
            )
        target_b = max(sizes) if (1 in sizes and len(non_unit) == 1) else next(iter(sizes))
        if xp.shape[0] != target_b:
            xp = xp.expand(target_b, self.embed_dim)
        if yp.shape[0] != target_b:
            yp = yp.expand(target_b, self.embed_dim)
        if idx.shape[0] != target_b:
            idx = idx.expand(target_b)
        return xp, yp, idx

    def role_depth(
        self,
        x_points: th.Tensor,
        y_points: th.Tensor,
        rel_ids: Union[int, Sequence[int], th.Tensor],
        crisp: bool = False,
    ) -> th.Tensor:
        """Compute signed product-box depth :math:`\\max_k \\min(\\operatorname{dep}(x, H_{r,k}), \\operatorname{dep}(y, T_{r,k}))`.

        Guarantees:
        - Returns a 1-D tensor of shape ``(B,)``.
        - Under ``agg="max"`` (or ``crisp=True``), entry ``b`` is strictly
          positive if and only if there is an active component ``k`` of role
          ``rel_ids[b]`` with ``x_points[b]`` strictly inside :math:`H_{r,k}`
          and ``y_points[b]`` strictly inside :math:`T_{r,k}`.
        - Roles with zero active components return the finite floor ``-1e6``.

        Non-guarantees:
        - Under ``agg="max"``, gradients flow only to the winning active role
          component and bottleneck coordinate.

        Parameters
        ----------
        x_points : torch.Tensor
            Source point coordinates of shape ``(B, d)`` (or ``(1, d)`` / ``(d,)``).
        y_points : torch.Tensor
            Target point coordinates of shape ``(B, d)`` (or ``(1, d)`` / ``(d,)``).
        rel_ids : Union[int, Sequence[int], torch.Tensor]
            Role indices of shape ``(B,)`` (or scalar).
        crisp : bool, default False
            If ``True``, forces hard ``max`` aggregation even when ``self.agg == "lse"``.

        Returns
        -------
        torch.Tensor
            Signed role depths of shape ``(B,)``.
        """
        xp, yp, idx = self._align_role_inputs(x_points, y_points, rel_ids)
        bsz = int(xp.shape[0])
        if bsz == 0:
            return th.empty((0,), dtype=xp.dtype, device=xp.device)

        h_reg, t_reg = self.role_regions(idx)
        h_cen = h_reg.cen.to(dtype=xp.dtype)
        h_w = h_reg.w.to(dtype=xp.dtype)
        t_cen = t_reg.cen.to(dtype=yp.dtype)
        t_w = t_reg.w.to(dtype=yp.dtype)

        h_dep = th.amin(h_w - th.abs(xp.unsqueeze(1) - h_cen), dim=-1)  # (B, K_r)
        t_dep = th.amin(t_w - th.abs(yp.unsqueeze(1) - t_cen), dim=-1)  # (B, K_r)
        slot_depth = th.minimum(h_dep, t_dep)  # (B, K_r)
        return self._aggregate_slot_depth(slot_depth, h_reg.active, crisp=crisp)

    def role_membership(
        self,
        x_points: th.Tensor,
        y_points: th.Tensor,
        rel_ids: Union[int, Sequence[int], th.Tensor],
    ) -> th.Tensor:
        """Compute soft role membership :math:`\\rho(x, y, r) = \\sigma(\\operatorname{role\\_depth}(x, y, r) / \\tau)`.

        Guarantees:
        - Returns a 1-D tensor of shape ``(B,)`` with values in ``[0, 1]``.

        Non-guarantees:
        - At finite ``tau > 0``, values lie in ``(0, 1)`` unless saturated.

        Parameters
        ----------
        x_points : torch.Tensor
            Source point coordinates of shape ``(B, d)`` (or ``(1, d)`` / ``(d,)``).
        y_points : torch.Tensor
            Target point coordinates of shape ``(B, d)`` (or ``(1, d)`` / ``(d,)``).
        rel_ids : Union[int, Sequence[int], th.Tensor]
            Role indices of shape ``(B,)`` (or scalar).

        Returns
        -------
        torch.Tensor
            Soft role memberships of shape ``(B,)``.
        """
        dep = self.role_depth(x_points, y_points, rel_ids)
        return th.sigmoid(dep / self.tau.to(dtype=dep.dtype))

    def expand_components(
        self,
        new_K: int,
        new_role_K: Optional[int] = None,
    ) -> None:
        """Expand class capacity to ``new_K`` (and optionally role capacity to ``new_role_K``) by appending inactive zero-width components.

        Implements requirement R3 (monotone capacity).

        Guarantees:
        - Appends ``new_K - n_components`` inactive zero-width class components
          (and ``new_role_K - n_role_components`` inactive zero-width role
          components when ``new_role_K`` is provided).
        - Outputs of :meth:`depth`, :meth:`membership`, :meth:`role_depth`, and
          :meth:`role_membership` are bitwise identical before and after the call.
        - Raises :class:`ValueError` if ``new_K < n_components`` (or
          ``new_role_K < n_role_components``).

        Non-guarantees:
        - Replaces ``nn.Parameter`` instances with larger tensors, so any
          external optimizer must be re-initialised if training continues.

        Parameters
        ----------
        new_K : int
            Target number of class components :math:`\\ge \\text{n\\_components}`.
        new_role_K : Optional[int], default None
            Optional target number of role components :math:`\\ge \\text{n\\_role\\_components}`.
        """
        target_k = _validate_int_ge("new_K", new_K, self.n_components)
        if target_k > self.n_components:
            extra = target_k - self.n_components
            device = self.class_centers.device
            dtype = self.class_centers.dtype
            with th.no_grad():
                pad_cen = th.zeros(
                    (self.nb_classes, extra, self.embed_dim), dtype=dtype, device=device
                )
                pad_raw = th.full(
                    (self.nb_classes, extra, self.embed_dim),
                    _INACTIVE_RAW_WIDTH,
                    dtype=dtype,
                    device=device,
                )
                pad_act = th.zeros(
                    (self.nb_classes, extra), dtype=th.bool, device=device
                )
                self.class_centers = nn.Parameter(
                    th.cat([self.class_centers.data, pad_cen], dim=1)
                )
                self.class_raw_widths = nn.Parameter(
                    th.cat([self.class_raw_widths.data, pad_raw], dim=1)
                )
                self.class_active = th.cat([self.class_active, pad_act], dim=1)
                self.n_components = target_k

        if new_role_K is not None:
            target_rk = _validate_int_ge(
                "new_role_K", new_role_K, self.n_role_components
            )
            if target_rk > self.n_role_components:
                extra_r = target_rk - self.n_role_components
                device = self.role_head_centers.device
                dtype = self.role_head_centers.dtype
                with th.no_grad():
                    pad_cen = th.zeros(
                        (self.nb_rels, extra_r, self.embed_dim),
                        dtype=dtype,
                        device=device,
                    )
                    pad_raw = th.full(
                        (self.nb_rels, extra_r, self.embed_dim),
                        _INACTIVE_RAW_WIDTH,
                        dtype=dtype,
                        device=device,
                    )
                    pad_act = th.zeros(
                        (self.nb_rels, extra_r), dtype=th.bool, device=device
                    )
                    self.role_head_centers = nn.Parameter(
                        th.cat([self.role_head_centers.data, pad_cen], dim=1)
                    )
                    self.role_head_raw_widths = nn.Parameter(
                        th.cat([self.role_head_raw_widths.data, pad_raw], dim=1)
                    )
                    self.role_tail_centers = nn.Parameter(
                        th.cat([self.role_tail_centers.data, pad_cen], dim=1)
                    )
                    self.role_tail_raw_widths = nn.Parameter(
                        th.cat([self.role_tail_raw_widths.data, pad_raw], dim=1)
                    )
                    self.role_active = th.cat([self.role_active, pad_act], dim=1)
                    self.n_role_components = target_rk

    def exact_limit_init(
        self,
        least_model_labels: Union[th.Tensor, np.ndarray],
        edges: Optional[
            Union[
                th.Tensor,
                np.ndarray,
                Iterable[Tuple[int, int, int]],
                Mapping[int, Iterable[Tuple[int, int]]],
            ]
        ] = None,
        delta: float = 0.05,
        seed: int = 0,
        dead_mask: Optional[Union[th.Tensor, np.ndarray]] = None,
    ) -> None:
        """Initialise parameters to the exact-limit §5 construction from GIVEN least-model labels and edges.

        Requires ``n_components >= nb_elements``. Places elements at random
        generic positions rescaled so that :math:`\\|p_i - p_j\\|_\\infty \\ge 4\\delta > 2\\delta`
        for all :math:`i \\neq j` (ensuring boxes of distinct elements are
        strictly disjoint). Sets component :math:`k` of class :math:`A` to
        :math:`\\operatorname{Box}(p_k, \\delta)` if and only if
        ``least_model_labels[k, A]`` is truthy (and marks unused components
        inactive with width ``0``). Sets each role edge :math:`(x, y, r)` to a
        product component :math:`\\operatorname{Box}(p_x, \\delta) \\times \\operatorname{Box}(p_y, \\delta)`.

        Guarantees:
        - Does not call any reasoner or completion algorithm; uses only the
          caller-supplied ``least_model_labels`` and ``edges``.
        - Distinct element boxes :math:`\\operatorname{Box}(p_i, \\delta)` and
          :math:`\\operatorname{Box}(p_j, \\delta)` are separated by at least
          :math:`2\\delta` in :math:`\\ell_\\infty` norm.
        - Unused class and role components have width ``0.0`` and ``active == False``.
        - Raises :class:`ValueError` if ``n_components < nb_elements`` or if
          ``least_model_labels`` does not have shape ``(nb_elements, nb_classes)``.

        Non-guarantees:
        - Soundness of the resulting configuration holds if and only if the
          supplied ``least_model_labels`` and ``edges`` form a valid model of the
          ontology.

        Parameters
        ----------
        least_model_labels : Union[torch.Tensor, numpy.ndarray]
            Boolean or binary matrix of shape ``(nb_elements, nb_classes)`` where
            entry ``(k, A)`` is truthy iff element ``k`` belongs to class ``A``
            in the least model.
        edges : Optional[Union[torch.Tensor, numpy.ndarray, Iterable[Tuple[int, int, int]], Mapping[int, Iterable[Tuple[int, int]]]]], default None
            Role edges as an iterable/tensor of ``(x, y, r)`` triples or a
            mapping ``{r: [(x, y), ...]}``.
        delta : float, default 0.05
            Positive box half-width :math:`\\delta > 0` around each element.
        seed : int, default 0
            Random seed for generic element position generation.
        dead_mask : Optional[Union[torch.Tensor, numpy.ndarray]], default None
            Optional boolean mask of shape ``(nb_elements,)`` marking dead
            elements; when ``None``, derived from ``least_model_labels`` (a class
            element ``k < nb_classes`` is live iff ``labels[k, k]`` is truthy and
            ``k != bot_id``; an individual element ``k >= nb_classes`` is live iff
            ``labels[k].any()`` is truthy).
        """
        if self.n_components < self.nb_elements:
            raise ValueError(
                f"exact_limit_init requires n_components >= nb_elements ({self.nb_elements}), "
                f"got n_components={self.n_components}"
            )
        delta_f = _validate_positive_float("delta", delta)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")

        device = self.class_centers.device
        dtype = self.class_centers.dtype
        labels_t = th.as_tensor(least_model_labels, device=device)
        if labels_t.shape != (self.nb_elements, self.nb_classes):
            raise ValueError(
                f"least_model_labels must have shape ({self.nb_elements}, {self.nb_classes}), "
                f"got {tuple(labels_t.shape)}"
            )
        labels_bool = labels_t > 0

        # Parse edges into per-role lists of (x, y) pairs
        edges_by_role: List[List[Tuple[int, int]]] = [
            [] for _ in range(self.nb_rels)
        ]
        if edges is not None:
            if isinstance(edges, Mapping):
                for r_key, pairs in edges.items():
                    r_idx = _validate_int_ge("role_id", int(r_key), 0)
                    if r_idx >= self.nb_rels:
                        raise ValueError(
                            f"Role index {r_idx} out of range [0, {self.nb_rels})"
                        )
                    for x_i, y_i in pairs:
                        xi = _validate_int_ge("x", int(x_i), 0)
                        yi = _validate_int_ge("y", int(y_i), 0)
                        if xi >= self.nb_elements or yi >= self.nb_elements:
                            raise ValueError(
                                f"Edge element index ({xi}, {yi}) out of range [0, {self.nb_elements})"
                            )
                        edges_by_role[r_idx].append((xi, yi))
            else:
                if isinstance(edges, (th.Tensor, np.ndarray)):
                    edges_arr = th.as_tensor(edges, dtype=th.long).cpu()
                    if edges_arr.numel() == 0:
                        triples_list: List[Tuple[int, int, int]] = []
                    elif edges_arr.ndim == 2 and edges_arr.shape[1] == 3:
                        triples_list = [
                            (int(row[0]), int(row[1]), int(row[2]))
                            for row in edges_arr.tolist()
                        ]
                    else:
                        raise ValueError(
                            f"Tensor edges must have shape (E, 3), got {tuple(edges_arr.shape)}"
                        )
                else:
                    triples_list = [
                        (int(t[0]), int(t[1]), int(t[2])) for t in edges
                    ]
                for xi, yi, ri in triples_list:
                    if not (0 <= ri < self.nb_rels):
                        raise ValueError(
                            f"Role index {ri} out of range [0, {self.nb_rels})"
                        )
                    if not (0 <= xi < self.nb_elements and 0 <= yi < self.nb_elements):
                        raise ValueError(
                            f"Edge element index ({xi}, {yi}) out of range [0, {self.nb_elements})"
                        )
                    edges_by_role[ri].append((xi, yi))

        max_role_edges = max((len(lst) for lst in edges_by_role), default=0)
        if max_role_edges > self.n_role_components:
            self.expand_components(self.n_components, new_role_K=max_role_edges)

        # Generate generic random positions on CPU with explicit Generator, then rescale
        gen = th.Generator(device="cpu")
        gen.manual_seed(int(seed))
        raw_pos = th.rand((self.nb_elements, self.embed_dim), dtype=th.float64, generator=gen)
        # Ensure strict separation in coordinate 0 even if random draws collide in 1-D
        perm = th.randperm(self.nb_elements, generator=gen).to(dtype=th.float64)
        raw_pos[:, 0] = perm + 0.5 * raw_pos[:, 0]

        if self.nb_elements > 1:
            diff_inf = th.amax(
                th.abs(raw_pos.unsqueeze(1) - raw_pos.unsqueeze(0)), dim=-1
            )
            eye_mask = th.eye(self.nb_elements, dtype=th.bool)
            diff_inf = diff_inf.masked_fill(eye_mask, float("inf"))
            min_sep = float(diff_inf.min().item())
            target_sep = 4.0 * delta_f
            if min_sep < target_sep:
                raw_pos = raw_pos * (target_sep / max(min_sep, 1e-12))

        pos_dev = raw_pos.to(device=device, dtype=dtype)
        raw_delta = _inv_softplus(delta_f)

        # Determine dead_mask
        if dead_mask is not None:
            dead_bool = th.as_tensor(dead_mask, dtype=th.bool, device=device)
            if dead_bool.shape != (self.nb_elements,):
                raise ValueError(
                    f"dead_mask must have shape ({self.nb_elements},), got {tuple(dead_bool.shape)}"
                )
        else:
            cls_ids = th.arange(self.nb_classes, device=device)
            live_cls = labels_bool[cls_ids, cls_ids].clone()
            if self.bot_id is not None:
                live_cls[self.bot_id] = False
                live_cls = live_cls & (~labels_bool[: self.nb_classes, self.bot_id])
            if self.nb_elements > self.nb_classes:
                live_inds = labels_bool[self.nb_classes :].any(dim=1)
                if self.bot_id is not None:
                    live_inds = live_inds & (
                        ~labels_bool[self.nb_classes :, self.bot_id]
                    )
                live_all = th.cat([live_cls, live_inds], dim=0)
            else:
                live_all = live_cls
            dead_bool = ~live_all

        with th.no_grad():
            self.elem_pos.weight.copy_(pos_dev)
            self.dead_logits.copy_(
                th.where(
                    dead_bool,
                    th.full_like(self.dead_logits, _DEAD_LOGIT),
                    th.full_like(self.dead_logits, _LIVE_LOGIT),
                )
            )

            # Initialize all class components to inactive zero-width
            self.class_centers.zero_()
            self.class_raw_widths.fill_(_INACTIVE_RAW_WIDTH)
            self.class_active.zero_()

            # Component k (0 <= k < nb_elements) of class A is Box(p_k, delta) iff labels[k, A]
            active_ca = labels_bool.t().contiguous()  # (nb_classes, nb_elements)
            if self.bot_id is not None:
                active_ca[self.bot_id, :] = False

            pos_expanded = pos_dev.unsqueeze(0).expand(
                self.nb_classes, self.nb_elements, self.embed_dim
            )
            self.class_centers[:, : self.nb_elements, :] = th.where(
                active_ca.unsqueeze(-1),
                pos_expanded,
                th.zeros_like(pos_expanded),
            )
            self.class_raw_widths[:, : self.nb_elements, :] = th.where(
                active_ca.unsqueeze(-1),
                th.full_like(pos_expanded, raw_delta),
                th.full_like(pos_expanded, _INACTIVE_RAW_WIDTH),
            )
            self.class_active[:, : self.nb_elements] = active_ca

            # Initialize all role components to inactive zero-width, then populate edges
            self.role_head_centers.zero_()
            self.role_head_raw_widths.fill_(_INACTIVE_RAW_WIDTH)
            self.role_tail_centers.zero_()
            self.role_tail_raw_widths.fill_(_INACTIVE_RAW_WIDTH)
            self.role_active.zero_()

            for r_idx, r_edges in enumerate(edges_by_role):
                for k_idx, (x_i, y_i) in enumerate(r_edges):
                    self.role_head_centers[r_idx, k_idx].copy_(pos_dev[x_i])
                    self.role_head_raw_widths[r_idx, k_idx].fill_(raw_delta)
                    self.role_tail_centers[r_idx, k_idx].copy_(pos_dev[y_i])
                    self.role_tail_raw_widths[r_idx, k_idx].fill_(raw_delta)
                    self.role_active[r_idx, k_idx] = True
