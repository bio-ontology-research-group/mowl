"""The parts of GroundedEL: a finite domain of elements, classes and relations around them.

GroundedEL reasons over a *finite domain*: one element per class (its witness), then one per
individual. Classes and relations are regions of the space the elements live in, and an
axiom is a constraint on the elements (:mod:`mowl.representation.library.grounded_logic`).
The parts are independent objects, each owning its tables:

=========================  ===================================================================
:class:`ElementDomain`     **The shared part.** The positions of the elements (``elem_pos``,
                           optionally computed by an encoder from features), their deadness
                           :math:`\\delta_x \\in [0, 1]` (an element that cannot exist), and the
                           temperature ``tau`` at which a signed depth becomes a degree of
                           membership. Class model, relation model and individual model all
                           read it; none of them owns it.
:class:`UnionDepth`        The measure: the signed depth of a point in a union of boxes, the
                           maximum (or a log-sum-exp) over the active boxes of the smallest
                           margin over the coordinates; its soft reading
                           :math:`\\sigma(\\text{depth} / \\tau)` and its crisp reading
                           (depth > 0); the comparison of two unions of boxes.
:class:`BoxUnionClasses`   The class model: a union of ``n_components`` boxes per class, some
                           inactive (``centers``, ``raw_widths``, ``active``), with the
                           constant classes top (depth :math:`+10^6`) and bottom
                           (:math:`-10^6`).
:class:`ProductBoxRelations`  A relation model: a union of products of boxes
                           :math:`H_{r,k} \\times T_{r,k}` (a pair is related if its head
                           is in :math:`H` and its tail in :math:`T`).
:class:`TranslationBoxRelations`  The other relation model: a union of boxes of
                           translations :math:`V_{r,k}` (a pair is related if
                           :math:`y - x \\in V_r`); :math:`\\exists r.B` is then a union of
                           boxes again (:meth:`~TranslationBoxRelations.preimage_boxes`).
:class:`ElementPoints`     The individual model: an individual is the point of its element
                           (a view of the shared part, owning nothing).
=========================  ===================================================================

The class and relation models reach the shared part and the measure through references they
do not own (the shared part is not a sub-module of theirs, so its tables appear once in a
``state_dict``: under the owner of the combination, :class:`mowl.nn.GroundedELModule`).

The sampling of elements and of points in these regions (:mod:`mowl.nn.el.grounded.sampling`)
and the query algebra over unions of boxes (:mod:`mowl.nn.el.grounded.queries`) are general
machinery over this interface: they read ``positions``, ``class_regions``, ``role_regions``,
``depth`` and ``role_depth`` of any geometry that has them.

Interface summary (for unifying with other constructions)::

    ElementDomain(nb_elements, nb_classes, embed_dim, tau) -> positions(ids), dead(ids), tau
    UnionDepth(agg, agg_tau) -> depth(points, Regions), aggregate(slot_depth, active),
                                soft(depth, tau), crisp(depth), inside(a, b), apart(a, b)
    BoxUnionClasses(nb_classes, K, d, measure, domain) -> regions(ids), depth(points, ids),
                                membership(points, ids), forward(ids) -> MaskedUnionOfBoxes
    ProductBoxRelations / TranslationBoxRelations(nb_rels, K_r, d, measure, domain)
        -> regions(ids), depth(x, y, ids), membership(x, y, ids), grid_membership(...)
"""
from __future__ import annotations

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from mowl.nn.el.structural import Regions, inside_measure
from mowl.representation.class_models import Classes
from mowl.representation.measures import RegionMeasure
from mowl.representation.regions import ConstructorUnavailable
from mowl.representation.relation_models import Relations
from mowl.representation.library.other_regions import MaskedUnionOfBoxes, _BIG
from mowl.representation.library.shapes_regions import Point

__all__ = ["ElementDomain", "UnionDepth", "BoxUnionClasses", "ProductBoxRelations",
           "TranslationBoxRelations", "ElementPoints", "ExistsRegions", "index_tensor",
           "INACTIVE_RAW_WIDTH", "EMPTY_DEPTH_FLOOR", "TOP_DEPTH_CEIL", "GROUNDED_PARTS",
           "register_grounded_part"]

#: Raw width of an inactive (padding) component: softplus of it is ~0.
INACTIVE_RAW_WIDTH: float = -100.0
#: Depth of a point in a class with no active box, or in bottom.
EMPTY_DEPTH_FLOOR: float = -1e6
#: Depth of every point in top.
TOP_DEPTH_CEIL: float = 1e6


# -- validation helpers (shared by the parts and the module) -----------------------------------
def validate_positive_float(name, val):
    """``val`` as a strictly positive finite float."""
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


def validate_finite_float(name, val):
    """``val`` as a finite float."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {val!r}")
    v = float(val)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be finite, got {val!r}")
    return v


def validate_int_ge(name, val, min_val):
    """``val`` as an integer ``>= min_val`` (booleans rejected)."""
    if isinstance(val, bool) or not isinstance(val, int) or val < min_val:
        raise ValueError(f"{name} must be an int >= {min_val}, got {val!r}")
    return int(val)


def inv_softplus(y):
    """The inverse of softplus, :math:`\\log(e^y - 1)` for :math:`y > 0`."""
    if y <= 0.0:
        raise ValueError(f"delta must be > 0, got {y!r}")
    if y > 20.0:
        return y
    return y + math.log(-math.expm1(-y))


def validate_inverse_of(nb_rels, inverse_of):
    """``inverse_of`` (length ``nb_rels``, ``-1`` or a partner, symmetric, never itself) as a
    ``LongTensor``."""
    if inverse_of is None:
        return th.full((nb_rels,), -1, dtype=th.long)
    if isinstance(inverse_of, th.Tensor):
        if (inverse_of.dtype == th.bool or inverse_of.is_floating_point()
                or inverse_of.is_complex() or inverse_of.ndim != 1):
            raise ValueError(
                f"inverse_of tensor must be a 1-D integer tensor, got dtype={inverse_of.dtype}, "
                f"shape={tuple(inverse_of.shape)}")
        inv_list = [int(x) for x in inverse_of.detach().cpu().tolist()]
    elif isinstance(inverse_of, (list, tuple, np.ndarray)):
        if isinstance(inverse_of, np.ndarray) and (
                inverse_of.ndim != 1 or inverse_of.dtype == np.bool_):
            raise ValueError(
                f"inverse_of array must be 1-D integer array, got dtype={inverse_of.dtype}, "
                f"shape={inverse_of.shape}")
        inv_list = []
        for x in inverse_of:
            if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
                raise ValueError(f"inverse_of entries must be integers, got {x!r}")
            inv_list.append(int(x))
    else:
        raise TypeError(
            f"inverse_of must be a Sequence[int], Tensor, or None, got {type(inverse_of).__name__}")
    if len(inv_list) != nb_rels:
        raise ValueError(f"inverse_of must have length nb_rels ({nb_rels}), got {len(inv_list)}")
    for r, p in enumerate(inv_list):
        if p == -1:
            continue
        if not (0 <= p < nb_rels):
            raise ValueError(f"inverse_of[{r}] = {p} out of range [-1, {nb_rels})")
        if p == r:
            raise ValueError(f"inverse_of[{r}] cannot be self-inverse ({r})")
        if inv_list[p] != r:
            raise ValueError(
                f"inverse_of must be symmetric: inverse_of[{r}] = {p} but inverse_of[{p}] = "
                f"{inv_list[p]}")
    return th.tensor(inv_list, dtype=th.long)


def index_tensor(ids, max_bound, name, device):
    """``ids`` (int, sequence, array or tensor) as a ``LongTensor`` on ``device`` (a scalar
    becomes one entry), every entry in ``[0, max_bound)``."""
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
                f"{name} indices out of range [0, {max_bound}): min={min_i}, max={max_i}")
    return idx


def _one_dim(idx, name):
    if idx.ndim != 1:
        raise ValueError(f"{name} must be 1-D (or scalar), got shape {tuple(idx.shape)}")
    return idx


def align_points_and_ids(points, ids, max_bound, id_name, embed_dim, device):
    """Validate and broadcast ``points`` ``(B, d)`` and ``ids`` ``(B,)``."""
    if not isinstance(points, th.Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points).__name__}")
    if points.ndim == 1:
        if points.shape[0] != embed_dim:
            raise ValueError(
                f"1-D points must have length embed_dim={embed_dim}, got {tuple(points.shape)}")
        pts = points.unsqueeze(0)
    elif points.ndim == 2:
        if points.shape[1] != embed_dim:
            raise ValueError(
                f"points must have second dimension embed_dim={embed_dim}, got "
                f"{tuple(points.shape)}")
        pts = points
    else:
        raise ValueError(f"points must be 2-D (B, d), got shape {tuple(points.shape)}")
    idx = _one_dim(index_tensor(ids, max_bound, id_name, device), id_name)
    b_pts, b_idx = pts.shape[0], idx.shape[0]
    if b_pts == b_idx:
        return pts, idx
    if b_pts == 1 and b_idx > 1:
        return pts.expand(b_idx, embed_dim), idx
    if b_idx == 1 and b_pts > 1:
        return pts, idx.expand(b_pts)
    raise ValueError(f"Incompatible batch sizes between points ({b_pts}) and {id_name} ({b_idx})")


def _shared(obj, name, value):
    """Keep a reference to a part this part reads but does not own (not a sub-module, so
    its tables are not repeated in the ``state_dict``)."""
    obj.__dict__[name] = value


# =============================================================================================
# the shared part
# =============================================================================================
class ElementDomain(nn.Module):
    """The finite domain of GroundedEL and what every other part reads of it (the **shared
    part**): elements ``0 .. nb_classes - 1`` are the witnesses of the classes, the rest the
    individuals.

    Owns: ``elem_pos`` (an embedding, the positions :math:`p_x \\in \\mathbb{R}^d`),
    ``dead_logits`` (:math:`\\delta_x \\in [0, 1]`, used through a straight-through clamp),
    ``tau`` (the temperature of the soft memberships, annealed by the training regime),
    ``element_features`` and ``encoder`` (positions of data elements computed from features)
    and the score temperature ``raw_tau_s`` of an inductive encoder.

    :param dead_init: Initial deadness (clamped to ``[0, 1]``).
    :param element_features: ``(nb_elements, F)`` or ``(nb_elements - nb_classes, F)``.
    :param encoder: Module mapping features to positions.
    """

    part = "element domain"

    def __init__(self, nb_elements, nb_classes, embed_dim, tau=1.0, dead_init=0.0,
                 element_features=None, encoder=None):
        super().__init__()
        self.nb_elements, self.nb_classes, self.embed_dim = nb_elements, nb_classes, embed_dim
        tau_f = validate_positive_float("tau", tau)
        if element_features is not None:
            if not isinstance(element_features, th.Tensor) or element_features.ndim != 2:
                raise ValueError(
                    "element_features must be a 2-D torch.Tensor of shape "
                    "(nb_elements, F) or (nb_elements - nb_classes, F).")
            n_inds = nb_elements - nb_classes
            if element_features.shape[0] not in (nb_elements, n_inds):
                raise ValueError(
                    f"element_features.shape[0] must be nb_elements ({nb_elements}) "
                    f"or nb_elements - nb_classes ({n_inds}), got {element_features.shape[0]}")
            self.register_buffer("element_features",
                                 element_features.detach().clone().to(dtype=th.float32))
        else:
            self.element_features = None
        self.encoder = encoder
        self.elem_pos = nn.Embedding(nb_elements, embed_dim)
        nn.init.uniform_(self.elem_pos.weight, a=-0.5, b=0.5)
        self.dead_logits = nn.Parameter(
            th.full((nb_elements,), max(0.0, min(1.0, dead_init)), dtype=th.float32))
        self.register_buffer("tau", th.tensor(tau_f, dtype=th.float32))
        self.raw_tau_s = (nn.Parameter(th.tensor(inv_softplus(0.1), dtype=th.float32))
                          if encoder is not None else None)

    def __setattr__(self, name, value):
        if name == "tau" and "_buffers" in self.__dict__ and "tau" in self._buffers:
            self._buffers["tau"].fill_(validate_positive_float("tau", value))
            return
        super().__setattr__(name, value)
        if name == "encoder" and value is not None and "_parameters" in self.__dict__:
            if self.__dict__.get("raw_tau_s") is None and \
                    self._parameters.get("raw_tau_s") is None and "elem_pos" in self._modules:
                self.ensure_tau_s()

    @property
    def device(self):
        return self.elem_pos.weight.device

    def set_tau(self, t):
        """Set the membership temperature (a positive finite number)."""
        self._buffers["tau"].fill_(validate_positive_float("tau", t))

    def ensure_tau_s(self):
        """The score temperature parameter ``raw_tau_s`` (created at ``softplus = 0.1``)."""
        if self.raw_tau_s is None:
            w = self.elem_pos.weight
            self.raw_tau_s = nn.Parameter(
                th.tensor(inv_softplus(0.1), dtype=w.dtype, device=w.device))
        return self.raw_tau_s

    @property
    def tau_s(self):
        """The positive score temperature :math:`\\max(\\operatorname{softplus}(\\text{raw}),
        10^{-3})` of an inductive encoder (:meth:`ensure_tau_s`)."""
        return th.clamp(F.softplus(self.ensure_tau_s()), min=1e-3)

    def place(self, positions, dead, dead_value=1.0, live_value=0.0):
        """Set the positions ``(nb_elements, d)`` and write ``dead_value`` (``live_value``)
        as the deadness of the elements where ``dead`` is true (false); under ``no_grad``."""
        with th.no_grad():
            self.elem_pos.weight.copy_(positions)
            self.dead_logits.copy_(th.where(dead, th.full_like(self.dead_logits, dead_value),
                                            th.full_like(self.dead_logits, live_value)))

    def clamp_dead(self, fixed_dead=None, dead_value=1.0):
        """Project the deadness onto :math:`[0, 1]` (after an optimiser step) and keep the
        elements ``fixed_dead`` (an index or ``None``) at ``dead_value``."""
        with th.no_grad():
            self.dead_logits.clamp_(0.0, 1.0)
            if fixed_dead is not None:
                self.dead_logits[fixed_dead] = dead_value

    def positions(self, element_ids=None):
        """Positions :math:`p_x` of ``element_ids`` (all elements if ``None``), computed by the
        encoder from the features where features are attached."""
        device = self.device
        scalar_input = isinstance(element_ids, int) and not isinstance(element_ids, bool)
        if element_ids is None:
            idx = th.arange(self.nb_elements, dtype=th.long, device=device)
        else:
            idx = index_tensor(element_ids, self.nb_elements, "element_ids", device)
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
                    sel_feats = feats[flat_idx[is_ind] - self.nb_classes]
                    enc_pos = self.encoder(sel_feats) if self.encoder is not None else sel_feats
                    pos = pos.clone()
                    pos[is_ind] = enc_pos.to(dtype=pos.dtype)
        out = pos.reshape(*orig_shape, self.embed_dim)
        return out.squeeze(0) if scalar_input else out

    def dead(self, element_ids=None):
        """Deadness :math:`\\delta_x \\in [0, 1]` of ``element_ids`` (straight-through clamp of
        ``dead_logits``: unit gradient everywhere)."""
        scalar_input = isinstance(element_ids, int) and not isinstance(element_ids, bool)
        if element_ids is None:
            raw = self.dead_logits
        else:
            raw = self.dead_logits[index_tensor(element_ids, self.nb_elements, "element_ids",
                                                self.device)]
        clamped = th.clamp(raw, 0.0, 1.0)
        out = raw + (clamped - raw).detach()
        return out.squeeze(0) if scalar_input else out


# =============================================================================================
# the measure
# =============================================================================================
class ExistsRegions(Regions):
    """Materialised regions of :math:`\\exists r.B` carrying an ``is_top`` marker."""

    is_top: bool
    is_top_mask: th.Tensor

    def __new__(cls, cen, w, active, is_top=False, is_top_mask=None):
        obj = super().__new__(cls, cen, w, active)
        obj.is_top = bool(is_top)
        obj.is_top_mask = (is_top_mask if is_top_mask is not None else
                           th.zeros((cen.shape[0],), dtype=th.bool, device=cen.device))
        return obj


class UnionDepth(RegionMeasure):
    """The measure of GroundedEL: the signed depth of a point in a union of boxes.

    The depth of :math:`z` in one box is the smallest margin over the coordinates,
    :math:`\\min_m (w_m - |z_m - c_m|)` (positive inside); in a union, the best box:
    the maximum over the active boxes (``agg="max"``) or its smooth bound
    :math:`\\tau_{agg} \\log \\sum_k e^{d_k / \\tau_{agg}}` (``agg="lse"``). A union without
    an active box has depth :data:`EMPTY_DEPTH_FLOOR`. The *soft* reading of a depth is the
    degree :math:`\\sigma(\\text{depth} / \\tau)`, the *crisp* reading
    :math:`[\\text{depth} > 0]`. Between regions, ``inside`` and ``apart`` compare the boxes
    of two unions. Used by the class model and both relation models.
    """

    part = "bounds per coordinate"
    accepts = (MaskedUnionOfBoxes, Point)

    def __init__(self, agg="max", agg_tau=0.1):
        self.agg = agg
        self.agg_tau = agg_tau

    def __setattr__(self, name, value):
        if name == "agg" and (not isinstance(value, str) or value not in ("max", "lse")):
            raise ValueError(f"agg must be 'max' or 'lse', got {value!r}")
        if name == "agg_tau":
            value = validate_positive_float("agg_tau", value)
        object.__setattr__(self, name, value)

    # -- points against boxes -------------------------------------------------------------
    @staticmethod
    def box_depth(points, cen, w):
        """Depth of the points ``(..., d)`` in every box: ``(..., K)`` for boxes
        ``(..., K, d)`` (points broadcast against the box axis)."""
        return th.amin(w - th.abs(points.unsqueeze(-2) - cen), dim=-1)

    def aggregate(self, slot_depth, active, crisp=False):
        """The depth in the union of the boxes ``(B, K)`` of which ``active`` are there."""
        has_active = active.any(dim=1, keepdim=True)
        inactive_fill = th.where(has_active, th.full_like(slot_depth, float("-inf")),
                                 th.full_like(slot_depth, EMPTY_DEPTH_FLOOR))
        masked = th.where(active, slot_depth, inactive_fill)
        if self.agg == "max" or crisp:
            return th.amax(masked, dim=1)
        lse = self.agg_tau * th.logsumexp(masked / self.agg_tau, dim=1)
        return th.where(has_active.squeeze(1), lse, th.full_like(lse, EMPTY_DEPTH_FLOOR))

    @staticmethod
    def crisp_union(slot_depth, active):
        """The crisp depth in the union of boxes for any leading shape: ``slot_depth``
        ``(..., K)``, ``active`` broadcastable to it; a union without an active box has
        depth :data:`EMPTY_DEPTH_FLOOR`."""
        has_act = active.any(dim=-1, keepdim=True)
        fill = th.where(has_act, th.full_like(slot_depth, float("-inf")),
                        th.full_like(slot_depth, EMPTY_DEPTH_FLOOR))
        return th.amax(th.where(active, slot_depth, fill), dim=-1)

    def depth(self, points, regions, crisp=False):
        """Depth of ``points`` ``(B, d)`` in the unions ``regions`` (``Regions`` of
        ``(B, K, d)``)."""
        if points.dtype != regions.cen.dtype:
            regions = Regions(cen=regions.cen.to(dtype=points.dtype),
                              w=regions.w.to(dtype=points.dtype), active=regions.active)
        if self.agg == "max" or crisp:
            return inside_measure(points, regions)
        coord_depth = regions.w - th.abs(points.unsqueeze(1) - regions.cen)
        return self.aggregate(th.amin(coord_depth, dim=-1), regions.active, crisp=False)

    @staticmethod
    def soft(depth, tau):
        """The degree of membership of a depth: :math:`\\sigma(\\text{depth} / \\tau)`."""
        return th.sigmoid(depth / tau.to(dtype=depth.dtype))

    @staticmethod
    def crisp(depth):
        """The crisp membership :math:`[\\text{depth} > 0]` (floating point)."""
        if not isinstance(depth, th.Tensor):
            raise TypeError(f"depth must be a torch.Tensor, got {type(depth).__name__}")
        out_dtype = depth.dtype if depth.is_floating_point() else th.float32
        return (depth > 0).to(dtype=out_dtype)

    # -- regions against regions ----------------------------------------------------------
    def inside(self, a, b, margin=0.0, depth=None):
        """Zero if every active box of ``a`` lies in one active box of ``b`` (sufficient for
        inclusion), otherwise the smallest total violation over the choices of box. If ``a``
        is a point, the hinge of its negative depth (``depth(points, region)``, the reading
        of the geometry that owns the region)."""
        if isinstance(a, Point):
            return th.relu(-depth(a.centre, b)).reshape(-1, 1)
        viol = th.linalg.norm(th.relu(
            (a.cen[:, :, None] - b.cen[:, None]).abs() + a.w[:, :, None] - b.w[:, None]),
            dim=-1)
        viol = th.where(b.active[:, None], viol, th.full_like(viol, _BIG)).amin(dim=2)
        return th.where(a.active, viol, th.zeros_like(viol)).sum(dim=1, keepdim=True)

    def apart(self, a, b, margin=0.0):
        """Exact: zero iff no active box of ``a`` meets one of ``b``."""
        overlap = (a.w[:, :, None] + b.w[:, None] - (a.cen[:, :, None] - b.cen[:, None]).abs()
                   ).amin(dim=-1)
        both = a.active[:, :, None] & b.active[:, None]
        return th.where(both, th.relu(overlap), th.zeros_like(overlap)).sum(
            dim=(1, 2)).reshape(-1, 1)


def _boxes(centers, raw_widths, active):
    """``(cen, w, effective active)`` of boxes from raw tables: widths
    :math:`\\operatorname{softplus}(\\tilde w)` where active, else ``0``; a box with a zero
    width is not active."""
    w_pos = F.softplus(raw_widths)
    w = th.where(active.unsqueeze(-1), w_pos, th.zeros_like(w_pos))
    return centers, w, active & (th.amin(w, dim=-1) > 0.0)


def _pad(table, extra, value):
    pad = th.full((table.shape[0], extra) + tuple(table.shape[2:]), value, dtype=table.dtype,
                  device=table.device)
    return th.cat([table, pad], dim=1)


# =============================================================================================
# the class model
# =============================================================================================
class BoxUnionClasses(Classes):
    """A class is a union of ``n_components`` axis-parallel boxes, some inactive.

    Owns ``centers`` and ``raw_widths`` ``(nb_classes, K, d)`` (half-widths are
    :math:`\\operatorname{softplus}` of the raw widths) and the buffer ``active``
    ``(nb_classes, K)``. Reads the measure (depth in a union) and the shared part
    (temperature). ``top_id`` and ``bot_id`` are the constants top (every point, depth
    :data:`TOP_DEPTH_CEIL`, membership 1) and bottom (no point, :data:`EMPTY_DEPTH_FLOOR`,
    membership 0).

    Part (vocabulary value): ``union of boxes``. Used by: ``grounded``.
    """

    region = MaskedUnionOfBoxes
    part = "union of boxes"
    exact_conjunction = True

    def __init__(self, nb_classes, n_components, embed_dim, measure, domain, top_id=None,
                 bot_id=None):
        super().__init__()
        self.nb_classes, self.n_components = nb_classes, n_components
        self._dim = embed_dim
        self.top_id, self.bot_id = top_id, bot_id
        _shared(self, "measure", measure)
        _shared(self, "domain", domain)
        self.centers = nn.Parameter(th.empty((nb_classes, n_components, embed_dim),
                                             dtype=th.float32))
        self.raw_widths = nn.Parameter(th.empty((nb_classes, n_components, embed_dim),
                                                dtype=th.float32))
        nn.init.uniform_(self.centers, a=-0.5, b=0.5)
        nn.init.constant_(self.raw_widths, inv_softplus(0.2))
        self.register_buffer("active", th.ones((nb_classes, n_components), dtype=th.bool))

    def __len__(self):
        return self.nb_classes

    @property
    def embed_dim(self):
        return self._dim

    def _ids(self, class_ids):
        if class_ids is None:
            return th.arange(self.nb_classes, dtype=th.long, device=self.centers.device)
        return _one_dim(index_tensor(class_ids, self.nb_classes, "class_ids",
                                     self.centers.device), "class_ids")

    def regions(self, class_ids=None):
        """The boxes of ``class_ids`` (all if ``None``) as
        :class:`~mowl.nn.el.structural.Regions` ``(B, K, d)``."""
        idx = self._ids(class_ids)
        cen, w, active = _boxes(self.centers[idx], self.raw_widths[idx], self.active[idx])
        return Regions(cen=cen, w=w, active=active)

    def forward(self, ids):
        return MaskedUnionOfBoxes.of(self.regions(ids), ids)

    def all(self):
        return self(th.arange(self.nb_classes, device=self.centers.device))

    def _constants(self, values, idx, top_id, bot_id, top, bottom):
        eff_top = self.top_id if top_id is None else top_id
        eff_bot = self.bot_id if bot_id is None else bot_id
        if eff_top is not None:
            values = th.where(idx == eff_top, top(values), values)
        if eff_bot is not None:
            values = th.where(idx == eff_bot, bottom(values), values)
        return values

    def depth(self, points, class_ids, top_id=None, bot_id=None, crisp=False):
        """Signed depth ``(B,)`` of ``points`` ``(B, d)`` in the classes ``class_ids``."""
        pts, idx = align_points_and_ids(points, class_ids, self.nb_classes, "class_ids",
                                        self._dim, self.centers.device)
        if int(pts.shape[0]) == 0:
            return th.empty((0,), dtype=pts.dtype, device=pts.device)
        dep = self.measure.depth(pts, self.regions(idx), crisp=crisp)
        return self._constants(dep, idx, top_id, bot_id,
                               lambda d: th.full_like(d, TOP_DEPTH_CEIL),
                               lambda d: th.full_like(d, EMPTY_DEPTH_FLOOR))

    def membership(self, points, class_ids, top_id=None, bot_id=None):
        """Degree ``(B,)`` of membership (the soft reading of the depth; exactly 1 in top and
        0 in bottom)."""
        pts, idx = align_points_and_ids(points, class_ids, self.nb_classes, "class_ids",
                                        self._dim, self.centers.device)
        dep = self.depth(pts, idx, top_id=top_id, bot_id=bot_id)
        m = self.measure.soft(dep, self.domain.tau)
        return self._constants(m, idx, top_id, bot_id, th.ones_like, th.zeros_like)

    def depth_table(self, points, class_ids, chunk_size=512):
        """Crisp depth ``(L, C)`` of every point ``(L, d)`` in every class of ``class_ids``
        (in chunks of classes; top and bottom constants)."""
        n_points, n_classes = int(points.shape[0]), int(class_ids.numel())
        out = th.empty((n_points, n_classes), dtype=th.float32, device=points.device)
        if n_points == 0 or n_classes == 0:
            return out
        for c_s in range(0, n_classes, chunk_size):
            c_sub = class_ids[c_s:c_s + chunk_size]
            reg = self.regions(c_sub)                                  # (C_sub, K, d)
            slot = self.measure.box_depth(points.unsqueeze(1), reg.cen.unsqueeze(0),
                                          reg.w.unsqueeze(0))          # (L, C_sub, K)
            dep = self.measure.crisp_union(slot, reg.active.unsqueeze(0))
            out[:, c_s:c_s + chunk_size] = self._constants(
                dep, c_sub.unsqueeze(0), None, None,
                lambda d: th.full_like(d, TOP_DEPTH_CEIL),
                lambda d: th.full_like(d, EMPTY_DEPTH_FLOOR))
        return out

    def held_membership(self, points, class_ids):
        """``(depth, membership)`` with the boxes held fixed: the gradient reaches only the
        points (the soft, aggregated depth; top and bottom constants on the membership)."""
        reg = self.regions(class_ids)
        slot = self.measure.box_depth(points, reg.cen.detach(), reg.w.detach())
        dep = self.measure.aggregate(slot, reg.active, crisp=False)
        m = self.measure.soft(dep, self.domain.tau)
        idx = class_ids
        return dep, self._constants(m, idx, None, None, th.ones_like, th.zeros_like)

    def place_point_boxes(self, member, positions, raw_width):
        """Make component :math:`k` of class :math:`c` the box of raw half-width ``raw_width``
        around ``positions[k]`` where ``member[c, k]`` (``(nb_classes, N)``, ``N`` at most the
        number of components); every other component inactive with width zero. Used by
        :class:`~mowl.representation.library.finite_model_placement.FiniteModelPlacement`."""
        n = int(positions.shape[0])
        with th.no_grad():
            self.centers.zero_()
            self.raw_widths.fill_(INACTIVE_RAW_WIDTH)
            self.active.zero_()
            grid = positions.unsqueeze(0).expand(self.nb_classes, n, positions.shape[-1])
            keep = member.unsqueeze(-1)
            self.centers[:, :n, :] = th.where(keep, grid, th.zeros_like(grid))
            self.raw_widths[:, :n, :] = th.where(keep, th.full_like(grid, raw_width),
                                                 th.full_like(grid, INACTIVE_RAW_WIDTH))
            self.active[:, :n] = member

    def silence(self, class_id):
        """Deactivate every component of ``class_id`` (width zero): a class with no point."""
        with th.no_grad():
            self.active[class_id, :] = False
            self.raw_widths[class_id, :] = INACTIVE_RAW_WIDTH

    def expand(self, new_k):
        """Append inactive zero-width boxes up to ``new_k`` per class (every depth and
        membership stays bitwise the same)."""
        if new_k <= self.n_components:
            return
        extra = new_k - self.n_components
        with th.no_grad():
            self.centers = nn.Parameter(_pad(self.centers.data, extra, 0.0))
            self.raw_widths = nn.Parameter(_pad(self.raw_widths.data, extra,
                                                INACTIVE_RAW_WIDTH))
            self.active = _pad(self.active, extra, False)
        self.n_components = new_k


# =============================================================================================
# the relation models
# =============================================================================================
class _BoxRelations(Relations):
    """What the two relation models of GroundedEL share: ``nb_rels`` relations of
    ``n_components`` boxes, a buffer ``active`` and the buffer ``inverse_of`` (a relation
    whose partner has a lower index owns no parameters of its own: it is derived from the
    partner)."""

    accepts = (MaskedUnionOfBoxes,)

    def __init__(self, nb_rels, n_components, embed_dim, measure, domain, inverse_of=None):
        super().__init__()
        self.nb_rels, self.n_components, self._dim = nb_rels, n_components, embed_dim
        _shared(self, "measure", measure)
        _shared(self, "domain", domain)
        self.register_buffer("inverse_of", validate_inverse_of(nb_rels, inverse_of))

    def _finish(self):
        self.register_buffer("active", th.ones((self.nb_rels, self.n_components),
                                               dtype=th.bool))

    def __len__(self):
        return self.nb_rels

    @property
    def embed_dim(self):
        return self._dim

    def _ids(self, rel_ids, device):
        if rel_ids is None:
            return th.arange(self.nb_rels, dtype=th.long, device=device)
        return _one_dim(index_tensor(rel_ids, self.nb_rels, "rel_ids", device), "rel_ids")

    def _owners(self, idx):
        """``(owner ids, derived mask)``: a relation whose inverse partner has a lower index
        is read from the partner; ``None`` when no relation has a partner."""
        if idx.numel() > 0 and bool((self.inverse_of >= 0).any().item()):
            inv = self.inverse_of[idx]
            is_derived = (inv >= 0) & (inv < idx)
            return th.where(is_derived, inv, idx), is_derived
        return None, None

    def align(self, x_points, y_points, rel_ids):
        """Validate and broadcast heads, tails ``(B, d)`` and relation ids ``(B,)``."""
        d = self._dim
        for name, p in (("x_points", x_points), ("y_points", y_points)):
            if not isinstance(p, th.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(p).__name__}")
        xp = x_points.unsqueeze(0) if x_points.ndim == 1 else x_points
        yp = y_points.unsqueeze(0) if y_points.ndim == 1 else y_points
        if xp.ndim != 2 or xp.shape[1] != d:
            raise ValueError(f"x_points must have shape (B, {d}), got {tuple(x_points.shape)}")
        if yp.ndim != 2 or yp.shape[1] != d:
            raise ValueError(f"y_points must have shape (B, {d}), got {tuple(y_points.shape)}")
        idx = _one_dim(index_tensor(rel_ids, self.nb_rels, "rel_ids", self.active.device),
                       "rel_ids")
        sizes = {xp.shape[0], yp.shape[0], idx.shape[0]}
        non_unit = {s for s in sizes if s != 1}
        if len(non_unit) > 1:
            raise ValueError(
                f"Incompatible batch sizes for x_points ({xp.shape[0]}), "
                f"y_points ({yp.shape[0]}), and rel_ids ({idx.shape[0]})")
        target_b = max(sizes) if (1 in sizes and len(non_unit) == 1) else next(iter(sizes))
        if xp.shape[0] != target_b:
            xp = xp.expand(target_b, d)
        if yp.shape[0] != target_b:
            yp = yp.expand(target_b, d)
        if idx.shape[0] != target_b:
            idx = idx.expand(target_b)
        return xp, yp, idx

    def align_with_classes(self, rel_ids, class_ids, nb_classes):
        """Validate and broadcast relation ids and class ids to ``(B,)``."""
        r_idx = _one_dim(index_tensor(rel_ids, self.nb_rels, "rel_ids", self.active.device),
                         "rel_ids")
        c_idx = _one_dim(index_tensor(class_ids, nb_classes, "class_ids", self.active.device),
                         "class_ids")
        b_r, b_c = int(r_idx.shape[0]), int(c_idx.shape[0])
        if b_r == b_c:
            return r_idx, c_idx
        if b_r == 1 and b_c > 1:
            return r_idx.expand(b_c), c_idx
        if b_c == 1 and b_r > 1:
            return r_idx, c_idx.expand(b_r)
        raise ValueError(
            f"Incompatible batch sizes between rel_ids ({b_r}) and class_ids ({b_c})")

    def membership(self, x_points, y_points, rel_ids):
        """Degree ``(B,)`` to which the pairs are related: :math:`\\sigma(\\text{depth}/\\tau)`."""
        dep = self.depth(x_points, y_points, rel_ids)
        return self.measure.soft(dep, self.domain.tau)

    def grid_membership(self, px, py, rel_ids, need_crisp=False):
        """Degrees for every pair of a grid: heads ``px`` ``(B, sx, d)``, tails ``py``
        ``(B, sy, d)``, one relation per row; returns ``(B * sx * sy,)`` degrees and, with
        ``need_crisp``, the crisp (max) depths."""
        bsz, sx = px.shape[:2]
        sy = int(py.shape[1])
        slot_depth, active = self._grid_slot_depth(px, py, rel_ids)
        act_flat = active.view(bsz, 1, 1, -1).expand(bsz, sx, sy, -1).reshape(bsz * sx * sy, -1)
        dep_soft = self.measure.aggregate(slot_depth, act_flat, crisp=False)
        rho = self.measure.soft(dep_soft, self.domain.tau)
        if not need_crisp:
            return rho, None
        return rho, self.measure.aggregate(slot_depth, act_flat, crisp=True)

    def place_edges(self, edges, positions, raw_width):
        """Make component :math:`k` of relation :math:`r` the box (raw half-width
        ``raw_width``) of the :math:`k`-th pair of ``edges[r]`` (:meth:`edge_box`); every
        other component inactive with width zero. A relation derived from its inverse
        partner is skipped (the partner holds its edges)."""
        with th.no_grad():
            self.active.zero_()
            for name in self.TABLES:
                table = getattr(self, name)
                if name.endswith("raw_widths"):
                    table.fill_(INACTIVE_RAW_WIDTH)
                else:
                    table.zero_()
            for r, pairs in enumerate(edges):
                partner = int(self.inverse_of[r].item())
                if 0 <= partner < r:
                    continue
                for k, (x, y) in enumerate(pairs):
                    for name, value in self.edge_box(positions[x], positions[y]).items():
                        getattr(self, name)[r, k].copy_(value)
                    for name in self.TABLES:
                        if name.endswith("raw_widths"):
                            getattr(self, name)[r, k].fill_(raw_width)
                    self.active[r, k] = True

    def _expand_tables(self, names, new_k):
        if new_k <= self.n_components:
            return
        extra = new_k - self.n_components
        with th.no_grad():
            for name in names:
                raw = name.endswith("raw_widths")
                setattr(self, name, nn.Parameter(_pad(getattr(self, name).data, extra,
                                                      INACTIVE_RAW_WIDTH if raw else 0.0)))
            self.active = _pad(self.active, extra, False)
        self.n_components = new_k


class ProductBoxRelations(_BoxRelations):
    """A relation is a union of products of boxes :math:`H_{r,k} \\times T_{r,k}`: the pair
    :math:`(x, y)` is related iff for some active :math:`k`, :math:`x \\in H_{r,k}` and
    :math:`y \\in T_{r,k}`; its depth is :math:`\\max_k \\min(\\text{dep}(x, H_{r,k}),
    \\text{dep}(y, T_{r,k}))`. The inverse of a relation swaps head and tail.

    Owns ``head_centers``, ``head_raw_widths``, ``tail_centers``, ``tail_raw_widths``
    ``(nb_rels, K_r, d)``, ``active`` and ``inverse_of``.

    Part (vocabulary value): ``region over pairs``. Used by: ``grounded``
    (``role_mode="product"``).
    """

    part = "region over pairs"
    mode = "product"
    TABLES = ("head_centers", "head_raw_widths", "tail_centers", "tail_raw_widths")

    def __init__(self, nb_rels, n_components, embed_dim, measure, domain, inverse_of=None):
        super().__init__(nb_rels, n_components, embed_dim, measure, domain, inverse_of)
        for name in self.TABLES:
            setattr(self, name, nn.Parameter(th.empty((nb_rels, n_components, embed_dim),
                                                      dtype=th.float32)))
        if nb_rels > 0:
            nn.init.uniform_(self.head_centers, a=-0.5, b=0.5)
            nn.init.constant_(self.head_raw_widths, inv_softplus(0.2))
            nn.init.uniform_(self.tail_centers, a=-0.5, b=0.5)
            nn.init.constant_(self.tail_raw_widths, inv_softplus(0.2))
        self._finish()

    def regions(self, rel_ids=None):
        """``(head regions, tail regions)`` of ``rel_ids`` (shared ``active``)."""
        idx = self._ids(rel_ids, self.head_centers.device)
        owner, is_derived = self._owners(idx)
        if owner is not None:
            h_cen_o, h_raw_o = self.head_centers[owner], self.head_raw_widths[owner]
            t_cen_o, t_raw_o = self.tail_centers[owner], self.tail_raw_widths[owner]
            mask3 = is_derived.view(-1, 1, 1)
            h_cen = th.where(mask3, t_cen_o, h_cen_o)
            h_raw = th.where(mask3, t_raw_o, h_raw_o)
            t_cen = th.where(mask3, h_cen_o, t_cen_o)
            t_raw = th.where(mask3, h_raw_o, t_raw_o)
            act_buf = self.active[owner]
        else:
            h_cen, h_raw = self.head_centers[idx], self.head_raw_widths[idx]
            t_cen, t_raw = self.tail_centers[idx], self.tail_raw_widths[idx]
            act_buf = self.active[idx]
        h_w_pos, t_w_pos = F.softplus(h_raw), F.softplus(t_raw)
        h_w = th.where(act_buf.unsqueeze(-1), h_w_pos, th.zeros_like(h_w_pos))
        t_w = th.where(act_buf.unsqueeze(-1), t_w_pos, th.zeros_like(t_w_pos))
        eff_active = act_buf & (th.amin(h_w, dim=-1) > 0.0) & (th.amin(t_w, dim=-1) > 0.0)
        return (Regions(cen=h_cen, w=h_w, active=eff_active),
                Regions(cen=t_cen, w=t_w, active=eff_active))

    def pair_regions(self, rel_ids=None):
        """The relations as unions of boxes of pairs: ``(regions, 2)``, boxes
        :math:`H_{r,k} \\times T_{r,k}` of dimension :math:`2d` (two blocks); a point is the
        pair of its halves (:meth:`split_pair`)."""
        h_reg, t_reg = self.regions(rel_ids)
        return Regions(cen=th.cat([h_reg.cen, t_reg.cen], dim=-1),
                       w=th.cat([h_reg.w, t_reg.w], dim=-1), active=h_reg.active), 2

    def split_pair(self, points):
        """The pair ``(x, y)`` of a point ``(N, 2d)`` of :meth:`pair_regions`."""
        return points[:, :self._dim], points[:, self._dim:]

    def depth(self, x_points, y_points, rel_ids, crisp=False):
        xp, yp, idx = self.align(x_points, y_points, rel_ids)
        if int(xp.shape[0]) == 0:
            return th.empty((0,), dtype=xp.dtype, device=xp.device)
        h_reg, t_reg = self.regions(idx)
        h_dep = th.amin(h_reg.w.to(dtype=xp.dtype)
                        - th.abs(xp.unsqueeze(1) - h_reg.cen.to(dtype=xp.dtype)), dim=-1)
        t_dep = th.amin(t_reg.w.to(dtype=yp.dtype)
                        - th.abs(yp.unsqueeze(1) - t_reg.cen.to(dtype=yp.dtype)), dim=-1)
        return self.measure.aggregate(th.minimum(h_dep, t_dep), h_reg.active, crisp=crisp)

    def _grid_slot_depth(self, px, py, rel_ids):
        bsz, sx = px.shape[:2]
        sy = int(py.shape[1])
        h_reg, t_reg = self.regions(rel_ids)
        h_dep = th.amin(h_reg.w.to(dtype=px.dtype).unsqueeze(1)
                        - th.abs(px.unsqueeze(2) - h_reg.cen.to(dtype=px.dtype).unsqueeze(1)),
                        dim=-1)
        t_dep = th.amin(t_reg.w.to(dtype=py.dtype).unsqueeze(1)
                        - th.abs(py.unsqueeze(2) - t_reg.cen.to(dtype=py.dtype).unsqueeze(1)),
                        dim=-1)
        slot = th.minimum(h_dep.unsqueeze(2), t_dep.unsqueeze(1)).reshape(bsz * sx * sy, -1)
        return slot, h_reg.active

    @staticmethod
    def edge_box(x, y):
        """The centres of the component that holds the pair ``(x, y)``: head at ``x``, tail
        at ``y``."""
        return {"head_centers": x, "tail_centers": y}

    def preimage(self, ids, region):
        raise ConstructorUnavailable(
            "Only role_mode='translation' has a region for exists r.C; in 'product' "
            "mode a role is a set of pairs of boxes (see GroundedELModule.role_depth).")

    def expand(self, new_k):
        self._expand_tables(self.TABLES, new_k)


class TranslationBoxRelations(_BoxRelations):
    """A relation is a union of boxes of translations :math:`V_{r,k}`: the pair
    :math:`(x, y)` is related iff :math:`y - x \\in V_{r,k}` for some active :math:`k`. The
    inverse of a relation negates the centres. :math:`\\exists r.B` is the union of boxes
    :math:`B_j - V_{r,k}` (:meth:`preimage_boxes`).

    Owns ``centers``, ``raw_widths`` ``(nb_rels, K_r, d)``, ``active`` and ``inverse_of``.

    Part (vocabulary value): ``box of translations``. Used by: ``grounded``
    (``role_mode="translation"``).
    """

    part = "box of translations"
    mode = "translation"
    TABLES = ("centers", "raw_widths")

    def __init__(self, nb_rels, n_components, embed_dim, measure, domain, inverse_of=None):
        super().__init__(nb_rels, n_components, embed_dim, measure, domain, inverse_of)
        self.centers = nn.Parameter(th.empty((nb_rels, n_components, embed_dim),
                                             dtype=th.float32))
        self.raw_widths = nn.Parameter(th.empty((nb_rels, n_components, embed_dim),
                                                dtype=th.float32))
        if nb_rels > 0:
            nn.init.uniform_(self.centers, a=-0.5, b=0.5)
            nn.init.constant_(self.raw_widths, inv_softplus(0.2))
        self._finish()

    @staticmethod
    def edge_box(x, y):
        """The centre of the component that holds the pair ``(x, y)``: the translation
        :math:`y - x`."""
        return {"centers": y - x}

    def regions(self, rel_ids=None):
        """The boxes of translations of ``rel_ids``."""
        idx = self._ids(rel_ids, self.centers.device)
        owner, is_derived = self._owners(idx)
        if owner is not None:
            cen_owner = self.centers[owner]
            cen = th.where(is_derived.view(-1, 1, 1), -cen_owner, cen_owner)
            raw_w, act_buf = self.raw_widths[owner], self.active[owner]
        else:
            cen, raw_w, act_buf = self.centers[idx], self.raw_widths[idx], self.active[idx]
        cen, w, active = _boxes(cen, raw_w, act_buf)
        return Regions(cen=cen, w=w, active=active)

    def pair_regions(self, rel_ids=None):
        """The relations as unions of boxes of pairs: ``(regions, 1)``, the boxes of
        translations; a translation :math:`v` stands for the pair :math:`(0, v)`
        (:meth:`split_pair`)."""
        return self.regions(rel_ids), 1

    @staticmethod
    def split_pair(points):
        """The pair ``(0, v)`` of a translation ``v``."""
        return th.zeros_like(points), points

    def depth(self, x_points, y_points, rel_ids, crisp=False):
        xp, yp, idx = self.align(x_points, y_points, rel_ids)
        if int(xp.shape[0]) == 0:
            return th.empty((0,), dtype=xp.dtype, device=xp.device)
        v_reg = self.regions(idx)
        diff = yp - xp
        slot = th.amin(v_reg.w.to(dtype=diff.dtype)
                       - th.abs(diff.unsqueeze(1) - v_reg.cen.to(dtype=diff.dtype)), dim=-1)
        return self.measure.aggregate(slot, v_reg.active, crisp=crisp)

    def _grid_slot_depth(self, px, py, rel_ids):
        bsz, sx = px.shape[:2]
        sy = int(py.shape[1])
        d = self._dim
        t_reg = self.regions(rel_ids)
        t_cen, t_w = t_reg.cen.to(dtype=px.dtype), t_reg.w.to(dtype=px.dtype)
        diff = py.unsqueeze(1) - px.unsqueeze(2)
        slot = th.amin(t_w.view(bsz, 1, 1, -1, d)
                       - th.abs(diff.unsqueeze(3) - t_cen.view(bsz, 1, 1, -1, d)),
                       dim=-1).reshape(bsz * sx * sy, -1)
        return slot, t_reg.active

    def preimage_boxes(self, rel_ids, class_ids, classes, inverse=False):
        """The boxes of :math:`\\exists r.B` (``inverse``: :math:`\\exists r^-.B`) for the
        fillers ``class_ids`` of the class model ``classes``: box :math:`(j, k)` has centre
        :math:`c_{B,j} \\mp c_{r,k}` and half-width :math:`w_{B,j} + w_{r,k}`. Top gives one
        box of half-width :data:`TOP_DEPTH_CEIL`, bottom none."""
        r_idx, c_idx = self.align_with_classes(rel_ids, class_ids, classes.nb_classes)
        bsz = int(r_idx.shape[0])
        device, dtype = classes.centers.device, classes.centers.dtype
        d = self._dim
        n_pair = classes.n_components * self.n_components
        if bsz == 0:
            return ExistsRegions(
                cen=th.empty((0, n_pair, d), dtype=dtype, device=device),
                w=th.empty((0, n_pair, d), dtype=dtype, device=device),
                active=th.empty((0, n_pair), dtype=th.bool, device=device), is_top=False)
        top_id, bot_id = classes.top_id, classes.bot_id
        is_top_row = (c_idx == top_id) if top_id is not None else \
            th.zeros((bsz,), dtype=th.bool, device=device)
        if bool(is_top_row.all().item()):
            return ExistsRegions(
                cen=th.zeros((bsz, 1, d), dtype=dtype, device=device),
                w=th.full((bsz, 1, d), TOP_DEPTH_CEIL, dtype=dtype, device=device),
                active=th.ones((bsz, 1), dtype=th.bool, device=device),
                is_top=True, is_top_mask=is_top_row)
        cls_reg = classes.regions(c_idx)
        rel_reg = self.regions(r_idx)
        if inverse:
            cen_jk = cls_reg.cen.unsqueeze(2) + rel_reg.cen.unsqueeze(1)
        else:
            cen_jk = cls_reg.cen.unsqueeze(2) - rel_reg.cen.unsqueeze(1)
        w_jk = cls_reg.w.unsqueeze(2) + rel_reg.w.unsqueeze(1)
        act_jk = cls_reg.active.unsqueeze(2) & rel_reg.active.unsqueeze(1)
        cen = cen_jk.reshape(bsz, n_pair, d)
        w_raw = w_jk.reshape(bsz, n_pair, d)
        act = act_jk.reshape(bsz, n_pair)
        if bot_id is not None:
            is_bot_row = c_idx == bot_id
            if bool(is_bot_row.any().item()):
                act = act & (~is_bot_row.unsqueeze(1))
        if bool(is_top_row.any().item()):
            cen, w_raw, act = cen.clone(), w_raw.clone(), act.clone()
            cen[is_top_row] = 0.0
            w_raw[is_top_row] = 0.0
            w_raw[is_top_row, 0, :] = TOP_DEPTH_CEIL
            act[is_top_row] = False
            act[is_top_row, 0] = True
        w = th.where(act.unsqueeze(-1), w_raw, th.zeros_like(w_raw))
        return ExistsRegions(cen=cen, w=w, active=act, is_top=False, is_top_mask=is_top_row)

    def exists_depth(self, points, rel_ids, class_ids, classes, crisp=False):
        """Depth ``(B,)`` of ``points`` in :math:`\\exists r.B`."""
        d = self._dim
        if not isinstance(points, th.Tensor):
            raise TypeError(f"points must be a torch.Tensor, got {type(points).__name__}")
        pts = points.unsqueeze(0) if points.ndim == 1 else points
        if pts.ndim != 2 or pts.shape[1] != d:
            raise ValueError(f"points must have shape (B, {d}), got {tuple(points.shape)}")
        r_idx, c_idx = self.align_with_classes(rel_ids, class_ids, classes.nb_classes)
        b_pts, b_ids = int(pts.shape[0]), int(r_idx.shape[0])
        if b_pts != b_ids:
            if b_pts == 1 and b_ids > 1:
                pts = pts.expand(b_ids, d)
            elif b_ids == 1 and b_pts > 1:
                r_idx, c_idx = r_idx.expand(b_pts), c_idx.expand(b_pts)
            else:
                raise ValueError(
                    f"Incompatible batch sizes between points ({b_pts}) and (rel_ids, "
                    f"class_ids) ({b_ids})")
        if int(pts.shape[0]) == 0:
            return th.empty((0,), dtype=pts.dtype, device=pts.device)
        reg = self.preimage_boxes(r_idx, c_idx, classes)
        return self.measure.depth(pts, Regions(cen=reg.cen, w=reg.w, active=reg.active),
                                  crisp=crisp)

    def preimage(self, ids, region):
        """The region of :math:`\\exists r.C` for a named filler (needs the class model
        the geometry passes as ``region.classes``)."""
        raise ConstructorUnavailable("Use GroundedELModule.exists (it knows the class model).")

    def expand(self, new_k):
        self._expand_tables(self.TABLES, new_k)


class ElementPoints:
    """The individual model of GroundedEL: individual ``i`` is the point of element
    ``nb_classes + i`` of the shared part (a view; it owns no table).

    Part (vocabulary value): ``canonical element``."""

    part = "canonical element"
    region = Point

    def __init__(self, domain):
        self.domain = domain

    def __call__(self, ids):
        return Point(self.domain.positions(self.domain.nb_classes + ids))

    def __len__(self):
        return self.domain.nb_elements - self.domain.nb_classes


# =============================================================================================
# the registry of the parts of grounded methods
# =============================================================================================
#: The parts of grounded methods by kind and name: what a builder (``grounded(...)``,
#: :class:`~mowl.nn.GroundedELModule`, :class:`~mowl.models.GroundedEL`) accepts by name.
#: The kinds are ``classes``, ``relations``, ``measure`` (factories with the signature of the
#: registered class), ``objective``, ``element sampler``, ``region sampler`` and
#: ``evaluation`` (classes whose constructor takes the options).
GROUNDED_PARTS = {kind: {} for kind in ("classes", "relations", "measure", "objective",
                                        "element sampler", "region sampler", "evaluation")}


def register_grounded_part(kind, *names):
    """Decorator: make a part of ``kind`` available under its ``part`` name and ``names``."""
    def decorate(factory):
        for name in (getattr(factory, "part", None),) + names:
            if name is None:
                continue
            taken = GROUNDED_PARTS[kind].get(name)
            if taken is not None and taken is not factory:
                raise ValueError(f"{name!r} is already a {kind} part ({taken!r}).")
            GROUNDED_PARTS[kind][name] = factory
        return factory
    return decorate


register_grounded_part("classes", "box")(BoxUnionClasses)
register_grounded_part("relations", "product")(ProductBoxRelations)
register_grounded_part("relations", "translation")(TranslationBoxRelations)
register_grounded_part("measure", "union depth")(UnionDepth)
