"""Set algebra over batched unions of boxes with a bound on the number of boxes.

A region is a batch of unions of axis-parallel boxes: centres ``cen`` and non-negative
half-widths ``w`` of shape ``(B, M, d)`` and a mask ``active`` ``(B, M)`` (an inactive slot
is padding). :class:`BoundedBoxUnions` computes, exactly as long as the bound ``max_boxes``
is not exceeded:

* :meth:`~BoundedBoxUnions.points`: the width-0 boxes at points (the region of a constant);
* :meth:`~BoundedBoxUnions.minkowski`: the Minkowski sum :math:`R + V` with a union of boxes
  per row (the image of a region under a relation of translations,
  :math:`\\{x + v : x \\in R, v \\in V\\}`);
* :meth:`~BoundedBoxUnions.intersect`: the pairwise intersections, empty pairs dropped and
  the active boxes first;
* :meth:`~BoundedBoxUnions.union`: the boxes of both, in order;
* :meth:`~BoundedBoxUnions.depth`: the signed depth :math:`\\max_m \\min_j (w_{m,j} -
  |p_j - c_{m,j}|)` of points (crisp, or a log-sum-exp at temperature ``tau``);
* :meth:`~BoundedBoxUnions.overlap_depth`: :math:`\\max_{m,q} \\min_j (w_{m,j} + w_{q,j} -
  |c_{m,j} - c_{q,j}|)`, positive iff two active boxes meet in their interiors (and
  :meth:`~BoundedBoxUnions.overlap_table` against a table of unions);
* :meth:`~BoundedBoxUnions.union_depth`: the depth of points in unions of any broadcast
  shape, crisp or as a stable log-sum-exp (:meth:`~BoundedBoxUnions.stable_soft_max`);
* :meth:`~BoundedBoxUnions.select_rows`, :meth:`~BoundedBoxUnions.flatten`: rows of one of
  two regions, and the boxes of chosen unions of a table as one region.

When a result would have more than ``max_boxes`` boxes, the largest boxes by volume are kept
(:meth:`~BoundedBoxUnions.cap`). An empty region has depth :data:`EMPTY_DEPTH`. Results are
of the type ``box_type`` (called with ``cen=, w=, active=``), so the algebra serves any
named tuple or region class of that shape. Used by the logical query answering of
GroundedEL (:mod:`mowl.nn.el.grounded.queries`), where a query is compiled to such a region;
usable by any method whose regions are unions of boxes.
"""
from __future__ import annotations

from typing import NamedTuple

import torch as th

__all__ = ["Boxes", "BoundedBoxUnions", "EMPTY_DEPTH"]

#: Depth of every point in a region with no active box.
EMPTY_DEPTH: float = -1e6
_LOG_VOLUME_FLOOR: float = 1e-30


class Boxes(NamedTuple):
    """A batch of unions of boxes (the default result type)."""
    cen: th.Tensor
    w: th.Tensor
    active: th.Tensor


class BoundedBoxUnions:
    """The algebra of the module docstring.

    :param max_boxes: Largest number of boxes of a result (per row).
    :param box_type: Type of the results (called with ``cen=``, ``w=``, ``active=``).
    :param tau: Temperature of the smooth depth (``crisp=False``).
    """

    def __init__(self, max_boxes=64, box_type=Boxes, tau=0.1):
        self.max_boxes, self.box_type, self.tau = int(max_boxes), box_type, float(tau)

    def make(self, cen, w, active):
        return self.box_type(cen=cen, w=w, active=active)

    def empty(self, bsz, dim, like):
        """A batch of ``bsz`` regions without boxes."""
        return self.make(th.empty((bsz, 0, dim), dtype=like.dtype, device=like.device),
                         th.empty((bsz, 0, dim), dtype=like.dtype, device=like.device),
                         th.empty((bsz, 0), dtype=th.bool, device=like.device))

    def align(self, r1, r2):
        """Broadcast a batch of one region to the batch of the other."""
        if r1.cen.shape[2] != r2.cen.shape[2]:
            raise ValueError(
                f"Incompatible embedding dimensions: {r1.cen.shape[2]} vs {r2.cen.shape[2]}")
        b1, b2 = r1.cen.shape[0], r2.cen.shape[0]
        if b1 == b2:
            return r1, r2
        if b1 == 1 and b2 > 1:
            return self._expand(r1, b2), r2
        if b2 == 1 and b1 > 1:
            return r1, self._expand(r2, b1)
        raise ValueError(f"Incompatible MultiBox batch sizes: {b1} vs {b2}")

    def _expand(self, r, bsz):
        m, d = r.cen.shape[1], r.cen.shape[2]
        return self.make(r.cen.expand(bsz, m, d), r.w.expand(bsz, m, d),
                         r.active.expand(bsz, m))

    def masked(self, cen, w, active):
        """Zero the centres and widths of the inactive boxes; widths clamped at 0."""
        keep = active.unsqueeze(-1)
        return self.make(th.where(keep, cen, th.zeros_like(cen)),
                         th.where(keep, th.clamp_min(w, 0.0), th.zeros_like(w)), active)

    def compact(self, cen, w, active):
        """Reorder the boxes of every row (stably) so that the active ones come first."""
        bsz, total, dim = cen.shape
        if total == 0 or bsz == 0:
            return self.make(cen, th.clamp_min(w, 0.0), active)
        grid = th.arange(total, device=active.device).unsqueeze(0)
        order = th.argsort(th.where(active, grid, grid + total), dim=1)
        order_d = order.unsqueeze(-1).expand(bsz, total, dim)
        return self.masked(th.gather(cen, 1, order_d), th.gather(w, 1, order_d),
                           th.gather(active, 1, order))

    def cap(self, r, max_boxes=None):
        """At most ``max_boxes`` boxes per row: the largest active ones by volume."""
        k = self.max_boxes if max_boxes is None else int(max_boxes)
        bsz, total, dim = r.cen.shape
        if total <= k:
            return self.masked(r.cen, r.w, r.active)
        if bsz == 0:
            return self.make(r.cen[:, :k, :], th.clamp_min(r.w[:, :k, :], 0.0),
                             r.active[:, :k])
        log_vol = th.log(th.clamp_min(r.w, _LOG_VOLUME_FLOOR)).sum(dim=-1)
        score = th.where(r.active, log_vol, th.full_like(log_vol, float("-inf")))
        _, top = th.topk(score, k=k, dim=1, largest=True, sorted=True)
        top_d = top.unsqueeze(-1).expand(bsz, k, dim)
        return self.masked(th.gather(r.cen, 1, top_d), th.gather(r.w, 1, top_d),
                           th.gather(r.active, 1, top))

    def points(self, positions):
        """Width-0 boxes ``(B, 1, d)`` at ``positions`` ``(B, d)``."""
        cen = positions.unsqueeze(1)
        return self.make(cen, th.zeros_like(cen),
                         th.ones((cen.shape[0], 1), dtype=th.bool, device=cen.device))

    def minkowski(self, r, cen, w, active):
        """:math:`R + V` with the boxes ``(cen, w, active)`` ``(B, K, d)`` of ``V``."""
        bsz, m, dim = r.cen.shape
        total = m * cen.shape[1]
        if total == 0:
            return self.empty(bsz, dim, r.cen)
        cen_s = (r.cen.unsqueeze(2) + cen.to(dtype=r.cen.dtype).unsqueeze(1)).reshape(
            bsz, total, dim)
        w_s = (r.w.unsqueeze(2) + w.to(dtype=r.w.dtype).unsqueeze(1)).reshape(bsz, total, dim)
        act = (r.active.unsqueeze(2) & active.unsqueeze(1)).reshape(bsz, total)
        keep = act.unsqueeze(-1)
        out = self.make(th.where(keep, cen_s, th.zeros_like(cen_s)),
                        th.where(keep, w_s, th.zeros_like(w_s)), act)
        return self.cap(out) if total > self.max_boxes else out

    def intersect(self, r1, r2):
        """The non-empty pairwise intersections of the boxes of ``r1`` and ``r2``."""
        r1, r2 = self.align(r1, r2)
        bsz, m1, dim = r1.cen.shape
        total = m1 * r2.cen.shape[1]
        if total == 0:
            return self.empty(bsz, dim, r1.cen)
        lo = th.maximum(r1.cen.unsqueeze(2) - r1.w.unsqueeze(2),
                        r2.cen.unsqueeze(1) - r2.w.unsqueeze(1))
        hi = th.minimum(r1.cen.unsqueeze(2) + r1.w.unsqueeze(2),
                        r2.cen.unsqueeze(1) + r2.w.unsqueeze(1))
        cen, w = 0.5 * (lo + hi), 0.5 * (hi - lo)
        act = r1.active.unsqueeze(2) & r2.active.unsqueeze(1) & (th.amin(w, dim=-1) > 0.0)
        cen = cen.reshape(bsz, total, dim)
        w = th.clamp_min(w.reshape(bsz, total, dim), 0.0)
        act = act.reshape(bsz, total)
        if total <= self.max_boxes:
            return self.compact(cen, w, act)
        return self.cap(self.make(cen, w, act))

    def union(self, r1, r2):
        """The boxes of ``r1`` then those of ``r2``."""
        r1, r2 = self.align(r1, r2)
        out = self.masked(th.cat([r1.cen, r2.cen], dim=1), th.cat([r1.w, r2.w], dim=1),
                          th.cat([r1.active, r2.active], dim=1))
        return self.cap(out) if out.cen.shape[1] > self.max_boxes else out

    @staticmethod
    def stable_soft_max(values, active, tau):
        """Log-sum-exp at temperature ``tau`` of ``values`` over the active entries of the
        last axis, shifted by their detached maximum (finite gradients at any ``tau``);
        :data:`EMPTY_DEPTH` where no entry is active or the axis is empty."""
        lead = values.shape[:-1]
        if values.shape[-1] == 0:
            return values.new_full(lead, EMPTY_DEPTH)
        on = active.expand(values.shape)
        rows = on.any(dim=-1)
        masked = values.masked_fill(~on, float("-inf"))
        peak = th.where(rows, masked.amax(dim=-1), th.zeros(lead, dtype=values.dtype,
                                                              device=values.device)).detach()
        t = tau.to(dtype=values.dtype, device=values.device) if th.is_tensor(tau) else tau
        z = th.where(rows.unsqueeze(-1), (masked - peak.unsqueeze(-1)) / t, th.zeros_like(masked))
        return (peak + t * th.logsumexp(z, dim=-1)).masked_fill(~rows, EMPTY_DEPTH)

    def union_depth(self, points, cen, w, active, tau=None):
        """Depth of ``points`` in unions of boxes, all broadcast together: ``points``
        ``(..., 1, d)``, ``cen`` and ``w`` ``(..., M, d)``, ``active`` ``(..., M)``; the
        result is ``(...)``. Crisp (the largest box depth) without ``tau``, else
        :meth:`stable_soft_max` at ``tau``. A union without active boxes has depth
        :data:`EMPTY_DEPTH`."""
        slot = th.amin(w - th.abs(points - cen), dim=-1)
        active = active.expand(slot.shape)
        if tau is not None:
            return self.stable_soft_max(slot, active, tau)
        if slot.shape[-1] == 0:
            return th.full(slot.shape[:-1], EMPTY_DEPTH, dtype=slot.dtype, device=slot.device)
        return self._masked_max(slot, active)

    def overlap_table(self, r, table):
        """Overlap depth ``(B, N)`` of every region of ``r`` ``(B, M, d)`` with every union
        of ``table`` ``(N, K, d)``."""
        bsz, m, _ = r.cen.shape
        n, k, _ = table.cen.shape
        if m * k == 0:
            return th.full((bsz, n), EMPTY_DEPTH, dtype=r.cen.dtype, device=r.cen.device)
        t_cen = table.cen.to(dtype=r.cen.dtype, device=r.cen.device)
        t_w = table.w.to(dtype=r.w.dtype, device=r.w.device)
        pair = th.amin(r.w[:, None, :, None] + t_w[None, :, None]
                       - th.abs(r.cen[:, None, :, None] - t_cen[None, :, None]), dim=-1)
        act = r.active[:, None, :, None] & table.active.to(r.active.device)[None, :, None]
        return self._masked_max(pair.reshape(bsz, n, m * k), act.reshape(bsz, n, m * k))

    def select_rows(self, mask, r_true, r_false):
        """Row ``b`` of ``r_true`` where ``mask[b]``, else of ``r_false`` (the shorter one
        padded with inactive boxes)."""
        size = max(r_true.cen.shape[1], r_false.cen.shape[1])
        a, b = self.pad(r_true, size), self.pad(r_false, size)
        return self.make(th.where(mask[:, None, None], a.cen, b.cen),
                         th.where(mask[:, None, None], a.w, b.w),
                         th.where(mask[:, None], a.active, b.active))

    def pad(self, r, size):
        """``r`` with inactive zero boxes appended up to ``size`` per row."""
        bsz, m, dim = r.cen.shape
        if m >= size:
            return r
        zeros = r.cen.new_zeros((bsz, size - m, dim))
        return self.make(th.cat([r.cen, zeros], 1), th.cat([r.w, r.w.new_zeros(zeros.shape)], 1),
                         th.cat([r.active, r.active.new_zeros((bsz, size - m))], 1))

    def flatten(self, table, keep):
        """The boxes of all unions of ``table`` ``(N, K, d)`` as one region per row of
        ``keep`` ``(B, N)`` (a box counts where its union is kept and it is active),
        active boxes first, at most ``max_boxes``."""
        n, k, dim = table.cen.shape
        bsz = keep.shape[0]
        cen = table.cen.reshape(1, n * k, dim).expand(bsz, n * k, dim)
        w = table.w.reshape(1, n * k, dim).expand(bsz, n * k, dim)
        act = (keep[:, :, None] & table.active[None]).reshape(bsz, n * k)
        if n * k <= self.max_boxes:
            return self.compact(cen, w, act)
        return self.cap(self.make(cen, w, act))

    def _masked_max(self, values, active, crisp=True):
        """``max`` (or log-sum-exp) of ``values`` over the active slots of the last axis;
        :data:`EMPTY_DEPTH` where no slot is active."""
        has = active.any(dim=-1, keepdim=True)
        fill = th.where(has, th.full_like(values, float("-inf")),
                        th.full_like(values, EMPTY_DEPTH))
        masked = th.where(active, values, fill)
        if crisp:
            return th.amax(masked, dim=-1)
        lse = self.tau * th.logsumexp(masked / self.tau, dim=-1)
        return th.where(has.squeeze(-1), lse, th.full_like(lse, EMPTY_DEPTH))

    def depth(self, points, r, crisp=True):
        """Depth ``(B, N)`` of ``points`` ``(1, N, 1, d)`` or ``(B, N, 1, d)`` (already
        expanded) in ``r``."""
        bsz, m, _ = r.cen.shape
        n = points.shape[1]
        if bsz == 0 or n == 0:
            return th.empty((bsz, n), dtype=r.cen.dtype, device=r.cen.device)
        if m == 0:
            return th.full((bsz, n), EMPTY_DEPTH, dtype=r.cen.dtype, device=r.cen.device)
        p = points.to(dtype=r.cen.dtype, device=r.cen.device)
        slot = th.amin(r.w.unsqueeze(1) - th.abs(p - r.cen.unsqueeze(1)), dim=-1)
        return self._masked_max(slot, r.active.unsqueeze(1), crisp=crisp)

    def overlap_depth(self, r, q):
        """Overlap depth ``(B,)`` of ``r`` and ``q``."""
        r, q = self.align(r, q)
        bsz, m, _ = r.cen.shape
        total = m * q.cen.shape[1]
        if bsz == 0:
            return th.empty((0,), dtype=r.cen.dtype, device=r.cen.device)
        if total == 0:
            return th.full((bsz,), EMPTY_DEPTH, dtype=r.cen.dtype, device=r.cen.device)
        q_cen = q.cen.to(dtype=r.cen.dtype, device=r.cen.device)
        q_w = q.w.to(dtype=r.w.dtype, device=r.w.device)
        q_act = q.active.to(device=r.active.device)
        pair = th.amin(r.w.unsqueeze(2) + q_w.unsqueeze(1)
                       - th.abs(r.cen.unsqueeze(2) - q_cen.unsqueeze(1)), dim=-1)
        act = (r.active.unsqueeze(2) & q_act.unsqueeze(1)).reshape(bsz, total)
        return self._masked_max(pair.reshape(bsz, total), act)
