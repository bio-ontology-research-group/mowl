"""Re-seeding the components of a union of boxes at the points they must hold (a training
regime's move on the class model of a finite domain).

Gradient steps cannot bring a box to a point it does not touch. Between optimisation steps,
:class:`OwnPointReseeding` moves components of a
:class:`~mowl.representation.library.grounded_parts.BoxUnionClasses` (under ``no_grad``) to
where they are needed, reading only the positions of the elements and the class model:

1. every live class :math:`A` whose own element :math:`x_A` is not crisp-inside :math:`A`
   gets a box of half-width ``size`` at :math:`p_{x_A}`: an inactive component if there is
   one, else the active component that holds the fewest live elements (ties: the smallest
   volume);
2. the same for every stated membership :math:`C(a)` (the element of :math:`a` in
   :math:`C`) that does not hold;
3. with ``prune``, the active components that hold no live element are deactivated.

Top and bottom are never touched. The counts of live elements per component are exact
(:func:`slot_point_counts`, in chunks of ``chunk_size`` points, with a lossless prefilter
by spatial bins for large domains). Used by GroundedEL (the regime
:class:`~mowl.training.IdleComponentReseeding` calls the model's ``reseed_components``);
any geometry of an element domain and a union-of-boxes class model can use it.
"""
from __future__ import annotations

import torch as th

from mowl.representation.library.grounded_parts import (INACTIVE_RAW_WIDTH, UnionDepth,
                                                        inv_softplus, validate_positive_float)

__all__ = ["slot_point_counts", "OwnPointReseeding"]


def _counts_one_class(cen, w, active, points, chunk_size):
    counts = th.zeros((int(cen.shape[0]),), dtype=th.long, device=cen.device)
    if int(points.shape[0]) == 0 or not bool(active.any().item()):
        return counts
    for start in range(0, int(points.shape[0]), chunk_size):
        depth = UnionDepth.box_depth(points[start:start + chunk_size], cen.unsqueeze(0),
                                     w.unsqueeze(0))
        counts += ((depth > 0.0) & active.unsqueeze(0)).sum(dim=0)
    return counts


def _binned_counts(regions, points, query, chunk_size):
    """Counts of the queried slots through a 4^D spatial-bin prefilter (lossless)."""
    device, dim = regions.cen.device, int(regions.cen.shape[-1])
    d_bins = min(dim, 5)
    n_bins = 4 ** d_bins
    p_min = points[:, :d_bins].amin(dim=0)
    p_max = points[:, :d_bins].amax(dim=0)
    scale = 3.999 / (p_max - p_min + 1e-6)
    bins = th.clamp(((points[:, :d_bins] - p_min) * scale).to(th.long), 0, 3)
    key = (bins * (4 ** th.arange(d_bins, dtype=th.long, device=device)).unsqueeze(0)).sum(1)
    order = th.argsort(key)
    sorted_points = points[order]
    offsets = th.zeros((n_bins + 1,), dtype=th.long, device=device)
    offsets[1:] = th.cumsum(th.bincount(key, minlength=n_bins), dim=0)
    flat = th.nonzero(query.reshape(-1), as_tuple=False).view(-1)
    cen = regions.cen.reshape(-1, dim)[flat]
    w = regions.w.reshape(-1, dim)[flat]
    lo = th.clamp(((cen[:, :d_bins] - w[:, :d_bins] - 1e-6 - p_min) * scale).to(th.long), 0, 3)
    hi = th.clamp(((cen[:, :d_bins] + w[:, :d_bins] + 1e-6 - p_min) * scale).to(th.long), 0, 3)
    counts = th.zeros((flat.numel(),), dtype=th.long, device=device)
    for h in range(n_bins):
        s, e = int(offsets[h].item()), int(offsets[h + 1].item())
        if s == e:
            continue
        rem = h
        mask = th.ones((flat.numel(),), dtype=th.bool, device=device)
        for m in range(d_bins):
            b = rem & 3
            rem >>= 2
            mask &= (lo[:, m] <= b) & (hi[:, m] >= b)
        boxes = th.nonzero(mask, as_tuple=False).view(-1)
        if boxes.numel() == 0:
            continue
        for p_s in range(s, e, chunk_size):
            chunk = sorted_points[p_s:min(p_s + chunk_size, e)]
            for b_s in range(0, boxes.numel(), 256):
                sub = boxes[b_s:b_s + 256]
                depth = UnionDepth.box_depth(chunk, cen[sub], w[sub]).t()
                counts[sub] += (depth > 0.0).sum(dim=1)
    return flat, counts


def slot_point_counts(regions, points, query, chunk_size=8192, small=2048):
    """The number of ``points`` ``(L, d)`` crisp-inside each component of ``regions``
    (:class:`~mowl.nn.el.structural.Regions` ``(C, K, d)``) where ``query`` ``(C, K)`` is
    true; zero elsewhere. Per class in chunks of ``chunk_size`` points when there are at most
    ``small`` classes or points, else through a lossless spatial-bin prefilter."""
    n_classes, k = int(regions.cen.shape[0]), int(regions.cen.shape[1])
    counts = th.zeros((n_classes, k), dtype=th.long, device=regions.cen.device)
    if int(points.shape[0]) == 0 or not bool(query.any().item()):
        return counts
    if n_classes <= small or int(points.shape[0]) <= small:
        for c in th.nonzero(query.any(dim=1), as_tuple=False).view(-1).tolist():
            counts[c] = _counts_one_class(regions.cen[c], regions.w[c],
                                          regions.active[c] & query[c], points, chunk_size)
        return counts
    flat, found = _binned_counts(regions, points, query, chunk_size)
    counts.view(-1)[flat] = found
    return counts


def _log_volume(w):
    return th.log(th.clamp(w, min=1e-12)).sum(dim=-1)


class OwnPointReseeding:
    """The relocation rule of the module docstring.

    :param size: Half-width of a re-seeded box.
    :param prune: Deactivate the active components that hold no live element.
    :param chunk_size: Points per chunk of the counts.
    """

    def __init__(self, size=0.08, prune=True, chunk_size=8192):
        self.size = validate_positive_float("reseed_size", size)
        self.prune, self.chunk_size = bool(prune), int(chunk_size)

    def apply(self, geometry, assertions=None, top_id=None, bot_id=None):
        """Re-seed the class model of ``geometry`` (``positions()``, ``dead()``,
        ``class_regions()``, ``depth(points, ids, crisp=True)``, ``measure.box_depth`` and a
        class model with ``centers``, ``raw_widths``, ``active``). ``assertions`` are rows
        ``(individual, class)`` of stated memberships (the element of an individual is
        ``nb_classes + individual``). Returns the number of components changed."""
        with th.no_grad():
            return self._apply(geometry, assertions, top_id, bot_id)

    def _apply(self, g, assertions, top_id, bot_id):
        boxes, n_classes = g.class_model, g.nb_classes
        k_slots = boxes.n_components
        raw_target = inv_softplus(self.size)
        device = boxes.centers.device
        positions = g.positions()
        dead = g.dead() > 0.5
        if bot_id is not None:
            dead[bot_id] = True
        live = ~dead
        live_points = positions[live]
        regions = g.class_regions()
        cen, w, act = regions.cen, regions.w, regions.active
        own_depth = g.measure.box_depth(positions[:n_classes], cen, w)
        own_in_slot = (own_depth > 0.0) & act & live[:n_classes].unsqueeze(1)
        own_inside = own_in_slot.any(dim=1)
        live_classes = live[:n_classes].clone()
        for fixed in (top_id, bot_id):
            if fixed is not None:
                live_classes[fixed] = False
        need_own = live_classes & (~own_inside)
        query = (need_own & act.all(dim=1) & (k_slots > 1)).unsqueeze(1) & act

        elems = th.empty((0,), dtype=th.long, device=device)
        cls = th.empty((0,), dtype=th.long, device=device)
        if assertions is not None and len(assertions) > 0:
            rows = assertions.to(device=device, dtype=th.long)
            elems_all, cls_all = n_classes + rows[:, 0], rows[:, 1]
            keep = live[elems_all]
            for fixed in (top_id, bot_id):
                if fixed is not None:
                    keep = keep & (cls_all != fixed)
            elems, cls = elems_all[keep], cls_all[keep]
            if cls.numel() > 0 and k_slots > 1:
                before = g.depth(positions[elems], cls, crisp=True)
                candidates = cls[(before <= 0.0) | need_own[cls]]
                if candidates.numel() > 0:
                    mask = th.zeros((n_classes,), dtype=th.bool, device=device)
                    mask[candidates] = True
                    query = query | ((mask & act.all(dim=1)).unsqueeze(1) & act)
        if self.prune:
            prunable = act & (~own_in_slot)
            for fixed in (top_id, bot_id):
                if fixed is not None:
                    prunable[fixed, :] = False
            query = query | prunable

        counts = slot_point_counts(regions, live_points, query, chunk_size=self.chunk_size)
        counts = th.where(own_in_slot & (counts == 0), th.ones_like(counts), counts)
        moves = self._own_points(boxes, positions, need_own, w, act, counts, raw_target)
        if elems.numel() > 0:
            moves += self._stated(g, boxes, positions, elems, cls, counts, raw_target)
        if self.prune:
            unused = g.class_regions().active & (counts == 0)
            for fixed in (top_id, bot_id):
                if fixed is not None:
                    unused[fixed, :] = False
            n_pruned = int(unused.sum().item())
            if n_pruned > 0:
                boxes.active[unused] = False
                boxes.raw_widths[unused] = INACTIVE_RAW_WIDTH
                moves += n_pruned
        return moves

    @staticmethod
    def _own_points(boxes, positions, need_own, w, act, counts, raw_target):
        """Step 1: a box at the own element of every class that misses it."""
        ids = th.nonzero(need_own, as_tuple=False).view(-1)
        if ids.numel() == 0:
            return 0
        act_n, cnt_n = act[ids], counts[ids]
        lvol = th.where(act_n, _log_volume(w[ids]),
                        th.full(act_n.shape, float("-inf"), dtype=w.dtype, device=w.device))
        best = th.zeros((ids.numel(),), dtype=th.long, device=w.device)
        b_act, b_cnt, b_vol = act_n[:, 0], cnt_n[:, 0], lvol[:, 0]
        for k in range(1, int(act.shape[1])):
            c_act, c_cnt, c_vol = act_n[:, k], cnt_n[:, k], lvol[:, k]
            better = (~c_act & b_act) | ((c_act == b_act) & (
                (c_cnt < b_cnt) | ((c_cnt == b_cnt) & (c_vol < b_vol))))
            best = th.where(better, th.full_like(best, k), best)
            b_act = th.where(better, c_act, b_act)
            b_cnt = th.where(better, c_cnt, b_cnt)
            b_vol = th.where(better, c_vol, b_vol)
        boxes.centers[ids, best] = positions[ids]
        boxes.raw_widths[ids, best] = raw_target
        boxes.active[ids, best] = True
        counts[ids, best] = th.clamp(counts[ids, best], min=1)
        return int(ids.numel())

    @staticmethod
    def _stated(g, boxes, positions, elems, cls, counts, raw_target):
        """Step 2: a box at the element of every stated membership that does not hold."""
        k_slots = boxes.n_components
        depth = g.depth(positions[elems], cls, crisp=True)
        missing = th.nonzero(depth <= 0.0, as_tuple=False).view(-1)
        moves = 0
        if missing.numel() == 0:
            return 0
        for el, c in zip(elems[missing].detach().cpu().tolist(),
                         cls[missing].detach().cpu().tolist()):
            reg = boxes.regions(th.tensor([c], device=boxes.centers.device))
            cen_c, w_c, act_c = reg.cen[0], reg.w[0], reg.active[0]
            if bool(((g.measure.box_depth(positions[el], cen_c, w_c) > 0.0) & act_c).any().item()):
                continue
            if bool((~act_c).any().item()):
                k = int(th.nonzero(~act_c, as_tuple=False)[0].item())
            elif k_slots == 1:
                k = 0
            else:
                cnt = counts[c].cpu().tolist()
                lv = _log_volume(w_c).cpu().tolist()
                k = min(range(k_slots), key=lambda j: (int(cnt[j]), float(lv[j])))
            boxes.centers[c, k].copy_(positions[el])
            boxes.raw_widths[c, k].fill_(raw_target)
            boxes.active[c, k] = True
            counts[c, k] = max(int(counts[c, k].item()), 1)
            moves += 1
        return moves
