"""Samplers of a grounded method: which elements of the finite domain and which points of
regions one step of training (or one evaluation) reads.

A grounded method reads an axiom at finitely many places: at elements of the domain (the
element terms, rules over degrees of membership) and at points drawn from the regions the
axiom talks about (the region terms, hinges on depths). Two parts decide those places:

- :class:`ElementSampler` (part ``"elements of the finite domain"``): rows of an axiom
  table, the candidate elements of a row (its own elements, random ones, elements of the
  batch and elements that are currently inside the classes of the left-hand side), fillers
  of an existential, grids over the domain for the role axioms, probes for a minimality
  penalty and corrupted assertions.
- :class:`RegionSampler` (part ``"points of a union of boxes"``): points of a union of
  boxes: a random box of the union, a uniform point in it (re-parametrised, so the
  gradient reaches the box), the corners of the box, and optional signed-gradient ascent on
  a hinge inside the box.

Both are written against the part interfaces only: a class model that gives regions
``classes(ids)`` with ``cen``, ``w``, ``active`` (a union of boxes per row) and
``depth_table``; a relation model that gives ``pair_regions(ids)`` (a union of boxes of
pairs and how a point of it is a pair) and, for an existential, ``preimage_boxes``. Every
random number comes from the generator passed in, so a run is a function of the seed.
"""
from __future__ import annotations

from typing import NamedTuple

import torch as th

from mowl.representation.library.grounded_parts import register_grounded_part

__all__ = ["RegionSample", "ElementSampler", "RegionSampler", "box_corners", "ascend"]

#: Points are kept this far (relative) inside the box they are drawn from.
INTERIOR = 1e-6


class RegionSample(NamedTuple):
    """Points drawn from regions: ``points`` ``(N, d)``, ``valid`` ``(N,)`` (the row had a
    non-empty active box), and the box each point was drawn from, ``lower`` and ``upper``."""

    points: th.Tensor
    valid: th.Tensor
    lower: th.Tensor
    upper: th.Tensor


def _draw(kind, gen, device, *args, **kwargs):
    """A draw of ``th.rand`` / ``th.randint`` / ``th.randperm`` from ``gen``, moved to
    ``device``."""
    fn = {"rand": th.rand, "randint": th.randint, "randperm": th.randperm}[kind]
    out = fn(*args, device=gen.device, generator=gen, **kwargs)
    return out if out.device == th.device(device) else out.to(device=device)


def box_corners(lower, upper, blocks=1, inset=INTERIOR):
    """Corners of the boxes ``[lower, upper]`` ``(B, D)``, pulled ``inset`` (relative)
    inside: ``(B, M, D)``. A box of dimension ``D = blocks * d`` is a product of ``blocks``
    boxes of dimension ``d`` (a box of pairs: ``blocks=2``); the corners are taken with the
    same signs in every block. For ``d <= 4`` the :math:`2^d` corners, else the :math:`2d`
    centres of the faces. The corners stay functions of ``lower`` and ``upper``."""
    centre = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * (1.0 - inset)
    d = lower.shape[1] // blocks
    if d <= 4:
        signs = th.tensor([[1.0 if (m >> bit) & 1 else -1.0 for bit in range(d)]
                           for m in range(1 << d)], dtype=lower.dtype, device=lower.device)
    else:
        eye = th.eye(d, dtype=lower.dtype, device=lower.device)
        signs = th.cat([-eye, eye], dim=0)
    signs = signs.repeat(1, blocks)
    return centre.unsqueeze(1) + half.unsqueeze(1) * signs.unsqueeze(0)


def _relative(points, lower, upper):
    """Where ``points`` lie in their boxes, per coordinate in ``[0, 1]`` (the middle for a
    box of no width)."""
    span = upper - lower
    return th.where(span > 0.0, (points - lower) / span.clamp(min=1e-12),
                    th.full_like(points, 0.5))


def ascend(points, lower, upper, hinge, steps, step_size=None, valid=None):
    """Signed-gradient ascent of ``hinge`` (points ``(B, d)`` -> ``(B,)``) from ``points``
    inside the boxes ``[lower, upper]``, ``steps`` times, each step of ``step_size``
    (default a quarter of the box). The result has the values of the points found and the
    gradient of their relative place in the box (the ascent itself is not differentiated);
    rows that are not ``valid`` are left where they were and get no gradient."""
    lo, hi = lower.detach(), upper.detach()
    z = points.detach()
    if z.numel() == 0:
        return points
    if steps == 0:
        rel = _relative(z, lo, hi).clamp(min=INTERIOR, max=1.0 - INTERIOR)
    else:
        step = 0.25 * (hi - lo) if step_size is None else (
            step_size.detach().to(z) if isinstance(step_size, th.Tensor) else float(step_size))
        start = z.clone()
        for _ in range(steps):
            with th.enable_grad():
                probe = z.detach().requires_grad_(True)
                grad, = th.autograd.grad(hinge(probe).sum(), probe)
            z = th.minimum(th.maximum(z + step * grad.sign(), lo), hi).detach()
        if valid is not None:
            z = th.where(valid.unsqueeze(-1), z, start)
        rel = _relative(z, lo, hi).clamp(min=0.05, max=0.95)
    moving = lower + (upper - lower) * rel.detach()
    if valid is not None:
        moving = th.where(valid.unsqueeze(-1), moving, th.zeros_like(moving))
    return z + (moving - moving.detach())


class RegionSampler:
    """Points of unions of boxes (module docstring).

    :param n_samples: Points per row: one uniform point in a random non-empty active box of
        the row, then (for ``n_samples > 1`` and ``corners``) up to ``n_samples - 1`` corners
        of that box and the rest uniform points in it.
    :param n_adv: Steps of signed-gradient ascent on the hinge of the term the points are
        for (``0``: plain uniform points).
    :param corners: Whether corners are among the points.
    :param step_size: Step of the ascent (default: a quarter of the box).
    :param doubled_gradient: The published reading of the first uniform point of a row: its
        value ``lower + (upper - lower) u`` already depends on the box, and the direct code
        adds the re-parametrisation ``c + w (2u - 1)`` once more, so the gradient of the
        point with respect to the centre of its box is 2 (the other points of the row get
        1). Default ``False``: every point has the gradient of its place in the box (1 with
        respect to the centre). GroundedEL selects ``True``.

    Part (vocabulary value): ``points of a union of boxes``. Used by: ``grounded`` (the
    region terms of its objective and the region checks of its evaluation).
    """

    part = "points of a union of boxes"

    def __init__(self, n_samples=4, n_adv=2, corners=True, step_size=None,
                 doubled_gradient=False):
        self.n_samples, self.n_adv = int(n_samples), int(n_adv)
        self.corners, self.step_size = bool(corners), step_size
        self.doubled_gradient = bool(doubled_gradient)

    # -- one point per row --------------------------------------------------------------------
    def one(self, regions, gen, hinge=None, n_adv=0):
        """One point per row of ``regions`` (``cen``, ``w`` ``(B, K, D)``, ``active``)."""
        cen, w = regions.cen, regions.w
        bsz, k, dim = cen.shape
        usable = regions.active & (th.amin(w, dim=-1) > 0.0)
        valid = usable.any(dim=1)
        if bsz == 0:
            empty = cen.new_zeros((0, dim))
            return RegionSample(empty, valid, empty, empty)
        u_box = _draw("rand", gen, cen.device, (bsz, k), dtype=cen.dtype)
        box = th.argmax(th.where(usable, u_box, th.full_like(u_box, -1.0)), dim=1)
        pick = box.view(bsz, 1, 1).expand(bsz, 1, dim)
        c = th.where(valid.unsqueeze(-1), th.gather(cen, 1, pick).squeeze(1), cen.new_zeros(()))
        r = th.where(valid.unsqueeze(-1), th.gather(w, 1, pick).squeeze(1), w.new_zeros(()))
        lower, upper = c - r, c + r
        u = _draw("rand", gen, cen.device, (bsz, dim), dtype=cen.dtype)
        u = u.clamp(min=INTERIOR, max=1.0 - INTERIOR)
        points = lower + (upper - lower) * u
        if self.doubled_gradient:
            live = c + r * (2.0 * u - 1.0)
            points = points + (live - live.detach())
        points = th.where(valid.unsqueeze(-1), points, th.zeros_like(lower))
        if n_adv > 0 and hinge is not None and bool(valid.any()):
            points = ascend(points, lower, upper, hinge, n_adv, self.step_size, valid)
        return RegionSample(points, valid, lower, upper)

    # -- n_samples points per row -------------------------------------------------------------
    def __call__(self, regions, gen, hinge=None, blocks=1, n_samples=None, n_adv=None):
        """``n_samples`` points per row of ``regions``, sample-major (``N = n * B``): every
        row once, then again. ``blocks``: the boxes are products of that many boxes (see
        :func:`box_corners`)."""
        n = self.n_samples if n_samples is None else int(n_samples)
        steps = self.n_adv if n_adv is None else int(n_adv)
        if n == 1 or not self.corners:
            if n > 1:
                regions = type(regions)(regions.cen.repeat(n, 1, 1), regions.w.repeat(n, 1, 1),
                                        regions.active.repeat(n, 1))
            return self.one(regions, gen, hinge, steps)
        bsz, _, dim = regions.cen.shape
        if bsz == 0:
            return self.one(regions, gen)
        base = self.one(regions, gen)
        ends = box_corners(base.lower, base.upper, blocks)
        ends = th.where(base.valid.view(bsz, 1, 1), ends, th.zeros_like(ends))
        n_corners = min(ends.shape[1], n - 1)
        n_uniform = n - n_corners
        lower, upper = base.lower.repeat(n_uniform, 1), base.upper.repeat(n_uniform, 1)
        valid = base.valid.repeat(n_uniform)
        u = _draw("rand", gen, lower.device, (n_uniform * bsz, dim), dtype=lower.dtype)
        u = u.clamp(min=INTERIOR, max=1.0 - INTERIOR)
        points = th.where(valid.unsqueeze(-1), lower + (upper - lower) * u, th.zeros_like(lower))
        if steps > 0 and hinge is not None and bool(valid.any()):
            points = ascend(points, lower, upper, hinge, steps, self.step_size, valid)
        if n_corners > 0:
            if n_corners < ends.shape[1]:
                ends = ends[:, _draw("randperm", gen, lower.device, ends.shape[1])[:n_corners]]
            points = th.cat([points, ends.transpose(0, 1).reshape(n_corners * bsz, dim)])
        return RegionSample(points, base.valid.repeat(n), base.lower.repeat(n, 1),
                            base.upper.repeat(n, 1))

    # -- the regions of the parts -------------------------------------------------------------
    def classes(self, class_model, ids, gen, hinge=None, **kw):
        """Points of the classes ``ids``."""
        return self(class_model(ids), gen, hinge, **kw)

    def conjunction(self, class_model, a_ids, b_ids, gen, hinge=None, **kw):
        """Points of the intersections of the classes ``a_ids`` and ``b_ids`` (the exact
        conjunction of the regions: every pair of boxes)."""
        region, _ = class_model(a_ids).intersect(class_model(b_ids))
        return self(region, gen, hinge, **kw)

    def pairs(self, relation_model, ids, gen, hinge=None, **kw):
        """Points of the regions of pairs of the relations ``ids`` (the relation model says
        what a point is: ``relation_model.split_pair(points)`` is the pair ``(x, y)``)."""
        regions, blocks = relation_model.pair_regions(ids)
        return self(regions, gen, hinge, blocks=blocks, **kw)

    def exists(self, relation_model, class_model, rel_ids, class_ids, gen, hinge=None, **kw):
        """Points of :math:`\\exists r.B` (the preimage of the fillers' boxes)."""
        return self(relation_model.preimage_boxes(rel_ids, class_ids, class_model), gen,
                    hinge, **kw)


class ElementSampler:
    """Elements of the finite domain a step reads (module docstring).

    :param batch_size: Rows of an axiom table per step (all if there are fewer).
    :param n_random: Random elements per row (and per filler row).
    :param n_inside: Elements currently inside the left-hand side classes per row.
    :param n_fillers: Filler elements per existential row (each row keeps its own filler).
    :param whole_domain: A domain of at most this many elements is read whole.
    :param min_batch: At least this many elements of the batch are read with every row.

    Part (vocabulary value): ``elements of the finite domain``. Used by: ``grounded``.
    """

    part = "elements of the finite domain"

    def __init__(self, batch_size=4096, n_random=4, n_inside=32, n_fillers=64,
                 whole_domain=32, min_batch=8):
        self.batch_size, self.n_random = int(batch_size), int(n_random)
        self.n_inside, self.n_fillers = int(n_inside), int(n_fillers)
        self.whole_domain, self.min_batch = int(whole_domain), int(min_batch)

    def rows(self, table, gen, device):
        """Up to ``batch_size`` rows of ``table`` (with replacement when there are more)."""
        if int(table.shape[0]) <= self.batch_size:
            return table.to(device=device, dtype=th.long)
        pick = _draw("randint", gen, table.device, 0, int(table.shape[0]), (self.batch_size,),
                     dtype=th.long)
        return table[pick].to(device=device, dtype=th.long)

    def batch(self, ids, n_elements, gen):
        """Elements read with every row: the whole domain if it is small, else up to
        ``max(n_random, min_batch)`` of the distinct ``ids``."""
        if n_elements <= self.whole_domain:
            return th.arange(n_elements, dtype=th.long, device=ids.device)
        distinct = th.unique(ids)
        cap = max(self.n_random, self.min_batch)
        if distinct.numel() <= cap:
            return distinct
        return distinct[_draw("randperm", gen, distinct.device, distinct.numel())[:cap]]

    @th.no_grad()
    def inside(self, class_model, lhs, positions, gen):
        """``(B, I)``: up to ``n_inside`` elements per row that are crisp inside all the
        left-hand side classes ``lhs`` ``(B, L)`` (first) or at least one of them, chosen at
        random; short rows are padded with the row's first class element. ``None`` when the
        domain is read whole."""
        if self.n_inside <= 0 or positions.shape[0] <= self.whole_domain:
            return None
        lhs = lhs.reshape(lhs.shape[0], -1)
        bsz, n_lhs = lhs.shape
        if bsz == 0:
            return lhs.new_empty((0, 0))
        pool = None
        points = positions.detach()
        distinct, where_in = th.unique(lhs.reshape(-1), return_inverse=True)
        if points.shape[0] * distinct.numel() > 65536:
            size = min(points.shape[0], max(self.n_inside * 8, 128))
            pool = _draw("randint", gen, lhs.device, 0, points.shape[0], (size,), dtype=th.long)
            points = points[pool]
        if points.shape[0] == 0:
            return lhs.new_empty((bsz, 0))
        inside = (class_model.depth_table(points, distinct) > 0.0)[:, where_in]
        inside = inside.reshape(points.shape[0], bsz, n_lhs)
        key = _draw("rand", gen, lhs.device, (points.shape[0], bsz), dtype=points.dtype)
        rank = th.where(inside.all(-1), key, th.where(inside.any(-1), key + 1.0,
                                                       th.full_like(key, 2.0)))
        k = min(self.n_inside, points.shape[0])
        best, order = th.topk(rank, k=k, dim=0, largest=False)
        chosen = best < 2.0
        width = min(k, int(chosen.sum(0).max()))
        if width == 0:
            return lhs.new_empty((bsz, 0))
        found = order[:width].T
        if pool is not None:
            found = pool[found]
        return th.where(chosen[:width].T, found, lhs[:, :1].expand(bsz, width))

    def candidates(self, lhs, n_elements, gen, extra=()):
        """``(B, S)``: the row's own elements ``lhs`` ``(B,)`` or ``(B, L)``, ``n_random``
        random elements and the ``extra`` element sets (``(M,)`` for every row or
        ``(B, M)``)."""
        own = lhs.reshape(lhs.shape[0], -1).to(dtype=th.long)
        bsz = own.shape[0]
        out = [own]
        if self.n_random > 0:
            out.append(_draw("randint", gen, own.device, 0, n_elements, (bsz, self.n_random),
                             dtype=th.long))
        for more in extra:
            if more is not None and more.numel() > 0:
                out.append(more.to(own.device).expand(bsz, more.shape[-1]))
        return th.cat(out, dim=1)

    def for_rows(self, class_model, positions, lhs_classes, batch_ids, gen, own=None):
        """The candidate elements ``(B, S)`` of rows whose left-hand side classes are
        ``lhs_classes``: the class elements ``own`` (default the left-hand side classes),
        random elements, elements of the batch (``batch_ids``) and inside elements."""
        n = int(positions.shape[0])
        batch = self.batch(batch_ids.reshape(-1), n, gen)
        inside = self.inside(class_model, lhs_classes, positions, gen)
        return self.candidates(lhs_classes if own is None else own, n, gen, (batch, inside))

    def fillers(self, rows, filler_mask, gen):
        """``(B, F)`` elements for the fillers of existential rows ``(r, B, ...)``: the
        fillers of the relations of the batch (``filler_mask`` ``(R, N)``) and each row's own
        filler class, at most ``n_fillers`` (then the row's own first), then ``n_random``
        random elements."""
        bsz, n = int(rows.shape[0]), int(filler_mask.shape[1])
        used = th.zeros(filler_mask.shape[0], dtype=th.bool, device=rows.device)
        used[rows[:, 0]] = True
        mask = (filler_mask & used.unsqueeze(1)).any(0)
        mask[rows[:, 1]] = True
        found = th.nonzero(mask).squeeze(1)
        own = None
        if found.numel() > self.n_fillers:
            keep = max(self.n_fillers - 1, 0)
            found = found[_draw("randperm", gen, found.device, found.numel())[:keep]] \
                if keep > 0 else found[:0]
            own = rows[:, 1:2]
        out = [found.unsqueeze(0).expand(bsz, found.numel())]
        if self.n_random > 0:
            out.append(_draw("randint", gen, rows.device, 0, n, (bsz, self.n_random),
                             dtype=th.long))
        out = th.cat(out, dim=1)
        return out if own is None else th.cat([own, out], dim=1)

    def grid(self, n_elements, limit, gen, device):
        """Elements of a grid over the domain for role axioms: the batch elements of the
        first ``limit`` elements."""
        return self.batch(th.arange(min(n_elements, limit), device=device), n_elements, gen)

    def _probe_count(self, n_elements):
        return min(self.batch_size, max(256, n_elements * self.n_random))

    def class_probes(self, n_elements, class_ids, gen):
        """``(elements, classes)`` pairs a minimality penalty reads: all of them if there are
        at most 1024, else random ones."""
        device = class_ids.device
        n_c = int(class_ids.numel())
        if n_elements * n_c <= 1024:
            grid = th.cartesian_prod(th.arange(n_elements, device=device),
                                     th.arange(n_c, device=device))
            return grid[:, 0], class_ids[grid[:, 1]]
        m = self._probe_count(n_elements)
        x = _draw("randint", gen, device, 0, n_elements, (m,), dtype=th.long)
        return x, class_ids[_draw("randint", gen, device, 0, n_c, (m,), dtype=th.long)]

    def relation_probes(self, n_elements, n_relations, gen, device):
        """``(x, y, relations)`` triples a minimality penalty reads: all of them if there
        are at most 1024, else random ones."""
        if n_elements * n_elements * n_relations <= 1024:
            grid = th.cartesian_prod(th.arange(n_elements, device=device),
                                     th.arange(n_elements, device=device),
                                     th.arange(n_relations, device=device))
            return grid[:, 0], grid[:, 1], grid[:, 2]
        m = self._probe_count(n_elements)
        return tuple(_draw("randint", gen, device, 0, hi, (m,), dtype=th.long)
                     for hi in (n_elements, n_elements, n_relations))

    def corrupted(self, rows, n_individuals, gen, per_row=1):
        """Corrupted role assertions ``(a, r, b)``: ``min(len(rows), batch_size) * per_row``
        random rows with the head or the tail (at random) replaced by a random individual."""
        device = rows.device
        m = min(int(rows.shape[0]), self.batch_size) * per_row
        picked = rows[_draw("randint", gen, device, 0, int(rows.shape[0]), (m,), dtype=th.long)]
        other = _draw("randint", gen, device, 0, n_individuals, (m,), dtype=th.long)
        head = _draw("randint", gen, device, 0, 2, (m,), dtype=th.long) == 0
        return th.stack([th.where(head, other, picked[:, 0]), picked[:, 1],
                         th.where(head, picked[:, 2], other)], dim=1)




register_grounded_part("region sampler")(RegionSampler)


register_grounded_part("element sampler")(ElementSampler)
