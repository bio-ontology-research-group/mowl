"""A finite domain of points with classes and relations as unions of boxes, read by the query
evaluators of :mod:`mowl.evaluation.query_search`.

:class:`ElementBoxFamily` adapts any geometry that has ``positions(ids)`` (the points of the
elements), ``class_regions(ids)`` and ``translation_regions(ids)`` (unions of boxes:
``cen``, ``w``, ``active``), ``nb_classes``, ``nb_elements``, ``embed_dim`` and optional
``top_id`` / ``bot_id``; elements ``0 .. nb_classes - 1`` are the classes. It gives

* regions (:class:`~mowl.representation.library.box_union_algebra.BoundedBoxUnions`): the
  point of an element, the union of boxes of a class (top: one box of half-width
  :math:`10^6`; bottom: none), the image of a region under a relation of translations (a
  Minkowski sum; *exact*: through the elements inside the region), the classes a region
  overlaps;
* degrees, crisp or soft (temperature ``tau``): of a pair of elements in a relation (the
  depth of :math:`y` in :math:`x + V_r`), of an element in a class, of a target in the
  class of a witness.

GroundedEL answers queries through it (:mod:`mowl.nn.el.grounded.queries`); any geometry of
that shape can.
"""
from __future__ import annotations

import torch as th

from mowl.representation.library.box_union_algebra import BoundedBoxUnions, Boxes

__all__ = ["ElementBoxFamily", "TOP_HALF_WIDTH"]

#: Half-width of the box of the class top.
TOP_HALF_WIDTH: float = 1e6


class ElementBoxFamily:
    """The adapter of the module docstring.

    :param geometry: The geometry read.
    :param positions: Positions of all elements to read instead of ``geometry.positions``
        (for example detached ones).
    :param box_type: Type of the regions (``cen=``, ``w=``, ``active=``).
    :param tau: Temperature of the smooth depth of a region.
    """

    def __init__(self, geometry, positions=None, box_type=Boxes, tau=0.1):
        self.geometry, self._positions = geometry, positions
        self.box_type, self.tau = box_type, tau
        ref = positions if positions is not None else geometry.positions(th.arange(1))
        self.dtype, self.device = ref.dtype, ref.device
        self.n_elements, self.n_classes = geometry.nb_elements, geometry.nb_classes
        self.embed_dim = geometry.embed_dim
        self.parts_per_degree = max(1, int(getattr(geometry, "n_role_components", 1)))

    def algebra(self, max_boxes=64):
        return BoundedBoxUnions(max_boxes, box_type=self.box_type, tau=self.tau)

    def positions(self, ids):
        return self._positions[ids] if self._positions is not None else \
            self.geometry.positions(ids)

    # -- regions ----------------------------------------------------------------------
    def class_region(self, ids=None):
        """The union of boxes of every class of ``ids`` (all if ``None``)."""
        g = self.geometry
        if ids is None:
            ids = th.arange(self.n_classes, device=self.device)
        reg = g.class_regions(ids)
        cen, w, act = reg.cen, reg.w, reg.active
        bot, top = getattr(g, "bot_id", None), getattr(g, "top_id", None)
        if bot is not None and ids.numel() > 0:
            act = act & (ids != bot).unsqueeze(1)
            w = th.where(act.unsqueeze(-1), w, th.zeros_like(w))
        if top is not None and ids.numel() > 0:
            first = th.arange(act.shape[1], device=act.device) == 0
            is_top = (ids == top)
            whole_w = th.where(first[None, :, None], th.full_like(w, TOP_HALF_WIDTH),
                               th.zeros_like(w))
            cen = th.where(is_top[:, None, None], th.zeros_like(cen), cen)
            w = th.where(is_top[:, None, None], whole_w, w)
            act = th.where(is_top[:, None], first[None].expand_as(act), act)
        return self.box_type(cen=cen, w=w, active=act)

    def anchor(self, elements):
        return self.algebra().points(self.geometry.positions(elements))

    def placeholder(self, bsz):
        """One box of width zero at the origin per row (a region that is read no further)."""
        zeros = th.zeros((bsz, 1, self.embed_dim), dtype=self.dtype, device=self.device)
        return self.box_type(cen=zeros, w=zeros.clone(),
                             active=th.ones((bsz, 1), dtype=th.bool, device=self.device))

    def project(self, region, role_ids, max_boxes):
        """The image of ``region`` under the relations ``role_ids``: :math:`R + V_r`."""
        rel = self.geometry.translation_regions(role_ids)
        return self.algebra(max_boxes).minkowski(region, rel.cen, rel.w, rel.active)

    def project_exact(self, region, role_ids, max_boxes, element_ids=None, max_witnesses=256):
        """The image through the elements inside ``region`` (at most ``max_witnesses``, the
        deepest): :math:`\\bigcup_x (p_x + V_r)`."""
        alg = self.algebra(max_boxes)
        elements = th.arange(self.n_elements, device=self.device) if element_ids is None \
            else element_ids
        rel = self.geometry.translation_regions(role_ids)
        bsz = region.cen.shape[0]
        if bsz == 0 or elements.numel() == 0 or rel.cen.shape[1] == 0:
            return alg.empty(bsz, self.embed_dim, region.cen)
        depth = self.entity_scores(region, elements)
        inside = depth > 0.0
        keep = min(elements.numel(), max_witnesses)
        _, top = th.topk(th.where(inside, depth, th.full_like(depth, float("-inf"))), k=keep,
                         dim=1, largest=True, sorted=True)
        points = alg.make(self.geometry.positions(elements)[top].to(dtype=region.cen.dtype),
                          th.zeros((bsz, keep, self.embed_dim), dtype=region.w.dtype,
                                   device=self.device), th.gather(inside, 1, top))
        sums = BoundedBoxUnions(max(1, keep * rel.cen.shape[1]), box_type=self.box_type) \
            .minkowski(points, rel.cen, rel.w, rel.active)
        return alg.cap(sums) if sums.cen.shape[1] > max_boxes else \
            alg.compact(sums.cen, sums.w, sums.active)

    def isa_region(self, region, max_boxes):
        """The union of the boxes of the classes that ``region`` overlaps."""
        table = self.class_region()
        keep = self.concept_scores(region, table=table) > 0.0
        table = self.box_type(cen=table.cen.to(dtype=region.cen.dtype),
                              w=table.w.to(dtype=region.w.dtype), active=table.active)
        return self.algebra(max_boxes).flatten(table, keep)

    def intersect(self, a, b, max_boxes):
        return self.algebra(max_boxes).intersect(a, b)

    def union(self, a, b, max_boxes):
        return self.algebra(max_boxes).union(a, b)

    def select_rows(self, rows, a, b):
        return self.algebra().select_rows(rows, a, b)

    def cap(self, region, max_boxes):
        """At most ``max_boxes`` boxes (a region within the bound is returned as it is)."""
        return self.algebra(max_boxes).cap(region) if region.cen.shape[1] > max_boxes \
            else region

    # -- scores -----------------------------------------------------------------------
    def entity_scores(self, region, elements=None, chunk=65536, max_cells=1 << 27):
        """Crisp depth ``(B, N)`` of the elements in ``region``."""
        if elements is None:
            elements = th.arange(self.n_elements, device=self.device)
        bsz, m, dim = region.cen.shape
        if not th.is_grad_enabled() and self.device.type == "cpu":
            max_cells = min(max_cells, 1 << 21)
        step = max(1, min(chunk, max_cells // max(1, bsz * max(1, m) * dim)))
        alg = self.algebra()
        out = [alg.depth(self.geometry.positions(elements[s:s + step]).unsqueeze(0).unsqueeze(2),
                         region) for s in range(0, max(1, elements.numel()), step)]
        return out[0] if len(out) == 1 else th.cat(out, dim=1)

    def concept_scores(self, region, class_ids=None, table=None, max_cells=1 << 27):
        """Overlap depth ``(B, C)`` of ``region`` with the classes."""
        table = self.class_region(class_ids) if table is None else table
        n = table.cen.shape[0]
        bsz, m, dim = region.cen.shape
        if bsz == 0 or n == 0:
            return th.empty((bsz, n), dtype=region.cen.dtype, device=region.cen.device)
        step = max(1, max_cells // max(1, bsz * max(1, m * table.cen.shape[1]) * dim))
        alg = self.algebra()
        out = [alg.overlap_table(region, self.box_type(cen=table.cen[s:s + step],
                                                       w=table.w[s:s + step],
                                                       active=table.active[s:s + step]))
               for s in range(0, n, step)]
        return out[0] if len(out) == 1 else th.cat(out, dim=1)

    def member_scores(self, class_ids, chunk=65536):
        """Crisp depth ``(B, N)`` of every element in the class of each row."""
        return self.entity_scores(self.class_region(class_ids), chunk=chunk)

    # -- degrees ----------------------------------------------------------------------
    def role_degree(self, ids, role_ids, targets, tau=None, budget=1 << 22):
        """Degree ``(B, M, K)`` of ``targets[b, m]`` being related by ``role_ids[b]`` from the
        witness ``ids[b, k]``: the depth of the target in :math:`p_z + V_r`."""
        rel = self.geometry.translation_regions(role_ids)
        cen = self.positions(ids).to(dtype=self.dtype).unsqueeze(2) + \
            rel.cen.to(dtype=self.dtype).unsqueeze(1)
        w = rel.w.to(dtype=self.dtype)[:, None, None]
        act = rel.active[:, None, None]
        step = max(1, budget // max(1, ids.shape[0] * ids.shape[1] * cen.shape[2]
                                    * self.embed_dim))
        alg = self.algebra()
        out = [alg.union_depth(self.positions(targets[:, s:s + step]).to(dtype=self.dtype)
                               [:, :, None, None, :], cen[:, None], w, act, tau=tau)
               for s in range(0, max(1, targets.shape[1]), step)]
        return out[0] if len(out) == 1 else th.cat(out, dim=1)

    def class_degree(self, ids, classes=None, tau=None, budget=1 << 26):
        """Degree of the witness ``ids[b, k]`` in a class: ``(B, C, K)`` for all classes,
        or ``(B, M, K)`` for the classes ``classes[b, m]``."""
        table = self.class_region()
        z = self.positions(ids).to(dtype=self.dtype)[:, None, :, None, :]
        alg = self.algebra()
        if classes is not None:
            return alg.union_depth(z, table.cen[classes].to(dtype=self.dtype)[:, :, None],
                                   table.w[classes].to(dtype=self.dtype)[:, :, None],
                                   table.active[classes][:, :, None], tau=tau)
        n, m = table.cen.shape[:2]
        step = max(1, min(n, budget // max(1, ids.numel() * m * self.embed_dim)))
        out = [alg.union_depth(z, table.cen[s:s + step].to(dtype=self.dtype)[None, :, None],
                               table.w[s:s + step].to(dtype=self.dtype)[None, :, None],
                               table.active[s:s + step][None, :, None], tau=tau)
               for s in range(0, n, step)]
        return out[0] if len(out) == 1 else th.cat(out, dim=1)

    def member_degree(self, class_ids, targets, tau=None):
        """Degree ``(B, M, K)`` of ``targets[b, m]`` in the class ``class_ids[b, k]``."""
        bsz, k = class_ids.shape
        table = self.class_region(class_ids.reshape(-1))
        m = table.cen.shape[1]
        cen = table.cen.to(dtype=self.dtype).view(bsz, k, m, self.embed_dim)
        w = table.w.to(dtype=self.dtype).view(bsz, k, m, self.embed_dim)
        act = table.active.view(bsz, k, m)
        points = self.positions(targets).to(dtype=self.dtype)[:, :, None, None, :]
        return self.algebra().union_depth(points, cen[:, None], w[:, None], act[:, None],
                                          tau=tau)
