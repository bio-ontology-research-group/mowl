"""Crisp violations of the axioms in a grounded geometry: an evaluation regime.

:class:`CrispViolations` reads the geometry crisply (a point is in a class iff its depth is
positive; the class model's ``depth(..., crisp=True)``, the relation model's
``depth(..., crisp=True)``) and checks every axiom twice, like the objective reads it:

- **at elements**: the rule of the form on crisp memberships of up to 256 live elements
  (``elem_viol``: the share of violated grounded rules);
- **in regions**: points of the left-hand side (elements, points sampled with
  :class:`~mowl.representation.library.grounded_sampling.RegionSampler`, corners of the
  boxes) and the depth of the right-hand side at them (``region_viol``: the share of axioms
  with a counterexample). A containment (``A ⊑ B``) is violated at tolerance ``tol`` if
  some point of the left-hand side has right-hand depth ``< -tol`` (``<= 0`` at ``tol=0``),
  a disjointness (``A ⊑ ⊥``) if some point has left-hand depth ``> tol``.

Per form the check is a few lines: which points (:meth:`CrispViolations._points`), the
left-hand side and the right-hand side at them. With more than ``few_axioms`` region axioms
the points are drawn one per row for chunks of rows (no corners), else ``n_eval / axioms``
points with corners for every row.
"""
from __future__ import annotations

from typing import Dict, Mapping

import torch as th

from mowl.representation.library.grounded_parts import EMPTY_DEPTH_FLOOR, register_grounded_part
from mowl.representation.library.grounded_sampling import RegionSampler

__all__ = ["CrispViolations", "ExactDomainCheck", "TOLERANCES"]

#: The tolerances of ``n_axioms_violated_by_depth``.
TOLERANCES = (("0", 0.0), ("1e-5", 1e-5), ("1e-4", 1e-4), ("1e-3", 1e-3), ("1e-2", 1e-2))
#: The concept forms (checked at elements and in regions, in this order), then the role forms.
CONCEPT_FORMS = ("gci0", "gci0_bot", "gci1", "gci1_bot", "gci2", "gci3", "gci3_bot")
ROLE_FORMS = ("role_inclusion", "role_chain")
#: The columns of each concept form that are classes.
CLASS_COLUMNS = {"gci0": (0, 1), "gci0_bot": (0,), "gci1": (0, 1, 2), "gci1_bot": (0, 1),
                 "gci2": (0, 2), "gci3": (1, 2), "gci3_bot": (1,)}
_INF = float("inf")


def _text(form, row, c, r):
    """The axiom as text (short names ``c`` of classes, ``r`` of relations)."""
    def cn(i):
        return c.get(i, str(i))

    def rn(i):
        return r.get(i, str(i))
    return {"gci0": lambda: f"{cn(row[0])} <= {cn(row[1])}",
            "gci0_bot": lambda: f"{cn(row[0])} <= bot",
            "gci1": lambda: f"{cn(row[0])} cap {cn(row[1])} <= {cn(row[2])}",
            "gci1_bot": lambda: f"{cn(row[0])} cap {cn(row[1])} <= bot",
            "gci2": lambda: f"{cn(row[0])} <= exists {rn(row[1])}.{cn(row[2])}",
            "gci3": lambda: f"exists {rn(row[0])}.{cn(row[1])} <= {cn(row[2])}",
            "gci3_bot": lambda: f"exists {rn(row[0])}.{cn(row[1])} <= bot",
            "role_inclusion": lambda: f"{rn(row[0])} <= {rn(row[1])}",
            "role_chain": lambda: f"{rn(row[0])} o {rn(row[1])} <= {rn(row[2])}"}[form]()


def contained(lhs, rhs, extra=None):
    """Per row of ``(B, M)`` points: ``(has, worst, violated)``: whether a point is in the
    left-hand side, the smallest right-hand depth there, and whether one is not inside
    (or ``extra``)."""
    if lhs.shape[1] == 0:
        no = th.zeros(lhs.shape[0], dtype=th.bool, device=lhs.device)
        return no, th.full(no.shape, _INF, device=lhs.device), no
    has = lhs.any(1)
    worst = th.where(lhs, rhs, th.full_like(rhs, _INF)).amin(1)
    bad = (lhs & (rhs <= 0.0)).any(1)
    return has, worst, has & (bad if extra is None else bad | extra)


def overlapping(depth, counted, extra=None):
    """Per row of ``(B, M)`` points: ``(deepest, violated)``: the largest left-hand depth
    of the counted points (:data:`EMPTY_DEPTH_FLOOR` if none), and whether it is positive
    (or ``extra``)."""
    if depth.shape[1] == 0:
        deepest = th.full((depth.shape[0],), EMPTY_DEPTH_FLOOR, device=depth.device)
    else:
        deepest = th.where(counted, depth, th.full_like(depth, EMPTY_DEPTH_FLOOR)).amax(1)
    bad = deepest > 0.0
    return deepest, bad if extra is None else bad | extra


def all_box_corners(regions, gen, cap=32, inset=1e-4):
    """The corners (``inset`` inside) of every non-empty active box of ``regions``
    (``(1, K, d)``): all :math:`2^d` for ``d <= 5``, else ``cap`` random corners per box."""
    usable = regions.active & (th.amin(regions.w, dim=-1) > 0.0)
    cen, half = regions.cen[usable], regions.w[usable] * (1.0 - inset)
    m, d = cen.shape
    if m == 0:
        return cen.new_empty((0, regions.cen.shape[-1]))
    if d <= 5 and (1 << d) <= cap:
        signs = th.tensor([[1.0 if (k >> bit) & 1 else -1.0 for bit in range(d)]
                           for k in range(1 << d)], dtype=cen.dtype, device=cen.device)
        signs = signs.unsqueeze(0)
    else:
        signs = th.randint(0, 2, (m, cap, d), device=gen.device, generator=gen).to(cen) * 2.0 - 1.0
    return (cen.unsqueeze(1) + half.unsqueeze(1) * signs).reshape(-1, d)


class _Tally:
    """Counts of violated rules and axioms, the worst depth of every axiom."""

    def __init__(self, tol):
        self.tol = tol
        self.worst: Dict[str, Dict[str, float]] = {}
        self.by_tol = {k: 0 for k, _ in TOLERANCES}
        self.rules = self.rules_bad = self.axioms = self.axioms_bad = 0

    def rule(self, violation):
        self.rules_bad += int((violation > 0.0).sum())
        self.rules += int(violation.numel())

    def _axiom(self, form, text, value, keep, bad_at):
        bucket = self.worst.setdefault(form, {})
        bucket[text] = keep(bucket[text], value) if text in bucket else value
        for key, t in TOLERANCES:
            self.by_tol[key] += int(bad_at(t))
        self.axioms += 1
        self.axioms_bad += int(bad_at(self.tol))

    def containment(self, form, text, has, worst, violated):
        value = float(worst) if has else 0.0
        self._axiom(form, text, value, min,
                    lambda t: violated if t == 0.0 else (has and value < -t))

    def disjointness(self, form, text, deepest, violated):
        value = float(deepest)
        self._axiom(form, text, value, max, lambda t: violated if t == 0.0 else value > t)


class CrispViolations:
    """Crisp violation rates of the axioms in a grounded geometry (module docstring).

    :param n_eval: Region points budget over all axioms.
    :param tol: Tolerance of a region counterexample (``region_viol``).
    :param n_elements: Live elements the element checks read (at most).
    :param n_fillers: Elements used as fillers ``y`` of existentials and role inclusions.
    :param n_chain: Elements of the grid of a role chain.
    :param few_axioms: Up to this many region axioms every row gets ``n_eval / axioms``
        points (and corners); above, chunks of rows get one point per draw.

    Part (vocabulary value): ``crisp violations``. Used by: ``grounded``
    (:meth:`mowl.models.GroundedEL.crisp_report`).
    """

    part = "crisp violations"

    def __init__(self, n_eval=10000, tol=0.0, n_elements=256, n_fillers=32, n_chain=16,
                 few_axioms=512, chunk=512):
        self.n_eval, self.tol = int(n_eval), float(tol)
        self.n_elements, self.n_fillers, self.n_chain = n_elements, n_fillers, n_chain
        self.few_axioms, self.chunk = few_axioms, chunk

    # =========================================================================================
    def report(self, geometry, axioms: Mapping, grounding, generator, class_names=None,
               relation_names=None, constraints=None):
        """``{"elem_viol", "region_viol", "worst_depth", "n_axioms_violated_by_depth",
        "n_empty_classes", "n_dead_elements"}`` of ``geometry`` on ``axioms``."""
        if constraints is None:
            from mowl.nn.el.grounded.constraints import ConstraintSet
            constraints = ConstraintSet()
        with th.no_grad():
            s = _Look(self, geometry, axioms, grounding, generator, constraints,
                      class_names or {}, relation_names or {})
            for form in CONCEPT_FORMS + ROLE_FORMS:
                if s.rows(form) is not None:
                    getattr(self, "elements_" + form)(s, s.rows(form))
            n_region = sum(len(s.rows(f)) for f in CONCEPT_FORMS + ROLE_FORMS
                           if s.rows(f) is not None)
            few = n_region <= self.few_axioms
            s.n_per = max(16, min(2000, self.n_eval // max(n_region, 1))) if few else \
                max(1, min(16, self.n_eval // max(n_region, 1)))
            for form in CONCEPT_FORMS:
                rows = s.rows(form)
                if rows is not None:
                    for chunk in (rows.split(1) if few else rows.split(self.chunk)):
                        self._record(s, form, chunk, getattr(self, "region_" + form)(s, chunk, few))
            s.n_per = max(16, min(2000, self.n_eval // max(n_region, 1)))
            for form in ROLE_FORMS:
                rows = s.rows(form)
                if rows is not None:
                    for chunk in rows.split(1):
                        self._record(s, form, chunk, getattr(self, "region_" + form)(s, chunk, True))
            self.elements_assertions(s)
            return s.result()

    @staticmethod
    def _record(s, form, rows, outcome):
        for i, row in enumerate(rows.tolist()):
            text = _text(form, row, s.class_names, s.relation_names)
            if len(outcome) == 3:
                s.tally.containment(form, text, bool(outcome[0][i]), float(outcome[1][i]),
                                    bool(outcome[2][i]))
            else:
                s.tally.disjointness(form, text, float(outcome[0][i]), bool(outcome[1][i]))

    # =========================================================================================
    # element checks: the rule of the form on crisp memberships of the live elements
    # =========================================================================================
    def elements_gci0(self, s, rows):
        s.tally.rule(s.C.gci0(s.m(rows[:, 0]), s.m(rows[:, 1]), dead=s.dead_at_rows(rows)))

    def elements_gci0_bot(self, s, rows):
        s.tally.rule(s.C.gci0_bot(s.m(rows[:, 0]), dead=s.dead_at_rows(rows)))

    def elements_gci1(self, s, rows):
        s.tally.rule(s.C.gci1(s.m(rows[:, 0]), s.m(rows[:, 1]), s.m(rows[:, 2]),
                              dead=s.dead_at_rows(rows)))

    def elements_gci1_bot(self, s, rows):
        s.tally.rule(s.C.gci1_bot(s.m(rows[:, 0]), s.m(rows[:, 1]), dead=s.dead_at_rows(rows)))

    def elements_gci2(self, s, rows):
        if s.n_at == 0:
            return
        r_depth = s.grid_relation(s.ids, rows[:, 2:3], rows[:, 1])       # (B, L, 1)
        dead_b = s.dead_f[rows[:, 2]].unsqueeze(0).expand(s.n_at, rows.shape[0])
        s.tally.rule(s.C.gci2(s.m(rows[:, 0]), s.crisp(r_depth[..., 0]).T,
                              dead=s.dead_at_rows(rows), dead_filler=dead_b))

    def elements_gci3(self, s, rows):
        if s.n_at == 0 or s.n_y == 0:
            return
        rho = s.crisp(s.grid_relation(s.ids, s.ys, rows[:, 0]))             # (B, L, Y)
        s.tally.rule(s.C.gci3(rho, s.m(rows[:, 1])[:s.n_y].T.unsqueeze(1).expand_as(rho),
                              s.m(rows[:, 2]).T.unsqueeze(2).expand_as(rho),
                              dead=s.dead_at.view(1, -1, 1).expand_as(rho)))

    def elements_gci3_bot(self, s, rows):
        if s.n_at == 0 or s.n_y == 0:
            return
        rho = s.crisp(s.grid_relation(s.ids, s.ys, rows[:, 0]))
        s.tally.rule(s.C.gci3_bot(rho, s.m(rows[:, 1])[:s.n_y].T.unsqueeze(1).expand_as(rho),
                                  dead=s.dead_at.view(1, -1, 1).expand_as(rho)))

    def elements_role_inclusion(self, s, rows):
        if s.n_at == 0 or s.n_y == 0:
            return
        s.tally.rule(s.C.role_inclusion(s.crisp(s.grid_relation(s.ids, s.ys, rows[:, 0])),
                                        s.crisp(s.grid_relation(s.ids, s.ys, rows[:, 1]))))

    def elements_role_chain(self, s, rows):
        c = s.ids[:self.n_chain]
        if c.numel() == 0:
            return
        xy = s.crisp(s.grid_relation(c, c, rows[:, 0]))                      # (B, x, y)
        yz = s.crisp(s.grid_relation(c, c, rows[:, 1]))                      # (B, y, z)
        xz = s.crisp(s.grid_relation(c, c, rows[:, 2]))                      # (B, x, z)
        s.tally.rule(s.C.role_chain(xy.unsqueeze(3).expand(-1, -1, -1, c.numel()),
                                    yz.unsqueeze(1).expand(-1, c.numel(), -1, -1),
                                    xz.unsqueeze(2).expand(-1, -1, c.numel(), -1)))

    def elements_assertions(self, s):
        for form in ("class_assertion", "object_property_assertion"):
            rows = s.rows(form)
            for part in (rows.split(4096) if rows is not None else ()):
                a = s.g.nb_classes + part[:, 0]
                if form == "class_assertion":
                    s.tally.rule(s.C.class_assertion(s.crisp(s.depth(s.pos[a], part[:, 1]))))
                else:
                    b = s.g.nb_classes + part[:, 2]
                    s.tally.rule(s.C.role_assertion(s.crisp(s.relations.depth(
                        s.pos[a], s.pos[b], part[:, 1], crisp=True))))

    # =========================================================================================
    # region checks: points of the left-hand side, depth of the right-hand side
    # =========================================================================================
    def region_gci0(self, s, rows, few):
        p = s.points(s.sampler.classes, rows, few, rows[:, 0], corners_of=rows[0, 0])
        return contained(s.inside(p, rows[:, 0]), s.depth_at(p, rows[:, 1]))

    def region_gci0_bot(self, s, rows, few):
        p = s.points(s.sampler.classes, rows, few, rows[:, 0])
        return overlapping(s.depth_at(p, rows[:, 0]), p.counted, p.sampled_any)

    def region_gci1(self, s, rows, few):
        p = s.points(s.sampler.conjunction, rows, few, rows[:, 0], rows[:, 1])
        return contained(s.inside(p, rows[:, 0]) & s.inside(p, rows[:, 1]),
                         s.depth_at(p, rows[:, 2]))

    def region_gci1_bot(self, s, rows, few):
        p = s.points(s.sampler.conjunction, rows, few, rows[:, 0], rows[:, 1],
                     instead=rows[:, 0] if few else None)
        both = th.minimum(s.depth_at(p, rows[:, 0]), s.depth_at(p, rows[:, 1]))
        return overlapping(both, p.counted, p.sampled_any)

    def region_gci2(self, s, rows, few):
        p = s.points(s.sampler.classes, rows, few, rows[:, 0], corners_of=rows[0, 0])
        d_a = s.depth_at(p, rows[:, 0])
        dead_b = s.dead_c[rows[:, 2]]
        rhs = th.where(dead_b.unsqueeze(1), -d_a, s.relation_to(p, rows[:, 2], rows[:, 1]))
        return contained((d_a > 0.0) & p.counted, rhs, dead_b)

    def region_gci3(self, s, rows, few):
        x, y, counted = s.pairs(rows, few)
        lhs = (s.relation_at(x, y, rows[:, 0]) > 0.0) & (s.depth_on(y, rows[:, 1]) > 0.0)
        return contained(lhs & counted, s.depth_on(x, rows[:, 2]))

    def region_gci3_bot(self, s, rows, few):
        x, y, counted = s.pairs(rows, few)
        both = th.minimum(s.relation_at(x, y, rows[:, 0]), s.depth_on(y, rows[:, 1]))
        return overlapping(both, counted)

    def region_role_inclusion(self, s, rows, few):
        sample = s.sampler.pairs(s.relations, rows[:, 0], s.gen, n_samples=s.n_per, n_adv=0)
        x, y = s.relations.split_pair(sample.points)
        x, y, counted = x.unsqueeze(0), y.unsqueeze(0), sample.valid.unsqueeze(0)
        return contained((s.relation_at(x, y, rows[:, 0]) > 0.0) & counted,
                         s.relation_at(x, y, rows[:, 1]))

    def region_role_chain(self, s, rows, few):
        if s.translation:
            c = s.pos[s.ids[:self.n_chain]]
            n = c.shape[0]
            x = c.view(n, 1, 1, -1).expand(n, n, n, -1).reshape(1, -1, c.shape[1])
            y = c.view(1, n, 1, -1).expand(n, n, n, -1).reshape(1, -1, c.shape[1])
            z = c.view(1, 1, n, -1).expand(n, n, n, -1).reshape(1, -1, c.shape[1])
            counted = th.ones(x.shape[:2], dtype=th.bool, device=x.device)
        else:
            first = s.sampler.pairs(s.relations, rows[:, 0], s.gen, n_samples=s.n_per, n_adv=0)
            second = s.sampler.pairs(s.relations, rows[:, 1], s.gen, n_samples=s.n_per, n_adv=0)
            (x, y), (_, z) = (s.relations.split_pair(first.points),
                              s.relations.split_pair(second.points))
            x, y, z = x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)
            counted = (first.valid & second.valid).unsqueeze(0)
        lhs = (s.relation_at(x, y, rows[:, 0]) > 0.0) & (s.relation_at(y, z, rows[:, 1]) > 0.0)
        return contained(lhs & counted, s.relation_at(x, z, rows[:, 2]))


class _Points:
    """Points ``at`` ``(B, M, d)`` of the rows (``counted``: the ones to read;
    ``sampled_any``: whether the left-hand side had a point). With ``elements`` the live
    elements come first and are read from the depth table (they are not in ``at``)."""

    def __init__(self, at, counted, sampled_any, elements=False):
        self.at, self.counted, self.sampled_any = at, counted, sampled_any
        self.elements = elements


class _Look:
    """What one report reads."""

    def __init__(self, regime, geometry, axioms, grounding, gen, constraints, class_names,
                 relation_names):
        self.r, self.g, self.gen, self.C = regime, grounding, gen, constraints
        self.class_names, self.relation_names = class_names, relation_names
        self.classes, self.relations = geometry.class_model, geometry.relation_model
        self.translation = getattr(self.relations, "mode", None) == "translation"
        self.sampler = RegionSampler(n_adv=0)
        self.axioms = axioms
        self.tally = _Tally(regime.tol)
        self.pos = geometry.shared.positions()
        self.device = self.pos.device
        dead_c = geometry.shared.dead() > 0.5
        self.n_dead = int(dead_c.sum()) - int(grounding.bot_id is not None
                                              and bool(dead_c[grounding.bot_id]))
        if grounding.bot_id is not None:
            dead_c[grounding.bot_id] = True
        self.dead_c, self.dead_f = dead_c, dead_c.to(dtype=th.float32)
        live = th.nonzero(~dead_c, as_tuple=False).view(-1)
        if live.numel() > regime.n_elements:
            live = live[th.randperm(live.numel(), device="cpu", generator=gen)[
                :regime.n_elements].to(device=self.device)]
        self.ids, self.n_at = live, int(live.numel())
        self.at, self.dead_at = self.pos[live], self.dead_f[live]
        self.n_y = min(self.n_at, regime.n_fillers)
        self.ys = live[:self.n_y]
        cols = [self.rows(f)[:, j] for f, js in CLASS_COLUMNS.items()
                if self.rows(f) is not None for j in js]
        mentioned = th.unique(th.cat(cols)) if cols else th.empty(0, dtype=th.long,
                                                                     device=self.device)
        self.col = th.full((grounding.nb_classes,), -1, dtype=th.long, device=self.device)
        self.col[mentioned] = th.arange(mentioned.numel(), device=self.device)
        self.table = self.classes.depth_table(self.at, mentioned, chunk_size=512)
        self.member = self.crisp(self.table)

    def rows(self, form):
        table = self.axioms.get(form)
        if table is None or len(table) == 0:
            return None
        return table[:].to(device=self.device, dtype=th.long)

    # -- crisp readings -----------------------------------------------------------------------
    @staticmethod
    def crisp(depth):
        return (depth > 0).to(dtype=depth.dtype if depth.is_floating_point() else th.float32)

    def m(self, class_ids):
        """Crisp memberships ``(L, B)`` of the live elements in the classes."""
        return self.member[:, self.col[class_ids]]

    def dead_at_rows(self, rows):
        return self.dead_at.unsqueeze(1).expand(self.n_at, rows.shape[0])

    def depth(self, points, class_ids):
        return self.classes.depth(points, class_ids, crisp=True)

    def depth_on(self, points, class_ids):
        """Crisp depth of ``(B, M, d)`` points in the class of each row ``(B, M)``."""
        b, m, d = points.shape
        return self.depth(points.reshape(-1, d), class_ids.repeat_interleave(m)).view(b, m)

    def depth_at(self, p, class_ids):
        depth = self.depth_on(p.at, class_ids)
        if p.elements:
            depth = th.cat([self.table[:, self.col[class_ids]].T, depth], 1)
        return depth

    def relation_to(self, p, filler_ids, rel_ids):
        """Crisp depth ``(B, M)`` of the pairs (point, filler element of the row)."""
        filler = self.pos[filler_ids].unsqueeze(1).expand_as(p.at)
        depth = self.relation_at(p.at, filler, rel_ids)
        if p.elements:
            depth = th.cat([self.grid_relation(self.ids, filler_ids.unsqueeze(1), rel_ids)[..., 0],
                            depth], 1)
        return depth

    def inside(self, p, class_ids):
        return (self.depth_at(p, class_ids) > 0.0) & p.counted

    def relation_at(self, x, y, rel_ids):
        """Crisp depth of the pairs ``(B, M, d)`` in the relation of each row ``(B, M)``."""
        b, m, d = x.shape
        return self.relations.depth(x.reshape(-1, d), y.reshape(-1, d),
                                    rel_ids.repeat_interleave(m), crisp=True).view(b, m)

    def grid_relation(self, xs, ys, rel_ids):
        """Crisp depth ``(B, X, Y)`` of every pair of elements ``xs`` x ``ys`` (``ys``
        ``(Y,)`` or one filler per row ``(B, Y)``) in the relation of the row."""
        b = rel_ids.shape[0]
        ys = ys.expand(b, -1) if ys.ndim == 2 else ys.unsqueeze(0).expand(b, -1)
        _, crisp = self.relations.grid_membership(
            self.pos[xs.unsqueeze(0).expand(b, -1)], self.pos[ys], rel_ids, need_crisp=True)
        return crisp.view(b, xs.numel(), ys.shape[1])

    # -- points of the left-hand side ---------------------------------------------------------
    def points(self, draw, rows, few, *ids, corners_of=None, instead=None):
        """The live elements and points drawn by ``draw`` (a method of the region sampler)
        from the region of ``ids``: ``n_per`` points with corners for one row (``few``,
        plus the corners of the boxes of ``corners_of`` and, if nothing was drawn, points of
        ``instead``), else one point per draw for every row."""
        b, n = rows.shape[0], self.n_per
        if few:
            sample = draw(self.classes, *ids, self.gen, n_samples=n)
            sampled_any = sample.valid.any().view(1)
            parts = [self.at, sample.points[sample.valid]]
            if instead is not None and not bool(sampled_any):
                more = self.sampler.classes(self.classes, instead, self.gen, n_samples=n)
                parts[1] = more.points[more.valid]
            if corners_of is not None:
                parts.append(all_box_corners(self.classes(int(corners_of)), self.gen))
            at = th.cat(parts).unsqueeze(0)
            return _Points(at, th.ones(at.shape[:2], dtype=th.bool, device=self.device),
                           sampled_any)
        sample = draw(self.classes, *(i.repeat_interleave(n) for i in ids), self.gen,
                      n_samples=1)
        counted = th.cat([th.ones((b, self.n_at), dtype=th.bool, device=self.device),
                          sample.valid.view(b, n)], 1)
        return _Points(sample.points.view(b, n, -1), counted, sample.valid.view(b, n).any(1),
                       elements=True)

    def pairs(self, rows, few):
        """``(x, y, counted)`` ``(B, M, d)``: pairs of the relation of each existential row:
        the live elements against the fillers (translations) or pairs drawn from the
        relation (products of boxes)."""
        b = rows.shape[0]
        if self.translation:
            x = self.at.unsqueeze(1).expand(-1, self.n_y, -1).reshape(1, -1, self.at.shape[1])
            y = self.pos[self.ys].unsqueeze(0).expand(self.n_at, -1, -1).reshape(
                1, -1, self.at.shape[1])
            x, y = x.expand(b, -1, -1), y.expand(b, -1, -1)
            return x, y, th.ones(x.shape[:2], dtype=th.bool, device=self.device)
        n = self.n_per
        ids = rows[:, 0] if few else rows[:, 0].repeat_interleave(n)
        sample = self.sampler.pairs(self.relations, ids, self.gen, n_samples=n if few else 1)
        x, y = self.relations.split_pair(sample.points)
        return x.reshape(b, -1, x.shape[1]), y.reshape(b, -1, y.shape[1]), \
            sample.valid.view(b, -1)

    # -- the result ---------------------------------------------------------------------------
    def result(self):
        t = self.tally
        empty = ~self.classes.regions().active.any(dim=1)
        for c in (self.g.bot_id, self.g.top_id):
            if c is not None:
                empty[c] = False
        return {"elem_viol": t.rules_bad / t.rules if t.rules else 0.0,
                "region_viol": t.axioms_bad / t.axioms if t.axioms else 0.0,
                "worst_depth": t.worst, "n_axioms_violated_by_depth": dict(t.by_tol),
                "n_empty_classes": int(empty.sum()), "n_dead_elements": self.n_dead}


register_grounded_part("evaluation")(CrispViolations)


class ExactDomainCheck:
    """Whether a grounded geometry is an exact model of its ontology: an evaluation regime
    over a :class:`CrispViolations` report.

    On top of the crisp violation rates (``elem_viol``, ``region_viol`` and the rate at
    tolerance :math:`10^{-4}`) it reads, crisply:

    - ``exist``: the share of the satisfiable classes whose own element is live and inside
      the class (without a reasoner: of the live classes);
    - ``dead_ok``: the dead classes are exactly the unsatisfiable ones (reasoner only), and
      ``n_false_dead`` the satisfiable classes that are dead;
    - ``faithful_named_precision`` / ``_recall``: for the satisfiable named classes (at most
      ``max_named``), the points of each class (its element, ``RegionSampler`` points and the
      corners of its boxes, those crisp-inside it) decide the inclusion of every ordered
      pair; *precision* is the share of the separated pairs that are not entailed, *recall*
      the share of the non-entailed pairs that are separated;
    - ``exact``: no violation, every class exists and the dead classes are right
      (``exact_tol1e-4`` at the tolerance :math:`10^{-4}`).

    The reasoner is an object with ``is_satisfiable(iri)`` and ``extension(iri)`` (a
    :class:`~mowl.geometric.completion.CanonicalModel`); ``None`` skips what needs it.

    :param chunk: Classes per chunk of the existence check.
    :param max_named: Largest number of named classes of the faithfulness check.
    """

    REGION_FORMS = CONCEPT_FORMS + ROLE_FORMS

    def __init__(self, chunk=8192, max_named=128):
        self.chunk, self.max_named = int(chunk), int(max_named)

    def report(self, geometry, base, axioms, generator, n_eval=10000, bot_id=None,
               reasoner=None, class_index=None, named_index=None, top=None, bottom=None):
        """The metrics dict; ``base`` is the :meth:`CrispViolations.report` of the
        geometry."""
        total = sum(len(axioms[k]) for k in self.REGION_FORMS if k in axioms and len(axioms[k]) > 0)
        out = {"elem_viol": base["elem_viol"], "region_viol": base["region_viol"],
               "region_viol_tol1e-4": (float(base["n_axioms_violated_by_depth"]["1e-4"] / total)
                                       if total > 0 else 0.0),
               "n_empty_classes": int(base["n_empty_classes"]),
               "n_dead_elements": int(base["n_dead_elements"])}
        n = geometry.nb_classes
        with th.no_grad():
            positions = geometry.positions()
            dead = (geometry.dead()[:n] > 0.5).cpu()
            if bot_id is not None:
                dead[bot_id] = True
            inside = th.zeros((n,), dtype=th.bool)
            for start in range(0, n, self.chunk):
                ids = th.arange(start, min(start + self.chunk, n), dtype=th.long,
                                device=positions.device)
                depth = geometry.depth(positions[ids], ids, crisp=True).cpu()
                inside[start:start + len(ids)] = (depth > 0.0) & (~dead[start:start + len(ids)])
            if reasoner is None:
                live = ~dead
                out.update(exist=float(inside[live].to(th.float32).mean().item())
                           if bool(live.any().item()) else 1.0, dead_ok=True,
                           n_false_dead=None, faithful_named_precision=None,
                           faithful_named_recall=None, faithful_named=(None, None))
            else:
                sat = th.zeros((n,), dtype=th.bool)
                for iri, c in class_index.items():
                    sat[c] = False if iri == bottom else bool(reasoner.is_satisfiable(iri))
                out["dead_ok"] = bool(th.equal(dead, ~sat))
                out["n_false_dead"] = int((dead & sat).sum().item())
                out["exist"] = (float(inside[sat].to(dtype=th.float32).mean().item())
                                if bool(sat.any().item()) else 1.0)
                named = [(iri, c) for iri, c in named_index.items()
                         if iri not in (top, bottom) and bool(sat[c].item())][:self.max_named]
                p, r = self._faithfulness(geometry, positions, named, reasoner, generator,
                                          n_eval)
                out.update(faithful_named_precision=p, faithful_named_recall=r,
                           faithful_named=(p, r))
        ok = out["exist"] == 1.0 and out["dead_ok"] and out["elem_viol"] == 0.0
        out["exact"] = bool(ok and out["region_viol"] == 0.0)
        out["exact_tol1e-4"] = bool(ok and out["region_viol_tol1e-4"] == 0.0)
        return out

    @staticmethod
    def _points_of(geometry, positions, c, generator, n_points):
        from mowl.representation.library.grounded_parts import index_tensor
        sampler = RegionSampler(n_points, 2, True, None, doubled_gradient=True)
        idx = index_tensor(c, geometry.nb_classes, "class_ids", positions.device)
        sample = sampler.classes(geometry.class_model, idx, generator, None)
        regions = geometry.class_regions()
        corners = all_box_corners(type(regions)(cen=regions.cen[c:c + 1], w=regions.w[c:c + 1],
                                                active=regions.active[c:c + 1]), generator)
        parts = [positions[c:c + 1]]
        if bool(sample.valid.any().item()):
            parts.append(sample.points[sample.valid])
        if corners.numel() > 0:
            parts.append(corners)
        points = th.cat(parts, dim=0)
        return points[geometry.depth(points, c, crisp=True) > 0.0]

    def _faithfulness(self, geometry, positions, named, reasoner, generator, n_eval):
        if len(named) < 2:
            return 1.0, 1.0
        n_points = max(16, min(256, n_eval // max(len(named), 1)))
        points = {c: self._points_of(geometry, positions, c, generator, n_points)
                  for _, c in named}
        tp = fp = tn = fn = 0
        for a_iri, a in named:
            for b_iri, b in named:
                if a == b:
                    continue
                entailed = bool(reasoner.extension(a_iri).issubset(reasoner.extension(b_iri)))
                pts = points[a]
                included = True if pts.shape[0] == 0 else bool(
                    (geometry.depth(pts, b, crisp=True) > 0.0).all().item())
                if entailed and included:
                    tp += 1
                elif not entailed and included:
                    fp += 1
                elif not entailed:
                    tn += 1
                else:
                    fn += 1
        precision = float(tn / (tn + fn)) if (tn + fn) > 0 else (1.0 if fp == 0 else 0.0)
        recall = float(tn / (tn + fp)) if (tn + fp) > 0 else (1.0 if fn == 0 else 0.0)
        return precision, recall


register_grounded_part("evaluation", "exact domain check")(ExactDomainCheck)
