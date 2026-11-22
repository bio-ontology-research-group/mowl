"""The objective of a grounded method: every axiom form read at sampled elements and at
sampled points of regions.

One step (:meth:`GroundedObjective.losses`) reads every axiom form twice:

- the **element term** of the form: rows of the form (:class:`ElementSampler.rows
  <mowl.representation.library.grounded_sampling.ElementSampler>`), candidate elements of
  each row, and the rule of the form (:class:`~mowl.nn.el.grounded.constraints.ConstraintSet`,
  in the logic of the method) on the degrees of membership the class model and the relation
  model give at those elements, with the deadness of the elements as the exemption;
- the **region term** of the form: points of the regions of the left-hand side
  (:class:`~mowl.representation.library.grounded_sampling.RegionSampler`; a class, an
  intersection of two classes, a region of pairs of a relation, the preimage of an
  existential) and the depth hinge of the right-hand side at them (with a relation model of
  translations, the rule on degrees at the point).

plus the existence of the class elements (``fact``), corrupted role assertions
(``neg``) and the minimality penalty. The table below is what :meth:`GroundedObjective.losses`
assembles; each entry is one short method of the class.

================== ============================== ========================================
form               element term (elements, rule)  region term (points of, hinge)
================== ============================== ========================================
gci0  A ⊑ B        x near A: A(x) -> B(x)         A: inside B (dead A: outside A)
gci0_bot  A ⊑ ⊥    x near A: A(x) -> ⊥            A: outside A
gci1  A ⊓ B ⊑ C    x near A, B: A, B -> C          A ∩ B: inside C
gci1_bot A ⊓ B ⊑ ⊥ x near A, B: A, B -> ⊥          A ∩ B: not in both
gci2  A ⊑ ∃r.B     x near A: A(x) -> r(x, x_B)    A: related to x_B (or in ∃r.B)
gci3  ∃r.B ⊑ C     x near C, fillers y: r, B -> C pairs of r (or ∃r.B): guarded inside C
gci3_bot ∃r.B ⊑ ⊥  x near B, fillers y: r, B -> ⊥ pairs of r (or ∃r.B): apart
role_inclusion     grid: r(x, y) -> s(x, y)       pairs of r: inside s
role_chain         grid: r, s -> t                pairs of r, s: guarded inside t
class_assertion    the individual: -> A(a)        -
role assertion     the individuals: -> r(a, b)    -
================== ============================== ========================================

The objective reads only the parts: ``geometry.shared`` (positions, deadness, temperature),
``geometry.class_model`` (``membership``, ``depth``, ``held_membership``, regions),
``geometry.relation_model`` (``membership``, ``depth``, ``grid_membership``, ``mode``,
``exists_depth``, ``pair_regions``) and ``geometry.measure`` (``soft``). A replaced part is
what the objective reads.
"""
from __future__ import annotations

from typing import Callable, Mapping, NamedTuple, Optional

import torch as th

from mowl.representation.library.grounded_parts import register_grounded_part

__all__ = ["GroundedObjective", "Grounding"]

#: The forms with an element term and a region term, in the order a step reads them.
FORMS = ("gci0", "gci0_bot", "gci1", "gci1_bot", "gci2", "gci3", "gci3_bot",
         "role_inclusion", "role_chain", "class_assertion", "object_property_assertion")


class Grounding(NamedTuple):
    """The finite domain the axioms are read on: the class elements that must exist
    (``live_classes``), the classes a minimality penalty reads (``ordinary_classes``), the
    elements whose deadness it reads (``dead_probes``), the filler elements of each relation
    (``fillers`` ``(R, N)``), and ``nb_classes`` (the first elements are the classes),
    ``nb_individuals``, ``top_id``, ``bot_id``."""

    live_classes: th.Tensor
    ordinary_classes: th.Tensor
    dead_probes: th.Tensor
    fillers: th.Tensor
    nb_classes: int
    nb_individuals: int
    top_id: Optional[int] = None
    bot_id: Optional[int] = None


def domain_index_sets(nb_classes, nb_elements, top_id=None, bot_id=None, device="cpu"):
    """The index sets of a :class:`Grounding` read off the size of the domain:
    ``(live_classes, ordinary_classes, dead_probes)``, the classes other than bottom, the
    classes other than top and bottom, and the elements other than the element of bottom."""
    def without(n, *drop):
        return th.tensor([i for i in range(n) if i not in drop], dtype=th.long, device=device)
    return (without(nb_classes, bot_id), without(nb_classes, top_id, bot_id),
            without(nb_elements, bot_id))


def relation_fillers(axioms, nb_rels, nb_elements, nb_classes, device="cpu"):
    """The filler and left-hand elements of every relation, in order of first occurrence:
    from :math:`A \\sqsubseteq \\exists r.B` (``A`` and ``B``) and the role assertions
    (the elements of the individuals). Returns ``(fillers, lhs, mask)``: dicts of index
    tensors per relation and the boolean ``(nb_rels, nb_elements)`` table of fillers."""
    fillers = {r: {} for r in range(nb_rels)}
    lhs = {r: {} for r in range(nb_rels)}
    for form, offset in (("gci2", 0), ("object_property_assertion", nb_classes)):
        rows = axioms[form] if form in axioms else None
        if rows is None or len(rows) == 0:
            continue
        for a, r, b in rows[:].detach().cpu().tolist():
            fillers[int(r)][offset + int(b)] = None
            lhs[int(r)][offset + int(a)] = None

    def tensors(table):
        return {r: th.tensor(list(v), dtype=th.long, device=device) for r, v in table.items()}
    fillers, lhs = tensors(fillers), tensors(lhs)
    mask = th.zeros((nb_rels, nb_elements), dtype=th.bool, device=device)
    for r, ids in fillers.items():
        mask[r, ids] = True
    return fillers, lhs, mask


class _Step:
    """What one step reads: the parts, the positions and deadness of the elements, the
    samplers and the generator."""

    def __init__(self, objective, geometry, grounding, elements, regions, gen):
        self.o, self.g, self.gen = objective, grounding, gen
        self.C = objective.constraints
        self.classes, self.relations = geometry.class_model, geometry.relation_model
        self.measure, self.shared = geometry.measure, geometry.shared
        self.elements, self.regions = elements, regions
        self.pos = self.shared.positions()
        dead = self.shared.dead()
        if grounding.bot_id is not None:
            dead = dead.clone()
            dead[grounding.bot_id] = 1.0
        self.dead = dead
        threshold = 0.75 if objective.dead_from_bot_only else 0.5
        self.gate = (dead.detach() > threshold).to(dtype=self.pos.dtype)
        self.device = self.pos.device
        self.n = int(self.pos.shape[0])
        self.translation = getattr(self.relations, "mode", None) == "translation"
        self.n_skipped_top = 0

    # -- elements ---------------------------------------------------------------------------
    def candidates(self, lhs, batch):
        """``(B, S)`` candidate elements of rows with left-hand side classes ``lhs``."""
        return self.elements.for_rows(self.classes, self.pos, lhs, batch, self.gen)

    @staticmethod
    def spread(col, s):
        return col.unsqueeze(1).expand(col.shape[0], s).reshape(-1)

    def inside(self, points, ids):
        return self.classes.depth(points, ids, crisp=True) > 0.0

    def depth(self, points, ids):
        return self.classes.depth(points, ids)

    def exists_depth(self, points, rel_ids, class_ids):
        return self.relations.exists_depth(points, rel_ids, class_ids, self.classes)

    def soft_exists(self, points, rel_ids, class_ids):
        return self.measure.soft(self.exists_depth(points, rel_ids, class_ids), self.shared.tau)

    # -- region points ------------------------------------------------------------------------
    def split(self, points):
        """The pair ``(x, y)`` a point of a region of pairs stands for (the relation model's
        reading)."""
        return self.relations.split_pair(points)

    def keep(self, points):
        """Region points as the left-hand side is trained through them (``region_lhs_grad``)."""
        return points if self.o.region_lhs_grad else points.detach()

    def rep(self, col):
        return col.repeat(self.regions.n_samples)

    def hinge(self, fn):
        """The hinge the region sampler ascends (``None`` without ascent)."""
        return fn if self.regions.n_adv > 0 else None

    @staticmethod
    def tile(col, points):
        """The row ids ``col`` for the (sample-major) points of a hinge."""
        return col.repeat(max(points.shape[0] // col.shape[0], 1))

    def without_top(self, rows):
        """Rows whose filler is not top (their region term is not read; counted)."""
        top = self.g.top_id
        if top is None:
            return rows
        is_top = rows[:, 1] == top
        n_top = int(is_top.sum())
        self.n_skipped_top += n_top
        return rows[~is_top] if n_top > 0 else rows

    def margin(self):
        return self.o.region_margin


def _grid(elements, cols, arity):
    """Every ``arity``-tuple of ``elements`` for every row: ``(element columns, row
    columns)``, flat."""
    b, s = int(cols[0].shape[0]), int(elements.numel())
    shape = (b,) + (s,) * arity
    at = [elements.view((1,) + tuple(s if i == j else 1 for j in range(arity))).expand(shape)
          .reshape(-1) for i in range(arity)]
    ids = [c.view((b,) + (1,) * arity).expand(shape).reshape(-1) for c in cols]
    return at, ids


class GroundedObjective:
    """The objective of a grounded method (module docstring).

    :param constraints: The constraints of the forms in a logic
        (:class:`~mowl.nn.el.grounded.constraints.ConstraintSet`; default Łukasiewicz).
    :param lambda_reg: Weight of the region terms.
    :param region_margin: Margin of the depth hinges.
    :param region_lhs_grad: Whether region terms train the left-hand side through the
        points (else the points are held).
    :param dead_from_bot_only: The deadness reading of GroundedEL (an element is silenced
        only when it is clearly dead and the box terms pull to full containment); ``False``
        is the earlier reading (ablation).
    :param neg_weight: Weight of the corrupted role assertions.
    :param neg_k: Corruptions per assertion.
    :param eps_min, eps_role, eps_dead: The minimality penalty on memberships, relations and
        deadness.
    :param agg, agg_tau: How a region term of an existential takes the best of several
        candidate fillers (default: what the measure of the geometry does).

    Part (vocabulary value): ``grounded constraints``. Used by: ``grounded``.
    """

    part = "grounded constraints"
    forms = FORMS

    def __init__(self, constraints=None, lambda_reg=1.0, region_margin=0.0,
                 region_lhs_grad=True, dead_from_bot_only=True, neg_weight=0.0, neg_k=1,
                 eps_min=1e-2, eps_role=1e-2, eps_dead=1e-3, agg=None, agg_tau=None):
        if constraints is None:
            from mowl.nn.el.grounded.constraints import ConstraintSet
            constraints = ConstraintSet()
        self.constraints = constraints
        self.lambda_reg, self.region_margin = float(lambda_reg), float(region_margin)
        self.region_lhs_grad, self.dead_from_bot_only = bool(region_lhs_grad), bool(
            dead_from_bot_only)
        self.neg_weight, self.neg_k = float(neg_weight), int(neg_k)
        self.eps_min, self.eps_role, self.eps_dead = eps_min, eps_role, eps_dead
        self.agg, self.agg_tau = agg, agg_tau

    # =========================================================================================
    # the assembly
    # =========================================================================================
    def losses(self, geometry, axioms: Mapping, grounding: Grounding, generator, elements,
               regions, weights=None, negatives: Optional[Callable] = None):
        """One step: ``{"fact", "axioms", "region", "minimality", "neg", "total",
        "w_fact", "w_axioms", "w_region", "n_region_skipped_top"}``.

        :param axioms: Form name -> table (a dataset or tensor of rows).
        :param elements, regions: The samplers (:class:`ElementSampler`, :class:`RegionSampler`).
        :param weights: ``{"fact", "axioms", "region"}`` multipliers (dual weights), or
            ``None`` (all 1).
        :param negatives: ``(table, generator) -> rows (a, r, b)`` of corrupted role
            assertions (default: :meth:`ElementSampler.corrupted`).
        """
        s = _Step(self, geometry, grounding, elements, regions, generator)
        zero = s.pos.new_zeros(())
        fact = self.fact(s)
        axiom_terms, region_terms = [], []
        do_region = self.lambda_reg > 0.0 and regions.n_samples > 0
        for form in FORMS:
            table = axioms.get(form)
            if table is None or len(table) == 0:
                continue
            rows = elements.rows(table[:], generator, s.device)
            axiom_terms.append(getattr(self, form)(s, rows))
            region = getattr(self, form + "_region", None)
            if do_region and region is not None:
                region_terms += region(s, rows)
        neg = self.negative(s, axioms.get("object_property_assertion"), negatives)
        loss_axioms = th.stack(axiom_terms).sum() if axiom_terms else zero
        loss_region = th.stack(region_terms).sum() if region_terms else zero
        loss_min = self.minimality(s)
        w = {"fact": 1.0, "axioms": 1.0, "region": 1.0} if weights is None else {
            k: float(weights[k]) for k in ("fact", "axioms", "region")}
        total = (fact if weights is None else w["fact"] * fact) + (
            loss_axioms if weights is None else w["axioms"] * loss_axioms) + (
            self.lambda_reg * loss_region if weights is None
            else w["region"] * self.lambda_reg * loss_region) + loss_min
        if self.neg_weight > 0.0:
            total = total + self.neg_weight * neg
        out = {"fact": fact, "axioms": loss_axioms, "region": loss_region,
               "minimality": loss_min, "neg": neg, "total": total}
        out.update({f"w_{k}": th.tensor(v, dtype=s.pos.dtype, device=s.device)
                    for k, v in w.items()})
        out["n_region_skipped_top"] = s.n_skipped_top
        return out

    # =========================================================================================
    # existence, negatives, minimality
    # =========================================================================================
    def fact(self, s):
        """Every class element is in its class (exempt when dead)."""
        ids = s.g.live_classes
        if ids.numel() == 0:
            return s.pos.new_zeros(())
        if ids.numel() > s.elements.batch_size:
            ids = s.elements.rows(ids.unsqueeze(1), s.gen, s.device).squeeze(1)
        return s.C.fact(s.classes.membership(s.pos[ids], ids), dead=s.gate[ids]).mean()

    def negative(self, s, table, negatives=None):
        """Corrupted role assertions are not related (mean degree)."""
        if self.neg_weight <= 0.0 or s.g.nb_individuals <= 0 or table is None or not len(table):
            return s.pos.new_zeros(())
        rows = negatives(table, s.gen) if negatives is not None else s.elements.corrupted(
            table[:].to(s.device), s.g.nb_individuals, s.gen, self.neg_k)
        rows = rows.to(device=s.device, dtype=th.long)
        a, b = s.g.nb_classes + rows[:, 0], s.g.nb_classes + rows[:, 2]
        return s.relations.membership(s.pos[a], s.pos[b], rows[:, 1]).mean()

    def minimality(self, s):
        """Few memberships, few related pairs, few dead elements."""
        m_rand = rho_rand = None
        ordinary = s.g.ordinary_classes
        if ordinary.numel() > 0 and self.eps_min > 0.0:
            x, c = s.elements.class_probes(s.n, ordinary, s.gen)
            m_rand = s.classes.membership(s.pos[x], c)
        n_rel = len(s.relations) if s.relations is not None else 0
        if n_rel > 0 and self.eps_role > 0.0:
            x, y, r = s.elements.relation_probes(s.n, n_rel, s.gen, s.device)
            rho_rand = s.relations.membership(s.pos[x], s.pos[y], r)
        probes = s.g.dead_probes
        dead = s.dead[probes] if probes.numel() > 0 else None
        return s.C.minimality(m_rand=m_rand, rho_rand=rho_rand, dead=dead, eps=self.eps_min,
                              eps_r=self.eps_role, eps_dead=self.eps_dead)

    # =========================================================================================
    # concept inclusions
    # =========================================================================================
    def gci0(self, s, rows):
        """A ⊑ B at candidates x of A."""
        x = s.candidates(rows[:, 0], rows).reshape(-1)
        a, b = s.spread(rows[:, 0], x.numel() // rows.shape[0]), s.spread(
            rows[:, 1], x.numel() // rows.shape[0])
        px, gate = s.pos[x], s.gate[x]
        ma, mb = s.classes.membership(px, a), s.classes.membership(px, b)
        own = x == a
        if self.dead_from_bot_only:
            # the box: x_A is in A, and x crisply in B counts as in B; x_A pulled into B
            box = s.C.gci0(th.where(own, th.ones_like(ma), ma),
                           th.where(s.inside(px, b), th.ones_like(mb), mb), dead=gate)
            xa = rows[:, 0]
            pull = (1.0 - s.gate[xa]) * s.C.gci0_region(s.depth(s.pos[xa], rows[:, 1]),
                                                        margin=s.margin())
            return box.mean() + pull.mean()
        own_dead = own & (gate > 0.5)
        broken = own & (s.inside(px, a) | own_dead) & (
            s.classes.depth(px, b, crisp=True) <= 0.0)
        dead_term = s.C.gci0(th.where(own_dead, th.ones_like(ma), ma.detach()), mb.detach(),
                             dead=th.where(broken, s.dead[x], th.ones_like(s.dead[x])))
        return (s.C.gci0(ma, mb, dead=gate) + dead_term).mean()

    def gci0_region(self, s, rows):
        a, b = rows[:, 0], rows[:, 1]
        hinge = s.hinge(lambda z: s.C.gci0_region(s.depth(z, s.tile(b, z)),
                                                  margin=s.margin()))
        sample = s.regions.classes(s.classes, a, s.gen, hinge)
        if not bool(sample.valid.any()):
            return []
        raw = sample.points[sample.valid]
        av, bv = s.rep(a)[sample.valid], s.rep(b)[sample.valid]
        term = (1.0 - s.gate[av]) * s.C.gci0_region(s.depth(s.keep(raw), bv),
                                                    margin=s.margin()) \
            + s.gate[av] * s.C.gci0_bot_region(s.depth(raw.detach(), av), margin=s.margin())
        return [term.mean()]

    def gci0_bot(self, s, rows):
        """A ⊑ ⊥ at candidates x of A."""
        a_col = rows[:, 0]
        x = s.candidates(a_col, a_col).reshape(-1)
        a = s.spread(a_col, x.numel() // a_col.shape[0])
        px = s.pos[x]
        ma = s.classes.membership(px, a)
        own = x == a
        ones = th.ones_like(s.dead[x])
        if self.dead_from_bot_only:
            crisp = own | s.inside(px, a) | (s.dead[x].detach() > 0.5)
            box = s.C.gci0_bot(ma, dead=th.where(own, s.gate[x], ones))
            dead = s.C.gci0_bot(th.where(crisp, th.ones_like(ma), ma.detach()),
                                dead=th.where(crisp, s.dead[x], ones))
        else:
            box = s.C.gci0_bot(ma, dead=s.gate[x])
            dead = s.C.gci0_bot(th.where(own, th.ones_like(ma), ma.detach()), dead=s.dead[x])
        return (box + dead).mean()

    def gci0_bot_region(self, s, rows):
        a = rows[:, 0]
        hinge = s.hinge(lambda z: s.C.gci0_bot_region(s.depth(z, s.tile(a, z)),
                                                      margin=s.margin()))
        sample = s.regions.classes(s.classes, a, s.gen, hinge)
        if not bool(sample.valid.any()):
            return []
        pts, av = sample.points[sample.valid].detach(), s.rep(a)[sample.valid]
        return [s.C.gci0_bot_region(s.depth(pts, av), margin=s.margin()).mean()]

    def gci1(self, s, rows):
        """A ⊓ B ⊑ C at candidates x of A and B."""
        x = s.candidates(rows[:, :2], rows).reshape(-1)
        k = x.numel() // rows.shape[0]
        px = s.pos[x]
        ma, mb, mc = (s.classes.membership(px, s.spread(rows[:, j], k)) for j in range(3))
        return s.C.gci1(ma, mb, mc, dead=s.gate[x]).mean()

    def gci1_region(self, s, rows):
        a, b, c = rows[:, 0], rows[:, 1], rows[:, 2]
        hinge = s.hinge(lambda z: s.C.gci1_region(s.depth(z, s.tile(c, z)), margin=s.margin()))
        sample = s.regions.conjunction(s.classes, a, b, s.gen, hinge)
        if not bool(sample.valid.any()):
            return []
        v = sample.valid
        av, bv, cv = s.rep(a)[v], s.rep(b)[v], s.rep(c)[v]
        live = (1.0 - s.gate[av]) * (1.0 - s.gate[bv])
        return [(live * s.C.gci1_region(s.depth(s.keep(sample.points[v]), cv),
                                        margin=s.margin())).mean()]

    def gci1_bot(self, s, rows):
        """A ⊓ B ⊑ ⊥ at candidates x of A and B."""
        x = s.candidates(rows[:, :2], rows[:, :2]).reshape(-1)
        k = x.numel() // rows.shape[0]
        a, b = s.spread(rows[:, 0], k), s.spread(rows[:, 1], k)
        px, gate = s.pos[x], s.gate[x]
        ma, mb = s.classes.membership(px, a), s.classes.membership(px, b)
        both = s.inside(px, a) & s.inside(px, b)
        dead_eff = th.where(both, s.dead[x], th.ones_like(s.dead[x]))
        if not self.dead_from_bot_only:
            return (s.C.gci1_bot(ma, mb, dead=gate)
                    + s.C.gci1_bot(ma.detach(), mb.detach(), dead=dead_eff)).mean()
        # boxes move only for the class elements; other elements are pushed out of one box
        own = (x == a) | (x == b)
        dep_a, ma_held = s.classes.held_membership(px, a)
        dep_b, mb_held = s.classes.held_membership(px, b)
        push = 0.5 * (1.0 - gate) * s.C.hinges.outside(th.minimum(dep_a, dep_b))
        box = s.C.gci1_bot(th.where(own, ma, ma_held), th.where(own, mb, mb_held),
                           dead=gate) + push
        xa, xb = rows[:, 0], rows[:, 1]
        apart = 0.5 * ((1.0 - s.gate[xa]) * s.C.hinges.outside(s.depth(s.pos[xa], xb))
                       + (1.0 - s.gate[xb]) * s.C.hinges.outside(s.depth(s.pos[xb], xa)))
        dead = s.C.gci1_bot(th.ones_like(ma), th.ones_like(mb), dead=dead_eff)
        return (box + dead).mean() + apart.mean()

    def gci1_bot_region(self, s, rows):
        a, b = rows[:, 0], rows[:, 1]
        hinge = s.hinge(lambda z: s.C.gci1_bot_region(
            s.depth(z, s.tile(a, z)), s.depth(z, s.tile(b, z)), margin=s.margin()))
        sample = s.regions.conjunction(s.classes, a, b, s.gen, hinge)
        if not bool(sample.valid.any()):
            return []
        pts, v = sample.points[sample.valid].detach(), sample.valid
        return [s.C.gci1_bot_region(s.depth(pts, s.rep(a)[v]), s.depth(pts, s.rep(b)[v]),
                                    margin=s.margin()).mean()]

    # =========================================================================================
    # existentials
    # =========================================================================================
    def gci2(self, s, rows):
        """A ⊑ ∃r.B at candidates x of A: related to the filler element x_B; a dead filler
        makes A empty."""
        x = s.candidates(rows[:, 0], rows[:, [0, 2]]).reshape(-1)
        k = x.numel() // rows.shape[0]
        a, r, b = (s.spread(rows[:, j], k) for j in range(3))
        px, gate = s.pos[x], s.gate[x]
        ma = s.classes.membership(px, a)
        rho = s.relations.membership(px, s.pos[b], r)
        dead_b = s.dead[b].detach()
        if self.dead_from_bot_only:
            sure = (x == a) | s.inside(px, a) | (s.dead[x].detach() > 0.5)
        else:
            sure = (x == a) & (gate > 0.5)
        ma_sure = th.where(sure, th.ones_like(ma), ma.detach())
        edge = s.C.gci2(ma, rho, dead=gate, dead_filler=None)
        dead = s.C.gci2_dead(ma, dead_filler=dead_b, dead=gate) + th.where(
            dead_b > 0.5, s.C.gci2_dead(ma_sure, dead_filler=dead_b, dead=s.dead[x]),
            th.zeros_like(ma))
        return (edge + dead).mean()

    def gci2_region(self, s, rows):
        a, r, b = rows[:, 0], rows[:, 1], rows[:, 2]
        if s.translation:
            hinge = s.hinge(lambda z: s.C.gci2_region_trans(
                s.classes.membership(z, s.tile(a, z)),
                s.soft_exists(z, s.tile(r, z), s.tile(b, z))))
        else:
            hinge = s.hinge(lambda z: s.C.gci2_region(
                s.relations.depth(z, s.pos[s.tile(b, z)].detach(), s.tile(r, z)),
                margin=s.margin(), agg=self._agg(s), agg_tau=self._agg_tau(s)))
        sample = s.regions.classes(s.classes, a, s.gen, hinge)
        if not bool(sample.valid.any()):
            return []
        v = sample.valid
        raw, av, rv, bv = sample.points[v], s.rep(a)[v], s.rep(r)[v], s.rep(b)[v]
        pts, held_a = s.keep(raw), s.depth(raw.detach(), av)
        if s.translation:
            term = s.C.gci2_region_trans(s.classes.membership(pts, av),
                                         s.soft_exists(pts, rv, bv) * (1.0 - s.gate[bv]))
        else:
            term = s.C.gci2_region(s.relations.depth(pts, s.pos[bv], rv), margin=s.margin(),
                                   agg=self._agg(s), agg_tau=self._agg_tau(s), depth_a=held_a,
                                   dead_filler=s.dead[bv].detach())
        term = (1.0 - s.gate[av]) * term + s.gate[av] * s.C.gci0_bot_region(
            held_a, margin=s.margin())
        return [term.mean()]

    def _existential_grid(self, s, rows, x, need_crisp):
        """Degrees of the relation on the grid of candidates ``x`` and fillers of the rows."""
        y = s.elements.fillers(rows, s.g.fillers, s.gen)
        rho, crisp = s.relations.grid_membership(s.pos[x], s.pos[y], rows[:, 0],
                                                 need_crisp=need_crisp)
        return y, rho, crisp

    @staticmethod
    def _over_grid(values, bsz, sx, sy, axis):
        """Values per candidate (``axis=1``) or per filler (``axis=2``) on the grid."""
        shape = (bsz, sx, 1) if axis == 1 else (bsz, 1, sy)
        return values.view(shape).expand(bsz, sx, sy).reshape(-1)

    def gci3(self, s, rows):
        """∃r.B ⊑ C at candidates x of C and fillers y: r(x, y), B(y) -> C(x)."""
        x = s.candidates(rows[:, 2], rows[:, 1:])
        y, rho, _ = self._existential_grid(s, rows, x, False)
        bsz, sx, sy = rows.shape[0], x.shape[1], y.shape[1]
        mb = s.classes.membership(s.pos[y.reshape(-1)], s.spread(rows[:, 1], sy))
        mc = s.classes.membership(s.pos[x.reshape(-1)], s.spread(rows[:, 2], sx))
        return s.C.gci3(rho, self._over_grid(mb, bsz, sx, sy, 2),
                        self._over_grid(mc, bsz, sx, sy, 1),
                        dead=self._over_grid(s.gate[x], bsz, sx, sy, 1)).mean()

    def gci3_region(self, s, rows):
        rows = s.without_top(rows)
        if rows.shape[0] == 0:
            return []
        r, b, c = rows[:, 0], rows[:, 1], rows[:, 2]
        if s.translation:
            hinge = s.hinge(lambda z: s.C.gci3_region_trans(
                s.soft_exists(z, s.tile(r, z), s.tile(b, z)),
                s.classes.membership(z, s.tile(c, z))))
            sample = s.regions.exists(s.relations, s.classes, r, b, s.gen, hinge)
            if not bool(sample.valid.any()):
                return []
            v = sample.valid
            pts, rv, bv, cv = s.keep(sample.points[v]), s.rep(r)[v], s.rep(b)[v], s.rep(c)[v]
            m_exists = s.soft_exists(pts, rv, bv) * (1.0 - s.gate[bv])
            return [s.C.gci3_region_trans(m_exists, s.classes.membership(pts, cv)).mean()]

        def hinge_fn(xy):
            hx, hy = s.split(xy)
            return s.C.gci3_region(s.relations.depth(hx, hy, s.tile(r, xy)),
                                   s.depth(hy, s.tile(b, xy)), s.depth(hx, s.tile(c, xy)),
                                   margin=s.margin(), smooth=True)
        sample = s.regions.pairs(s.relations, r, s.gen, s.hinge(hinge_fn))
        if not bool(sample.valid.any()):
            return []
        v = sample.valid
        hx, hy = s.split(sample.points[v])
        rv, bv, cv = s.rep(r)[v], s.rep(b)[v], s.rep(c)[v]
        filler = s.pos[bv]
        at_pair = s.C.gci3_region(s.relations.depth(hx.detach(), hy.detach(), rv),
                                  s.depth(s.keep(hy), bv), s.depth(s.keep(hx), cv),
                                  margin=s.margin())
        at_filler = s.C.gci3_region(s.relations.depth(hx.detach(), filler, rv),
                                    s.depth(filler, bv), s.depth(s.keep(hx), cv),
                                    margin=s.margin())
        return [at_pair.mean(), at_filler.mean()]

    def gci3_bot(self, s, rows):
        """∃r.B ⊑ ⊥ at candidates x of B and fillers y: not r(x, y) with B(y)."""
        x = s.candidates(rows[:, 1], rows[:, 1])
        y, rho, r_crisp = self._existential_grid(s, rows, x, True)
        bsz, sx, sy = rows.shape[0], x.shape[1], y.shape[1]
        py, b = s.pos[y.reshape(-1)], s.spread(rows[:, 1], sy)
        mb = self._over_grid(s.classes.membership(py, b), bsz, sx, sy, 2)
        both = (r_crisp > 0.0) & self._over_grid(s.inside(py, b), bsz, sx, sy, 2)
        xg = self._over_grid(x, bsz, sx, sy, 1)
        dead_eff = th.where(both, s.dead[xg], th.ones_like(s.dead[xg]))
        box = s.C.gci3_bot(rho, mb, dead=s.gate[xg])
        if self.dead_from_bot_only:
            dead = s.C.gci3_bot(th.ones_like(rho), th.ones_like(mb), dead=dead_eff)
        else:
            dead = s.C.gci3_bot(rho.detach(), mb.detach(), dead=dead_eff)
        return (box + dead).mean()

    def gci3_bot_region(self, s, rows):
        rows = s.without_top(rows)
        if rows.shape[0] == 0:
            return []
        r, b = rows[:, 0], rows[:, 1]
        if s.translation:
            hinge = s.hinge(lambda z: s.C.gci3_bot_region_trans(
                s.soft_exists(z, s.tile(r, z), s.tile(b, z))))
            sample = s.regions.exists(s.relations, s.classes, r, b, s.gen, hinge)
            if not bool(sample.valid.any()):
                return []
            v = sample.valid
            pts, rv, bv = s.keep(sample.points[v]), s.rep(r)[v], s.rep(b)[v]
            return [s.C.gci3_bot_region_trans(
                s.soft_exists(pts, rv, bv) * (1.0 - s.gate[bv])).mean()]

        def hinge_fn(xy):
            hx, hy = s.split(xy)
            return s.C.gci3_bot_region(s.relations.depth(hx, hy, s.tile(r, xy)),
                                       s.depth(hy, s.tile(b, xy)), margin=s.margin())
        sample = s.regions.pairs(s.relations, r, s.gen, s.hinge(hinge_fn))
        if not bool(sample.valid.any()):
            return []
        v = sample.valid
        hx, hy = s.split(sample.points[v])
        return [s.C.gci3_bot_region(s.relations.depth(hx.detach(), hy.detach(), s.rep(r)[v]),
                                    s.depth(s.keep(hy), s.rep(b)[v]),
                                    margin=s.margin()).mean()]

    # =========================================================================================
    # role axioms
    # =========================================================================================
    def role_inclusion(self, s, rows):
        """r ⊑ s on a grid of elements."""
        el = s.elements.grid(s.n, 16, s.gen, s.device)
        (x, y), (r, t) = _grid(el, [rows[:, 0], rows[:, 1]], 2)
        px, py = s.pos[x], s.pos[y]
        return s.C.role_inclusion(s.relations.membership(px, py, r),
                                  s.relations.membership(px, py, t)).mean()

    def role_inclusion_region(self, s, rows):
        r, t = rows[:, 0], rows[:, 1]
        if s.translation:
            def hinge_fn(p):
                hx, hy = s.split(p)
                return s.C.role_inclusion_region_trans(
                    s.relations.membership(hx, hy, s.tile(r, p)),
                    s.relations.membership(hx, hy, s.tile(t, p)))
        else:
            def hinge_fn(p):
                hx, hy = s.split(p)
                return s.C.role_inclusion_region(s.relations.depth(hx, hy, s.tile(t, p)),
                                                 margin=s.margin())
        sample = s.regions.pairs(s.relations, r, s.gen, s.hinge(hinge_fn))
        if not bool(sample.valid.any()):
            return []
        v = sample.valid
        hx, hy = s.split(s.keep(sample.points[v]))
        if s.translation:
            return [s.C.role_inclusion_region_trans(
                s.relations.membership(hx, hy, s.rep(r)[v]),
                s.relations.membership(hx, hy, s.rep(t)[v])).mean()]
        return [s.C.role_inclusion_region(s.relations.depth(hx, hy, s.rep(t)[v]),
                                          margin=s.margin()).mean()]

    def role_chain(self, s, rows):
        """r ∘ s ⊑ t on a grid of at most 10 elements."""
        el = s.elements.grid(s.n, 12, s.gen, s.device)[:10]
        (x, y, z), (r, t1, t2) = _grid(el, [rows[:, 0], rows[:, 1], rows[:, 2]], 3)
        m = s.relations.membership
        px, py, pz = s.pos[x], s.pos[y], s.pos[z]
        return s.C.role_chain(m(px, py, r), m(py, pz, t1), m(px, pz, t2)).mean()

    def role_chain_region(self, s, rows):
        r, t1, t2 = rows[:, 0], rows[:, 1], rows[:, 2]
        first = s.regions.pairs(s.relations, r, s.gen, n_adv=0)
        second = s.regions.pairs(s.relations, t1, s.gen, n_adv=0)
        v = first.valid & second.valid
        if not bool(v.any()):
            return []
        rv, sv, tv = s.rep(r)[v], s.rep(t1)[v], s.rep(t2)[v]
        if s.translation:
            # pairs (0, v1) of r and (0, v2) of s compose to (0, v1 + v2)
            x1, v1 = s.split(s.keep(first.points[v]))
            _, v2 = s.split(s.keep(second.points[v]))
            m = s.relations.membership
            return [s.C.role_chain_region_trans(m(x1, v1, rv), m(x1, v2, sv),
                                                m(x1, v1 + v2, tv)).mean()]
        hx, hy = s.split(first.points[v])
        _, tz = s.split(second.points[v])
        d = s.relations.depth
        return [s.C.role_chain_region(d(hx.detach(), hy.detach(), rv),
                                      d(s.keep(hy), tz.detach(), sv),
                                      d(s.keep(hx), s.keep(tz), tv),
                                      margin=s.margin()).mean()]

    # =========================================================================================
    # assertions
    # =========================================================================================
    def class_assertion(self, s, rows):
        """A(a): the individual's element is in A."""
        a = s.g.nb_classes + rows[:, 0]
        return s.C.class_assertion(s.classes.membership(s.pos[a], rows[:, 1])).mean()

    def object_property_assertion(self, s, rows):
        """r(a, b): the individuals' elements are related."""
        a, b = s.g.nb_classes + rows[:, 0], s.g.nb_classes + rows[:, 2]
        return s.C.role_assertion(s.relations.membership(s.pos[a], s.pos[b], rows[:, 1])).mean()

    # -- options read from the measure by default ------------------------------------------
    def _agg(self, s):
        return self.agg if self.agg is not None else s.measure.agg

    def _agg_tau(self, s):
        return self.agg_tau if self.agg_tau is not None else s.measure.agg_tau


register_grounded_part("objective")(GroundedObjective)
