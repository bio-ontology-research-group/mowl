"""Answering logical queries over a finite domain: three evaluators for any region family.

A batch of queries of one structure (the nested-tuple format of
:mod:`mowl.evaluation.query_answering`: ``("e", ("r", "r"))`` with instances
``(a, (r1, r2))``, ``(q1, q2)`` an intersection, ``(q1, q2, ("u",))`` a union) is read into a
tree (:func:`read_batch`) whose leaves are anchored chains (anchors and, per hop, the
relation of every query) and whose inner nodes are chains over a sub-query, intersections
and unions. Two relation ids may be special, as in the query graphs of
:mod:`mowl.projection.query_graph`: ``isa`` (an element to the classes it belongs to) and
``isa_inv`` (a class to its members). :class:`HopPlan` says, for one hop of a batch, which
rows follow which kind of edge.

The three evaluators read the domain through a *family* (an adapter of a geometry; for
points and unions of boxes:
:class:`~mowl.representation.library.element_box_family.ElementBoxFamily`):

:class:`RegionCompiler`
    compiles every query to a region (``family.anchor``, ``family.project``,
    ``family.project_exact``, ``family.isa_region``, ``family.class_region``,
    ``family.intersect``, ``family.union``, ``family.select_rows``, ``family.cap``,
    ``family.placeholder``).
:class:`MaxMinBeam`
    scores every element: an anchor scores :data:`TOP` and everything else :data:`EMPTY`;
    a hop is the max-min composition (:func:`max_min`) of the ``k`` best elements
    (:func:`top_witnesses`) with the degree of an edge (``family.role_degree``,
    ``family.class_degree``, ``family.member_degree``); intersection is the minimum and
    union the maximum.
:class:`WitnessChain`
    the differentiable score of given targets: the witnesses of every hop are chosen by
    the beam without gradient, the degrees along the chosen witnesses are soft
    (``family.*_degree(..., tau=...)``, a stable log-sum-exp over the parts of a region).

A family of another kind (sets of a graph, cones, any region with a depth) plugs into the
same evaluators.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
import torch as th

__all__ = ["EMPTY", "TOP", "Anchored", "Chain", "Join", "read_batch", "HopPlan", "Anchors",
           "read_anchors", "max_min", "top_witnesses", "sparsify", "compose_all",
           "RegionCompiler", "MaxMinBeam", "WitnessChain", "as_target_table"]

#: Score of an element that is not an answer (the depth in an empty region).
EMPTY: float = -1e6
#: Score of an anchor (the depth in the region of everything).
TOP: float = 1e6


# -- the tree of a batch ------------------------------------------------------------------
class Anchored(NamedTuple):
    """A chain from anchor entities: ``entities[b]`` and, per hop, the relations."""
    entities: tuple
    hops: tuple


class Chain(NamedTuple):
    """Hops (relations per query) from the answers of the sub-query ``child``."""
    child: Any
    hops: tuple


class Join(NamedTuple):
    """``"and"`` (intersection) or ``"or"`` (union) of the children, left to right."""
    kind: str
    children: tuple


def _relation_columns(rels, n_hops, structure):
    hops = tuple(zip(*rels)) if n_hops > 0 else ()
    if len(hops) != n_hops:
        raise ValueError(f"Query relation tuples do not match structure {structure!r}")
    return hops


def _only_relations(spec):
    return isinstance(spec, tuple) and len(spec) > 0 and all(x == "r" for x in spec)


def read_batch(structure, queries):
    """The tree (:class:`Anchored`, :class:`Chain`, :class:`Join`) of a batch of queries of
    one ``structure``."""
    if structure == "e":
        return Anchored(tuple(queries), ())
    if isinstance(structure, tuple) and len(structure) >= 1 and structure[0] == "e":
        if len(structure) == 1:
            return Anchored(tuple(zip(*queries))[0], ())
        if len(structure) == 2 and isinstance(structure[1], tuple) and \
                all(x == "r" for x in structure[1]):
            columns = tuple(zip(*queries))
            if len(columns) != 2:
                raise ValueError(f"Queries do not match anchor structure {structure!r}")
            return Anchored(columns[0],
                            _relation_columns(columns[1], len(structure[1]), structure))
        raise ValueError(f"Unsupported anchor query structure: {structure!r}")
    if not isinstance(structure, tuple) or len(structure) < 2:
        raise ValueError(f"Unsupported query structure: {structure!r}")
    columns = tuple(zip(*queries))
    if len(structure) == 2 and _only_relations(structure[1]):
        if len(columns) != 2:
            raise ValueError(f"Queries do not match compound structure {structure!r}")
        return Chain(read_batch(structure[0], columns[0]),
                     _relation_columns(columns[1], len(structure[1]), structure))
    if structure[-1] == ("u",):
        if len(structure) < 3:
            raise ValueError("Union structure must have at least 2 disjuncts before ('u',), "
                             f"got {structure!r}")
        parts = structure[:-1]
        if len(columns) < len(parts):
            raise ValueError(f"Queries do not match union structure {structure!r}")
        return Join("or", tuple(read_batch(s, c) for s, c in zip(parts, columns)))
    if len(columns) != len(structure):
        raise ValueError(f"Queries do not match intersection structure {structure!r}")
    return Join("and", tuple(read_batch(s, c) for s, c in zip(structure, columns)))


# -- the anchors of a chain ---------------------------------------------------------------
class Anchors(NamedTuple):
    """What a chain starts from: the elements of the anchors (``elements``, safe to index),
    the class of every anchor (``classes``, ``-1``: none; ``None`` when not needed), whether
    all anchors are classes, and whether the first hop goes from the classes to their
    members only (then the elements are not read)."""
    elements: th.Tensor
    classes: Optional[th.Tensor]
    all_classes: bool
    members_only: bool


def _ids(values, device):
    return th.as_tensor(np.asarray(values, dtype=np.int64), device=device)


def _within(table, ids, name):
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= table.shape[0]):
        raise ValueError(f"Anchor entity id out of range [0, {table.shape[0]}) for {name}")
    return table[ids]


def read_anchors(entities, first_hop, n_classes, device, isa_inv=None, entity_to_element=None,
                 class_of_entity=None):
    """:class:`Anchors` of an anchored chain whose first hop has the relations
    ``first_hop`` (``None``: no hop). Entities are elements unless ``entity_to_element``
    maps them; their classes are ``class_of_entity`` if given, else (with ``isa_inv``) the
    elements that are classes."""
    ent = _ids(entities, device)
    classes, all_classes = None, False
    if class_of_entity is not None:
        classes = _within(class_of_entity, ent, "class_of_entity")
        all_classes = bool((classes >= 0).all())
    elif isa_inv is not None:
        mapped = ent if entity_to_element is None else _within(entity_to_element, ent,
                                                               "entity_to_element")
        classes = th.where((mapped >= 0) & (mapped < n_classes), mapped,
                           th.full_like(mapped, -1))
        all_classes = entity_to_element is None and bool(((ent >= 0) & (ent < n_classes)).all())
    elements = ent if entity_to_element is None else _within(entity_to_element, ent,
                                                             "entity_to_element")
    hop = set(first_hop) if first_hop is not None else set()
    members_only = first_hop is not None and isa_inv is not None and hop == {isa_inv} \
        and all_classes
    if not members_only and first_hop is not None and isa_inv is not None and \
            isa_inv in hop and classes is not None:
        replaced = (_ids(first_hop, device) == isa_inv) & (classes >= 0)
        elements = th.where(replaced & (elements < 0), th.zeros_like(elements), elements)
    return Anchors(elements, classes, all_classes, members_only)


# -- one hop ------------------------------------------------------------------------------
class HopPlan:
    """Which rows of a batch follow which kind of edge at one hop.

    ``kinds`` lists the kinds present, in the order they are computed and merged
    (``"role"``, ``"isa"``, ``"isa_inv"`` after the first hop, ``"members"``: ``isa_inv``
    at the first hop from anchors that are classes); ``only`` is the single kind when one
    computation answers every row. ``rows(kind)`` are the rows a kind decides (the role
    rows are what is left); ``role_ids`` are the relations with ``isa``/``isa_inv`` read as
    relation ``0`` (their rows are overwritten)."""

    def __init__(self, relations, device, isa=None, isa_inv=None, first=False, anchors=None):
        self.r = _ids(relations, device)
        rel = set(relations)
        classes = anchors.classes if anchors is not None else None
        all_classes = anchors.all_classes if anchors is not None else False
        members = first and isa_inv is not None and classes is not None and isa_inv in rel
        has_isa = isa is not None and isa in rel
        inv_later = not first and isa_inv is not None and isa_inv in rel
        special = {x for x, on in ((isa_inv, members and all_classes), (isa_inv, inv_later),
                                   (isa, has_isa)) if on}
        present = {"role": len(rel - special) > 0, "isa": has_isa, "isa_inv": inv_later,
                   "members": members}
        self.kinds = [k for k in ("role", "isa", "isa_inv", "members") if present[k]]
        self.only = None
        if members and rel == {isa_inv} and all_classes:
            self.only = "members"
        elif has_isa and not members and rel == {isa}:
            self.only = "isa"
        elif inv_later and rel == {isa_inv}:
            self.only = "isa_inv"
        elif not (members or has_isa or inv_later):
            self.only = "role"
        self.isa, self.isa_inv, self.classes = isa, isa_inv, classes
        ids = self.r
        for special_id in (isa_inv, isa):
            if special_id is not None:
                ids = th.where(ids == special_id, th.zeros_like(ids), ids)
        self.role_ids = ids

    def rows(self, kind):
        if kind == "isa":
            return self.r == self.isa
        if kind == "isa_inv":
            return self.r == self.isa_inv
        if kind == "members":
            return (self.r == self.isa_inv) & (self.classes >= 0)
        raise ValueError(kind)

    def member_classes(self):
        return th.clamp_min(self.classes, 0)

    def combine(self, compute, merge, start_members):
        """The result of every row: ``compute(kind)`` for each kind present, merged in order
        (``merge(rows, new, current)``); ``start_members(new, rows)`` when the members are
        the first kind present."""
        if self.only is not None:
            return compute(self.only)
        current = None
        for kind in self.kinds:
            new = compute(kind)
            if current is None:
                current = new if kind != "members" else start_members(new, self.rows(kind))
            else:
                current = merge(self.rows(kind), new, current)
        return current


# -- max-min composition ------------------------------------------------------------------
def max_min(scores, valid, degree):
    """:math:`\\max_z \\min(s_z, \\deg(x, z))` over the valid witnesses ``z``: ``scores``,
    ``valid`` ``(B, K)``, ``degree`` ``(B, N, K)``; :data:`EMPTY` for a row without a valid
    witness."""
    has = valid.any(dim=1, keepdim=True).unsqueeze(-1)
    value = th.minimum(scores.unsqueeze(1), degree)
    fill = th.where(has, th.full_like(value, float("-inf")), th.full_like(value, EMPTY))
    return th.where(valid.unsqueeze(1), value, fill).amax(dim=-1)


def top_witnesses(scores, k):
    """The ``k`` best elements of every row (stable order): ``(ids, scores, valid)``, the
    columns cut after the last one valid in some row (at least one)."""
    keep = min(k, scores.shape[1])
    ids = th.argsort(scores, dim=1, descending=True, stable=True)[:, :keep]
    best = th.gather(scores, 1, ids)
    valid = best > EMPTY
    used = max(1, int(valid.any(dim=0).sum()))
    return ids[:, :used], best[:, :used], valid[:, :used]


def sparsify(scores, k):
    """Keep the scores of the ``k`` best elements of every row, :data:`EMPTY` elsewhere."""
    ids, best, valid = top_witnesses(scores, k)
    out = th.full_like(scores, EMPTY)
    return out.scatter(1, ids, th.where(valid, best, th.full_like(best, EMPTY)))


def compose_all(family, degree, best, valid, budget=1 << 26):
    """:func:`max_min` against every element of the family, in chunks of elements:
    ``degree(targets)`` gives ``(B, M, K)`` for a ``(B, M)`` table of element ids."""
    bsz, n = best.shape[0], family.n_elements
    if bsz == 0 or n == 0:
        return th.empty((bsz, n), dtype=family.dtype, device=family.device)
    per = max(1, bsz * best.shape[1] * family.parts_per_degree * family.embed_dim)
    step = max(1, min(n, budget // per))
    out = []
    for start in range(0, n, step):
        ids = th.arange(start, min(n, start + step), device=family.device)
        out.append(max_min(best, valid, degree(ids.unsqueeze(0).expand(bsz, -1))))
    return out[0] if len(out) == 1 else th.cat(out, dim=1)


def _anchor_scores(elements, n, dtype, device):
    if elements.numel() and bool(((elements < 0) | (elements >= n)).any()):
        raise ValueError(f"Anchor element id out of range [0, {n})")
    grid = th.arange(n, dtype=th.long, device=device).unsqueeze(0)
    return th.where(grid == elements.unsqueeze(1), th.full((len(elements), n), TOP, dtype=dtype,
                                                           device=device),
                    th.full((len(elements), n), EMPTY, dtype=dtype, device=device))


class _Options(NamedTuple):
    isa: Optional[int]
    isa_inv: Optional[int]
    entity_to_element: Optional[th.Tensor]
    class_of_entity: Optional[th.Tensor]


def _options(isa, isa_inv, entity_to_element, class_of_entity):
    return _Options(isa, isa_inv, entity_to_element, class_of_entity)


def _anchors_of(node, family, opt):
    first = node.hops[0] if node.hops else None
    return read_anchors(node.entities, first, family.n_classes, family.device, opt.isa_inv,
                        opt.entity_to_element, opt.class_of_entity)


# -- 1. regions ---------------------------------------------------------------------------
class RegionCompiler:
    """Compile every query of a batch to a region of the family.

    :param max_boxes: Bound on the parts of a region (passed to the family).
    :param exact: Project later hops through the elements inside the region
        (``family.project_exact``) instead of the region itself.
    """

    def __init__(self, max_boxes=64, exact=False, element_ids=None, max_witnesses=256,
                 isa=None, isa_inv=None, entity_to_element=None, class_of_entity=None):
        self.max_boxes, self.exact = max_boxes, exact
        self.element_ids, self.max_witnesses = element_ids, max_witnesses
        self.opt = _options(isa, isa_inv, entity_to_element, class_of_entity)

    def compile(self, family, node):
        if isinstance(node, Join):
            parts = [self.compile(family, c) for c in node.children]
            step = family.intersect if node.kind == "and" else family.union
            out = parts[0]
            for part in parts[1:]:
                out = step(out, part, self.max_boxes)
            return out
        if isinstance(node, Chain):
            region, anchors = self.compile(family, node.child), None
        else:
            anchors = _anchors_of(node, family, self.opt)
            region = (family.placeholder(len(node.entities)) if anchors.members_only
                      else family.anchor(anchors.elements))
        for i, hop in enumerate(node.hops):
            first = anchors is not None and i == 0
            region = self.hop(family, region, hop, first, anchors if first else None)
        return region

    def hop(self, family, region, relations, first, anchors):
        plan = HopPlan(relations, family.device, self.opt.isa, self.opt.isa_inv, first, anchors)
        if not first and self.opt.isa_inv is not None and self.opt.isa_inv in set(relations):
            raise NotImplementedError(
                "compile_query does not support type_inv_rel at a non-first hop (class -> its "
                "instances after an intermediate step); use score_query_beam or "
                "score_query_witness for such queries")
        b = self.max_boxes

        def compute(kind):
            if kind == "members":
                return family.cap(family.class_region(plan.member_classes()), b)
            if kind == "isa":
                return family.isa_region(region, b)
            if self.exact and not first:
                return family.project_exact(region, plan.role_ids, b, self.element_ids,
                                            self.max_witnesses)
            return family.project(region, plan.role_ids, b)
        out = plan.combine(compute, family.select_rows, lambda new, rows: new)
        return out if plan.only is not None else family.cap(out, b)


# -- 2. scores of all elements ------------------------------------------------------------
class MaxMinBeam:
    """Score every element of the family as an answer (max-min search over the ``k`` best
    witnesses of every hop)."""

    def __init__(self, k=64, isa=None, isa_inv=None, entity_to_element=None,
                 class_of_entity=None):
        self.k = k
        self.opt = _options(isa, isa_inv, entity_to_element, class_of_entity)

    def scores(self, family, node):
        if isinstance(node, Join):
            parts = [self.scores(family, c) for c in node.children]
            out = parts[0]
            for part in parts[1:]:
                out = th.minimum(out, part) if node.kind == "and" else th.maximum(out, part)
            return out
        if isinstance(node, Chain):
            scores, anchors = self.scores(family, node.child), None
        else:
            anchors, scores = self.start(family, node)
        for i, hop in enumerate(node.hops):
            first = anchors is not None and i == 0
            scores = self.hop(family, scores, hop, first, anchors if first else None)
        return scores

    def start(self, family, node):
        anchors = _anchors_of(node, family, self.opt)
        n, b = family.n_elements, len(node.entities)
        if anchors.members_only:
            return anchors, th.full((b, n), EMPTY, dtype=family.dtype, device=family.device)
        return anchors, _anchor_scores(anchors.elements, n, family.dtype, family.device)

    def hop(self, family, scores, relations, first, anchors):
        plan = HopPlan(relations, family.device, self.opt.isa, self.opt.isa_inv, first, anchors)

        def compute(kind):
            if kind == "members":
                return family.member_scores(plan.member_classes())
            if kind == "isa_inv":
                return self.through_classes(family, scores)
            if kind == "isa":
                return self.to_classes(family, scores)
            return self.along(family, scores, plan.role_ids)
        return plan.combine(
            compute, lambda rows, new, cur: th.where(rows.unsqueeze(1), new, cur),
            lambda new, rows: th.where(rows.unsqueeze(1), new, th.full_like(new, EMPTY)))

    def along(self, family, scores, role_ids):
        """A role hop: max-min of the best witnesses with ``family.role_degree``."""
        ids, best, valid = top_witnesses(scores, self.k)
        return compose_all(family, lambda x: family.role_degree(ids, role_ids, x), best, valid)

    def to_classes(self, family, scores):
        """An ``isa`` hop: the classes of the best witnesses (individuals score
        :data:`EMPTY`)."""
        ids, best, valid = top_witnesses(scores, self.k)
        classes = max_min(best, valid, family.class_degree(ids))  # (B, classes)
        rest = scores.shape[1] - classes.shape[1]
        if rest <= 0:
            return classes
        return th.cat([classes, th.full((scores.shape[0], rest), EMPTY, dtype=scores.dtype,
                                        device=scores.device)], dim=1)

    def through_classes(self, family, scores):
        """An ``isa_inv`` hop after the first: the members of the best classes."""
        n_classes = min(family.n_classes, scores.shape[1])
        if min(self.k, n_classes) == 0:
            return th.full_like(scores, EMPTY)
        ids, best, valid = top_witnesses(scores[:, :n_classes], self.k)
        has = valid.any(dim=1, keepdim=True)
        out = th.where(has, th.full_like(scores, float("-inf")), th.full_like(scores, EMPTY))
        for j in range(ids.shape[1]):
            value = th.minimum(best[:, j:j + 1], family.member_scores(ids[:, j]).to(scores.dtype))
            out = th.maximum(out, th.where(valid[:, j:j + 1], value,
                                           th.full_like(value, float("-inf"))))
        return out


# -- 3. differentiable scores of targets --------------------------------------------------
class WitnessChain:
    """Differentiable scores of target elements: witnesses chosen by :class:`MaxMinBeam`
    (on ``search``, a family read without gradient), soft degrees along them (on
    ``family``, at the temperature ``tau``)."""

    def __init__(self, k=16, isa=None, isa_inv=None, entity_to_element=None,
                 class_of_entity=None):
        self.k = k
        self.opt = _options(isa, isa_inv, entity_to_element, class_of_entity)
        self.beam = MaxMinBeam(k, isa, isa_inv, entity_to_element, class_of_entity)

    def scores(self, family, search, node, targets, tau):
        if isinstance(node, Join):
            parts = [self.scores(family, search, c, targets, tau) for c in node.children]
            out = parts[0]
            for part in parts[1:]:
                out = th.minimum(out, part) if node.kind == "and" else th.maximum(out, part)
            return out
        if isinstance(node, Anchored):
            return self._anchored(family, search, node, targets, tau)
        with th.no_grad():
            dense = self.beam.scores(search, node.child)
            chosen = [top_witnesses(dense, self.k)]
            for hop in node.hops[:-1]:
                dense = self.beam.hop(search, dense, hop, False, None)
                chosen.append(top_witnesses(dense, self.k))
        ids, _, valid = chosen[0]
        best = self.scores(family, search, node.child, ids, tau)
        return self._follow(family, node.hops, chosen[1:], ids, best, valid, targets, tau)

    def _anchored(self, family, search, node, targets, tau):
        anchors = _anchors_of(node, family, self.opt)
        elements = th.zeros_like(anchors.elements) if anchors.members_only else anchors.elements
        if not anchors.members_only and elements.numel() and \
                bool(((elements < 0) | (elements >= family.n_elements)).any()):
            raise ValueError(f"Anchor element id out of range [0, {family.n_elements})")
        if not node.hops:
            return th.where(targets == elements.unsqueeze(1),
                            th.full(targets.shape, TOP, dtype=family.dtype, device=family.device),
                            th.full(targets.shape, EMPTY, dtype=family.dtype,
                                    device=family.device))
        if len(node.hops) == 1:
            return self.first(family, elements, node.hops[0], targets, anchors, tau)
        with th.no_grad():
            dense = (th.full((len(elements), family.n_elements), EMPTY, dtype=family.dtype,
                             device=family.device) if anchors.members_only else
                     _anchor_scores(elements, family.n_elements, family.dtype, family.device))
            chosen = []
            for i, hop in enumerate(node.hops[:-1]):
                first = i == 0
                dense = self.beam.hop(search, dense if first else sparsify(dense, self.k), hop,
                                      first, anchors if first else None)
                chosen.append(top_witnesses(dense, self.k))
        ids, _, valid = chosen[0]
        best = self.first(family, elements, node.hops[0], ids, anchors, tau)
        return self._follow(family, node.hops[1:], chosen[1:], ids, best, valid, targets, tau)

    def _follow(self, family, hops, chosen, ids, best, valid, targets, tau):
        """Soft hops along ``hops`` from the witnesses ``ids``; ``chosen[i]`` are the
        witnesses after ``hops[i]`` (all hops but the last), the last hop reaches
        ``targets``."""
        for i, hop in enumerate(hops):
            last = i == len(hops) - 1
            best = self.later(family, ids, best, valid, hop, targets if last else chosen[i][0],
                              tau)
            if not last:
                ids, valid = chosen[i][0], chosen[i][2]
        return best

    def first(self, family, elements, relations, targets, anchors, tau):
        """The soft first hop from the anchors to ``targets``."""
        plan = HopPlan(relations, family.device, self.opt.isa, self.opt.isa_inv, True, anchors)
        b = len(elements)
        ids = elements.unsqueeze(1)
        best = th.full((b, 1), TOP, dtype=family.dtype, device=family.device)
        valid = th.ones((b, 1), dtype=th.bool, device=family.device)

        def compute(kind):
            if kind == "members":
                return family.member_degree(plan.member_classes().unsqueeze(1), targets,
                                            tau=tau).squeeze(-1)
            if kind == "isa":
                return self.to_classes(family, ids, best, valid, targets, tau)
            return max_min(best, valid, family.role_degree(ids, plan.role_ids, targets, tau=tau))
        return plan.combine(
            compute, lambda rows, new, cur: th.where(rows.unsqueeze(1), new, cur),
            lambda new, rows: th.where(rows.unsqueeze(1), new, th.full_like(new, EMPTY)))

    @staticmethod
    def to_classes(family, ids, best, valid, targets, tau):
        """A soft ``isa`` hop: the degree of the witnesses in the target classes (a target
        that is not a class scores :data:`EMPTY`)."""
        are = (targets >= 0) & (targets < family.n_classes)
        degree = family.class_degree(ids, th.where(are, targets, th.zeros_like(targets)), tau=tau)
        out = max_min(best, valid, degree)
        return th.where(are, out, th.full_like(out, EMPTY))

    def later(self, family, ids, best, valid, relations, targets, tau):
        """A soft hop from witnesses ``ids`` (scores ``best``) to ``targets``."""
        plan = HopPlan(relations, family.device, self.opt.isa, self.opt.isa_inv, False, None)

        def compute(kind):
            if kind == "isa_inv":
                are_classes = (ids >= 0) & (ids < family.n_classes)
                degree = family.member_degree(th.where(are_classes, ids, th.zeros_like(ids)),
                                              targets, tau=tau)
                return max_min(best, valid & are_classes, degree)
            if kind == "isa":
                return self.to_classes(family, ids, best, valid, targets, tau)
            return max_min(best, valid, family.role_degree(ids, plan.role_ids, targets, tau=tau))
        return plan.combine(compute, lambda rows, new, cur: th.where(rows.unsqueeze(1), new, cur),
                            lambda new, rows: new)


def as_target_table(ids, bsz, n, device, name, allow_empty=False) -> Tuple[th.Tensor, bool]:
    """Target elements as a ``(B, M)`` table and whether they were given one per query."""
    if isinstance(ids, th.Tensor):
        if ids.dtype == th.bool or ids.is_floating_point() or ids.is_complex():
            raise ValueError(f"{name} must contain integer indices, got dtype={ids.dtype}")
        table = ids.to(device=device, dtype=th.long)
    else:
        arr = np.asarray(ids)
        if arr.size and (arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer)):
            raise ValueError(f"{name} must contain integer indices, got dtype={arr.dtype}")
        table = th.as_tensor(arr.astype(np.int64), device=device)
    one = table.ndim < 2
    if table.ndim > 2:
        raise ValueError(f"{name} must be 1-D (B,) or 2-D (B, M), got shape {tuple(table.shape)}")
    table = table.reshape(-1, 1) if one else table
    if table.shape[0] != bsz:
        raise ValueError(f"{name} batch size ({table.shape[0]}) does not match queries batch "
                         f"size ({bsz})")
    if not allow_empty and bsz > 0 and table.shape[1] < 1:
        raise ValueError(f"{name} must have at least 1 column per query, got shape "
                         f"{tuple(table.shape)}")
    if table.numel() and bool(((table < 0) | (table >= n)).any()):
        raise ValueError(f"{name} indices out of range [0, {n})")
    return table, one
