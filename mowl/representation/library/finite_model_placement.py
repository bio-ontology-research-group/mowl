"""Placing a finite interpretation into a geometry of points and unions of boxes (an
initialisation part).

A finite interpretation is given by *labels* (which element belongs to which class: a boolean
``(elements, classes)`` table) and *edges* (which pairs of elements each relation holds of).
:class:`FiniteModelPlacement` realises it exactly in a geometry whose elements are points and
whose classes and relations are unions of boxes:

1. the elements get generic positions at :math:`\\ell_\\infty` distance at least
   :math:`4\\delta` from each other (:func:`separated_points`; with a relation model of
   translations, also the differences :math:`p_y - p_x` of distinct pairs are that far
   apart, so that a box of half-width :math:`\\delta` around a difference holds no other
   difference);
2. component :math:`k` of class :math:`A` becomes the box of half-width :math:`\\delta`
   around :math:`p_k` iff element :math:`k` belongs to :math:`A`; the other components are
   inactive with width zero (:meth:`BoxUnionClasses.place_point_boxes
   <mowl.representation.library.grounded_parts.BoxUnionClasses.place_point_boxes>`);
3. component :math:`k` of relation :math:`r` becomes the box around the :math:`k`-th edge
   :math:`(x, y)` of :math:`r` (the product :math:`\\mathrm{Box}(p_x) \\times \\mathrm{Box}(p_y)`
   or the translation box around :math:`p_y - p_x`; ``place_edges`` of the relation model);
4. the elements that belong to no class (an individual) or not to their own class (a class
   witness), and the witness of bottom, are dead (:meth:`ElementDomain.place
   <mowl.representation.library.grounded_parts.ElementDomain.place>`).

The result is a model of every axiom the interpretation satisfies, at every temperature:
used by GroundedEL (``init="elements"``) from a least model, and usable by any geometry made
of an :class:`~mowl.representation.library.grounded_parts.ElementDomain`, a
:class:`~mowl.representation.library.grounded_parts.BoxUnionClasses` and one of its relation
models. Reasoner-free: the labels and the edges are the caller's.
"""
from __future__ import annotations

from typing import List, Mapping, Tuple

import numpy as np
import torch as th

from mowl.representation.library.grounded_parts import (inv_softplus, validate_int_ge,
                                                        validate_positive_float)

__all__ = ["separated_points", "live_elements", "edges_by_relation", "FiniteModelPlacement",
           "interpretation_labels", "close_labels", "edge_triples", "syntactic_edges"]


def _min_offdiagonal_distance(points):
    """The smallest :math:`\\ell_\\infty` distance between two distinct rows of ``points``."""
    diff = th.amax(th.abs(points.unsqueeze(1) - points.unsqueeze(0)), dim=-1)
    diff = diff.masked_fill(th.eye(points.shape[0], dtype=th.bool), float("inf"))
    return float(diff.min().item())


def separated_points(n, dim, delta, seed=0, distinct_differences=False, max_pairs=256):
    """``n`` generic points of :math:`\\mathbb{R}^{dim}` (float64, CPU) at :math:`\\ell_\\infty`
    distance at least :math:`4\\delta` from each other.

    The points are uniform draws of a generator seeded with ``seed``; the first coordinate is
    a random permutation of ``0 .. n - 1`` plus half the draw (distinct even when the draws
    collide). The set is scaled up until the smallest distance is :math:`4\\delta`. With
    ``distinct_differences`` (and ``n <= max_pairs``) the differences :math:`p_j - p_i` of
    distinct pairs are also scaled to be :math:`4\\delta` from zero and from each other."""
    gen = th.Generator(device="cpu")
    gen.manual_seed(int(seed))
    raw = th.rand((n, dim), dtype=th.float64, generator=gen)
    perm = th.randperm(n, generator=gen).to(dtype=th.float64)
    raw[:, 0] = perm + 0.5 * raw[:, 0]
    if n <= 1:
        return raw
    target = 4.0 * delta
    min_sep = _min_offdiagonal_distance(raw)
    if min_sep < target:
        raw = raw * (target / max(min_sep, 1e-12))
    if distinct_differences and n <= max_pairs:
        diffs = (raw.unsqueeze(0) - raw.unsqueeze(1)).reshape(-1, dim)
        diffs = diffs[~th.eye(n, dtype=th.bool).reshape(-1)]   # entry (i, j): p_j - p_i
        if diffs.shape[0] > 0:
            sep = float(th.amax(th.abs(diffs), dim=-1).min().item())
            if diffs.shape[0] > 1:
                sep = min(sep, _min_offdiagonal_distance(diffs))
            if sep < target:
                raw = raw * (target / max(sep, 1e-12))
    return raw


def live_elements(labels, nb_classes, bot_id=None):
    """Which elements of a labelled domain exist: the witness :math:`x_A` of a class iff it
    belongs to :math:`A` (and not to bottom; never the witness of bottom), an individual iff
    it belongs to some class (and not to bottom). ``labels`` is boolean ``(N, nb_classes)``
    with the class witnesses first."""
    ids = th.arange(nb_classes, device=labels.device)
    live = labels[ids, ids].clone()
    if bot_id is not None:
        live[bot_id] = False
        live = live & (~labels[:nb_classes, bot_id])
    if labels.shape[0] > nb_classes:
        rest = labels[nb_classes:].any(dim=1)
        if bot_id is not None:
            rest = rest & (~labels[nb_classes:, bot_id])
        live = th.cat([live, rest], dim=0)
    return live


def _checked(x, y, nb_elements):
    if not (0 <= x < nb_elements and 0 <= y < nb_elements):
        raise ValueError(f"Edge element index ({x}, {y}) out of range [0, {nb_elements})")
    return x, y


def edges_by_relation(edges, nb_rels, nb_elements, inverse_of=None) -> List[List[Tuple[int, int]]]:
    """The edges per relation: ``[[(x, y), ...] for r in range(nb_rels)]``.

    ``edges`` is ``None``, a mapping ``{r: [(x, y), ...]}``, an ``(E, 3)`` tensor or array of
    ``(x, y, r)`` rows, or an iterable of such triples. With ``inverse_of`` (``-1`` or the
    partner of each relation), an edge :math:`(x, y)` of a relation whose partner has a lower
    index adds :math:`(y, x)` to the partner (which owns the parameters of both)."""
    out: List[List[Tuple[int, int]]] = [[] for _ in range(nb_rels)]
    if edges is not None and isinstance(edges, Mapping):
        for key, pairs in edges.items():
            r = validate_int_ge("role_id", int(key), 0)
            if r >= nb_rels:
                raise ValueError(f"Role index {r} out of range [0, {nb_rels})")
            for x, y in pairs:
                xi = validate_int_ge("x", int(x), 0)
                yi = validate_int_ge("y", int(y), 0)
                out[r].append(_checked(xi, yi, nb_elements))
    elif edges is not None:
        if isinstance(edges, (th.Tensor, np.ndarray)):
            rows = th.as_tensor(edges, dtype=th.long).cpu()
            if rows.numel() == 0:
                triples = []
            elif rows.ndim == 2 and rows.shape[1] == 3:
                triples = [tuple(int(v) for v in row) for row in rows.tolist()]
            else:
                raise ValueError(f"Tensor edges must have shape (E, 3), got {tuple(rows.shape)}")
        else:
            triples = [(int(t[0]), int(t[1]), int(t[2])) for t in edges]
        for x, y, r in triples:
            if not (0 <= r < nb_rels):
                raise ValueError(f"Role index {r} out of range [0, {nb_rels})")
            out[r].append(_checked(x, y, nb_elements))
    if inverse_of is not None and nb_rels > 0 and bool((inverse_of >= 0).any().item()):
        for r, partner in enumerate(inverse_of.detach().cpu().tolist()):
            if 0 <= partner < r:
                owned = set(out[partner])
                for x, y in out[r]:
                    if (y, x) not in owned:
                        out[partner].append((y, x))
                        owned.add((y, x))
    return out


class FiniteModelPlacement:
    """The initialisation that realises a finite interpretation exactly (module docstring).

    :param delta: Half-width :math:`\\delta > 0` of the box around every element.
    :param seed: Seed of the generic positions.
    :param dead_value, live_value: Deadness written for dead and live elements.
    """

    def __init__(self, delta=0.05, seed=0, dead_value=1.0, live_value=0.0):
        self.delta = validate_positive_float("delta", delta)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        self.seed, self.dead_value, self.live_value = seed, dead_value, live_value

    def place(self, domain, classes, relations, labels, edges=None, dead_mask=None,
              bot_id=None, expand=None):
        """Write the interpretation into the tables of ``domain`` (an
        :class:`~mowl.representation.library.grounded_parts.ElementDomain`), ``classes`` (a
        :class:`~mowl.representation.library.grounded_parts.BoxUnionClasses` with at least
        as many components as elements) and ``relations`` (a relation model with
        ``place_edges``).

        :param labels: ``(elements, classes)``, truthy where the element is in the class.
        :param edges: The edges (:func:`edges_by_relation`).
        :param dead_mask: Boolean ``(elements,)``; default :func:`live_elements` negated.
        :param bot_id: The class bottom (no component of it is placed).
        :param expand: Called with the largest number of edges of a relation when it exceeds
            the components of the relation model (to add components first).
        """
        n, n_classes = domain.nb_elements, classes.nb_classes
        if classes.n_components < n:
            raise ValueError(
                f"exact_limit_init requires n_components >= nb_elements ({n}), "
                f"got n_components={classes.n_components}")
        device, dtype = classes.centers.device, classes.centers.dtype
        labels = th.as_tensor(labels, device=device)
        if labels.shape != (n, n_classes):
            raise ValueError(f"least_model_labels must have shape ({n}, {n_classes}), "
                             f"got {tuple(labels.shape)}")
        member = labels > 0
        per_relation = edges_by_relation(edges, len(relations), n, relations.inverse_of)
        most = max((len(e) for e in per_relation), default=0)
        if most > relations.n_components and expand is not None:
            expand(most)
        translations = getattr(relations, "mode", None) == "translation"
        positions = separated_points(n, domain.embed_dim, self.delta, self.seed,
                                     distinct_differences=translations).to(device=device,
                                                                           dtype=dtype)
        if dead_mask is not None:
            dead = th.as_tensor(dead_mask, dtype=th.bool, device=device)
            if dead.shape != (n,):
                raise ValueError(f"dead_mask must have shape ({n},), got {tuple(dead.shape)}")
        else:
            dead = ~live_elements(member, n_classes, bot_id)
        raw_delta = inv_softplus(self.delta)
        in_class = member.t().contiguous()
        if bot_id is not None:
            in_class[bot_id, :] = False
        with th.no_grad():
            domain.place(positions, dead, self.dead_value, self.live_value)
            classes.place_point_boxes(in_class, positions, raw_delta)
            relations.place_edges(per_relation, positions, raw_delta)
        return positions


# -- reading a finite interpretation given by the caller ---------------------------------------
def interpretation_labels(raw, nb_elements, class_index, element_index, device="cpu"):
    """The labels ``(nb_elements, classes)`` of a finite interpretation, as a boolean tensor.

    ``raw`` is a tensor or array of that shape, or a mapping either from classes to the
    elements in them or from elements to their classes (keys and values are names of
    ``class_index`` / ``element_index`` or indices)."""
    n_classes = len(class_index)
    if isinstance(raw, (th.Tensor, np.ndarray)):
        labels = th.as_tensor(raw, dtype=th.bool, device=device).clone()
        if labels.shape != (nb_elements, n_classes):
            raise ValueError(f"init_labels tensor must have shape ({nb_elements}, {n_classes}), "
                             f"got {tuple(labels.shape)}")
        return labels
    if not isinstance(raw, Mapping):
        raise TypeError(f"Unsupported init_labels type {type(raw).__name__}; expected Tensor, "
                        "ndarray, tuple, or Mapping")
    labels = th.zeros((nb_elements, n_classes), dtype=th.bool, device=device)
    for key, values in raw.items():
        if isinstance(key, str) and key in class_index:
            c = class_index[key]
            for v in values:
                labels[element_index[v] if isinstance(v, str) else int(v), c] = True
        else:
            e = element_index[key] if isinstance(key, str) else int(key)
            for v in values:
                labels[e, class_index[v] if isinstance(v, str) else int(v)] = True
    return labels


def _witness_live(labels, nb_classes):
    """Live elements before deadness by bottom: a witness in its own class, an individual in
    some class."""
    ids = th.arange(nb_classes, device=labels.device)
    live = th.zeros((labels.shape[0],), dtype=th.bool, device=labels.device)
    live[:nb_classes] = labels[ids, ids]
    if labels.shape[0] > nb_classes:
        live[nb_classes:] = labels[nb_classes:].any(dim=1)
    return live


def close_labels(labels, nb_classes, top_id=None, bot_id=None):
    """Make labels respect top and bottom (in place): nothing is in bottom and the witness of
    bottom is in nothing; every live element (and the witness of top, if any element lives)
    is in top."""
    if bot_id is not None:
        labels[:, bot_id] = False
        labels[bot_id, :] = False
    if top_id is not None:
        live = _witness_live(labels, nb_classes)
        if bool(live.any().item()) and not bool(labels[top_id, top_id].item()):
            labels[top_id, top_id] = True
            live[top_id] = True
        labels[live, top_id] = True
    return labels


def edge_triples(raw, role_index):
    """Edges given by the caller as a list of ``(x, y, r)`` (``r`` a name of ``role_index``
    or an index): from a mapping ``{r: [(x, y), ...]}``, an ``(E, 3)`` tensor or an iterable."""
    out = []
    if isinstance(raw, Mapping):
        for key, pairs in raw.items():
            r = role_index[key] if isinstance(key, str) else int(key)
            out.extend((int(u), int(v), int(r)) for u, v in pairs)
    elif isinstance(raw, (th.Tensor, np.ndarray)):
        rows = th.as_tensor(raw, dtype=th.long).cpu()
        if rows.numel() > 0:
            out.extend((int(a), int(b), int(c)) for a, b, c in rows.tolist())
    else:
        for u, v, r in raw:
            out.append((int(u), int(v), int(role_index[r] if isinstance(r, str) else r)))
    return out


def _rows(tables, form):
    table = tables.get(form) if form in tables else None
    if table is None or len(table) == 0:
        return []
    return [tuple(int(v) for v in row) for row in table[:].detach().cpu().tolist()]


def syntactic_edges(labels, tables, nb_classes, inverse_of=None):
    """The edges of the least model read off the axioms and the labels, without a reasoner.

    From the role assertions :math:`r(a, b)` (the elements of the individuals) and from every
    :math:`A \\sqsubseteq \\exists r.B` with a live :math:`x_B`: an edge :math:`(x, x_B)` for
    every live element :math:`x` of :math:`A`. Then closed under the inverses
    (``inverse_of``), the role inclusions and the role chains of ``tables`` (``{form:
    rows}``). Returns the sorted ``(x, y, r)`` triples."""
    edges = set()
    live = _witness_live(labels, nb_classes)
    for a, r, b in _rows(tables, "object_property_assertion"):
        edges.add((nb_classes + a, nb_classes + b, r))
    for a, r, b in _rows(tables, "gci2"):
        if not bool(live[b].item()):
            continue
        for x in th.nonzero(labels[:, a] & live, as_tuple=False).view(-1).cpu().tolist():
            edges.add((int(x), b, r))
    inclusions = _rows(tables, "role_inclusion")
    chains = _rows(tables, "role_chain")
    if not (inclusions or chains or inverse_of is not None):
        return sorted(edges)
    changed = True
    while changed:
        changed = False
        for u, v, r in list(edges):
            if inverse_of is not None and 0 <= r < len(inverse_of) and inverse_of[r] >= 0:
                inv = int(inverse_of[r])
                if (v, u, inv) not in edges:
                    edges.add((v, u, inv))
                    changed = True
            for sub, sup in inclusions:
                if r == sub and (u, v, sup) not in edges:
                    edges.add((u, v, sup))
                    changed = True
        if chains:
            by_role = {}
            for u, v, r in edges:
                by_role.setdefault(r, []).append((u, v))
            for r1, r2, sup in chains:
                for u, mid in by_role.get(r1, ()):
                    for mid2, w in by_role.get(r2, ()):
                        if mid == mid2 and (u, w, sup) not in edges:
                            edges.add((u, w, sup))
                            changed = True
    return sorted(edges)
