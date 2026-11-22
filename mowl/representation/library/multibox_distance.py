"""The evaluation distance of the volume-based MultiBoxEL (Aharoni 2024, Section 7.1.2).

An evaluation regime ranks candidate axioms by a distance, not by the training loss: for a
subsumption the fraction of points drawn from the boxes of the left side that fall outside the
right side. It reads the regions of the class model and the relation model of a geometry
(unions of boxes, roles over pairs) and the sampled measure
(:class:`~mowl.representation.library.multibox_parts.SampledVolume`), so it works for every
such geometry, not only the module of the paper.
"""

import torch as th

from mowl.representation.library.multibox_parts import membership_sampler

__all__ = ["sampled_distance", "ranking_ratio", "box_samples_outside"]


def _unions(geometry, ids):
    region = geometry.classes(ids)
    return region.centre, region.offset


def box_samples_outside(unit_points, left, right, valid=None):
    """The fraction of the points drawn from the boxes ``left = (lo, hi)`` (``(B, nb, k)``
    corners, ``valid`` ``(B, nb)`` which boxes count) that lie outside every box of ``right =
    (centre, offset)`` (``(B, nr, k)``); 1 where no box counts."""
    lo, hi = left
    centre, offset = right
    batch, boxes, dim = lo.shape
    u = unit_points.view(1, 1, -1, dim)
    pts = (lo.unsqueeze(2) + (hi - lo).unsqueeze(2) * u).flatten(1, 2)
    if valid is None:
        valid = th.ones(batch, boxes, dtype=th.bool, device=lo.device)
    w = valid.unsqueeze(-1).expand(-1, -1, u.shape[2]).flatten(1)
    rlo, rhi = (centre - offset).unsqueeze(1), (centre + offset).unsqueeze(1)
    inside = ((pts.unsqueeze(2) >= rlo) & (pts.unsqueeze(2) <= rhi)).all(dim=-1).any(dim=-1)
    hits = (inside & w).sum(dim=-1).to(th.float32)
    total = w.sum(dim=-1).to(th.float32)
    return th.where(total > 0, 1.0 - hits / total.clamp(min=1.0), th.ones_like(hits))


def _sample_distance(geometry, data, gci_name, unit_points):
    lc, lo = _unions(geometry, data[:, 0])
    if gci_name == "gci0":
        rc, ro = _unions(geometry, data[:, 1])
        return box_samples_outside(unit_points, (lc - lo, lc + lo), (rc, ro))
    c2, o2 = _unions(geometry, data[:, 1])
    rc, ro = _unions(geometry, data[:, 2])
    lo_corner = th.maximum((lc - lo).unsqueeze(2), (c2 - o2).unsqueeze(1))
    hi_corner = th.minimum((lc + lo).unsqueeze(2), (c2 + o2).unsqueeze(1))
    valid = (hi_corner >= lo_corner).all(dim=-1).flatten(1)
    k = lc.shape[-1]
    lo_corner = lo_corner.reshape(len(data), -1, k)
    hi_corner = hi_corner.reshape(len(data), -1, k)
    return box_samples_outside(unit_points, (lo_corner, hi_corner), (rc, ro), valid)


def ranking_ratio(geometry, data, gci_name, step=None):
    """The training loss of ``gci2`` / ``gci3`` rows, computed for ranking: the volume ratio
    of the measure on its fixed point set, with the regions and their memberships made once
    per *distinct* operand and not once per row (the candidates of a query usually differ in
    one operand only): the class, or the preimage of the filler under the relation, which the
    relation model makes once per distinct relation. Points in chunks of the measure's
    ``chunk_size``, operands in chunks of ``step``. Returns ``(N,)``."""
    measure, relations = geometry.measure, geometry.relation_model

    def concept(keys):
        return geometry.classes(keys[:, 0])

    def existential(keys):
        return relations.preimage(keys[:, 0], geometry.classes(keys[:, 1]))

    if gci_name == "gci2":
        operands = ((data[:, :1], concept), (data[:, 1:3], existential))
    else:
        operands = ((data[:, :2], existential), (data[:, 2:3], concept))
    distinct = [th.unique(keys, dim=0, return_inverse=True) for keys, _ in operands]
    step = step or max(max(len(keys) for keys, _ in distinct), 1)
    samplers = [[membership_sampler(region_of(keys[i:i + step]), measure.temperature,
                                    measure.aggregate) for i in range(0, len(keys), step)]
                for (keys, _), (_, region_of) in zip(distinct, operands)]
    (_, left_rows), (_, right_rows) = distinct
    points = measure.points(geometry.class_model.embed_dim, data.device)
    shared = th.zeros(len(data), device=data.device)
    own = th.zeros(len(data), device=data.device)
    for chunk in th.split(points, measure.chunk_size):
        left, right = (th.cat([fn(chunk) for fn in fns]) for fns in samplers)
        for rows in th.arange(len(data), device=data.device).split(step):
            a, b = left[left_rows[rows]], right[right_rows[rows]]
            shared[rows] += th.minimum(a, b).sum(dim=-1)
            own[rows] += a.sum(dim=-1)
    total = points.shape[0]
    return 1.0 - (shared / total) / (own / total + measure.eps)


@th.no_grad()
def sampled_distance(geometry, data, gci_name, unit_points, chunk=1024, step=None):
    """The evaluation distance of ``gci_name`` rows (lower is better), ``(N,)``.

    ``gci0``: :math:`1 - |\\{p \\in P_C : p \\in \\llbracket D \\rrbracket\\}| / |P_C|` with
    :math:`P_C` the points ``unit_points`` mapped into every box of :math:`C`, membership in
    :math:`D` exact; ``gci1``: the points are drawn from the non-empty pairwise intersections
    of the boxes of the two left concepts; ``gci2``, ``gci3``: the thesis gives no distance, the
    training loss is used (:func:`ranking_ratio`).
    """
    if gci_name in ("gci2", "gci3"):
        return ranking_ratio(geometry, data, gci_name, step)
    if gci_name not in ("gci0", "gci1"):
        raise NotImplementedError(
            f"The thesis (Aharoni 2024) defines no distance for '{gci_name}'.")
    return th.cat([_sample_distance(geometry, data[i:i + chunk], gci_name, unit_points)
                   for i in range(0, len(data), chunk)])
