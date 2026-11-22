"""The evaluation distance of the volume-based MultiBoxEL (Aharoni 2024, Section 7.1.2).

An evaluation regime ranks candidate axioms by a distance, not by the training loss: for a
subsumption the fraction of points drawn from the boxes of the left side that fall outside the
right side. It reads the regions of the class model and the relation model of a geometry
(unions of boxes, roles over pairs) and the sampled measure
(:class:`~mowl.representation.library.multibox_parts.SampledVolume`), so it works for every
such geometry, not only the module of the paper.
"""

import torch as th

import mowl.nn.el.mcmultibox.volume as V

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
    on the fixed point set, with the memberships computed once per *distinct* operand and
    not once per row (the candidates of a query usually differ in one operand only), points in
    chunks of the measure's ``chunk_size``, operands in chunks of ``step``. Returns ``(N,)``."""
    measure, relations = geometry.measure, geometry.relation_model
    if gci_name == "gci2":
        left_keys, right_keys = data[:, :1], data[:, 1:3]
    else:
        left_keys, right_keys = data[:, :2], data[:, 2:3]
    left_u, left_inv = th.unique(left_keys, dim=0, return_inverse=True)
    right_u, right_inv = th.unique(right_keys, dim=0, return_inverse=True)
    step = step or max(len(left_u), len(right_u), 1)
    temperature, aggregate = measure.temperature, measure.aggregate
    d = relations.embed_dim

    def concept(keys, col):
        return _unions(geometry, keys[:, col])

    def concept_operand(keys):
        c, o = concept(keys, 0)
        return lambda p: V.soft_membership(p, c, o, temperature, aggregate)

    def projection(keys):
        # the membership in the domain boxes of a role depends on the role only: computed
        # once per distinct role; the gate depends on (role, filler) but is small
        roles, inv = th.unique(keys[:, 0], return_inverse=True)
        rc, ro = relations._boxes(roles)
        fc, fo = concept(keys, 1)
        gate = V.project_gate(rc[inv], ro[inv], fc, fo, measure.bound, temperature, aggregate)

        def fn(p):
            per_box = V.soft_box_membership(p, rc[..., :d], ro[..., :d], temperature,
                                            aggregate)
            return th.minimum(per_box[inv], gate.unsqueeze(1)).amax(dim=-1)
        return fn

    left_op, right_op = ((concept_operand, projection) if gci_name == "gci2"
                         else (projection, concept_operand))
    points = measure.points(geometry.class_model.embed_dim, data.device)
    total = points.shape[0]
    num = th.zeros(len(data), device=data.device)
    den = th.zeros(len(data), device=data.device)
    for start in range(0, total, measure.chunk_size):
        p = points[start:start + measure.chunk_size]
        left = th.cat([left_op(left_u[i:i + step])(p) for i in range(0, len(left_u), step)])
        right = th.cat([right_op(right_u[i:i + step])(p)
                        for i in range(0, len(right_u), step)])
        for j in range(0, len(data), step):
            a, b = left[left_inv[j:j + step]], right[right_inv[j:j + step]]
            num[j:j + step] += th.minimum(a, b).sum(dim=-1)
            den[j:j + step] += a.sum(dim=-1)
    return V.containment_loss(num / total, den / total)


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
