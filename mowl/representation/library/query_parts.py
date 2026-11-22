"""The parts of the query embedding methods (Query2Box, BetaE, ConE, AConE).

A query method is five parts:

=========================  =====================================================
entity model               a table that gives an entity the region of an anchor
relation model             how a relation turns the region of a query into the region
                           of its projection (:class:`~mowl.representation.Relations`)
intersection operator      the region of an intersection of query regions
negation operator          the region of a complement (optional)
distance                   how far an entity is from a query region
=========================  =====================================================

The relation models are ``image`` maps that also serve as the relation model of
:class:`~mowl.models.ComposedEL` (``preimage`` is :math:`\\exists r.C`), where the query
methods use a separate relation (an inverse) for it: ``inverse_of`` says how to find it.

Which parts fit together is decided by the region family: ``box`` (Query2Box),
``distribution`` (BetaE) and ``sector cone`` (ConE, AConE).
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from mowl.nn.query.base import _clamp_positive
from mowl.representation.library.query_regions import (
    BetaRegion, SectorCone, as_state, beta_kl, from_state, to_aperture, to_axis,
    wrap_angle)
from mowl.representation.methods import register_relation_model
from mowl.representation.regions import Box, ConstructorUnavailable
from mowl.representation.relation_models import Relations

__all__ = ["PointBoxes", "BetaPoints", "ConePoints", "BetaNetwork",
           "ConeNetwork", "ConeRotation", "BoxAttentionIntersection",
           "BoxExactIntersection", "BetaAttentionIntersection", "BetaMeanIntersection", "ConeAttentionIntersection",
           "BetaInversion", "ConeComplement", "BoxDistance", "BetaDistance", "ConeDistance",
           "embedding"]


def embedding(parameter):
    """An :class:`~torch.nn.Embedding` whose weight *is* ``parameter`` (not a copy)."""
    table = nn.Embedding(parameter.shape[0], parameter.shape[1])
    table.weight = parameter
    return table


def _uniform(size, dim, low, high):
    table = nn.Embedding(size, dim)
    nn.init.uniform_(table.weight, a=low, b=high)
    return table


# ---------------------------------------------------------------------------------------
# entity models: the region of an anchor
# ---------------------------------------------------------------------------------------

class _Entities(nn.Module):
    region = None
    part = ""

    def __len__(self):
        return self.table.num_embeddings

    def flat(self):
        """One flat vector per entity, for inspection."""
        raise NotImplementedError


class PointBoxes(_Entities):
    """An entity is a point, a box of offset zero (Query2Box).

    :param table: An existing embedding (or a parameter) to share, ``(n, d)``.
    """

    region = Box
    part = "box"

    def __init__(self, size=None, dim=None, table=None, init_range=0.1):
        super().__init__()
        if isinstance(table, nn.Parameter):
            table = embedding(table)
        self.table = table if table is not None else _uniform(size, dim, -init_range,
                                                              init_range)

    def forward(self, ids):
        c = self.table(ids)
        return Box(c, th.zeros_like(c))

    def flat(self):
        return self.table.weight.detach()


class BetaPoints(_Entities):
    """An entity is a Beta distribution per coordinate; the table holds ``(alpha, beta)``
    and the parameters are clamped to ``[0.05, 1e9]`` (BetaE)."""

    region = BetaRegion
    part = "distribution"

    def __init__(self, size=None, dim=None, table=None, init_range=0.1):
        super().__init__()
        if isinstance(table, nn.Parameter):
            table = embedding(table)
        self.table = table if table is not None else _uniform(size, 2 * dim, -init_range,
                                                              init_range)
        self.dim = self.table.embedding_dim // 2

    def forward(self, ids):
        e = _clamp_positive(self.table(ids))
        return BetaRegion(e[..., :self.dim], e[..., self.dim:])

    def flat(self):
        return _clamp_positive(self.table.weight.detach())


class ConePoints(_Entities):
    """An entity is a sector of aperture zero: the axis ``tanh(x / range) * pi`` (ConE)."""

    region = SectorCone
    part = "sector cone"

    def __init__(self, size=None, dim=None, table=None, angle_range=0.1):
        super().__init__()
        if isinstance(table, nn.Parameter):
            table = embedding(table)
        self.table = table if table is not None else _uniform(size, dim, -angle_range,
                                                              angle_range)
        self.angle_range = angle_range

    def forward(self, ids):
        axis = to_axis(self.table(ids) / self.angle_range)
        return SectorCone(axis, th.zeros_like(axis))

    def flat(self):
        return to_axis(self.table.weight.detach() / self.angle_range)


# ---------------------------------------------------------------------------------------
# relation models
# ---------------------------------------------------------------------------------------

class _QueryRelations(Relations):
    """Relation models with an ``image``; ``preimage`` follows the inverse relation if
    ``inverse_of`` (a function from relation ids to the ids of their inverses) is given,
    else the model's own derived inverse."""

    inverse_of = None

    def preimage(self, ids, region):
        if self.inverse_of is not None:
            return self.image(self.inverse_of(ids), region)
        return self._derived_preimage(ids, region)

    def _derived_preimage(self, ids, region):
        raise ConstructorUnavailable(
            f"{type(self).__name__} has no inverse of its own: give inverse_of (the rows "
            f"of the inverse relations).")

    def flat(self):
        """One flat vector per relation, for inspection."""
        return self.vectors.weight.detach()


class BetaNetwork(_QueryRelations):
    """A relation is an embedding and a network: the projected Beta distribution is a
    function of the distribution and the relation embedding (BetaE)."""

    part = "neural network"
    accepts = (BetaRegion,)

    def __init__(self, size=None, embed_dim=None, hidden=128, layers=1, tables=None,
                 inverse_of=None, init_range=0.1):
        super().__init__()
        if tables is not None:
            self.vectors = embedding(tables.relation_embedding)
            self.proj_in, self.proj_hidden, self.proj_out = (
                tables.proj_in, tables.proj_hidden, tables.proj_out)
            self.dim = tables.dim
        else:
            self.dim = embed_dim
            self.vectors = _uniform(size, embed_dim, -init_range, init_range)
            self.proj_in = nn.Linear(3 * embed_dim, hidden)
            self.proj_hidden = nn.ModuleList([nn.Linear(hidden, hidden)
                                              for _ in range(layers - 1)])
            self.proj_out = nn.Linear(hidden, 2 * embed_dim)
            for layer in (self.proj_in, self.proj_out, *self.proj_hidden):
                nn.init.xavier_uniform_(layer.weight)
        self.inverse_of = inverse_of

    def image(self, ids, region):
        x = th.cat([region.alpha, region.beta, self.vectors(ids)], -1)
        x = F.relu(self.proj_in(x))
        for layer in self.proj_hidden:
            x = F.relu(layer(x))
        x = _clamp_positive(self.proj_out(x))
        return BetaRegion(x[..., :self.dim], x[..., self.dim:])


class _ConeRelations(_QueryRelations):
    """A relation is a pair of angles per coordinate (an axis and an aperture)."""

    def __init__(self, size=None, embed_dim=None, axes=None, apertures=None,
                 angle_range=0.1, inverse_of=None):
        super().__init__()
        if isinstance(axes, nn.Parameter):
            axes = embedding(axes)
        if isinstance(apertures, nn.Parameter):
            apertures = embedding(apertures)
        self.vectors = axes if axes is not None else _uniform(
            size, embed_dim, -angle_range, angle_range)
        self.apertures = apertures if apertures is not None else _uniform(
            len(self), self.vectors.embedding_dim, -angle_range, angle_range)
        self.angle_range = angle_range
        self.dim = self.vectors.embedding_dim
        self.inverse_of = inverse_of

    def angles(self, ids):
        """``(axis, aperture)`` of the relations ``ids``."""
        return (to_axis(self.vectors(ids) / self.angle_range),
                to_axis(self.apertures(ids) / self.angle_range))

    def flat(self):
        ids = th.arange(len(self), device=self.vectors.weight.device)
        return th.cat(self.angles(ids), -1).detach()


class ConeNetwork(_ConeRelations):
    """ConE: the projected cone is a network of the sum of the cone's and the relation's
    angles."""

    part = "neural network"
    accepts = (SectorCone,)

    def __init__(self, size=None, embed_dim=None, hidden=128, layers=1, tables=None,
                 angle_range=0.1, inverse_of=None):
        if tables is not None:
            super().__init__(axes=tables.axis_embedding, apertures=tables.aperture_embedding,
                             angle_range=tables.embedding_range, inverse_of=inverse_of)
            self.proj_in, self.proj_hidden, self.proj_out = (
                tables.proj_in, tables.proj_hidden, tables.proj_out)
        else:
            super().__init__(size, embed_dim, angle_range=angle_range, inverse_of=inverse_of)
            self.proj_in = nn.Linear(2 * embed_dim, hidden)
            self.proj_hidden = nn.ModuleList([nn.Linear(hidden, hidden)
                                              for _ in range(layers - 1)])
            self.proj_out = nn.Linear(hidden, 2 * embed_dim)
            for layer in (self.proj_in, self.proj_out, *self.proj_hidden):
                nn.init.xavier_uniform_(layer.weight)

    def image(self, ids, region):
        r_axis, r_aperture = self.angles(ids)
        x = th.cat([region.axis + r_axis, region.aperture + r_aperture], -1)
        x = F.relu(self.proj_in(x))
        for layer in self.proj_hidden:
            x = F.relu(layer(x))
        x = self.proj_out(x)
        return SectorCone(to_axis(x[..., :self.dim]), to_aperture(x[..., self.dim:]))


class ConeRotation(_ConeRelations):
    """AConE: a relation rotates a cone by the angle :math:`\\theta_r` per coordinate and
    changes its aperture by :math:`\\delta_r`, by adding its angles to the axis and the
    aperture (multiplication of unit complex numbers).

    As a set of pairs of points (angles per coordinate): :math:`(x, y)` is in the relation
    iff :math:`y` lies within :math:`|\\delta_r|` of :math:`x + \\theta_r` on the circle, in
    every coordinate (:meth:`related`), a rotation with a tolerance. Its image of a sector is
    the sector rotated by :math:`\\theta_r` and widened by :math:`|\\delta_r|`, its preimage
    (the region of :math:`\\exists r.C`, no second relation needed) the sector rotated by
    :math:`-\\theta_r` and also widened by :math:`|\\delta_r|`.

    :param aperture: ``"clip"`` (default): the aperture stays in ``[0, pi]``. ``"wrap"``: it
        is wrapped to ``(-pi, pi]`` as in the published AConE (negative apertures).
    :param aperture_change: ``"tolerance"`` (default): image and preimage widen by
        :math:`|\\delta_r|`, as above. ``"signed"``: the image adds :math:`\\delta_r` (it
        narrows a cone where :math:`\\delta_r < 0`, as the published AConE projection does)
        and the preimage subtracts it; no set of pairs of points has these as image and
        preimage (the preimage of a cone is then *narrower* than the cone, also for
        :math:`\\delta_r > 0`).

    Part (vocabulary value): ``rotation``. Used by: ``acone``, ``cone_qe``.
    """

    part = "rotation"
    accepts = (SectorCone,)

    def __init__(self, size=None, embed_dim=None, tables=None, angle_range=0.1,
                 inverse_of=None, aperture="clip", aperture_change="tolerance"):
        if aperture_change not in ("tolerance", "signed"):
            raise ValueError("aperture_change must be 'tolerance' or 'signed'.")
        self.aperture_change = aperture_change
        if tables is not None:
            super().__init__(axes=tables.axis_embedding, apertures=tables.aperture_embedding,
                             angle_range=tables.embedding_range, inverse_of=inverse_of)
        else:
            super().__init__(size, embed_dim, angle_range=angle_range, inverse_of=inverse_of)
        if aperture not in ("clip", "wrap"):
            raise ValueError("aperture must be 'clip' or 'wrap'.")
        self.aperture = aperture

    def _change(self, ids):
        d_axis, d_aperture = self.angles(ids)
        return d_axis, (d_aperture if self.aperture_change == "signed" else d_aperture.abs())

    def related(self, ids, x, y):
        """Whether :math:`y` lies within :math:`|\\delta_r|` of :math:`x + \\theta_r` in
        every coordinate (``aperture_change="tolerance"``), ``(B, P)``."""
        if self.aperture_change != "tolerance":
            raise ConstructorUnavailable(
                "With aperture_change='signed' the rotation is not a set of pairs of points.")
        d_axis, d_aperture = self._change(ids)
        turn = y.unsqueeze(0) - x.unsqueeze(0) - d_axis.unsqueeze(1)
        gap = th.abs(th.atan2(th.sin(turn), th.cos(turn)))
        return (gap <= d_aperture.unsqueeze(1) + 1e-6).all(dim=-1)

    def image(self, ids, region):
        d_axis, d_aperture = self._change(ids)
        return region.rotate(d_axis, d_aperture, aperture=self.aperture)

    def _derived_preimage(self, ids, region):
        d_axis, d_aperture = self._change(ids)
        if self.aperture_change == "signed":
            d_aperture = -d_aperture
        return region.rotate(-d_axis, d_aperture, aperture=self.aperture)


register_relation_model("angle rotation")(ConeRotation)


# ---------------------------------------------------------------------------------------
# intersection operators: a list of at least two regions to one
# ---------------------------------------------------------------------------------------

class _Intersection(nn.Module):
    #: Whether the result is the set intersection of the regions.
    exact = False
    #: The region class the operator reads (``None``: any).
    region = None

    def gap(self, regions):
        """Cost ``(B, 1)`` that is zero iff the intersection is not empty; a learned
        operator never reports an empty result."""
        return th.zeros_like(as_state(regions[0])[0][..., :1])


class BoxAttentionIntersection(_Intersection):
    """Query2Box: the centre is an attention-weighted average of the centres (weights from
    a one-layer network), the offset the minimum of the offsets times a sigmoid gate
    (DeepSets). Not the set intersection."""

    region = Box

    def __init__(self, dim=None, tables=None):
        super().__init__()
        if tables is not None:
            self.center_layer1, self.center_layer2 = tables.center_layer1, tables.center_layer2
            self.offset_layer1, self.offset_layer2 = tables.offset_layer1, tables.offset_layer2
        else:
            self.center_layer1, self.center_layer2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.offset_layer1, self.offset_layer2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
            for layer in (self.center_layer1, self.center_layer2, self.offset_layer1,
                          self.offset_layer2):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, regions):
        centers = th.stack([r.centre for r in regions])
        offsets = th.stack([r.offset for r in regions])
        attention = F.softmax(self.center_layer2(F.relu(self.center_layer1(centers))), dim=0)
        gate = th.sigmoid(self.offset_layer2(F.relu(self.offset_layer1(offsets)).mean(0)))
        return Box((attention * centers).sum(0), offsets.min(0)[0] * gate)


class BoxExactIntersection(_Intersection):
    """The set intersection of boxes: the largest lower and the smallest upper bound per
    coordinate. An empty intersection has offset zero in the coordinates where it is empty
    (the box degenerates to a point there) and :meth:`gap` is positive. No parameters."""

    exact = True
    region = Box

    def forward(self, regions):
        lower = th.stack([r.lower for r in regions]).max(0)[0]
        upper = th.stack([r.upper for r in regions]).min(0)[0]
        return Box((lower + upper) / 2, F.relu(upper - lower) / 2)

    def gap(self, regions):
        lower = th.stack([r.lower for r in regions]).max(0)[0]
        upper = th.stack([r.upper for r in regions]).min(0)[0]
        return th.linalg.norm(F.relu(lower - upper), dim=-1, keepdim=True)


class BetaAttentionIntersection(_Intersection):
    """BetaE: the attention-weighted average of the parameters."""

    region = BetaRegion

    def __init__(self, dim=None, tables=None):
        super().__init__()
        if tables is not None:
            self.int_layer1, self.int_layer2 = tables.int_layer1, tables.int_layer2
        else:
            self.int_layer1 = nn.Linear(2 * dim, 2 * dim)
            self.int_layer2 = nn.Linear(2 * dim, dim)
            for layer in (self.int_layer1, self.int_layer2):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, regions):
        alpha = th.stack([r.alpha for r in regions])
        beta = th.stack([r.beta for r in regions])
        attention = F.softmax(
            self.int_layer2(F.relu(self.int_layer1(th.cat([alpha, beta], -1)))), dim=0)
        return BetaRegion((attention * alpha).sum(0), (attention * beta).sum(0))


class BetaMeanIntersection(_Intersection):
    """The plain average of the parameters of the distributions: BetaE's intersection with
    equal attention weights and no parameters (an ablation of BetaE's attention network).
    Not the set intersection."""

    region = BetaRegion

    def forward(self, regions):
        alpha = th.stack([r.alpha for r in regions])
        beta = th.stack([r.beta for r in regions])
        return BetaRegion(alpha.mean(0), beta.mean(0))


class ConeAttentionIntersection(_Intersection):
    """ConE: the axis is the angle of the attention-weighted average of the unit vectors
    of the axes (SemanticAverage), the aperture the minimum of the apertures times a gate
    (CardMin)."""

    region = SectorCone

    def __init__(self, dim=None, dropout=0.0, tables=None):
        super().__init__()
        if tables is not None:
            self.axis_layer1, self.axis_layer2 = tables.axis_layer1, tables.axis_layer2
            self.aperture_layer1, self.aperture_layer2 = (tables.aperture_layer1,
                                                          tables.aperture_layer2)
            self.dropout = tables.dropout
        else:
            self.axis_layer1, self.axis_layer2 = nn.Linear(2 * dim, dim), nn.Linear(dim, dim)
            self.aperture_layer1 = nn.Linear(2 * dim, dim)
            self.aperture_layer2 = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)
            for layer in (self.axis_layer1, self.aperture_layer1, self.axis_layer2,
                          self.aperture_layer2):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, regions):
        axes = th.stack([r.axis for r in regions])
        apertures = th.stack([r.aperture for r in regions])
        feats = th.cat([axes - apertures, axes + apertures], -1)
        attention = F.softmax(self.axis_layer2(F.relu(self.axis_layer1(feats))), dim=0)
        axis = th.atan2((attention * th.sin(axes)).sum(0), (attention * th.cos(axes)).sum(0))
        gate = th.sigmoid(self.aperture_layer2(F.relu(self.aperture_layer1(feats)).mean(0)))
        return SectorCone(axis, self.dropout(apertures).min(0)[0] * gate)


# ---------------------------------------------------------------------------------------
# negation operators
# ---------------------------------------------------------------------------------------

class BetaInversion(nn.Module):
    """BetaE negation: ``alpha, beta`` to ``1/alpha, 1/beta``.

    The construction is exact for the model (the inverted density is the complement the
    method defines): ``exact = True``, which ``geometry.parts["negation"]`` reports.
    """

    exact = True
    region = BetaRegion

    def forward(self, region):
        return region.invert()


class ConeComplement(nn.Module):
    """ConE negation: the closure of the complement of every sector, with the absolute
    aperture; ``signed=True`` is the published ``pi - aperture``. It is taken per coordinate,
    not the complement of the product of sectors: ``exact = False`` (``approximate``)."""

    exact = False
    region = SectorCone

    def __init__(self, signed=False):
        super().__init__()
        self.signed = bool(signed)

    def forward(self, region):
        return region.complement(self.signed)


# ---------------------------------------------------------------------------------------
# distances: query region (B, 1, d) against entity regions (B, K, d), result (B, K)
# ---------------------------------------------------------------------------------------

class BoxDistance(nn.Module):
    """Query2Box: the L1 distance of the point to the box plus ``cen`` times the L1
    distance of the point to the box's centre, cut at the box (out-of-box and in-box
    parts)."""

    part = "point-to-region distance"
    region = Box

    def __init__(self, center_weight=0.02, norm=1):
        super().__init__()
        if norm not in (1, 2):
            raise ValueError("norm must be 1 (Query2Box) or 2.")
        self.cen = center_weight
        self.norm = norm

    def forward(self, query, entity):
        delta = (entity.centre - query.centre).abs()
        outside, inside = F.relu(delta - query.offset), th.min(delta, query.offset)
        if self.norm == 1:
            return outside.sum(-1) + self.cen * inside.sum(-1)
        return th.sqrt((outside ** 2).sum(-1) + 1e-12) + self.cen * th.sqrt(
            (inside ** 2).sum(-1) + 1e-12)


class BetaDistance(nn.Module):
    """BetaE: ``KL(entity || query)`` summed over the coordinates; ``reverse=True``:
    ``KL(query || entity)``."""

    part = "divergence"
    region = BetaRegion

    def __init__(self, reverse=False):
        super().__init__()
        self.reverse = bool(reverse)

    def forward(self, query, entity):
        if self.reverse:
            return beta_kl(query.alpha, query.beta, entity.alpha, entity.beta).sum(-1)
        return beta_kl(entity.alpha, entity.beta, query.alpha, query.beta).sum(-1)


class ConeDistance(nn.Module):
    """ConE: chord distance of the entity's axis to the cone's boundary (zero inside) plus
    ``cen`` times the distance to the axis cut at the boundary, scaled by a learnable
    modulus."""

    part = "angle"
    region = SectorCone

    def __init__(self, center_weight=0.02, modulus=None, angle_range=0.1):
        super().__init__()
        self.cen = center_weight
        self.modulus = modulus if modulus is not None else nn.Parameter(
            th.tensor(0.5 * angle_range))

    def forward(self, query, entity):
        axis, aperture, e = query.axis, query.aperture, entity.axis
        lower, upper = axis - aperture, axis + aperture
        to_axis_ = th.abs(th.sin((e - axis) / 2))
        base = th.abs(th.sin(aperture / 2))
        outside = th.min(th.abs(th.sin((e - lower) / 2)), th.abs(th.sin((e - upper) / 2)))
        outside = th.where(to_axis_ < base, th.zeros_like(outside), outside)
        inside = th.min(to_axis_, base)
        return self.modulus * (outside.sum(-1) + self.cen * inside.sum(-1))


__all__ += ["from_state", "wrap_angle"]
