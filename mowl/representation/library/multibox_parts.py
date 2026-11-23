"""Parts of the box-with-bump and multi-box family (group ``multibox``).

Regions
    :class:`BumpedBox` (a box that carries the bump vector of its class),
    :class:`UnionOfBoxes` (a class that is a union of ``k`` boxes), and, for sampled
    measures, :class:`MembershipIntersection`, :class:`DomainProjection` and
    :class:`ProductRegion` (regions known by a point-membership function).

Class models
    :class:`BumpedBoxClasses` (Box2EL: a box and a bump vector), :class:`UnionClasses`
    (MultiBoxEL, MultiELBE: ``k`` boxes, with or without a bump vector) and
    :class:`PackedUnionClasses` (the single ``(N, n, 2d)`` parameter of the volume-based
    multibox).

Relation models
    :class:`HeadTailRelations` (a head box and a tail box; the bump vectors of the
    arguments move the boxes) and :class:`PairUnions` (a union of boxes over pairs).

Measures
    :class:`UnionMeasure` lifts any box measure to unions of boxes, with the hard or the soft
    (:func:`soft_minimum`, temperature ``tau``) aggregation over the components;
    :class:`SampledVolume` scores every relation as a volume ratio estimated on sampled points
    and works for every region that has a point-membership function
    (:func:`membership_sampler`, :func:`box_membership`, :func:`sobol_points`).
"""

import math
from dataclasses import dataclass, replace

import torch as th
import torch.nn as nn
from torch.quasirandom import SobolEngine
from torch.utils.checkpoint import checkpoint

from mowl.representation.class_models import BoxClasses, Classes
from mowl.representation.measures import Bounds, RegionMeasure
from mowl.representation.methods import (DEFAULT_MEASURE, register_class_model, register_measure,
                                         register_relation_model)
from mowl.representation.regions import (Ball, Box, ConstructorNotImplemented, ConstructorUnavailable,
                                         Region)
from mowl.representation.relation_models import Relations

__all__ = ["BumpedBox", "UnionOfBoxes", "MembershipIntersection", "DomainProjection",
           "ProductRegion", "BumpedBoxClasses", "UnionClasses", "PackedUnionClasses",
           "HeadTailRelations", "PairUnions", "UnionMeasure", "SampledVolume",
           "check_temperature", "soft_minimum", "box_membership", "membership_sampler",
           "sobol_points"]


# ====================================================================================
# soft aggregation
# ====================================================================================

def check_temperature(tau, name="tau"):
    """``tau`` as a float: a positive finite number (or a one-element tensor of one)."""
    if isinstance(tau, th.Tensor) and tau.numel() == 1 and tau.dtype != th.bool:
        value = float(tau.detach())
    elif isinstance(tau, (int, float)) and not isinstance(tau, bool):
        value = float(tau)
    else:
        raise ValueError(f"{name} must be a positive number, got {tau!r}")
    if not (value > 0.0 and math.isfinite(value)):
        raise ValueError(f"{name} must be > 0 and finite, got {tau!r}")
    return value


def soft_minimum(values, tau, dim=-1):
    """The soft minimum :math:`-\\tau \\log \\frac{1}{K} \\sum_j e^{-v_j / \\tau}` along
    ``dim``: between the minimum and the mean of the values, the minimum as
    :math:`\\tau \\to 0`, the value itself where there is one; its gradient is the soft argmin
    :math:`\\operatorname{softmax}(-v / \\tau)`. Computed around the minimum, so that no
    exponential overflows."""
    t = check_temperature(tau)
    if values.shape[dim] == 1:
        return values.squeeze(dim)
    low = values.amin(dim=dim, keepdim=True)
    spread = th.exp((low - values) / t).mean(dim=dim, keepdim=True)
    return (low - t * th.log(spread)).squeeze(dim)


# ====================================================================================
# regions
# ====================================================================================

@dataclass(frozen=True)
class BumpedBox(Box):
    """A box with the bump vector of its class (Box2EL). Only a relation model that moves
    its boxes by the bump of an argument reads it; for every measure it is a
    :class:`~mowl.representation.Box`.

    :param bump: ``(B, d)``, or ``None`` (no bump, as a zero bump).
    """

    bump: th.Tensor = None


@dataclass(frozen=True)
class UnionOfBoxes(Region):
    """The union of ``k`` boxes per class. All boxes are there (no mask); the intersection of
    two unions keeps every pair of boxes, each with its distance, and the cost is the distance
    of the nearest pair. Where boxes can be inactive (defined classes, queries of the grounded
    model) the region is
    :class:`~mowl.representation.library.other_regions.MaskedUnionOfBoxes`, a different
    region with the exact set intersection. Used by: ``multiboxel``, ``multielbe``,
    ``mcmultibox``.

    :param centre: ``(B, k, d)``.
    :param offset: ``(B, k, d)``, non-negative half-widths.
    :param bump: ``(B, d)`` bump vector of the class (shared by all its boxes), or ``None``.
    """

    space = "euclidean"

    centre: th.Tensor
    offset: th.Tensor
    bump: th.Tensor = None

    #: A union of boxes is made of boxes: a relation model that maps boxes maps it box by
    #: box (:meth:`~mowl.representation.Relations.maps`).
    component = Box

    @property
    def components(self):
        return self.centre.shape[1]

    def flat_components(self):
        """``(boxes, k)``: the ``B * k`` boxes of the unions, row by row, and ``k``."""
        batch, k, d = self.centre.shape
        return Box(self.centre.reshape(-1, d), self.offset.reshape(-1, d)), k

    def from_components(self, boxes):
        """The unions of ``boxes`` (``B * k`` of them, as :meth:`flat_components` gives
        them); the bump of a mapped union is that of its class (none)."""
        batch, k, d = self.centre.shape
        return UnionOfBoxes(boxes.centre.reshape(batch, k, -1),
                            boxes.offset.reshape(batch, k, -1))

    def contains(self, points):
        """Whether the points ``(P, d)`` lie in some box of the union, ``(B, P)``."""
        batch, k, d = self.centre.shape
        boxes = Box(self.centre.reshape(-1, d), self.offset.reshape(-1, d))
        return boxes.contains(points).reshape(batch, k, -1).any(dim=1)

    def is_empty(self):
        """Every box of the union is empty (negative half-width in some coordinate)."""
        return (self.offset < 0).any(dim=-1).all(dim=-1)

    #: The union volume sums over the subsets of the boxes: at most this many boxes.
    MAX_EXACT_VOLUME_BOXES = 10

    def volume(self):
        """``(B, 1)``: the exact volume of the union by inclusion and exclusion,
        :math:`\\sum_{\\emptyset \\ne S} (-1)^{|S|+1} \\mathrm{Vol}(\\bigcap_{i \\in S} b_i)`
        (:math:`2^k - 1` intersections of boxes; more than
        :attr:`MAX_EXACT_VOLUME_BOXES` boxes is refused)."""
        from itertools import combinations
        k = self.components
        if k > self.MAX_EXACT_VOLUME_BOXES:
            raise ConstructorNotImplemented(
                f"the volume of a union of {k} boxes (inclusion and exclusion has 2^k terms; "
                f"at most {self.MAX_EXACT_VOLUME_BOXES} boxes are summed).")
        lower, upper = self.centre - self.offset, self.centre + self.offset
        total = th.zeros_like(self.centre[:, 0, :1])
        for size in range(1, k + 1):
            sign = 1.0 if size % 2 else -1.0
            for subset in combinations(range(k), size):
                idx = list(subset)
                lo = lower[:, idx].amax(dim=1)
                hi = upper[:, idx].amin(dim=1)
                total = total + sign * th.relu(hi - lo).prod(dim=-1, keepdim=True)
        return total

    def translate(self, vector):
        return UnionOfBoxes(self.centre + vector.unsqueeze(1), self.offset)

    def grow(self, amount):
        """Every box enlarged by ``amount`` (the Minkowski sum distributes over the union)."""
        if isinstance(amount, th.Tensor) and amount.dim() == 2:
            amount = amount.unsqueeze(1)
        return replace(self, offset=self.offset + amount)

    def contract(self):
        """Every box contracted to its centre."""
        return replace(self, offset=th.zeros_like(self.offset))

    def detach_size(self):
        return replace(self, offset=self.offset.detach())

    def scale(self, factor):
        return UnionOfBoxes(self.centre * factor.unsqueeze(1),
                            self.offset * factor.abs().unsqueeze(1))

    def intersect(self, other):
        """The union of the intersections of all pairs of boxes, and a cost that is zero iff
        at least one pair intersects."""
        other = _union(other)
        a, b = self, other
        batch, ka, d = a.centre.shape
        kb = b.centre.shape[1]
        box, gap = _pair_boxes(a, b).intersect(_pair_partners(a, b))
        region = UnionOfBoxes(box.centre.reshape(batch, ka * kb, d),
                              box.offset.reshape(batch, ka * kb, d))
        return region, gap.reshape(batch, ka * kb).amin(dim=1, keepdim=True)


def _union(region):
    """The region as a union of boxes (a box is a union of one)."""
    if isinstance(region, UnionOfBoxes):
        return region
    if isinstance(region, Box):
        return UnionOfBoxes(region.centre.unsqueeze(1), region.offset.unsqueeze(1),
                            getattr(region, "bump", None))
    raise ConstructorUnavailable(
        f"Expected boxes or unions of boxes, got {type(region).__name__}.")


def _expand(a, b):
    """The boxes of ``a`` and ``b`` as flat ``(B * ka * kb, d)`` tensors of all pairs."""
    batch, ka, d = a.centre.shape
    kb = b.centre.shape[1]
    shape = (batch, ka, kb, d)
    return (a.centre.unsqueeze(2).expand(shape).reshape(-1, d),
            a.offset.unsqueeze(2).expand(shape).reshape(-1, d),
            b.centre.unsqueeze(1).expand(shape).reshape(-1, d),
            b.offset.unsqueeze(1).expand(shape).reshape(-1, d))


def _pair_boxes(a, b):
    ca, oa, _, _ = _expand(a, b)
    return Box(ca, oa)


def _pair_partners(a, b):
    _, _, cb, ob = _expand(a, b)
    return Box(cb, ob)


class _SampledRegion(Region):
    """A region known by a point-membership function: see :func:`membership_sampler`."""

    def __len__(self):
        return self.batch


@dataclass(frozen=True)
class MembershipIntersection(_SampledRegion):
    """The intersection of regions, as the pointwise minimum of their memberships."""

    parts: tuple

    @property
    def batch(self):
        return len(self.parts[0])

    @property
    def dim(self):
        return space_dim(self.parts[0])

    def sampler(self, temperature, aggregate):
        fns = [membership_sampler(p, temperature, aggregate) for p in self.parts]

        def fn(points):
            out = fns[0](points)
            for other in fns[1:]:
                out = th.minimum(out, other(points))
            return out
        return fn

    def intersect(self, other):
        return MembershipIntersection(self.parts + (other,)), \
            th.zeros(self.batch, 1, device=self.parts[0].centre.device)


@dataclass(frozen=True)
class DomainProjection(_SampledRegion):
    """The domain of a union of role boxes restricted to a filler: the domain box of a role box
    counts if its codomain box meets the filler (Aharoni 2024, :math:`\\pi[R \\cap (S
    \\times D)]`); the weight of each role box is a soft overlap gate.

    :param role_centre, role_offset: ``(B, n, 2d)``.
    :param filler: The filler region, a union of boxes in :math:`\\mathbb{R}^d`.
    :param bound: Half-width :math:`b_o` of the space.
    """

    role_centre: th.Tensor
    role_offset: th.Tensor
    filler: Region
    bound: float = 1.0
    #: ``(B,)``: the row of ``role_centre`` of every row (``None``: row ``b`` is row ``b``).
    #: The relation model gives each distinct relation one row, so that the membership in its
    #: domain boxes is computed once however many fillers it comes with.
    rows: th.Tensor = None

    @property
    def batch(self):
        return self.role_centre.shape[0] if self.rows is None else self.rows.shape[0]

    @property
    def dim(self):
        return self.role_centre.shape[-1] // 2

    def _per_row(self, tensor):
        return tensor if self.rows is None else tensor[self.rows]

    def sampler(self, temperature, aggregate):
        filler = _union(self.filler)
        d = self.dim
        # a role box counts as much as its codomain box meets the filler (inside the space)
        role_low = self._per_row(self.role_centre - self.role_offset)[..., d:]
        role_high = self._per_row(self.role_centre + self.role_offset)[..., d:]
        lower = th.maximum(role_low[:, :, None], (filler.centre - filler.offset)[:, None])
        upper = th.minimum(role_high[:, :, None], (filler.centre + filler.offset)[:, None])
        gate = soft_indicator(upper.clamp(max=self.bound) - lower.clamp(min=-self.bound),
                              temperature, aggregate).amax(dim=-1)

        def fn(points):
            domain = box_membership(points, self.role_centre[..., :d], self.role_offset[..., :d],
                                    temperature, aggregate)
            return th.minimum(self._per_row(domain), gate[:, None, :]).amax(dim=-1)
        return fn


@dataclass(frozen=True)
class ProductRegion(_SampledRegion):
    """The product of two regions: a point of the product space is a member iff its first
    part is a member of the first region and its second part of the second."""

    first: Region
    second: Region

    @property
    def batch(self):
        return len(self.first)

    @property
    def dim(self):
        return space_dim(self.first) + space_dim(self.second)

    def sampler(self, temperature, aggregate):
        d = space_dim(self.first)
        f = membership_sampler(self.first, temperature, aggregate)
        g = membership_sampler(self.second, temperature, aggregate)
        return lambda points: th.minimum(f(points[:, :d]), g(points[:, d:]))


def space_dim(region):
    """Dimension of the space a region lives in."""
    if hasattr(region, "dim"):
        return region.dim
    return region.centre.shape[-1]


def _device(region):
    for name in ("centre", "role_centre"):
        if hasattr(region, name):
            return getattr(region, name).device
    return _device(region.parts[0] if hasattr(region, "parts") else region.first)


def soft_indicator(slack, temperature=1.0, aggregate="min"):
    """``relu(agg(sigmoid(slack / T)) - 0.5)`` over the last dimension: positive iff every
    entry of ``slack`` is (``aggregate="min"``); ``"mean"`` is the literal formula of Aharoni
    (2024), positive everywhere for a box of positive width."""
    if aggregate not in ("min", "mean"):
        raise ValueError(f"aggregate must be 'min' or 'mean', got {aggregate!r}.")
    soft = th.sigmoid(slack / temperature)
    inner = soft.amin(dim=-1) if aggregate == "min" else soft.mean(dim=-1)
    return th.relu(inner - 0.5)


def box_membership(points, centre, offset, temperature=1.0, aggregate="min"):
    """The soft membership of every point ``(P, d)`` in every box ``(B, n, d)``, ``(B, P, n)``:
    :func:`soft_indicator` of the distances of the point to the faces of the box (Aharoni
    2024, Section 7.1.1)."""
    gap = points[None, :, None, :] - centre[:, None]
    half = offset[:, None]
    return soft_indicator(th.cat([half - gap, half + gap], dim=-1), temperature, aggregate)


def sobol_points(num_points, dim, bound, seed=0, device="cpu"):
    """A scrambled Sobol point set ``(num_points, dim)`` in :math:`[-b, b]^{dim}`, the same for
    the same seed."""
    unit = SobolEngine(dimension=dim, scramble=True, seed=seed).draw(num_points)
    return (unit.to(th.float32) * 2.0 - 1.0).mul(bound).to(device)


def mean_memberships(fn, points, chunk_size):
    """The means over ``points`` of the memberships ``fn(points) -> (m_1, ..., m_n)`` (each
    ``(B, P)``), chunk by chunk; under gradients each chunk is recomputed in the backward pass
    instead of kept (memory of one chunk)."""
    def sums(chunk):
        return th.stack([m.sum(dim=-1) for m in fn(chunk)])

    total = None
    for chunk in th.split(points, chunk_size):
        part = checkpoint(sums, chunk, use_reentrant=False) if th.is_grad_enabled() \
            else sums(chunk)
        total = part if total is None else total + part
    return tuple(total / points.shape[0])


def membership_sampler(region, temperature=1.0, aggregate="min"):
    """A function from points ``(P, d)`` to the soft memberships ``(B, P)`` of the points in
    the regions of the batch. Defined for balls, boxes, unions of boxes and the regions of
    this module; a new region family only needs a ``sampler(temperature, aggregate)`` method.
    """
    if hasattr(region, "sampler"):
        return region.sampler(temperature, aggregate)
    if isinstance(region, UnionOfBoxes) or isinstance(region, Box):
        union = _union(region)
        return lambda points: box_membership(points, union.centre, union.offset,
                                             temperature, aggregate).amax(dim=-1)
    if isinstance(region, Ball):
        # positive iff the point is inside, like the box membership
        def fn(points):
            dist = th.linalg.norm(points.unsqueeze(0) - region.centre.unsqueeze(1), dim=-1)
            return th.relu(th.sigmoid((region.radius - dist) / temperature) - 0.5)
        return fn
    refusal = ConstructorNotImplemented if getattr(region, "space", None) == "euclidean" \
        else ConstructorUnavailable
    raise refusal(f"{type(region).__name__} has no point-membership function (a `sampler` "
                  f"method).")


# ====================================================================================
# class models
# ====================================================================================

def _unit_rows(table, parts=1, low=-1, high=1):
    """Uniform in ``[low, high]`` (default ``[-1, 1]``), every one of ``parts`` consecutive
    blocks of a row of length one (the initialisation of Box2EL, of its multi-box variants
    and of ELBE)."""
    with th.no_grad():
        nn.init.uniform_(table.weight, a=low, b=high)
        if table.num_embeddings > 0:
            view = table.weight.view(table.num_embeddings, parts, -1)
            view.div_(th.linalg.norm(view, dim=-1, keepdim=True))
    return table


class BumpedBoxClasses(BoxClasses):
    """Box2EL's class: a box (centre, half-widths) and a bump vector, which only the relation
    model uses (:class:`HeadTailRelations`). Everything but the bump is a
    :class:`~mowl.representation.BoxClasses`; the region is a :class:`BumpedBox`.

    :param bumps: An existing table to use instead of a new one.

    Part (vocabulary value): ``box``. Used by: ``box2el``.
    """

    region = BumpedBox
    part = "box"

    def __init__(self, size=None, embed_dim=None, centres=None, offsets=None, bumps=None):
        super().__init__(size, embed_dim, centres=centres, offsets=offsets)
        self.bumps = bumps if bumps is not None else _unit_rows(
            nn.Embedding(len(self), self.embed_dim))

    def forward(self, ids):
        return BumpedBox(self.centres(ids), th.abs(self.offsets(ids)), self.bumps(ids))


class UnionClasses(Classes):
    """A class is a union of ``k`` boxes (MultiBoxEL, MultiELBE). The tables have one row of
    ``k * d`` numbers per class; ``k = 1`` is :class:`~mowl.representation.BoxClasses`.

    :param components: ``k``.
    :param bump: Whether every class has a bump vector (shared by its boxes), for
        :class:`HeadTailRelations`.
    :param centres, offsets, bumps: Existing tables to use instead of new ones.

    Part (vocabulary value): ``union of boxes``. Used by: ``multiboxel``, ``multielbe``.
    """

    region = UnionOfBoxes
    part = "union of boxes"
    exact_conjunction = False

    def __init__(self, size=None, embed_dim=None, components=2, bump=False, centres=None,
                 offsets=None, bumps=None):
        super().__init__()
        if components < 1:
            raise ValueError("components must be at least 1.")
        self.components = int(components)
        self.centres = centres if centres is not None else _unit_rows(
            nn.Embedding(size, self.components * embed_dim), self.components)
        self.offsets = offsets if offsets is not None else _unit_rows(
            nn.Embedding(len(self), self.centres.embedding_dim), self.components)
        if bumps is None and bump:
            bumps = _unit_rows(nn.Embedding(len(self), self.embed_dim))
        self.bumps = bumps

    @property
    def embed_dim(self):
        return self.centres.embedding_dim // self.components

    def forward(self, ids):
        batch = ids.shape[0]
        shape = (batch, self.components, self.embed_dim)
        bump = None if self.bumps is None else self.bumps(ids)
        return UnionOfBoxes(self.centres(ids).view(shape), th.abs(self.offsets(ids)).view(shape),
                            bump)


class PackedUnionClasses(Classes):
    """A class is a union of ``n`` boxes stored as one parameter of shape ``(N, n, 2d)``
    (``[centre, offset]`` of every box), as in the volume-based multibox. Initialised as
    there: centres uniform in :math:`[-b_o, b_o]`, offsets uniform in :math:`[0.4, 0.9] b_o`.

    :param table: An existing parameter to use instead of a new one.
    :param individuals: Whether the table is one of individuals (shape ``(N, 1, 2d)``).
    """

    region = UnionOfBoxes
    part = "union of boxes"
    exact_conjunction = False

    def __init__(self, size=None, embed_dim=None, components=5, bound=1.0, table=None,
                 generator=None):
        super().__init__()
        if table is None:
            table = nn.Parameter(_packed_init(size, components, embed_dim, bound, generator))
        self.centres = table

    def __len__(self):
        return self.centres.shape[0]

    @property
    def components(self):
        return self.centres.shape[1]

    @property
    def embed_dim(self):
        return self.centres.shape[-1] // 2

    def all(self):
        return self(th.arange(len(self), device=self.centres.device))

    def forward(self, ids):
        w = self.centres[ids]
        d = self.embed_dim
        return UnionOfBoxes(w[..., :d], w[..., d:].abs())


def _packed_init(count, boxes, dim, bound, generator):
    centre = (th.rand(count, boxes, dim, generator=generator) * 2 - 1) * bound
    offset = (0.4 + 0.5 * th.rand(count, boxes, dim, generator=generator)) * bound
    return th.cat([centre, offset], dim=-1)


register_class_model("box with bump")(BumpedBoxClasses)
register_class_model("union of boxes")(UnionClasses)


# ====================================================================================
# relation models
# ====================================================================================

class HeadTailRelations(Relations):
    """A relation is a head box and a tail box (Box2EL, MultiBoxEL), given by centre and
    half-width. A pair :math:`(x, y)` is in the relation if :math:`x` shifted by the bump of
    :math:`y` lies in the head box and :math:`y` shifted by the bump of :math:`x` lies in the
    tail box: a statement about *regions* (containment, read by any measure of boxes, balls
    or unions). BoxE has the same geometry but scores a pair of points by a distance and not
    by containment, with its own parametrisation (base, delta, size): that is
    :class:`~mowl.representation.library.other_graph.BoxERelations`
    (``"head and tail boxes (distance score)"``), a different relation model. Used by:
    ``box2el``, ``multiboxel``.

    The bump belongs to the arguments: the relation model reads the ``bump`` field of the
    regions it is given (a :class:`BumpedBox` or :class:`UnionOfBoxes` made by a class model
    with bumps) and treats a region without a bump as having a zero one. So the split is
    clean in the data (the class model produces regions with a bump; the relation model is
    the only reader), and only the training of the bump is shared state: a regulariser on the
    bump table (:func:`~mowl.representation.library.multibox_objective.bump_penalty`).

    * ``preimage(r, D)``: the head box moved by :math:`-b_D`: the region of
      :math:`\\exists r.D`, since :math:`x + b_D \\in H_r`.
    * ``image(r, C)``: the tail box moved by :math:`-b_C`.

    The two together give the inclusion :math:`C \\sqsubseteq \\exists r.D` its two
    conditions (:meth:`subsumed`). The result is a region of the family of the filler (a
    union of one box for unions, a ball for ``shape="ball"``).

    :param shape: ``"box"`` (default) or ``"ball"`` (head and tail are balls: not Box2EL, to
        use the relation model with ball classes).
    :param head_centres, head_offsets, tail_centres, tail_offsets: Existing tables.
    """

    part = "head and tail boxes"
    #: Not a set of pairs of points: whether :math:`(x, y)` is related depends on the bumps
    #: of the *classes* of :math:`x` and :math:`y`, so :meth:`related` is not defined; the
    #: relation is read on regions (:meth:`subsumed`).
    accepts = (Box, Ball, UnionOfBoxes)

    def __init__(self, size=None, embed_dim=None, shape="box", head_centres=None,
                 head_offsets=None, tail_centres=None, tail_offsets=None):
        super().__init__()
        if shape not in ("box", "ball"):
            raise ValueError("shape must be 'box' or 'ball'.")
        self.shape = shape
        width = embed_dim if shape == "box" else 1

        def table(given, size_, dim):
            return given if given is not None else _unit_rows(nn.Embedding(size_, dim))

        self.head_centres = table(head_centres, size, embed_dim)
        self.head_offsets = table(head_offsets, size, width)
        self.tail_centres = table(tail_centres, size, embed_dim)
        self.tail_offsets = table(tail_offsets, size, width)

    @property
    def vectors(self):
        return self.head_centres

    def _make(self, like, centre, size):
        if isinstance(like, UnionOfBoxes):
            return UnionOfBoxes(centre.unsqueeze(1), size.unsqueeze(1))
        if self.shape == "ball":
            return Ball(centre, size, size)
        return Box(centre, size)

    def head(self, ids, like=None):
        """The head boxes (balls) of the relations ``ids``."""
        return self._make(like, self.head_centres(ids), th.abs(self.head_offsets(ids)))

    def tail(self, ids, like=None):
        """The tail boxes (balls) of the relations ``ids``."""
        return self._make(like, self.tail_centres(ids), th.abs(self.tail_offsets(ids)))

    @staticmethod
    def _bump(region):
        bump = getattr(region, "bump", None)
        return 0.0 if bump is None else bump

    def preimage(self, ids, region):
        return self._make(region, self.head_centres(ids) - self._bump(region),
                          th.abs(self.head_offsets(ids)))

    def image(self, ids, region):
        return self._make(region, self.tail_centres(ids) - self._bump(region),
                          th.abs(self.tail_offsets(ids)))

    def subsumed(self, measure, sub, ids, filler, margin):
        """Cost of :math:`C \\sqsubseteq \\exists r.D`: the mean of the inclusion of
        :math:`C` in the preimage of :math:`D` and the inclusion of (a component of)
        :math:`D` in the image of :math:`C`."""
        first = measure.inside(sub, self.preimage(ids, filler), margin)
        second = _some(measure)(filler, self.image(ids, sub), margin)
        return (first + second) / 2

    def role_inclusion(self, sub, sup, measure, margin=0.0):
        """:math:`r \\sqsubseteq s`: head in head, tail in tail (the mean of the two)."""
        head = measure.inside(self.head(sub), self.head(sup), margin)
        tail = measure.inside(self.tail(sub), self.tail(sup), margin)
        return (head + tail) / 2

    def role_chain(self, first, second, sup, measure, margin=0.0):
        """:math:`r_1 \\circ r_2 \\sqsubseteq s`: the head of :math:`r_1` in the head of
        :math:`s` and the tail of :math:`r_2` in the tail of :math:`s`."""
        head = measure.inside(self.head(first), self.head(sup), margin)
        tail = measure.inside(self.tail(second), self.tail(sup), margin)
        return (head + tail) / 2

    def subsumed_negative(self, measure, sub, ids, filler, margin, delta):
        """The negative loss of Box2EL: :math:`(\\delta - \\mu(C + b_D, H_r))^2 +
        (\\delta - \\mu(D + b_C, T_r))^2` with :math:`\\mu` the distance of the
        two boxes (``meets`` with margin ``margin``); the mean over components for unions."""
        first = _each(measure)(sub, self.preimage(ids, filler), margin)
        second = _each(measure)(filler, self.image(ids, sub), margin)
        return _mean_over(sub, (delta - first).square()) \
            + _mean_over(filler, (delta - second).square())


def _mean_over(region, values):
    """The mean over the components of ``region`` of ``values`` ``(B, k)``: over the slots
    that count where the region says which (``gate``: structural slots; zero if none does)."""
    gate = getattr(region, "gate", None)
    if gate is None:
        return values.mean(dim=1, keepdim=True)
    total = th.where(gate, values, th.zeros_like(values)).sum(dim=1, keepdim=True)
    return total / gate.sum(dim=1, keepdim=True).clamp_min(1)


def _some(measure):
    return getattr(measure, "inside_exists", measure.inside)


def _each(measure):
    """Per component of the first argument: ``(B, k)``; for a measure of single regions
    ``(B, 1)``."""
    return getattr(measure, "meets_each", measure.meets)


class PairUnions(Relations):
    """A relation is a union of ``n`` boxes in the space of pairs :math:`\\mathbb{R}^{2d}`
    (first ``d`` coordinates: domain, last ``d``: codomain), stored as one parameter of shape
    ``(R, n, 4d)`` (``[centre, offset]``), as in the volume-based multibox.

    * ``preimage(r, D)``: the domain of the role boxes whose codomain meets :math:`D`
      (:class:`DomainProjection`): the region of :math:`\\exists r.D`.
    * :meth:`region`, :meth:`domain`, :meth:`codomain`: the relation as a region of pairs, and
      its two projections (the role boxes projected on the domain resp. codomain).

    :param bound: :math:`b_o`, the half-width of the space in which the gate of the
        projection clips the boxes.
    :param table: An existing parameter to use instead of a new one.

    Part (vocabulary value): ``region over pairs``. Used by: ``mcmultibox``.
    """

    part = "region over pairs"
    accepts = (Box, UnionOfBoxes)

    def __init__(self, size=None, embed_dim=None, components=5, bound=1.0, table=None,
                 generator=None):
        super().__init__()
        self.bound = float(bound)
        if table is None:
            table = nn.Parameter(_packed_init(size, components, 2 * embed_dim, bound,
                                              generator))
        self.vectors = table

    def __len__(self):
        return self.vectors.shape[0]

    @property
    def embed_dim(self):
        return self.vectors.shape[-1] // 4

    def _boxes(self, ids):
        w = self.vectors[ids]
        k = w.shape[-1] // 2
        return w[..., :k], w[..., k:].abs()

    def region(self, ids):
        """The relations ``ids`` as unions of boxes in the space of pairs."""
        centre, offset = self._boxes(ids)
        return UnionOfBoxes(centre, offset)

    def domain(self, ids):
        centre, offset = self._boxes(ids)
        d = self.embed_dim
        return UnionOfBoxes(centre[..., :d], offset[..., :d])

    def codomain(self, ids):
        centre, offset = self._boxes(ids)
        d = self.embed_dim
        return UnionOfBoxes(centre[..., d:], offset[..., d:])

    def related(self, ids, x, y):
        """Whether the pair :math:`(x, y)` lies in the union of boxes of the relation,
        ``(B, P)``. (:meth:`preimage` is the soft projection of the volume-based multibox,
        an approximation of :math:`\\exists r.D` that only the sampled measure reads.)"""
        centre, offset = self._boxes(ids)
        pairs = th.cat([x, y], dim=-1)
        batch, k, width = centre.shape
        boxes = Box(centre.reshape(-1, width), offset.reshape(-1, width))
        return boxes.contains(pairs).reshape(batch, k, -1).any(dim=1)

    def preimage(self, ids, region):
        roles, rows = th.unique(ids, return_inverse=True)
        centre, offset = self._boxes(roles)
        return DomainProjection(centre, offset, region, self.bound, rows)

    def role_inclusion(self, sub, sup, measure, margin=0.0):
        """:math:`r \\sqsubseteq s`: the region of pairs of :math:`r` inside that of
        :math:`s`."""
        return measure.inside(self.region(sub), self.region(sup))

    def role_chain(self, first, second, sup, measure, margin=0.0):
        """The simplified chain loss of the thesis (not fully semantic): the domain of
        :math:`r_1` times the codomain of :math:`r_2` inside :math:`s`."""
        both = ProductRegion(self.domain(first), self.codomain(second))
        return measure.inside(both, self.region(sup))


register_relation_model("head and tail boxes")(HeadTailRelations)
register_relation_model("union of boxes over pairs")(PairUnions)


# ====================================================================================
# measures
# ====================================================================================

class UnionMeasure(RegionMeasure):
    """A measure of unions of boxes, built from a measure of boxes (``base``, default
    :class:`~mowl.representation.Bounds`).

    The cost of a relation between two unions is the cost of the relation between pairs of
    boxes, aggregated:

    * ``inside(A, B)``: every component :math:`A_i` has to be in *some*
      component of :math:`B`, so :math:`\\operatorname{mean}_i \\operatorname{agg}_j
      \\mathrm{cost}(A_i, B_j)`; ``agg`` is the minimum (``aggregation="hard"``) or the
      soft minimum with temperature ``tau`` (``"soft"``).
    * ``meets(A, B)``: with ``meets_reading="some"`` (default) *some* component of :math:`A`
      meets *some* component of :math:`B` (:math:`\\operatorname{agg}_{i,j}`: zero exactly
      when the unions overlap); ``"every"``: every component of :math:`A` meets some
      component of :math:`B` (the mean over :math:`i` as for ``inside``: a sufficient
      condition, which the published MultiBoxEL and MultiELBE losses use).
    * ``apart(A, B)``: every pair has to be apart, the mean over all pairs.
    * ``empty(A)``: the mean over the components.
    * ``intersection_inside``: the designated pair of components (the pair that intersects
      most) pays for the emptiness, all pairs that intersect pay for the inclusion in a
      component of the target (the gated loss of MultiBoxEL and MultiELBE).

    For one component each of these is the cost of ``base`` exactly. Boxes are accepted and
    read as a union of one.

    Further methods used by the objectives of this module: :meth:`meets_all` (mean over all
    pairs), :meth:`meets_each` (per component of the first argument), :meth:`inside_exists`
    (*some* component of the first argument inside the second).

    :param base: The measure of boxes.
    :param aggregation: ``"hard"`` or ``"soft"``.
    :param tau: Temperature of the soft minimum (a float, or a tensor that is updated in
        place, as the buffer of the modules).
    :param meets_reading: ``"some"`` (default) or ``"every"``, see ``meets``.

    Part (vocabulary value): ``bounds per coordinate``. Used by: ``multiboxel`` and
    ``multielbe`` (``meets_reading="every"``).
    """

    accepts = (UnionOfBoxes, Box)
    part = "bounds per coordinate"
    exact_intersection = False

    def __init__(self, base=None, aggregation="hard", tau=1.0, meets_reading="some"):
        if aggregation not in ("hard", "soft"):
            raise ValueError("aggregation must be 'hard' or 'soft'.")
        if meets_reading not in ("some", "every"):
            raise ValueError("meets_reading must be 'some' or 'every'.")
        self.meets_reading = meets_reading
        self.base = base if base is not None else Bounds()
        self.aggregation = aggregation
        self.tau = tau

    # -- helpers --------------------------------------------------------------------------
    def _pairs(self, function, a, b, margin):
        a, b = _union(a), _union(b)
        batch, ka = a.centre.shape[:2]
        kb = b.centre.shape[1]
        ca, oa, cb, ob = _expand(a, b)
        return function(Box(ca, oa), Box(cb, ob), margin).reshape(batch, ka, kb)

    def _agg(self, x, dim):
        if self.aggregation == "hard":
            return th.amin(x, dim=dim)
        return soft_minimum(x, self.tau, dim=dim)

    # -- the relations --------------------------------------------------------------------
    def inside(self, a, b, margin=0.0):
        return self._agg(self._pairs(self.base.inside, a, b, margin), 2).mean(
            dim=1, keepdim=True)

    def inside_exists(self, a, b, margin=0.0):
        """Some component of ``a`` is inside ``b``."""
        per = self._agg(self._pairs(self.base.inside, a, b, margin), 2)
        return self._agg(per, 1).unsqueeze(1)

    def meets(self, a, b, margin=0.0):
        pairs = self._pairs(self.base.meets, a, b, margin)
        if self.meets_reading == "some":
            return self._agg(self._agg(pairs, 2), 1).unsqueeze(1)
        return self._agg(pairs, 2).mean(dim=1, keepdim=True)

    def meets_all(self, a, b, margin=0.0):
        return self._pairs(self.base.meets, a, b, margin).mean(dim=(1, 2)).unsqueeze(1)

    def meets_each(self, a, b, margin=0.0):
        return self._agg(self._pairs(self.base.meets, a, b, margin), 2)

    def apart(self, a, b, margin=0.0):
        return self._pairs(self.base.apart, a, b, margin).mean(dim=(1, 2)).unsqueeze(1)

    def outside(self, a, b, margin=0.0):
        """Zero only if :math:`A \\not\\subseteq B`, by a sufficient condition: the centre of
        some component of ``a`` lies apart (by ``margin``) from every component of ``b``, so
        a point of ``a`` is in no box of ``b`` (the mean over the components of ``b``, the
        minimum over those of ``a``)."""
        a = _union(a)
        contracted = replace(a, offset=th.zeros_like(a.offset))
        pairs = self._pairs(self.base.apart, contracted, b, margin)
        return pairs.mean(dim=2).amin(dim=1, keepdim=True)

    def empty(self, a, margin=0.0):
        a = _union(a)
        batch, k, d = a.centre.shape
        each = self.base.empty(Box(a.centre.reshape(-1, d), a.offset.reshape(-1, d)))
        return each.reshape(batch, k).mean(dim=1, keepdim=True)

    def nonempty(self, a, epsilon=0.0):
        """Zero iff some component has extent at least ``epsilon``."""
        a = _union(a)
        batch, k, d = a.centre.shape
        each = self.base.nonempty(Box(a.centre.reshape(-1, d), a.offset.reshape(-1, d)),
                                  epsilon)
        return each.reshape(batch, k).amin(dim=1, keepdim=True)

    def intersection_inside(self, a, b, c, margin=0.0):
        """With a base measure that reads the conjunction exactly (``Bounds(conjunction=
        "exact")``, the default): every intersection of a component of ``a`` with one of
        ``b`` lies in some component of ``c`` (empty ones do: a sufficient condition for
        :math:`A \\cap B \\subseteq C`). Otherwise the gated loss of MultiBoxEL and
        MultiELBE described in the class."""
        if getattr(self.base, "conjunction", None) == "exact":
            a, b = _union(a), _union(b)
            batch, ka, d = a.centre.shape
            kb = b.centre.shape[1]
            ca, oa, cb, ob = _expand(a, b)
            both, _ = Box(ca, oa).intersect(Box(cb, ob), signed=True)
            pairs = UnionOfBoxes(both.centre.reshape(batch, ka * kb, d),
                                 both.offset.reshape(batch, ka * kb, d))
            return self.inside(pairs, c, margin)
        a, b, c = _union(a), _union(b), _union(c)
        batch, ka, d = a.centre.shape
        kb, kc = b.centre.shape[1], c.centre.shape[1]
        ca, oa, cb, ob = _expand(a, b)
        lower = th.maximum(ca - oa, cb - ob)
        upper = th.minimum(ca + oa, cb + ob)
        centre = ((lower + upper) / 2).reshape(batch, ka, kb, 1, d)
        offset = (th.abs(upper - lower) / 2).reshape(batch, ka, kb, 1, d)
        empt = th.linalg.norm(th.relu(lower - upper), dim=-1).reshape(batch, ka * kb)
        shape = (batch, ka, kb, kc, d)
        inclusion = self.base.inside(
            Box(centre.expand(shape).reshape(-1, d), offset.expand(shape).reshape(-1, d)),
            Box(c.centre.reshape(batch, 1, 1, kc, d).expand(shape).reshape(-1, d),
                c.offset.reshape(batch, 1, 1, kc, d).expand(shape).reshape(-1, d)),
            margin).reshape(batch, ka, kb, kc)
        non_empty = (lower.detach() <= upper.detach()).all(dim=-1).reshape(batch, ka * kb)
        # the designated pair, which pays for the emptiness: the pair that intersects most
        # (hard: a one-hot weight; soft: the soft argmin of the gaps, temperature tau)
        if self.aggregation == "hard":
            weights = th.zeros_like(empt).scatter_(1, empt.detach().argmin(dim=1,
                                                                            keepdim=True), 1.0)
        else:
            weights = th.softmax(-empt / check_temperature(self.tau), dim=1)
        # every pair that intersects, and the designated one, pays for the inclusion
        gate = th.maximum(non_empty.to(empt.dtype), weights.detach())
        covered = self._agg(inclusion, 3).reshape(batch, ka * kb)
        gated = (gate * covered).sum(dim=1) / gate.sum(dim=1)
        return (gated + (weights * empt).sum(dim=1)).unsqueeze(1)


register_measure("union of bounds", default_for=(UnionOfBoxes,))(UnionMeasure)
DEFAULT_MEASURE[BumpedBox] = Bounds


class SampledVolume(RegionMeasure, nn.Module):
    """Every relation as a volume ratio, estimated on sampled points (Aharoni 2024, Section
    7.1.1): :math:`\\mathrm{inside}(A, B) = 1 - \\mathrm{Vol}(A \\cap B) / \\mathrm{Vol}(A)`.

    It works for every region with a point-membership function
    (:func:`membership_sampler`): balls, boxes, unions of boxes, and the regions of this
    module. Memberships are soft (positive iff the point is inside); the intersection of
    regions is the minimum of their memberships, so conjunctions work for every region, also
    for balls, and are not exact in the sense of :class:`~mowl.representation.Box`.

    ``apart(A, B)`` is the fraction of :math:`A` covered by :math:`B` (zero iff, on the
    sample, they do not meet): the thesis defines no such loss, it is an extension. The
    measure has no margin: a margin other than 0 is refused (``ValueError``); ``meets``,
    ``empty`` and ``nonempty`` are not defined.

    :param points: ``None`` (draw Sobol points when a region of a dimension is first seen),
        or a list of point sets, one per dimension used (``(P, d)`` tensors, kept as buffers).
    :param num_points: Number of points per dimension when they are drawn.
    :param bound: Half-width :math:`b_o` of the sampled cube.
    :param temperature: Temperature of the sigmoid.
    :param aggregate: ``"min"`` (membership positive iff inside) or ``"mean"`` (the literal
        formula of the thesis, positive everywhere).
    :param chunk_size: Points processed at once (memory only).
    :param seed: Seed of the Sobol scrambling.

    Part (vocabulary value): ``sampled volume``. Used by: ``mcmultibox``.
    """

    accepts = (Region,)
    part = "sampled volume"
    exact_intersection = False

    def __init__(self, points=None, num_points=20000, bound=1.0, temperature=1.0,
                 aggregate="min", chunk_size=2048, seed=0, eps=1e-8):
        nn.Module.__init__(self)
        if aggregate not in ("mean", "min"):
            raise ValueError("aggregate must be 'mean' or 'min'.")
        self.num_points = num_points
        self.bound = float(bound)
        self.temperature = float(temperature)
        self.aggregate = aggregate
        self.chunk_size = chunk_size
        self.seed = seed
        self.eps = eps
        self._dims = []
        for p in (points or ()):
            self._add(p)

    def refuses(self, region):
        """Regions without a point-membership function cannot be sampled."""
        try:
            membership_sampler(region, self.temperature, self.aggregate)
        except ConstructorUnavailable as error:
            return str(error)
        return None

    def _add(self, points):
        self.register_buffer(f"points_{points.shape[1]}", points, persistent=False)
        self._dims.append(points.shape[1])

    def points(self, dim, device=None):
        """The point set of dimension ``dim`` (drawn on first use)."""
        if dim not in self._dims:
            self._add(sobol_points(self.num_points, dim, self.bound, self.seed + len(
                self._dims), device or "cpu"))
        return getattr(self, f"points_{dim}")

    def _ratio(self, numerator_region, denominator_region):
        dim = space_dim(denominator_region)
        fn_a = membership_sampler(denominator_region, self.temperature, self.aggregate)
        fn_n = membership_sampler(numerator_region, self.temperature, self.aggregate)
        points = self.points(dim, _device(denominator_region))
        return mean_memberships(lambda p: (fn_n(p), fn_a(p)), points, self.chunk_size)


    def covered(self, a, b):
        """``(B,)``: the fraction :math:`\\mathrm{Vol}(A \\cap B) / \\mathrm{Vol}(A)` of ``a``
        that ``b`` covers (zero where ``a`` has no volume on the sample)."""
        num, den = self._ratio(MembershipIntersection((a, b)), a)
        return num / (den + self.eps)

    def inside(self, a, b, margin=0.0):
        """:math:`1 - \\mathrm{Vol}(A \\cap B)/\\mathrm{Vol}(A)`; with a strict margin, of
        ``a`` enlarged by it (zero iff the enlarged region lies inside ``b`` on the
        sample); a slack is refused."""
        a = self.grown(a, margin, "SampledVolume")
        return (1.0 - self.covered(a, b)).unsqueeze(1)

    def apart(self, a, b, margin=0.0):
        a = self.grown(a, margin, "SampledVolume")
        return self.covered(a, b).unsqueeze(1)

    def intersect(self, a, b):
        """The minimum of the memberships; the gap is zero (a sampled region has no
        emptiness test)."""
        return MembershipIntersection((a, b)), th.zeros(len(a), 1, device=_device(a))

    def intersection_inside(self, a, b, c, margin=0.0):
        return self.inside(MembershipIntersection((a, b)), c, margin)

    def intersection_apart(self, a, b, c, margin=0.0):
        return self.apart(MembershipIntersection((a, b)), c, margin)

    def intersection_nonempty(self, a, b, margin=0.0, epsilon=0.0):
        raise ConstructorNotImplemented("SampledVolume has no measure of emptiness.")

    def meets(self, a, b, margin=0.0):
        raise ConstructorNotImplemented("SampledVolume does not define meets.")

    def empty(self, a, margin=0.0):
        raise ConstructorNotImplemented("SampledVolume does not define empty.")

    def nonempty(self, a, epsilon=0.0):
        raise ConstructorNotImplemented("SampledVolume does not define nonempty.")


register_measure("sampled volume")(SampledVolume)
