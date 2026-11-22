"""Do the measures mean what their names say?

Every measure turns a relation between regions into a cost. These tests check, by sampling
points, that a zero cost means the relation holds in the sets of points the regions are
(:meth:`Region.contains`), with the measure's DEFAULT options:

* ``inside(a, b)`` zero iff every point of ``a`` lies in ``b``; with a positive (strict)
  margin a zero cost implies the inclusion with room (``a`` grown by the margin still lies
  in ``b``); with a negative margin (a slack) the inclusion may be violated by at most the
  margin (``a`` lies in ``b`` grown by it);
* ``apart`` zero iff no point is in both, ``meets`` zero iff some point is in both,
  ``empty`` zero iff no sampled point is in the region (a region of no volume, such as a point
  or a flat box, is empty for a sampled interpretation), ``nonempty(epsilon)`` zero only if
  the region holds a cube (ball) of half-width ``epsilon`` around its centre;
* costs are never negative, and one step of gradient descent on a violated relation lowers
  its cost.

"Zero implies the relation" (soundness) is checked on random regions: the claim of a zero
cost is refuted by a point sampled inside ``a`` (for ``inside``, ``apart``, ``empty``) or by a
witness that has to lie in both regions (for ``meets``). "The relation implies zero"
(completeness) is checked on regions for which it holds by construction: nested, equal,
disjoint, touching, overlapping, empty, a point.

Measures whose zero has no truth condition declare it (``reading = "score"``); measures and
constructors that cannot be exact declare it in the vocabulary (smoothed or sampled volumes,
``exact_intersection = False``); see the table at the end of the report.
"""

import math

import pytest
import torch as th

import mowl.representation as R
from mowl.representation import Ball, Box, CornerBox, ConstructorUnavailable
from mowl.representation.measures import Bounds, CentreDistance
from mowl.representation.library.boxrel_measures import (AveragedBounds, MaskedBounds,
                                                         VolumeShare)
from mowl.representation.library.ballfam_parts import BoundaryDepth
from mowl.representation.library.hierarchy_measures import LogVolumeRatio
from mowl.representation.library.multibox_parts import UnionMeasure, UnionOfBoxes
from mowl.representation.library.query_regions import ConeBounds, SectorCone

D = 2
N_PTS = 400
TOL = 1e-6


def _gen(seed):
    return th.Generator().manual_seed(seed)


# -- sampling ---------------------------------------------------------------------------------

def bbox(region):
    """Lower and upper corners ``(B, d)`` of boxes around every region of the batch."""
    if isinstance(region, Ball):
        r = region.radius
        return region.centre - r, region.centre + r
    if isinstance(region, Box):
        return th.minimum(region.lower, region.upper), th.maximum(region.lower, region.upper)
    if isinstance(region, UnionOfBoxes):
        lo = (region.centre - region.offset.abs()).amin(dim=1)
        hi = (region.centre + region.offset.abs()).amax(dim=1)
        return lo, hi
    if isinstance(region, SectorCone):
        return th.full_like(region.axis, -math.pi), th.full_like(region.axis, math.pi)
    raise TypeError(type(region).__name__)


def points_in(region, n=N_PTS, seed=0):
    """``(B, n, d)`` points sampled in the bounding box of every region, and ``(B, n)``
    whether they lie in it."""
    lo, hi = bbox(region)
    u = th.rand(len(region), n, lo.shape[1], generator=_gen(seed))
    x = lo.unsqueeze(1) + u * (hi - lo).unsqueeze(1)
    return x, x


def member(region, x):
    """``(B, n)``: whether the points ``x[b]`` (``(B, n, d)``) lie in region ``b``."""
    return th.stack([region_row(region, i).contains(x[i])[0] for i in range(len(region))])


def region_row(region, i):
    """Row ``i`` of a batch of regions."""
    from dataclasses import fields, replace
    values = {}
    for f in fields(region):
        v = getattr(region, f.name)
        values[f.name] = v[i:i + 1] if isinstance(v, th.Tensor) and v.dim() > 0 \
            and v.shape[0] == len(region) else v
    return replace(region, **values)


def refuted_inside(a, b, grow_a=0.0):
    """Rows where a point of ``a`` (grown by ``grow_a``) is found outside ``b``."""
    x, _ = points_in(a if not grow_a else a.grow(_amount(a, grow_a)))
    in_a = member(a if not grow_a else a.grow(_amount(a, grow_a)), x)
    in_b = member(b, x)
    return (in_a & ~in_b).any(dim=1)


def refuted_apart(a, b):
    x, _ = points_in(a)
    return (member(a, x) & member(b, x)).any(dim=1)


def refuted_empty(a):
    x, _ = points_in(a)
    return member(a, x).any(dim=1)


def _amount(region, m):
    if isinstance(region, Ball):
        return th.full_like(region.radius, m)
    return th.full_like(region.offset, m)


def witness(a, b):
    """A point that lies in both ``a`` and ``b`` if they meet (boxes, balls)."""
    if isinstance(a, Ball):
        # the middle of the overlap of the two balls on the line through their centres
        diff = b.centre - a.centre
        d = th.linalg.norm(diff, dim=1, keepdim=True).clamp_min(1e-12)
        near = th.maximum(d - b.radius, -a.radius)
        far = th.minimum(a.radius, d + b.radius)
        return a.centre + (near + far) / 2 * diff / d
    lo = th.maximum(a.lower, b.lower)
    hi = th.minimum(a.upper, b.upper)
    return (lo + hi) / 2


def holds_meets(a, b):
    w = witness(a, b)
    rows = []
    for i in range(len(a)):
        point = w[i:i + 1]
        rows.append(region_row(a, i).contains(point)[0, 0] & region_row(b, i).contains(point)[0, 0])
    return th.stack(rows)


# -- regions ----------------------------------------------------------------------------------

def random_balls(n, seed):
    g = _gen(seed)
    return Ball(th.randn(n, D, generator=g), th.rand(n, 1, generator=g) * 1.5)


def random_boxes(n, seed):
    g = _gen(seed)
    return Box(th.randn(n, D, generator=g), th.rand(n, D, generator=g) * 1.5)


def ball_cases():
    """(a, b, relation flags) built so that the relation holds."""
    c = th.tensor([[0.0, 0.0]])
    one = th.ones(1, 1)
    big = Ball(c, 2 * one)
    small = Ball(c + 0.5, 0.5 * one)
    far = Ball(c + 10, one)
    touching = Ball(c + th.tensor([[3.0, 0.0]]), one)
    point = Ball(c + 0.3, 0 * one)
    return {
        "nested": (small, big, {"inside": True, "meets": True}),
        "equal": (big, big, {"inside": True, "meets": True}),
        "disjoint": (big, far, {"apart": True}),
        "touching": (big, touching, {"apart": True, "meets": True}),
        "overlapping": (big, Ball(c + th.tensor([[2.5, 0.0]]), one), {"meets": True}),
        "point inside": (point, big, {"inside": True, "meets": True}),
    }


def box_cases():
    c = th.zeros(1, D)
    one = th.ones(1, D)
    big = Box(c, 2 * one)
    empty = Box(c + 0.3, -0.2 * one)
    return {
        "nested": (Box(c + 0.5, 0.5 * one), big, {"inside": True, "meets": True}),
        "equal": (big, big, {"inside": True, "meets": True}),
        "disjoint": (big, Box(c + 10, one), {"apart": True}),
        "disjoint in one coordinate": (big, Box(th.tensor([[10.0, 0.0]]), one),
                                       {"apart": True}),
        "touching": (big, Box(th.tensor([[3.0, 0.0]]), one), {"apart": True, "meets": True}),
        "overlapping": (big, Box(c + th.tensor([[2.5, 0.0]]), one), {"meets": True}),
        "point inside": (Box(c + 0.3, 0 * one), big, {"inside": True, "meets": True}),
        "empty": (empty, big, {"inside": True, "apart": True}),
        "empty corner box": (CornerBox.from_corners(c + 1, c), big,
                             {"inside": True, "apart": True}),
    }


# -- the measures -----------------------------------------------------------------------------
# Default options, as compose would build them; (region factory, constructed cases).
POINT_SET_MEASURES = {
    "centre distance": (CentreDistance, random_balls, ball_cases),
    "bounds per coordinate": (Bounds, random_boxes, box_cases),
    "boundary and depth (boxes)": (BoundaryDepth, random_boxes, box_cases),
    "boundary and depth (balls)": (BoundaryDepth, random_balls, ball_cases),
    "averaged bounds": (AveragedBounds, random_boxes, box_cases),
    "masked bounds": (MaskedBounds, random_boxes, box_cases),
    "volume share": (VolumeShare, random_boxes, box_cases),
}
RELATIONS = ("inside", "meets", "apart")


def _cost(measure, relation, a, b, margin):
    try:
        return getattr(measure, relation)(a, b, margin).detach().reshape(-1)
    except ConstructorUnavailable:
        return None


@pytest.mark.parametrize("name", sorted(POINT_SET_MEASURES))
def test_zero_cost_implies_the_relation(name):
    factory, regions, _ = POINT_SET_MEASURES[name]
    measure = factory()
    a, b = regions(64, 1), regions(64, 2)
    # make some pairs nested and some overlapping, so that zero costs occur
    if isinstance(a, Ball):
        b = Ball(th.cat([a.centre[:16], b.centre[16:]]),
                 th.cat([a.radius[:16] + 0.5, b.radius[16:]]))
    else:
        b = Box(th.cat([a.centre[:16], b.centre[16:]]),
                th.cat([a.offset[:16] + 0.5, b.offset[16:]]))
    for margin in (0.0, 0.2):
        inside = _cost(measure, "inside", a, b, margin)
        if inside is not None:
            zero = inside <= TOL
            assert not (zero & refuted_inside(a, b, margin)).any(), (name, margin)
        apart = _cost(measure, "apart", a, b, margin)
        if apart is not None:
            assert not ((apart <= TOL) & refuted_apart(a, b)).any(), (name, margin)
        meets = _cost(measure, "meets", a, b, margin)
        if meets is not None:
            assert not ((meets <= TOL) & ~holds_meets(a, b)).any(), (name, margin)
        try:
            empty = measure.empty(a, margin).detach().reshape(-1)
            assert not ((empty <= TOL) & refuted_empty(a)).any(), (name, margin)
        except ConstructorUnavailable:
            pass


@pytest.mark.parametrize("name", sorted(POINT_SET_MEASURES))
def test_slack_bounds_the_violation(name):
    factory, regions, _ = POINT_SET_MEASURES[name]
    measure = factory()
    if isinstance(measure, VolumeShare):
        pytest.skip("A slack of a volume ratio is a share of volume, not a distance.")
    a, b = regions(64, 3), regions(64, 4)
    m = 0.3
    inside = _cost(measure, "inside", a, b, -m)
    grown = b.grow(_amount(b, m * (1 + 1e-4)))
    if isinstance(measure, (Bounds, BoundaryDepth)) and isinstance(a, Box):
        # the norm of the violations over the coordinates is at most m: each one is
        grown = b.grow(_amount(b, m * (1 + 1e-4)))
    assert not ((inside <= TOL) & refuted_inside(a, grown)).any()


@pytest.mark.parametrize("name", sorted(POINT_SET_MEASURES))
def test_the_relation_implies_zero_cost(name):
    factory, _, cases = POINT_SET_MEASURES[name]
    measure = factory()
    for case, (a, b, holds) in cases().items():
        for relation in RELATIONS:
            if not holds.get(relation):
                continue
            if case == "touching" and relation == "apart":
                continue     # closed regions share the touching point; see test_touching
            cost = _cost(measure, relation, a, b, 0.0)
            if cost is None:
                continue
            if relation == "apart" and isinstance(measure, VolumeShare) \
                    and measure.temperature is not None:
                continue     # the smoothed volume of BoxEL is never exactly zero (declared)
            assert float(cost.max()) <= 1e-5, (name, case, relation, cost)


@pytest.mark.parametrize("name", sorted(POINT_SET_MEASURES))
def test_constructed_cases_are_not_misread(name):
    """The cases where a relation is false give a positive cost."""
    factory, _, cases = POINT_SET_MEASURES[name]
    measure = factory()
    c = cases()
    false = [("disjoint", "inside"), ("disjoint", "meets"), ("equal", "apart"),
             ("overlapping", "inside"), ("overlapping", "apart"), ("nested", "apart")]
    if "empty" in c:
        false += [("empty", "meets"), ("empty corner box", "meets")]
    for case, relation in false:
        a, b, _ = c[case]
        cost = _cost(measure, relation, a, b, 0.0)
        if cost is not None:
            assert float(cost.min()) > 0, (name, case, relation)


@pytest.mark.parametrize("name", sorted(POINT_SET_MEASURES))
def test_costs_are_never_negative(name):
    factory, regions, _ = POINT_SET_MEASURES[name]
    measure = factory()
    a, b = regions(64, 5), regions(64, 6)
    for margin in (-0.3, 0.0, 0.3):
        for relation in RELATIONS:
            cost = _cost(measure, relation, a, b, margin)
            if cost is not None:
                assert float(cost.min()) >= 0, (name, relation, margin)
        for rel in ("empty", "nonempty"):
            try:
                cost = getattr(measure, rel)(a, abs(margin))
            except ConstructorUnavailable:
                continue
            assert float(cost.min()) >= 0, (name, rel)


def _leaf(region):
    if isinstance(region, Ball):
        return Ball(region.centre.clone().requires_grad_(True),
                    region.radius.clone().requires_grad_(True))
    return Box(region.centre.clone().requires_grad_(True),
               region.offset.clone().requires_grad_(True))


def _moved(tensor, lr):
    return tensor - lr * (tensor.grad if tensor.grad is not None else 0.0)


def _step(region, lr):
    with th.no_grad():
        if isinstance(region, Ball):
            return Ball(_moved(region.centre, lr), _moved(region.radius, lr).clamp_min(0))
        return Box(_moved(region.centre, lr), _moved(region.offset, lr).clamp_min(0))


@pytest.mark.parametrize("name", sorted(POINT_SET_MEASURES))
def test_a_gradient_step_repairs_a_violation(name):
    factory, regions, _ = POINT_SET_MEASURES[name]
    measure = factory()
    for relation in RELATIONS + ("empty", "nonempty"):
        a, b = _leaf(regions(32, 7)), _leaf(regions(32, 8))
        args = (a,) if relation in ("empty", "nonempty") else (a, b)
        margin = 0.1 if relation == "nonempty" else 0.0
        try:
            cost = getattr(measure, relation)(*args, margin)
        except ConstructorUnavailable:
            continue
        if float(cost.sum()) <= 0 or not cost.requires_grad:
            continue
        cost.sum().backward()
        lr = 1e-3
        new = getattr(measure, relation)(*(_step(r, lr) for r in args), margin)
        assert float(new.sum()) < float(cost.sum()), (name, relation)


def test_touching_regions_are_a_boundary_case():
    """Closed regions that touch share a point: ``meets`` is zero and ``apart`` (whose
    margin 0 does not separate them) is zero as well; with a strict margin only ``apart``
    of separated regions is zero. This is the boundary of the sets, of measure zero for a
    sampled interpretation, and the reason a strict margin is the one that guarantees
    something."""
    a, b = Box(th.zeros(1, 2), th.ones(1, 2)), Box(th.tensor([[2.0, 0.0]]), th.ones(1, 2))
    m = Bounds()
    assert float(m.meets(a, b)) == 0 and float(m.apart(a, b)) == 0
    assert float(m.apart(a, b, 0.1)) > 0


# -- what used to be wrong ----------------------------------------------------------------------

def test_empty_boxes_are_read_as_empty_sets():
    """A box with a negative half-width (the exact intersection of two boxes that are apart,
    the bottom class of ELK-Em) used to be compared by its tensors: it met boxes and was not
    inside them."""
    empty = Box(th.zeros(1, 2), th.tensor([[-0.5, 1.0]]))
    big = Box(th.zeros(1, 2), th.full((1, 2), 3.0))
    m = Bounds()
    assert float(m.inside(empty, Box(th.full((1, 2), 9.0), th.ones(1, 2) * 0.1))) == 0
    assert float(m.meets(empty, big)) > 0
    assert float(m.apart(empty, big)) == 0
    assert float(m.empty(empty)) == 0 and float(m.nonempty(empty, 0.01)) > 0
    old = Bounds(empty_boxes="formula", extent="norm")
    assert float(old.meets(empty, big)) == 0 and float(old.empty(empty)) > 0


def test_flat_boxes_have_no_extent():
    """``nonempty(epsilon)`` reads the thinnest half-width: a flat box has no volume (the
    norm of the half-widths, the earlier default and the reading of ELBE, says it has)."""
    flat = Box(th.zeros(1, 2), th.tensor([[1.0, 0.0]]))
    assert float(Bounds().nonempty(flat, 0.01)) > 0 and float(Bounds().empty(flat)) == 0
    old = Bounds(extent="norm")
    assert float(old.nonempty(flat, 0.01)) == 0 and float(old.empty(flat)) > 0


def test_the_conjunction_of_balls_is_read_soundly():
    """``A ⊓ B ⊑ C`` with ``a = b = ball(0, 1)`` and ``c = ball(0.9, 1)``: the centre of
    ``c`` lies in both and ``c`` is as large as the smaller ball, so the published loss of
    ELEmbeddings is zero; but ``a ∩ b = a`` is not inside ``c``."""
    a = Ball(th.zeros(1, 2), th.ones(1, 1))
    c = Ball(th.tensor([[0.9, 0.0]]), th.ones(1, 1))
    assert float(CentreDistance(conjunction="published").intersection_inside(a, a, c)) == 0
    assert float(CentreDistance().intersection_inside(a, a, c)) > 0
    # and a negative axiom: ``c`` far larger, centred outside both: DELE's loss is zero but
    # the intersection lies inside ``c``
    huge = Ball(th.tensor([[3.0, 0.0]]), th.full((1, 1), 10.0))
    assert float(CentreDistance(conjunction="published").intersection_apart(a, a, huge)) == 0
    assert float(CentreDistance().intersection_apart(a, a, huge)) > 0


def _lens_points(a, b, n=4000, seed=0):
    x, _ = points_in(a, n, seed)
    inside = member(a, x) & member(b, x)
    return x, inside


def test_conjunction_of_balls_sampled():
    th.manual_seed(0)
    a, b, c = random_balls(64, 11), random_balls(64, 12), random_balls(64, 13)
    c = Ball(c.centre * 0.3, c.radius + 1.0)
    m = CentreDistance()
    for margin in (0.0, 0.1):
        cost = m.intersection_inside(a, b, c, margin).reshape(-1)
        x, lens = _lens_points(a, b)
        outside_c = ~member(c, x)
        refuted = (lens & outside_c).any(dim=1)
        assert not ((cost <= TOL) & refuted).any()
        assert bool((cost <= TOL).any())          # the test has zero costs to check
        neg = m.intersection_apart(a, b, c, margin).reshape(-1)
        # zero: the intersection is not empty and has a point outside c
        has_outside = (lens & outside_c).any(dim=1)
        assert not ((neg <= TOL) & ~has_outside).any()


def test_conjunction_of_boxes_is_exact():
    th.manual_seed(0)
    a, b, c = random_boxes(64, 21), random_boxes(64, 22), random_boxes(64, 23)
    c = Box(c.centre * 0.3, c.offset + 1.0)
    m = Bounds()
    cost = m.intersection_inside(a, b, c).reshape(-1)
    x, lens = _lens_points(a, b)
    refuted = (lens & ~member(c, x)).any(dim=1)
    assert not ((cost <= TOL) & refuted).any()
    # disjoint a and b: the axiom holds, the cost is zero (it was the gap before)
    far = Box(a.centre + 20, a.offset)
    assert float(m.intersection_inside(a, far, c).max()) == 0
    assert float(Bounds(conjunction="nonempty").intersection_inside(a, far, c).min()) > 0


def test_union_of_boxes():
    g = _gen(5)
    a = UnionOfBoxes(th.randn(32, 2, D, generator=g), th.rand(32, 2, D, generator=g))
    b = UnionOfBoxes(th.cat([a.centre[:8], th.randn(24, 2, D, generator=g)]),
                     th.cat([a.offset[:8] + 0.3, th.rand(24, 2, D, generator=g)]))
    m = UnionMeasure()
    x, _ = points_in(a, 2000)
    in_a, in_b = member(a, x), member(b, x)
    inside = m.inside(a, b).reshape(-1)
    assert not ((inside <= TOL) & (in_a & ~in_b).any(1)).any()
    assert bool((inside[:8] <= TOL).all())
    apart = m.apart(a, b).reshape(-1)
    assert not ((apart <= TOL) & (in_a & in_b).any(1)).any()
    empty = m.empty(a).reshape(-1)
    assert not ((empty <= TOL) & in_a.any(1)).any()


def test_union_meets_needs_every_component():
    """``UnionMeasure.meets`` is zero only if *every* component of ``a`` meets ``b``: a
    sufficient condition for the overlap of two unions (sound, not complete)."""
    a = UnionOfBoxes(th.tensor([[[0.0, 0.0], [10.0, 10.0]]]), th.ones(1, 2, 2) * 0.5)
    b = Box(th.zeros(1, 2), th.ones(1, 2))
    assert float(UnionMeasure().meets(a, b)) > 0      # they do meet (first component)


def test_sector_cones():
    g = _gen(9)
    axis = (th.rand(64, D, generator=g) * 2 - 1) * math.pi
    a = SectorCone(axis, th.rand(64, D, generator=g) * 1.5)
    b = SectorCone(th.cat([axis[:16], (th.rand(48, D, generator=g) * 2 - 1) * math.pi]),
                   th.cat([a.aperture[:16] + 0.3, th.rand(48, D, generator=g) * 3]))
    m = ConeBounds()
    x = (th.rand(64, 3000, D, generator=g) * 2 - 1) * math.pi
    in_a, in_b = member(a, x), member(b, x)
    inside = m.inside(a, b).reshape(-1)
    assert not ((inside <= TOL) & (in_a & ~in_b).any(1)).any()
    assert bool((inside[:16] <= TOL).all())
    apart = m.apart(a, b).reshape(-1)
    assert not ((apart <= TOL) & (in_a & in_b).any(1)).any()
    # a sector of no aperture in one coordinate: no sampled point; empty used to need all
    flat = SectorCone(th.zeros(1, 2), th.tensor([[1.0, 0.0]]))
    assert float(m.empty(flat)) == 0 and float(m.nonempty(flat, 0.01)) > 0


def test_log_volume_ratio_hard_boxes():
    m = LogVolumeRatio()
    a = CornerBox.from_corners(th.zeros(1, 2), th.ones(1, 2))
    b = CornerBox.from_corners(-th.ones(1, 2), 2 * th.ones(1, 2))
    far = CornerBox.from_corners(5 + th.zeros(1, 2), 6 + th.ones(1, 2))
    assert float(m.inside(a, b)) <= 1e-9 and float(m.inside(b, a)) > 0
    assert float(m.apart(a, far)) <= 1e-9 and float(m.apart(a, b)) > 0


def test_a_strict_margin_of_a_volume_ratio_asks_for_room():
    """A positive margin used to be ignored by ``VolumeShare``: a zero cost did not mean
    the inclusion with room that a strict margin promises."""
    a = Box(th.zeros(1, 2), th.ones(1, 2))
    touching = Box(th.tensor([[0.5, 0.0]]), th.tensor([[0.5, 1.0]]))
    assert float(VolumeShare(None).inside(touching, a, 0.2)) > 0
    assert float(VolumeShare(None, strict_margin="ignore").inside(touching, a, 0.2)) == 0


# -- regions that are not sets of points of a space ----------------------------------------------

def test_fuzzy_measures_read_the_pointwise_order():
    """Fuzzy sets: ``MeanViolation.inside`` is zero iff the degree of membership of ``A``
    is at most that of ``B`` for every element of the domain; ``apart`` iff the conjunction
    is 0 everywhere, ``empty`` iff every degree is 0. ``ConflictMeasure`` reads the
    S-implication: zero iff :math:`\\neg A \\sqcup B` has degree 1 everywhere, a sufficient
    condition for ``A <= B`` (sound, not complete for degrees strictly between 0 and 1)."""
    from mowl.representation.library.logic_fuzzy import (
        ConflictMeasure, FuzzySet, Goedel, MeanViolation, Product)
    g = _gen(3)
    for family in (Goedel(), Product()):
        a = FuzzySet(th.rand(64, 6, generator=g), family)
        b = FuzzySet(th.maximum(a.values, th.rand(64, 6, generator=g)), family)
        c = FuzzySet(th.rand(64, 6, generator=g), family)
        mv = MeanViolation()
        assert float(mv.inside(a, b).max()) == 0
        assert bool(((mv.inside(a, c).reshape(-1) == 0) == (a.values <= c.values).all(1)).all())
        both = family.and_(a.values, c.values)
        assert bool(((mv.apart(a, c).reshape(-1) == 0) == (both == 0).all(1)).all())
        conflict = ConflictMeasure().inside(a, c).reshape(-1)
        assert not ((conflict <= TOL) & ~(a.values <= c.values).all(1)).any()


def test_fuzzy_measures_refuse_logits_and_the_other_way_round():
    """A fuzzy set of logits (Al-cone's half-axis cones) used to get the measure of degrees
    by default: its ``empty`` was the mean logit (negative, and zero for a non-empty cone)."""
    from mowl.representation.library.logic_fuzzy import (
        ConeMeasure, LogitSet, LogitGoedel, MeanViolation)
    cone = LogitSet(th.tensor([[5.0, -5.0]]), LogitGoedel())
    with pytest.raises(ConstructorUnavailable):
        MeanViolation().empty(cone)
    assert float(ConeMeasure().empty(cone)) > 0
    g = R.compose(3, 1, embed_dim=2, classes="half-axis cone", relations=None)
    assert isinstance(g.measure, ConeMeasure)


def test_order_violation_is_the_order_of_corner_cones():
    """An order cone ``x`` is the set of points above ``x`` (``"product"``) or below it
    (``"reversed"``, order embeddings: the origin is the top); inclusion of those sets is the
    componentwise order the measure reads."""
    from mowl.representation.library.hierarchy_measures import OrderViolation
    from mowl.representation.library.hierarchy_regions import OrderCone
    g = _gen(4)
    x, y = th.rand(64, 3, generator=g), th.rand(64, 3, generator=g)
    a, b = OrderCone(x), OrderCone(y)
    for direction, holds in (("reversed", (x >= y).all(1)), ("product", (x <= y).all(1))):
        cost = OrderViolation(direction).inside(a, b).reshape(-1)
        assert bool(((cost <= 1e-9) == holds).all()), direction


def test_entailment_cones_are_reflexive_and_sound():
    from mowl.representation.library.hierarchy_classes import ConePoints
    from mowl.representation.library.hierarchy_measures import ConeAngle
    th.manual_seed(1)
    cones = ConePoints(16, 2, K=0.1)
    a = cones(th.arange(16))
    m = ConeAngle()
    assert float(m.inside(a, a).max()) <= 1e-6          # every cone lies in itself
    child, parent = cones(th.arange(8)), cones(th.arange(8, 16))
    cost = m.inside(child, parent).reshape(-1)
    apex_in = th.stack([region_row(parent, i).contains(child.centre[i:i + 1])[0, 0]
                        for i in range(8)])
    assert bool(((cost <= 1e-6) == apex_in).all())


def test_disks_protrusion():
    from mowl.representation.library.hierarchy_measures import Protrusion
    from mowl.representation.library.hierarchy_regions import Disk
    from mowl.nn.order.manifolds import Euclidean
    e = Euclidean()
    a = Disk(th.zeros(1, 2), e, None, th.tensor([1.0]))
    b = Disk(th.tensor([[0.5, 0.0]]), e, None, th.tensor([2.0]))
    m = Protrusion()
    assert float(m.inside(a, b)) <= 1e-5 and float(m.inside(b, a)) > 0


def test_score_measures_declare_themselves():
    """Measures whose cost is a ranking score of a published method, with no truth
    condition, say so (``reading = "score"``): the distance and depth of HiT, the triple
    score of knowledge graph models, the divergence of BetaE, the cosine order of EIKE."""
    scores = {name for name, factory in R.MEASURES.items()
              if getattr(factory, "reading", "truth") == "score"}
    assert {"distance and depth", "triple score", "divergence", "cosine order"} <= scores
    for name, factory in R.MEASURES.items():
        assert getattr(factory, "reading", "truth") in ("truth", "score"), name


def test_inexact_measures_declare_themselves():
    """Smoothed and sampled volumes cannot reach zero for some true relations; they say so
    in the vocabulary (part ``smoothed volume ratio`` / ``sampled volume``)."""
    assert VolumeShare().part == "smoothed volume ratio"
    assert VolumeShare(None).part == "volume ratio"
    assert R.MEASURES["gumbel volume ratio"]().part == "smoothed volume ratio"
    assert R.MEASURES["sampled volume"]().part == "sampled volume"
    a = Box(th.zeros(1, 2), th.ones(1, 2))
    far = Box(th.full((1, 2), 9.0), th.ones(1, 2))
    near = Box(th.full((1, 2), 2.5), th.ones(1, 2))      # apart, but close
    assert float(VolumeShare().apart(a, near)) > 1e-3    # never zero (smoothed)
    assert float(VolumeShare(None).apart(a, far)) <= 1e-9  # the floor of the volumes


def test_wrong_regions_are_an_error_that_says_so():
    """A measure given a region of a family it does not compare raises
    ``ConstructorUnavailable`` (a ``TypeError``), instead of computing with tensors it was
    not written for."""
    from mowl.representation.library.shapes_measures import OctagonBounds, PointDistance
    from mowl.representation.library.shapes_regions import Point
    box = Box(th.zeros(1, 2), th.ones(1, 2))
    for measure in (OctagonBounds(), PointDistance(), CentreDistance()):
        with pytest.raises(ConstructorUnavailable):
            measure.inside(box, box)
    with pytest.raises(ConstructorUnavailable):
        PointDistance().empty(Point(th.zeros(1, 2)))     # a point is never empty
