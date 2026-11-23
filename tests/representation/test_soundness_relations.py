"""Do the relation models and the constructors mean what their names say?

Every relation model that is a set of pairs of points states which pairs it relates
(``related(ids, x, y)``). These tests check, by sampling, that with the DEFAULT options

* ``preimage(r, C)`` (the region of :math:`\\exists r.C`) contains exactly the points that
  have an r-successor in ``C``, and ``image(r, A)`` exactly the successors of points of ``A``;
* a zero cost of ``r ⊑ s`` implies that every sampled pair of ``r`` is a pair of ``s``, and a
  zero cost of ``r ∘ t ⊑ s`` that every sampled pair of the composition is one of ``s``;
* the constructors of regions are exact where the family is closed under them
  (``intersect`` of boxes, ``complement`` and ``union`` of fuzzy sets) and declared
  approximate otherwise.
"""

import math

import pytest
import torch as th

from mowl.representation import Ball, Box, ConstructorUnavailable, Translation
from mowl.representation.measures import Bounds, CentreDistance
from mowl.representation.library.ballfam_parts import ToleranceTranslation
from mowl.representation.library.boxrel_relations import Affine, AffineBoxes, TranslationBoxes
from mowl.representation.library.query_parts import ConeRotation
from mowl.representation.library.query_regions import SectorCone

D = 2


def _gen(seed):
    return th.Generator().manual_seed(seed)


def _boxes(n, seed, spread=1.0):
    g = _gen(seed)
    return Box(th.randn(n, D, generator=g) * spread, th.rand(n, D, generator=g) + 0.1)


def _balls(n, seed):
    g = _gen(seed)
    return Ball(th.randn(n, D, generator=g), th.rand(n, 1, generator=g) + 0.1)


def _points(n, seed, scale=4.0):
    return (th.rand(n, D, generator=_gen(seed)) * 2 - 1) * scale


def _box_meets(lo1, hi1, lo2, hi2):
    return (th.maximum(lo1, lo2) <= th.minimum(hi1, hi2)).all(dim=-1)


# -- the region of exists r.C ----------------------------------------------------------------

def test_translation_preimage_and_image():
    th.manual_seed(0)
    r = Translation(3, D)
    ids = th.tensor([0, 1, 2])
    for region in (_boxes(3, 1), _balls(3, 2)):
        x = _points(3000, 3)
        succ = r.successor(ids, x)                                  # (B, P, d)
        has = th.stack([region.contains(succ[i])[i] for i in range(3)])
        assert th.equal(r.preimage(ids, region).contains(x), has)
        # image: y is a successor of a point of the region iff y - v lies in it
        back = x.unsqueeze(0) - r.vectors(ids).unsqueeze(1)
        source = th.stack([region.contains(back[i])[i] for i in range(3)])
        assert th.equal(r.image(ids, region).contains(x), source)
        # related agrees with successor
        assert bool(r.related(ids, x, succ[0]).all(dim=1)[0])


def test_affine_preimage_and_image():
    th.manual_seed(0)
    r = Affine(3, D)
    with th.no_grad():
        r.scales.weight.uniform_(-2, 2)                           # also negative factors
    ids = th.tensor([0, 1, 2])
    region = _boxes(3, 4)
    x = _points(3000, 5)
    succ = r.successor(ids, x)
    has = th.stack([region.contains(succ[i])[i] for i in range(3)])
    got = r.preimage(ids, region).contains(x)
    # points on the border may fall either side by rounding
    assert float((got != has).float().mean()) < 2e-3
    f, t = r.factor(ids), r.vectors(ids)
    back = (x.unsqueeze(0) - t.unsqueeze(1)) / f.unsqueeze(1)
    source = th.stack([region.contains(back[i])[i] for i in range(3)])
    assert float((r.image(ids, region).contains(x) != source).float().mean()) < 2e-3


@pytest.mark.parametrize("family", ["box", "ball"])
def test_tolerance_translation_preimage(family):
    th.manual_seed(0)
    r = ToleranceTranslation(3, D, sigma_init=0.4)
    ids = th.tensor([0, 1, 2])
    region = _boxes(3, 6) if family == "box" else _balls(3, 7)
    x = _points(4000, 8)
    centre = r.successor(ids, x)                                 # x + v
    sigma = r.tolerance(ids).unsqueeze(1)                        # (B, 1, 1)
    if family == "box":
        # the successors of x are the cube of half-width sigma around x + v
        lo, hi = region.lower.unsqueeze(1), region.upper.unsqueeze(1)
        has = _box_meets(centre - sigma, centre + sigma, lo, hi)
        cls = Box
    else:
        d = th.linalg.norm(centre - region.centre.unsqueeze(1), dim=-1)
        has = d <= region.radius + sigma[..., 0]
        cls = Ball
    got = r.preimage(ids, region).contains(x)
    assert float((got != has).float().mean()) < 2e-3
    # related() says the same as the witness: the point of the successor set nearest C
    y = th.stack([centre[i, 0] for i in range(3)])
    assert bool(r.related(ids, x[:1].expand(3, -1)[:1], y[:1], region=cls)[0, 0])


@pytest.mark.parametrize("family", ["box", "ball"])
def test_box_of_translations_preimage(family):
    th.manual_seed(0)
    r = TranslationBoxes(3, D)
    ids = th.tensor([0, 1, 2])
    region = _boxes(3, 9) if family == "box" else _balls(3, 10)
    x = _points(4000, 11)
    box = r.box(ids)
    if family == "box":
        lo = x.unsqueeze(0) + (box.centre - box.offset).unsqueeze(1)
        hi = x.unsqueeze(0) + (box.centre + box.offset).unsqueeze(1)
        has = _box_meets(lo, hi, region.lower.unsqueeze(1), region.upper.unsqueeze(1))
    else:
        # ball geometries: the translations are the ball of radius |o| around c
        reach = th.linalg.norm(box.offset, dim=-1, keepdim=True)
        moved = x.unsqueeze(0) + box.centre.unsqueeze(1)
        d = th.linalg.norm(moved - region.centre.unsqueeze(1), dim=-1)
        has = d <= region.radius + reach
    got = r.preimage(ids, region).contains(x)
    assert float((got != has).float().mean()) < 2e-3
    # a pair (x, x + c) is related; one far away is not
    y = x[:5] + box.centre[0]
    assert bool(r.related(ids[:1], x[:5], y, region=Box if family == "box" else Ball).all())


def test_affine_boxes_preimage():
    th.manual_seed(0)
    r = AffineBoxes(3, D)
    with th.no_grad():
        r.range_offsets.weight.uniform_(0.5, 1.5)
        r.error_offsets.weight.uniform_(0.05, 0.3)
    ids = th.tensor([0, 1, 2])
    region = _boxes(3, 12, spread=0.3)
    x = _points(4000, 13, scale=2.0)
    m, c, e = (t.unsqueeze(1) for t in (r.slope(ids), r.vectors(ids), r.error(ids)))
    rng = r.range(ids)
    # successors of x: y in R with |x - (m y + c)| <= e, a box
    lo = th.maximum((x.unsqueeze(0) - c - e) / m, rng.lower.unsqueeze(1))
    hi = th.minimum((x.unsqueeze(0) - c + e) / m, rng.upper.unsqueeze(1))
    has = _box_meets(lo, hi, region.lower.unsqueeze(1), region.upper.unsqueeze(1))
    got = r.preimage(ids, region).contains(x)
    assert float((got != has).float().mean()) < 2e-3
    assert bool(has.any())


def test_affine_boxes_preimage_of_an_empty_restriction_is_empty():
    r = AffineBoxes(1, D)
    with th.no_grad():
        r.range_centres.weight.fill_(5.0)
        r.range_offsets.weight.fill_(0.1)
    far = Box(th.zeros(1, D), th.full((1, D), 0.2))
    pre = r.preimage(th.tensor([0]), far)
    assert not bool(pre.contains(_points(2000, 1)).any())
    assert float(Bounds().empty(pre)) == 0
    assert float(AffineBoxes(1, D, sound=False).preimage(th.tensor([0]), far).offset.min()) \
        >= -1e9          # (the unsound option only changes the error term)


def test_rotation_preimage_and_image():
    """The rotation of AConE as a relation: y within |delta| of x + theta. Its preimage used
    to rotate back and *narrow* the cone by delta, so that it was not the set of points with
    a successor in the cone (and image and preimage disagreed)."""
    th.manual_seed(0)
    r = ConeRotation(3, D, angle_range=0.5)
    ids = th.tensor([0, 1, 2])
    g = _gen(14)
    cone = SectorCone((th.rand(3, D, generator=g) * 2 - 1) * math.pi,
                      th.rand(3, D, generator=g) * 1.2)
    x = (th.rand(4000, D, generator=g) * 2 - 1) * math.pi
    theta, delta = r.angles(ids)
    turn = x.unsqueeze(0) + theta.unsqueeze(1) - cone.axis.unsqueeze(1)
    arc = th.abs(th.atan2(th.sin(turn), th.cos(turn)))
    has = (arc <= delta.abs().unsqueeze(1) + cone.aperture.unsqueeze(1)).all(dim=-1)
    got = r.preimage(ids, cone).contains(x)
    assert float((got != has).float().mean()) < 2e-3
    old = ConeRotation(3, D, angle_range=0.5, aperture_change="signed")
    old.load_state_dict(r.state_dict())
    assert float((old.preimage(ids, cone).contains(x) != has).float().mean()) > 1e-2
    # image: the successors of the cone's points
    turn = x.unsqueeze(0) - theta.unsqueeze(1) - cone.axis.unsqueeze(1)
    arc = th.abs(th.atan2(th.sin(turn), th.cos(turn)))
    succ = (arc <= delta.abs().unsqueeze(1) + cone.aperture.unsqueeze(1)).all(dim=-1)
    assert float((r.image(ids, cone).contains(x) != succ).float().mean()) < 2e-3


def test_relation_models_that_are_not_sets_of_pairs_of_points_say_so():
    """Head and tail boxes (Box2EL) relate classes through their bumps, BoxE and ExpressivE
    score triples, FALCON's network scores pairs of domain vectors: none of them states a
    set of pairs of points."""
    from mowl.representation.library.multibox_parts import HeadTailRelations
    for relations in (HeadTailRelations(2, D),):
        with pytest.raises(ConstructorUnavailable):
            relations.related(th.tensor([0]), _points(2, 0), _points(2, 1))


def test_fuzzy_existential_restriction():
    """Fuzzy relations: the degree of an element in exists r.C is the supremum over the
    domain of the conjunction of r(a, b) and C(b), in the family of the fuzzy sets."""
    from mowl.representation.library.logic_fuzzy import (FuzzySet, Goedel, Product,
                                                         TruthMatrix)
    r = TruthMatrix(2, 4)
    ids = th.tensor([0, 1])
    for family in (Goedel(), Product()):
        c = FuzzySet(th.rand(2, 4, generator=_gen(1)), family)
        got = r.preimage(ids, c).values
        m = r.matrix(ids)
        want = th.tensor([[max(float(family.and_(m[k, a, b], c.values[k, b]))
                               for b in range(4)) for a in range(4)] for k in range(2)])
        assert th.allclose(got, want)
    with pytest.raises(ConstructorUnavailable):          # a domain of another size
        r.preimage(ids, FuzzySet(th.rand(2, 5), Goedel()))


# -- role axioms ------------------------------------------------------------------------------

def _pairs_translation(r, ids, x):
    return x, r.successor(ids, x)[0]


def test_translation_role_axioms():
    r = Translation(4, D)
    with th.no_grad():
        r.vectors.weight[1] = r.vectors.weight[0]
        r.vectors.weight[3] = r.vectors.weight[0] + r.vectors.weight[2]
    x = _points(500, 2)
    assert float(r.role_inclusion(th.tensor([0]), th.tensor([1]))) <= 1e-6
    y = r.successor(th.tensor([0]), x)[0]
    assert bool(r.related(th.tensor([1]), x, y).all())
    assert float(r.role_chain(th.tensor([0]), th.tensor([2]), th.tensor([3]))) <= 1e-6
    z = r.successor(th.tensor([2]), y)[0]
    assert bool(r.related(th.tensor([3]), x, z).all())
    assert float(r.role_inclusion(th.tensor([0]), th.tensor([2]))) > 0


def test_affine_role_axioms_need_equal_maps():
    """A role of BoxEL is a function, so ``r ⊑ s`` holds only if the two maps are equal. The
    loss of the BoxEL code is zero as soon as the factor of ``r`` does not exceed that of
    ``s`` (kept as ``role_scale="at_most"``, selected by the method ``boxel``)."""
    for role_scale, sound in (("equal", True), ("at_most", False)):
        r = Affine(3, D, role_scale=role_scale)
        with th.no_grad():
            r.vectors.weight[1] = r.vectors.weight[0]
            r.scales.weight[0] = th.tensor([0.5, 0.5])
            r.scales.weight[1] = th.tensor([1.0, 1.0])
        cost = float(r.role_inclusion(th.tensor([0]), th.tensor([1])))
        x = _points(200, 3)
        y = r.successor(th.tensor([0]), x)[0]
        holds = bool(r.related(th.tensor([1]), x, y).all())
        assert holds is False
        assert (cost > 0) is sound
    r = Affine(3, D)
    with th.no_grad():
        r.vectors.weight[1] = r.vectors.weight[0]
        r.scales.weight[1] = r.scales.weight[0]
    assert float(r.role_inclusion(th.tensor([0]), th.tensor([1]))) <= 1e-6
    # chain: s is the composition of the two maps
    with th.no_grad():
        f0, f1 = r.factor(th.tensor([0])), r.factor(th.tensor([1]))
        t0, t1 = r.vectors.weight[0], r.vectors.weight[1]
        r.scales.weight[2] = (f0 * f1)[0] - r.eps
        r.vectors.weight[2] = f1[0] * t0 + t1
    assert float(r.role_chain(th.tensor([0]), th.tensor([1]), th.tensor([2]))) <= 1e-5
    x = _points(200, 4)
    z = r.successor(th.tensor([1]), r.successor(th.tensor([0]), x)[0])[0]
    assert bool(r.related(th.tensor([2]), x, z, tolerance=1e-4).all())


def _sample_box_pairs(r, ids, n, seed):
    """Pairs (x, y) of a box of translations: y = x + a point of the box."""
    g = _gen(seed)
    x = _points(n, seed)
    box = r.box(ids)
    u = th.rand(n, D, generator=g) * 2 - 1
    return x, x + r.sign * (box.centre + u * box.offset)


def test_box_of_translations_role_axioms():
    r = TranslationBoxes(4, D)
    with th.no_grad():
        r.vectors.weight[1] = r.vectors.weight[0]
        r.offsets.weight[1] = r.offsets.weight[0].abs() + 0.2
    m = Bounds()
    assert float(r.role_inclusion(th.tensor([0]), th.tensor([1]), m)) == 0
    x, y = _sample_box_pairs(r, th.tensor([0]), 1000, 1)
    assert bool(r.related(th.tensor([1]), x, y).all())
    assert float(r.role_inclusion(th.tensor([1]), th.tensor([0]), m)) > 0
    with th.no_grad():
        r.vectors.weight[3] = r.vectors.weight[0] + r.vectors.weight[2]
        r.offsets.weight[3] = r.offsets.weight[0].abs() + r.offsets.weight[2].abs()
    assert float(r.role_chain(th.tensor([0]), th.tensor([2]), th.tensor([3]), m)) <= 1e-6
    x, y = _sample_box_pairs(r, th.tensor([0]), 1000, 2)
    _, z = _sample_box_pairs(r, th.tensor([2]), 1000, 3)
    z = y + (z - _points(1000, 3))
    assert bool(r.related(th.tensor([3]), x, z).all())


def test_affine_boxes_role_inclusion_is_sound():
    th.manual_seed(0)
    r = AffineBoxes(2, D)
    with th.no_grad():
        for name in ("vectors", "slopes", "range_centres"):
            getattr(r, name).weight[1] = getattr(r, name).weight[0]
        r.range_offsets.weight[1] = r.range_offsets.weight[0].abs() + 0.1
        r.error_offsets.weight[1] = r.error_offsets.weight[0].abs() + 0.1
    m = Bounds()
    assert float(r.role_inclusion(th.tensor([0]), th.tensor([1]), m)) <= 1e-6
    # pairs of r: y in the range, x in T(y) + E
    g = _gen(5)
    rng = r.range(th.tensor([0]))
    y = rng.lower + th.rand(1000, D, generator=g) * (rng.upper - rng.lower)
    e = r.error(th.tensor([0]))
    x = y * r.slope(th.tensor([0])) + r.vectors(th.tensor([0])) \
        + (th.rand(1000, D, generator=g) * 2 - 1) * e
    assert bool(r.related(th.tensor([0]), x, y).all())
    assert bool(r.related(th.tensor([1]), x, y).all())


# -- constructors ------------------------------------------------------------------------------

def test_box_intersection_is_exact():
    a, b = _boxes(64, 20), _boxes(64, 21)
    both, gap = a.intersect(b)
    x = _points(4000, 22)
    assert th.equal(both.contains(x), a.contains(x) & b.contains(x))
    assert th.equal(gap[:, 0] == 0, ~both.is_empty())
    # the conjunction of the geometry is that intersection
    from mowl.representation import compose
    g = compose(4, 1, embed_dim=D, classes="box")
    region, _ = g.conjunction(a, b)
    assert th.equal(region.contains(x), a.contains(x) & b.contains(x))


def test_union_of_boxes_intersection_is_exact():
    from mowl.representation.library.multibox_parts import UnionOfBoxes
    g = _gen(23)
    a = UnionOfBoxes(th.randn(16, 2, D, generator=g), th.rand(16, 2, D, generator=g))
    b = UnionOfBoxes(th.randn(16, 3, D, generator=g), th.rand(16, 3, D, generator=g))
    both, gap = a.intersect(b)
    x = _points(4000, 24)
    assert th.equal(both.contains(x), a.contains(x) & b.contains(x))


def test_fuzzy_constructors_are_the_operators_of_the_family():
    from mowl.representation.library.logic_fuzzy import FuzzySet, Goedel, Lukasiewicz, Product
    g = _gen(25)
    for family in (Goedel(), Product(), Lukasiewicz()):
        a = FuzzySet(th.rand(8, 5, generator=g), family)
        b = FuzzySet(th.rand(8, 5, generator=g), family)
        assert th.allclose(a.intersect(b)[0].values, family.and_(a.values, b.values))
        assert th.allclose(a.union(b).values, family.or_(a.values, b.values))
        assert th.allclose(a.complement().values, 1 - a.values)
        # crisp sets: the operators are the set operations
        ca = FuzzySet((a.values > 0.5).float(), family)
        cb = FuzzySet((b.values > 0.5).float(), family)
        both = ca.intersect(cb)[0].values
        assert th.equal(both, ((ca.values > 0) & (cb.values > 0)).float())


def test_cone_negation_is_declared_approximate():
    """The complement of a product of sectors is not the product of the complements: ConE's
    negation misses points that are outside in one coordinate and inside in another, and the
    class model and the catalogue say so."""
    from mowl.representation import REPRESENTATIONS, compose
    cone = SectorCone(th.zeros(1, 2), th.full((1, 2), 1.0))
    x = th.tensor([[0.0, 3.0], [3.0, 3.0]])           # outside in one coordinate / in both
    assert not bool(cone.contains(x)[0, 0])
    assert not bool(cone.complement().contains(x)[0, 0])      # missed by the complement
    assert bool(cone.complement().contains(x)[0, 1])
    g = compose(3, 1, embed_dim=2, classes="sector cone", relations=None)
    assert g.parts["negation"] == "approximate"
    for key in ("cone_qe", "acone"):
        assert REPRESENTATIONS[key].negation.value == "approximate"


def test_ball_intersection_is_not_a_ball():
    a = Ball(th.zeros(1, 2), th.ones(1, 1))
    lens, gap = a.intersect(a)               # a lens (it has a volume), not a Ball
    assert not isinstance(lens, Ball) and float(gap) == 0.0
    assert CentreDistance.exact_intersection is False
