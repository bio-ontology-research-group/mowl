"""What the regions of every class model are, and whether the parts say so.

For every registered class model this file states its semantics and checks it:

* regions that are sets of points of a space have ``contains(points)``, and their emptiness
  (``is_empty``) agrees with it;
* a class model that declares its conjunction exact (``exact_conjunction``) intersects its
  regions exactly (the points of ``intersect(a, b)`` are those in both), and its default
  measure says the same (``exact_intersection``); the catalogue says the same as ``parts``;
* the regions of the other class models (fuzzy sets, logits of half-axes, order cones, points
  of a manifold, distributions) have another meaning, which is tested in
  ``test_soundness_measures``; they have no point-membership test and say so.
"""

import math

import pytest
import torch as th

import mowl.representation as R
from mowl.representation import ConstructorUnavailable, compose

DIM = 4

#: Class model -> what its region is.
SEMANTICS = {
    "ball": "closed balls of R^d",
    "box": "closed axis-parallel boxes of R^d (negative half-width: empty)",
    "box with lower corner": "boxes [m, m + exp(delta)] of R^d",
    "box with fixed classes": "boxes; fixed classes are the empty box and a very large box",
    "box with bump": "boxes (the bump is read by the relation model only)",
    "union of boxes": "finite unions of boxes",
    "union of boxes with slots": "finite unions of boxes in K slots: an inactive slot or a slot "
                                 "of negative half-width is the empty box",
    "unit box": "boxes in the unit cube",
    "soft box": "Gumbel boxes: a distribution of boxes, compared by expected volumes",
    "point": "points (a class of one element)",
    "octagon product": "products of octagons on pairs of coordinates",
    "triangle x box": "a triangle with fixed normals times a box",
    "ellipsoid": "axis-parallel ellipsoids",
    "sector cone": "products of circular sectors (one per coordinate)",
    "geodesic cone": "entailment cones of the Poincare ball",
    "disk": "disks (balls of a manifold, of signed radius)",
    "manifold point": "points of a manifold, compared by a ranking score",
    "half-space point": "points of the Poincare half-space, compared by a ranking score",
    "subspace": "axis-parallel subspaces of R^2d given by relaxed 0/1 indicators (quantum "
                "logic: meet, span, orthogonal complement)",
    "binary subspace": "axis-parallel subspaces of R^2d, indicators started at 0/1",
    "order cone": "elements of an ordered vector space (the corner cone above or below)",
    "fuzzy set": "fuzzy sets over a finite domain (Goedel operators)",
    "fuzzy set (product)": "fuzzy sets over a finite domain (product operators)",
    "fuzzy set (lukasiewicz)": "fuzzy sets over a finite domain (Lukasiewicz operators)",
    "half-axis cone": "sets of half-axes given by logits (an element belongs iff its logit "
                      "is positive)",
}
NOT_POINT_SETS = {"manifold point", "half-space point",
                  "subspace", "binary subspace", "order cone", "fuzzy set", "fuzzy set (product)",
                  "fuzzy set (lukasiewicz)", "half-axis cone", "soft box"}


def test_every_class_model_has_a_stated_semantics():
    assert set(R.CLASS_MODELS) <= set(SEMANTICS), set(R.CLASS_MODELS) - set(SEMANTICS)


def _model(name):
    th.manual_seed(0)
    model = R.CLASS_MODELS[name](12, DIM)
    with th.no_grad():
        for p in model.parameters():
            p.mul_(1 + th.rand_like(p))
    return model


def _points(region, n=3000, seed=0):
    g = th.Generator().manual_seed(seed)
    centre = getattr(region, "centre", None)
    if centre is None:
        width = region.bounds.shape[1] * 2
    elif type(region).__name__ == "Prism":
        width = 2 + centre.shape[1]
    else:
        width = centre.shape[-1]
    if type(region).__name__ == "SectorCone":
        return (th.rand(n, width, generator=g) * 2 - 1) * math.pi
    if type(region).__name__ in ("ConePoint", "Disk") and region.manifold is not None:
        from mowl.nn.order.manifolds import Hyperboloid
        if isinstance(region.manifold, Hyperboloid):
            return Hyperboloid.from_spatial((th.rand(n, width - 1, generator=g) * 2 - 1))
        x = th.randn(n, width, generator=g)
        return x / x.norm(dim=1, keepdim=True) * th.rand(n, 1, generator=g) * 0.9
    return (th.rand(n, width, generator=g) * 2 - 1) * 3


@pytest.mark.parametrize("name", sorted(set(R.CLASS_MODELS) - NOT_POINT_SETS))
def test_point_sets_have_a_membership_test(name):
    region = _model(name)(th.arange(12))
    inside = region.contains(_points(region))
    assert inside.shape == (12, 3000) and inside.dtype == th.bool
    try:
        empty = region.is_empty()
    except ConstructorUnavailable:
        return
    # an empty region contains no point
    assert not bool(inside[empty].any())


@pytest.mark.parametrize("name", sorted(NOT_POINT_SETS))
def test_other_regions_say_they_are_not_point_sets(name):
    region = _model(name)(th.arange(3))
    with pytest.raises(ConstructorUnavailable):
        region.contains(th.zeros(2, DIM))


EXACT = sorted(n for n in set(R.CLASS_MODELS) - NOT_POINT_SETS
               if R.CLASS_MODELS[n](2, DIM).exact_conjunction)


@pytest.mark.parametrize("name", EXACT)
def test_a_conjunction_declared_exact_is_exact(name):
    model = _model(name)
    a, b = model(th.arange(6)), model(th.arange(6, 12))
    both, gap = a.intersect(b)
    pts = _points(a)
    want = a.contains(pts) & b.contains(pts)
    got = both.contains(pts)
    assert float((got != want).float().mean()) < 1e-3, name
    assert bool(((gap[:, 0] > 0) <= ~want.any(dim=1)).all())


@pytest.mark.parametrize("name", sorted(set(R.CLASS_MODELS)))
def test_parts_say_how_the_conjunction_is_made(name):
    try:
        g = compose(4, 1, embed_dim=DIM, classes=name, relations=None)
    except TypeError:
        pytest.skip("no default measure")
    model = g.class_model
    conjunction = g.parts["conjunction"]
    if model.exact_conjunction and model.conjunction_part is None:
        assert conjunction == "exact"
    if not g.measure.exact_intersection:
        # the default measure does not compute the intersection exactly: the class model
        # must not claim more than the measure does when the objective reads conjunctions
        assert conjunction != "exact" or name in KNOWN_DISAGREEMENTS, (name, conjunction)


#: Class models whose ``parts`` say "exact" while the default measure reads conjunctions
#: approximately: the regions intersect exactly, the measure does not use it.
KNOWN_DISAGREEMENTS = {
    "unit box": "LogVolumeRatio (default for CornerBox) inherits the exact box intersection; "
                "its exact_intersection is True",
}


def test_the_catalogue_says_what_the_parts_say():
    """For every method built from parts, the catalogue's conjunction agrees with the
    geometry's ``parts`` where both say exact or approximate."""
    disagree = []
    for key, build in R.METHODS.items():
        if R.METHOD_KINDS.get(key) != "axioms" or key not in R.REPRESENTATIONS:
            continue
        try:
            g = build(4, 2, 2, embed_dim=DIM)
        except Exception:      # noqa: BLE001 - methods that need tables or a dataset
            continue
        card = R.REPRESENTATIONS[key].conjunction.value
        parts = getattr(g, "parts", {}).get("conjunction")
        if parts in ("exact", "approximate") and card in ("exact", "approximate") \
                and parts != card:
            disagree.append((key, card, parts))
    assert not disagree, disagree
