"""Declared operations of measures and relation models, their generic definitions, exact
views of regions, and the parts that serve other methods because of them.

An objective asks a part for an operation by name
(:meth:`~mowl.representation.operations.Operations.operation`); the part has it as its own
formula, as the generic one of its kind, or refuses with
:class:`~mowl.representation.ConstructorUnavailable` naming the operation.
"""
import pytest
import torch as th

from mowl.representation import (Ball, Box, BoxClasses, ConstructorUnavailable, Objective,
                                 Translation)
from mowl.representation.measures import Bounds, CentreDistance, RegionMeasure
from mowl.representation.regions import CornerBox, MarginUnavailable
from mowl.representation.relation_models import Relations
from mowl.representation.library.boxrel_measures import VolumeShare
from mowl.representation.library.boxrel_relations import Affine, AffineBoxes, TranslationBoxes
from mowl.representation.library.hierarchy_measures import LogVolumeRatio, OrderViolation
from mowl.representation.library.multibox_parts import HeadTailRelations, UnionOfBoxes
from mowl.representation.library.shapes_measures import OctagonBounds, PrismBounds
from mowl.representation.library.shapes_regions import OctagonProduct, Point

D = 4


def boxes(n, seed, spread=1.0):
    g = th.Generator().manual_seed(seed)
    return Box(th.randn(n, D, generator=g) * spread, th.rand(n, D, generator=g) + 0.1)


def balls(n, seed):
    g = th.Generator().manual_seed(seed)
    return Ball(th.randn(n, D, generator=g), th.rand(n, 1, generator=g) + 0.1)


def points(n, seed, scale=4.0):
    g = th.Generator().manual_seed(seed)
    return (th.rand(n, D, generator=g) * 2 - 1) * scale


# -- the declaration ------------------------------------------------------------------------
def test_every_measure_and_relation_model_declares_its_operations():
    assert set(RegionMeasure.declared_operations()) >= {
        "outside", "negative", "product_overlap", "meets_shrunk", "is_empty", "host_for"}
    assert set(Relations.declared_operations()) >= {
        "subsumed_negative", "preimage_all", "range", "image_with_error"}
    # the specialised part has its own formula, the others the generic one or none
    assert VolumeShare.provides("product_overlap") == "own"
    assert Bounds.provides("product_overlap") == "generic"
    assert PrismBounds.provides("meets_shrunk") == "own"
    assert Bounds.provides("meets_shrunk") == "generic"
    assert HeadTailRelations.provides("subsumed_negative") == "own"
    assert Translation.provides("subsumed_negative") == "generic"
    assert TranslationBoxes.provides("preimage_all") == "own"
    assert Translation.provides("preimage_all") == "generic"
    assert AffineBoxes.provides("range") == "own"
    # no generic non-inclusion: a measure without one says so
    assert RegionMeasure.provides("outside") is None
    assert Bounds.provides("outside") == "own" and CentreDistance.provides("outside") == "own"
    with pytest.raises(KeyError):
        Bounds.provides("no such operation")


def test_a_missing_operation_is_refused_with_its_name_and_purpose():
    class Bare(RegionMeasure):
        accepts = (Box,)

    with pytest.raises(ConstructorUnavailable, match="'outside'.*gci0_negative"):
        Bare().operation("outside", "gci0_negative='outside'")


# -- the generic definitions ------------------------------------------------------------------
def test_generic_product_overlap_is_the_volume_product_of_the_exact_intersection():
    a, b = boxes(64, 1), boxes(64, 2)
    cost = Bounds().product_overlap(a, b)
    inter, _ = a.intersect(b, signed=True)
    expected = inter.volume() / (a.volume() * b.volume())
    assert th.allclose(cost, expected)
    # zero exactly where the boxes do not overlap
    disjoint = (a.lower > b.upper).any(1) | (b.lower > a.upper).any(1)
    assert bool((cost.squeeze(1)[disjoint] == 0).all())
    assert bool((cost.squeeze(1)[~disjoint] > 0).all())
    with pytest.raises(ConstructorUnavailable):    # balls do not intersect in a ball
        CentreDistance().product_overlap(balls(3, 1), balls(3, 2))


def test_generic_meets_shrunk_is_the_centre_of_a_meeting_b():
    a, b = boxes(64, 3), boxes(64, 4)
    cost = Bounds().meets_shrunk(a, b)
    inside = ((a.centre >= b.lower) & (a.centre <= b.upper)).all(1)
    assert bool((cost.squeeze(1)[inside] == 0).all())
    assert bool((cost.squeeze(1)[~inside] > 0).all())


def test_generic_subsumed_negative_pushes_both_pairs_delta_apart():
    th.manual_seed(0)
    rel = Translation(3, D)
    ids = th.tensor([0, 1, 2])
    c, d = boxes(3, 5), boxes(3, 6)
    measure = Bounds()
    cost = rel.subsumed_negative(measure, c, ids, d, 0.0, 2.0)
    first = measure.meets(c, rel.preimage(ids, d))
    second = measure.meets(d, rel.image(ids, c))
    assert th.allclose(cost, (2.0 - first) ** 2 + (2.0 - second) ** 2)


def test_preimage_all_of_a_functional_relation_is_its_preimage():
    th.manual_seed(0)
    rel = Translation(2, D)
    ids, d = th.tensor([0, 1]), boxes(2, 7)
    assert th.equal(rel.preimage_all(ids, d).centre, rel.preimage(ids, d).centre)
    with pytest.raises(ConstructorUnavailable, match="functional"):
        HeadTailRelations(2, D).operation("preimage_all")(ids, d)


def test_a_translation_has_no_range_constraint_and_affine_boxes_have_one():
    assert Translation(2, D).range(th.tensor([0, 1])) is None
    assert isinstance(AffineBoxes(2, D).range(th.tensor([0, 1])), Box)


@pytest.mark.parametrize("measure, make", [(Bounds(), boxes), (CentreDistance(), balls)])
def test_outside_zero_means_a_point_of_a_is_outside_b(measure, make):
    """Sound: where the cost of non-inclusion is zero, the point of ``a`` farthest out of
    ``b`` (in the coordinate that sticks out most; on the line of the centres) lies in ``a``
    and not in ``b``; an included region never has a zero cost."""
    a, b = make(400, 8), make(400, 9)
    around = a.grow(0.3 if isinstance(a, Box) else th.full_like(a.radius, 0.3))
    assert bool((measure.outside(a, around, 0.05) > 0).all())   # included: never zero
    cost = measure.outside(a, b, 0.05).squeeze(1)
    zero = cost == 0
    assert bool(zero.any())
    if isinstance(a, Ball):
        unit = a.centre - b.centre
        unit = unit / th.linalg.norm(unit, dim=1, keepdim=True)
        far = a.centre + a.radius * unit
        out = th.linalg.norm(far - b.centre, dim=1) > b.radius.squeeze(1)
    else:
        upper, lower = a.upper - b.upper, b.lower - a.lower
        stick = th.maximum(upper, lower)
        i = stick.argmax(dim=1)
        rows = th.arange(len(a))
        far = a.centre.clone()
        far[rows, i] = th.where(upper[rows, i] >= lower[rows, i], a.upper[rows, i],
                                a.lower[rows, i])
        out = ~b.contains(far).diagonal()
    assert bool(out[zero].all())


def test_outside_of_an_empty_box_is_never_zero():
    empty = Box(th.zeros(1, D), -th.ones(1, D))
    assert float(Bounds().outside(empty, boxes(1, 1))) > 0


def test_outside_of_an_order_is_the_energy_beyond_the_margin():
    from mowl.representation.library.hierarchy_regions import OrderCone
    m = OrderViolation()
    a, b = OrderCone(th.zeros(2, D)), OrderCone(th.tensor([[1.0] * D, [-1.0] * D]))
    energy = m.raw(a, b)
    assert th.allclose(m.outside(a, b, 0.1), th.relu(0.1 - energy))


# -- margins of measures without a margin of their own --------------------------------------
def test_a_strict_margin_of_the_log_volume_ratio_is_the_enlarged_box():
    m = LogVolumeRatio()
    a = CornerBox.from_corners(th.zeros(1, 2), th.ones(1, 2))
    b = CornerBox.from_corners(-0.05 * th.ones(1, 2), 1.05 * th.ones(1, 2))
    assert float(m.inside(a, b)) <= 1e-9
    assert float(m.inside(a, b, 0.2)) > 0            # no room of 0.2
    assert float(m.inside(a, b, 0.04)) <= 1e-9        # room of 0.04
    with pytest.raises(MarginUnavailable, match="slack"):
        m.inside(a, b, -0.1)
    with pytest.raises(ValueError):                   # still a ValueError, as before
        m.inside(a, b, -0.1)


# -- exact views -------------------------------------------------------------------------------
def test_a_point_is_compared_as_a_box_of_no_width():
    p = Point(th.zeros(1, D))
    assert float(Bounds().inside(p, Box(th.zeros(1, D), th.ones(1, D)))) == 0
    assert float(Bounds().inside(p, Box(3 + th.zeros(1, D), th.ones(1, D)))) > 0
    assert Point.viewable_as((Box,)) and not Point.viewable_as((OctagonProduct,))


def test_a_box_is_compared_as_an_octagon_product():
    a, b = boxes(32, 11), boxes(32, 12)
    m = OctagonBounds()
    included = Bounds().inside(a, b).squeeze(1) == 0
    assert bool((m.inside(a, b).squeeze(1)[included] <= 1e-6).all())
    assert bool((m.inside(a, b).squeeze(1)[~included] > 0).all())
    assert Box.viewable_as((OctagonProduct,))


def test_a_relation_of_boxes_maps_a_union_of_boxes_box_by_box():
    th.manual_seed(0)
    union = UnionOfBoxes(th.randn(3, 2, D), th.rand(3, 2, D))
    ids = th.tensor([0, 1, 2])
    rel = Affine(3, D)
    assert rel.maps(UnionOfBoxes)
    mapped = rel.preimage(ids, union)
    assert isinstance(mapped, UnionOfBoxes) and mapped.centre.shape == (3, 2, D)
    for k in range(2):
        one = rel.preimage(ids, Box(union.centre[:, k], union.offset[:, k]))
        assert th.allclose(mapped.centre[:, k], one.centre)
        assert th.allclose(mapped.offset[:, k], one.offset)


def test_the_image_of_affine_boxes_is_the_set_of_successors():
    th.manual_seed(0)
    rel = AffineBoxes(1, 2, sound=True)
    with th.no_grad():
        rel.range_offsets.weight.fill_(2.0)
    ids = th.tensor([0])
    a = Box(th.zeros(1, 2), 0.5 * th.ones(1, 2))
    image = rel.image(ids, a)
    y = (th.rand(4000, 2) * 2 - 1) * 3
    x = (th.rand(4000, 2) * 2 - 1) * 3
    pairs = rel.related(ids, x, y)[0] & a.contains(x)[0]
    # every successor of a point of a that was sampled lies in the image
    assert bool(image.contains(y[pairs])[0].all())


# -- parts read through the general interface -----------------------------------------------
def test_the_size_of_any_region_can_be_detached():
    classes = BoxClasses(3, D)
    region = classes.regions(th.tensor([0, 1]), detach_size=True)
    assert not region.offset.requires_grad and region.centre.requires_grad


def test_a_class_model_without_contexts_is_the_same_in_every_context():
    classes = BoxClasses(3, D)
    ids = th.tensor([0, 2])
    assert th.equal(classes.in_context(ids, th.tensor([1, 1])).centre, classes(ids).centre)


def test_the_cube_penalty_reads_any_class_model_of_boxes():
    from mowl.representation.library.boxrel_regularizers import LocationPenalty
    from mowl.representation import compose
    geometry = compose(4, 1, embed_dim=D, classes="box", relations="translation",
                       objective=Objective(0.0))
    assert float(LocationPenalty().penalty(geometry)) >= 0


def test_an_objective_of_another_method_takes_the_negatives_it_defines():
    """The forms with a negative loss follow a swapped objective, so that the probe does not
    ask a strict objective for a negative it does not have."""
    from mowl.representation.swap import build_method
    geometry = build_method("elbe", 4, 2, embed_dim=D, objective=Objective(0.1,
                                                                            negatives="gci0"))
    assert geometry.neg_capable_gcis == frozenset({"gci0"})
    # a ranking form has a negative cost too
    ranked = Objective(0.1, negatives="gci2", ranking={"gci0": 0.5})
    geometry = build_method("elbe", 4, 2, embed_dim=D, objective=ranked)
    assert geometry.neg_capable_gcis == frozenset({"gci0", "gci2", "object_property_assertion"})
