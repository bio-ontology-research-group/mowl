"""The general parts of the multi-box family on their own (round recomp4): the soft minimum and
the designated pair of the measure of unions of boxes, the slot algebra and the structural
reading on pieces, the sampled membership and volume, the point terms, the penalties, the
replacement of parts through the builders, and the regimes of the readings."""
from types import SimpleNamespace
from unittest import TestCase

import torch as th

from mowl.representation import Bounds, Objective, Translation
from mowl.representation.library.multibox_methods import mcmultibox, multiboxel, multielbe
from mowl.representation.library.multibox_objective import DiversityPenalty
from mowl.representation.library.multibox_parts import (
    SampledVolume, UnionClasses, UnionMeasure, UnionOfBoxes, box_membership, soft_minimum)
from mowl.representation.library.structural_points import (SoftMembership, VolumePenalty,
                                                           log_volume)
from mowl.representation.library.structural_reading import (ExistencePenalty,
                                                            StructuralReading, pair_size_of,
                                                            size_of)
from mowl.representation.library.structural_regions import (SlotBoxes, largest_side, side,
                                                            slot_depth, slot_pairs)
from mowl.representation.regions import Box


def slots(centre, w, active=None):
    centre, w = th.tensor(centre, dtype=th.float), th.tensor(w, dtype=th.float)
    if active is None:
        active = th.ones(centre.shape[:2], dtype=th.bool)
    return SlotBoxes.make(centre, w, th.tensor(active) if not th.is_tensor(active) else active)


class TestSoftMinimum(TestCase):

    def test_one_entry_is_the_entry(self):
        v = th.tensor([[3.0], [-1.0]])
        self.assertTrue(th.equal(soft_minimum(v, 0.5, dim=1), v[:, 0]))

    def test_between_minimum_and_mean_and_the_minimum_at_low_temperature(self):
        v = th.tensor([[0.0, 1.0, 4.0]])
        s = soft_minimum(v, 1.0, dim=1)
        self.assertGreaterEqual(float(s), 0.0)
        self.assertLessEqual(float(s), float(v.mean()))
        self.assertLess(float(soft_minimum(v, 1e-4, dim=1)), 1e-3)
        # no overflow for large values at a small temperature
        self.assertTrue(bool(th.isfinite(soft_minimum(v * 1e4, 1e-3, dim=1)).all()))

    def test_refuses_a_temperature_that_is_not_positive(self):
        with self.assertRaises(ValueError):
            soft_minimum(th.ones(1, 2), 0.0)


class TestDesignatedPair(TestCase):

    def test_soft_aggregation_with_unequal_numbers_of_components(self):
        """The soft designated pair needs no square grid of pairs any more."""
        th.manual_seed(0)
        a = UnionOfBoxes(th.randn(4, 2, 3), th.rand(4, 2, 3))
        b = UnionOfBoxes(th.randn(4, 3, 3), th.rand(4, 3, 3))
        c = UnionOfBoxes(th.randn(4, 2, 3), th.rand(4, 2, 3))
        measure = UnionMeasure(Bounds(separation="every", extent="norm",
                                      conjunction="nonempty"), "soft", 0.5)
        out = measure.intersection_inside(a, b, c)
        self.assertEqual(out.shape, (4, 1))
        self.assertTrue(bool(th.isfinite(out).all()))

    def test_hard_and_cold_soft_agree(self):
        th.manual_seed(1)
        a, b, c = (UnionOfBoxes(th.randn(3, 2, 2), th.rand(3, 2, 2)) for _ in range(3))
        base = Bounds(separation="every", extent="norm", conjunction="nonempty")
        hard = UnionMeasure(base, "hard").intersection_inside(a, b, c)
        cold = UnionMeasure(base, "soft", 1e-5).intersection_inside(a, b, c)
        self.assertTrue(th.allclose(hard, cold, atol=1e-3))


class TestSlotAlgebra(TestCase):

    def test_pairs_in_order_signed_and_active_iff_both(self):
        a = slots([[[0.0], [5.0]]], [[[1.0], [1.0]]], [[True, False]])
        b = slots([[[0.5], [10.0]]], [[[1.0], [1.0]]])
        centre, w, active = slot_pairs(a, b)
        self.assertEqual(active.tolist(), [[True, True, False, False]])
        self.assertAlmostEqual(float(w[0, 0, 0]), 0.75)   # [-0.5, 1] of [-1, 1] and [-0.5, 1.5]
        self.assertLess(float(w[0, 1, 0]), 0.0)          # apart: negative half-width

    def test_side_and_size(self):
        self.assertEqual(side(th.tensor([[0.5, -0.2]])).tolist(), [-0.4000000059604645])
        sizes = th.tensor([[1.0, 3.0]])
        self.assertEqual(largest_side(sizes, th.tensor([[True, False]])).tolist(), [1.0])
        with self.assertRaises(ValueError):
            largest_side(sizes, th.tensor([[False, False]]))

    def test_depth(self):
        region = slots([[[0.0], [4.0]]], [[[1.0], [1.0]]], [[True, True]])
        self.assertAlmostEqual(float(slot_depth(th.tensor([[3.5]]), region)[0]), 0.5)
        empty = slots([[[0.0]]], [[[1.0]]], [[False]])
        self.assertEqual(float(slot_depth(th.tensor([[0.0]]), empty)[0]), -1e6)

    def test_intersection_keeps_the_active_pairs_first(self):
        a = slots([[[0.0], [5.0], [0.0]]], [[[1.0], [1.0], [-1.0]]], [[True, True, False]])
        b = slots([[[0.5], [0.0], [0.0]]], [[[1.0], [-1.0], [-1.0]]], [[True, False, False]])
        both, gap = a.intersect(b)
        self.assertEqual(both.active.tolist(), [[True, True, False]])
        self.assertEqual(float(gap), 0.0)


class TestStructuralReadingOnPieces(TestCase):

    def geometry(self, regions):
        return SimpleNamespace(classes=lambda ids: regions[int(ids[0])], class_model=None)

    def test_zero_when_every_slot_is_included_and_positive_otherwise(self):
        inner = slots([[[0.0], [3.0]]], [[[0.5], [0.5]]])
        outer = slots([[[0.0], [3.0]]], [[[1.0], [1.0]]])
        reading = StructuralReading(0.0)
        g = self.geometry({0: inner, 1: outer})
        self.assertEqual(float(reading.gci0(g, th.tensor([[0, 1]]))), 0.0)
        self.assertGreater(float(reading.gci0(g, th.tensor([[1, 0]]))), 0.0)

    def test_a_box_is_one_piece(self):
        reading = StructuralReading(0.0)
        g = self.geometry({0: Box(th.zeros(1, 2), th.full((1, 2), 0.5)),
                           1: Box(th.zeros(1, 2), th.ones(1, 2))})
        self.assertEqual(float(reading.gci0(g, th.tensor([[0, 1]]))), 0.0)
        self.assertGreater(float(reading.gci0_bot(g, th.tensor([[0, 0]]))), 0.0)

    def test_sizes_and_existence(self):
        a = slots([[[0.0]]], [[[1.0]]])
        b = slots([[[1.5]]], [[[1.0]]])
        self.assertAlmostEqual(float(size_of(a)), 2.0)
        self.assertAlmostEqual(float(pair_size_of(a, b)), 0.5)
        model = SimpleNamespace(device="cpu", classes=lambda ids: a)
        penalty = ExistencePenalty(th.tensor([0]), th.empty(0, 2, dtype=th.long), 1.0)
        self.assertEqual(float(penalty.penalty(model)), 0.0)
        penalty = ExistencePenalty(th.tensor([0]), th.empty(0, 2, dtype=th.long), 3.0)
        self.assertAlmostEqual(float(penalty.penalty(model)), 1.0)


class TestSampledVolume(TestCase):

    def test_membership_positive_iff_inside(self):
        points = th.tensor([[0.0, 0.0], [2.0, 0.0]])
        m = box_membership(points, th.zeros(1, 1, 2), th.ones(1, 1, 2), 0.1)
        self.assertGreater(float(m[0, 0, 0]), 0.0)
        self.assertEqual(float(m[0, 1, 0]), 0.0)

    def test_refuses_a_margin(self):
        measure = SampledVolume(num_points=64)
        a = UnionOfBoxes(th.zeros(1, 1, 2), th.ones(1, 1, 2))
        self.assertAlmostEqual(float(measure.inside(a, a)), 0.0, places=5)
        # a strict margin is read as the enlarged region (no room inside itself); a slack
        # is refused
        self.assertGreater(float(measure.inside(a, a, margin=0.1)), 0.0)
        with self.assertRaises(ValueError):
            measure.inside(a, a, margin=-0.1)
        with self.assertRaises(ValueError):
            mcmultibox(3, 1, embed_dim=2, num_boxes=1, num_points=16, margin=0.1)


class TestPointParts(TestCase):

    def test_union_membership_is_the_noisy_or(self):
        m = SoftMembership(0.1)
        region = UnionOfBoxes(th.tensor([[[0.0], [0.0]]]), th.tensor([[[1.0], [1.0]]]))
        one = m.box_probability(th.zeros(1, 1), region.centre[:, :1], region.offset[:, :1])
        self.assertAlmostEqual(float(m.probability(th.zeros(1, 1), region)),
                               float(1 - (1 - one) ** 2), places=6)

    def test_volume_penalty(self):
        self.assertEqual(float(log_volume(th.empty(0, 2))), 0.0)
        self.assertAlmostEqual(float(log_volume(th.full((2, 1, 3), 0.5))), 0.0, places=5)
        classes = UnionClasses(3, 2, components=1)
        model = SimpleNamespace(class_model=classes, device="cpu",
                                classes=lambda ids: classes(ids))
        self.assertIsNone(VolumePenalty(0.0).penalty(model))
        self.assertTrue(bool(th.isfinite(VolumePenalty(1.0, [0, 1]).penalty(model))))

    def test_diversity_of_two_equal_boxes_is_one_and_has_a_gradient(self):
        classes = UnionClasses(1, 20, components=2)
        with th.no_grad():
            classes.centres.weight.zero_()
            classes.offsets.weight.fill_(0.3)
        pen = DiversityPenalty(1.0).penalty(SimpleNamespace(class_model=classes))
        pen.backward()
        self.assertAlmostEqual(float(pen), 1.0, places=5)
        self.assertGreater(float(classes.centres.weight.grad.norm()), 0.0)


class TestBuildersReplaceParts(TestCase):
    """Every part of the methods can be replaced through the builder; none is ignored."""

    def test_multiboxel(self):
        relations = Translation(2, 3)
        measure = UnionMeasure(Bounds())
        objective = Objective(0.1)
        extra = VolumePenalty(1.0)
        g = multiboxel(5, 2, embed_dim=3, classes="union of boxes", relations=relations,
                       measure=measure, objective=objective, regularizers=(extra,))
        self.assertIs(g.relation_model, relations)
        self.assertIs(g.measure, measure)
        self.assertIs(g.objective, objective)
        self.assertIn(extra, g.regularizers)
        self.assertIsInstance(g.class_model, UnionClasses)
        self.assertEqual(g(th.tensor([[0, 1]]), "gci0").shape, (1, 1))

    def test_multiboxel_structural_and_multielbe(self):
        objective = StructuralReading(0.05)
        g = multiboxel(5, 2, embed_dim=3, reading="structural", objective=objective)
        self.assertIs(g.objective, objective)
        measure = UnionMeasure(Bounds())
        g = multielbe(5, 2, embed_dim=3, measure=measure, regularizers=(VolumePenalty(),))
        self.assertIs(g.measure, measure)
        self.assertTrue(any(isinstance(r, VolumePenalty) for r in g.regularizers))

    def test_mcmultibox(self):
        measure = SampledVolume(num_points=32)
        g = mcmultibox(4, 1, embed_dim=2, num_boxes=1, num_points=32, measure=measure)
        self.assertIs(g.measure, measure)
        self.assertEqual(g(th.tensor([[0, 1]]), "gci0").shape, (1, 1))

    def test_refused_with_shared_tables(self):
        g = multielbe(5, 2, embed_dim=3)
        g.tau = 1.0
        with self.assertRaises(ValueError):
            multielbe(None, None, embed_dim=3, tables=SimpleNamespace(
                class_embed=g.class_model.centres, class_offset=g.class_model.offsets,
                ind_embed=None, ind_offset=None, rel_embed=g.relation_model.vectors,
                tau=1.0, point_supervision=None), measure=UnionMeasure(Bounds()))
