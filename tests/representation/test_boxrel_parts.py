"""The parts of the boxes group work outside their papers: they combine with parts of other
methods, ComposedEL trains with them, and their semantics hold."""
from unittest import TestCase

import torch as th

from mowl.models import ComposedEL
from mowl.representation import (BallClasses, Bounds, BoxClasses, ConstructorUnavailable,
                                 Objective, compose, METHODS, RELATION_MODELS,
                                 CLASS_MODELS, MEASURES, ComposedGeometry, Relations)
from mowl.representation.regions import Box
from mowl.representation.library.boxrel_boxes import MinDeltaBoxes, signed_intersection
from mowl.representation.library.boxrel_measures import (
    AveragedBounds, MaskedBounds, VolumeShare)
from mowl.representation.library.boxrel_relations import (
    Affine, AffineBoxes, TranslationBoxes)
from tests.datasetFactory import FamilyDataset


def box(lower, upper):
    lower, upper = th.tensor([lower], dtype=th.float), th.tensor([upper], dtype=th.float)
    return Box((lower + upper) / 2, (upper - lower) / 2)


class TestRegistered(TestCase):

    def test_names(self):
        for key in ("transbox", "boxel", "boxsel", "elkem"):
            self.assertIn(key, METHODS)
        for key in ("box of translations", "affine map",
                    "affine map with range and error box"):
            self.assertIn(key, RELATION_MODELS)
        for key in ("box with lower corner", "box with fixed classes"):
            self.assertIn(key, CLASS_MODELS)
        for key in ("volume share", "averaged bounds", "masked bounds"):
            self.assertIn(key, MEASURES)

    def test_vocabulary_values(self):
        from mowl.representation import ClassModel, Measure, RelationModel
        self.assertEqual(TranslationBoxes.part, RelationModel.TRANSLATION_BOX.value)
        self.assertEqual(Affine.part, RelationModel.AFFINE.value)
        self.assertEqual(AffineBoxes.part, RelationModel.AFFINE_BOXES.value)
        self.assertEqual(MinDeltaBoxes.part, ClassModel.BOX.value)
        self.assertEqual(VolumeShare().part, Measure.SOFT_VOLUME_RATIO.value)
        self.assertEqual(VolumeShare(None).part, Measure.VOLUME_RATIO.value)


class TestMeasures(TestCase):

    def test_exact_volume_ratio(self):
        m = VolumeShare(temperature=None)
        a, b = box([0, 0], [1, 1]), box([0, 0], [0.5, 1])
        self.assertAlmostEqual(float(m.inside(b, a)), 0.0, places=6)
        self.assertAlmostEqual(float(m.inside(a, b)), 0.5, places=6)
        self.assertAlmostEqual(float(m.probability(a, b)), 0.5, places=6)
        self.assertAlmostEqual(float(m.apart(a, box([2, 2], [3, 3]))), 0.0, places=6)
        self.assertAlmostEqual(float(m.apart(a, b)), 0.5, places=6)

    def test_slack_and_ignored_strict_margin(self):
        m = VolumeShare(temperature=None)
        a, b = box([0, 0], [1, 1]), box([0, 0], [0.5, 1])
        self.assertAlmostEqual(float(m.inside(a, b, -0.6)), 0.0, places=6)
        self.assertAlmostEqual(float(m.inside(b, a, 0.3)), 0.0, places=6)

    def test_smoothed_approaches_exact(self):
        a, b = box([0, 0], [1, 1]), box([0.2, 0], [1.5, 1])
        exact = float(VolumeShare(None).inside(a, b))
        costs = [abs(float(VolumeShare(t).inside(a, b)) - exact) for t in (1, 10, 100)]
        self.assertGreater(costs[0], costs[1])
        self.assertGreater(costs[1], costs[2])

    def test_conditional(self):
        m = VolumeShare(None)
        c, d = box([0, 0], [1, 1]), box([0, 0], [0.5, 1])
        self.assertAlmostEqual(float(m.conditional(c, d, 0.4, 0.6)), 0.0, places=6)
        self.assertAlmostEqual(float(m.conditional(c, d, 0.6, 0.9)), 0.1, places=6)
        self.assertAlmostEqual(float(m.conditional(c, d, 0.1, 0.3)), 0.2, places=6)

    def test_averaged_bounds_empty_box_is_inside_everything(self):
        m = AveragedBounds()
        empty = Box(th.zeros(1, 2), -th.ones(1, 2))
        self.assertEqual(float(m.inside(empty, box([0, 0], [1, 1]), 0.1)), 0.0)
        self.assertTrue(bool(m.is_empty(empty)))
        self.assertGreater(float(m.inside(box([0, 0], [1, 1]), box([0, 0], [0.5, 0.5]))), 0)

    def test_masked_bounds_charges_the_gap(self):
        c, d, e = box([0, 0], [1, 1]), box([3, 0], [4, 1]), box([0, 0], [4, 1])
        zero = MaskedBounds(empty_weight=0.5).intersection_inside(c, d, e)
        self.assertAlmostEqual(float(zero), 1.0, places=5)   # 0.5 * gap of 2
        c2 = box([0, 0], [2, 1])
        d2 = box([1, 0], [3, 1])
        self.assertAlmostEqual(
            float(MaskedBounds().intersection_inside(c2, d2, box([0, 0], [3, 1]))), 0.0,
            places=6)

    def test_signed_intersection_is_associative(self):
        a, b, c = box([0, 0], [1, 1]), box([2, 0], [3, 1]), box([0, 0], [4, 1])
        left = signed_intersection(signed_intersection(a, b), c)
        right = signed_intersection(a, signed_intersection(b, c))
        self.assertTrue(th.allclose(left.lower, right.lower))
        self.assertTrue(th.allclose(left.upper, right.upper))


class TestRelations(TestCase):

    def test_box_of_translations_contains_the_pairs(self):
        th.manual_seed(0)
        relations = TranslationBoxes(2, 3, convention="x-y")   # the TransBox convention
        ids = th.tensor([1])
        target = box([0, 0, 0], [1, 1, 1])
        target = Box(target.centre.expand(1, 3), target.offset.expand(1, 3))
        region = relations.preimage(ids, target)
        r = relations.box(ids)
        for _ in range(50):
            y = target.lower + th.rand(1, 3) * 2 * target.offset
            t = r.lower + th.rand(1, 3) * 2 * r.offset
            x = y + t
            self.assertTrue(bool((x >= region.lower - 1e-6).all()))
            self.assertTrue(bool((x <= region.upper + 1e-6).all()))
        image = relations.image(ids, region)
        self.assertTrue(bool((image.lower <= target.lower + 1e-6).all()))

    def test_role_axioms_of_translation_boxes(self):
        relations = TranslationBoxes(3, 2)
        with th.no_grad():
            relations.vectors.weight.copy_(th.tensor([[0., 0.], [1., 0.], [1., 0.]]))
            relations.offsets.weight.copy_(th.tensor([[1., 1.], [1., 1.], [2., 2.]]))
        measure = Bounds()
        one = th.tensor([0]), th.tensor([1]), th.tensor([2])
        # box(0) = [-1,1]^2 and box(1) = [0,2]x[-1,1]: the sum has centre (1,0), half-width 2,
        # which is box(2).
        self.assertEqual(float(relations.role_inclusion(one[0], one[0], measure)), 0.0)
        self.assertGreater(float(relations.role_inclusion(one[1], one[0], measure)), 0)
        self.assertAlmostEqual(
            float(relations.role_chain(one[0], one[1], one[2], measure)), 0.0, places=6)
        self.assertGreater(float(relations.role_chain(one[0], one[1], one[0], measure)), 0)

    def test_affine_image_and_preimage(self):
        th.manual_seed(1)
        relations = Affine(2, 3)
        ids = th.tensor([0, 1])
        b = Box(th.randn(2, 3), th.rand(2, 3))
        back = relations.preimage(ids, relations.image(ids, b))
        self.assertTrue(th.allclose(back.centre, b.centre, atol=1e-5))
        self.assertTrue(th.allclose(back.offset, b.offset, atol=1e-5))
        # A negative factor mirrors the box instead of inverting it.
        with th.no_grad():
            relations.scales.weight.mul_(-1)
        image = relations.image(ids, b)
        self.assertTrue(bool((image.offset >= 0).all()))
        self.assertTrue(th.allclose(relations.preimage(ids, image).centre, b.centre,
                                    atol=1e-5))

    def test_affine_role_axioms(self):
        relations = Affine(3, 2, eps=0.0)
        with th.no_grad():
            relations.vectors.weight.copy_(th.tensor([[1., 0.], [0., 2.], [1., 2.]]))
            relations.scales.weight.copy_(th.tensor([[2., 1.], [1., 3.], [2., 3.]]))
        ids = [th.tensor([i]) for i in range(3)]
        self.assertEqual(float(relations.role_inclusion(ids[0], ids[0])), 0.0)
        self.assertGreater(float(relations.role_inclusion(ids[0], ids[1])), 0.0)
        # x -> f0 x + t0, then f1 (.) + t1: scale (2, 3), shift f1 t0 + t1 = (1, 2): relation 2.
        self.assertAlmostEqual(float(relations.role_chain(ids[0], ids[1], ids[2])), 0.0,
                               places=6)

    def test_affine_boxes_preimage_is_sound(self):
        relations = AffineBoxes(1, 2)
        ids = th.tensor([0])
        far = box([100, 100], [101, 101])
        self.assertTrue(bool((relations.preimage(ids, far).offset < 0).any()))
        unsound = AffineBoxes(1, 2, sound=False)
        with th.no_grad():
            unsound.load_state_dict(relations.state_dict())
            unsound.error_offsets.weight.fill_(1000.0)
        self.assertTrue(bool((unsound.preimage(ids, far).offset > 0).all()))

    def test_affine_boxes_role_axioms(self):
        relations = AffineBoxes(2, 3)
        with th.no_grad():   # relation 0: the identity on a large range, no error
            relations.range_centres.weight[0].zero_()
            relations.range_offsets.weight[0].fill_(10.0)
            relations.error_offsets.weight[0].zero_()
            relations.slopes.weight[0].fill_(1.0)
            relations.vectors.weight[0].zero_()
        relations.eps = 0.0
        measure = AveragedBounds()
        zero, one = th.tensor([0]), th.tensor([1])
        self.assertAlmostEqual(float(relations.role_inclusion(one, one, measure).detach()), 0.0,
                               places=6)
        # the identity followed by r1 is r1
        self.assertAlmostEqual(float(relations.role_chain(zero, one, one, measure)), 0.0,
                               places=6)

    def test_boxes_only(self):
        ball = BallClasses(3, 2)(th.tensor([0]))
        for relations in (Affine(1, 2), AffineBoxes(1, 2)):
            with self.assertRaises(ConstructorUnavailable):
                relations.preimage(th.tensor([0]), ball)
        # a box of translations widens a ball by the norm of its half-widths (enclosing)
        relations = TranslationBoxes(1, 2)
        grown = relations.preimage(th.tensor([0]), ball)
        self.assertTrue(th.allclose(
            grown.radius, ball.radius + th.linalg.norm(relations.box(th.tensor([0])).offset,
                                                       dim=-1, keepdim=True)))


class TestObjectives(TestCase):

    def test_role_forms_need_a_relation_model_that_has_them(self):
        class NoRoles(Relations):  # a relation model without role axioms
            def preimage(self, ids, region):
                return region

        g = ComposedGeometry(BoxClasses(4, 2), NoRoles(), Bounds(), Objective())
        self.assertFalse(g.role_axiom_capable)
        with self.assertRaises(ConstructorUnavailable):
            g(th.tensor([[0, 1]]), "role_inclusion")

    def test_role_forms_in_any_box_combination(self):
        for classes, relations, measure in (
                (BoxClasses(4, 3), TranslationBoxes(3, 3), Bounds()),
                (MinDeltaBoxes(4, 3), TranslationBoxes(3, 3), VolumeShare()),
                (BoxClasses(4, 3), Affine(3, 3), Bounds()),
                (BoxClasses(4, 3), AffineBoxes(3, 3), AveragedBounds())):
            g = ComposedGeometry(classes, relations, measure, Objective(0.01))
            for form, rows in (("role_inclusion", [[0, 1], [1, 2]]),
                               ("role_chain", [[0, 1, 2], [2, 2, 0]])):
                cost = g(th.tensor(rows), form)
                self.assertEqual(cost.shape, (2, 1))
                self.assertTrue(bool((cost >= 0).all()))

    def test_role_axioms_can_be_learned(self):
        th.manual_seed(0)
        for relations, measure in ((TranslationBoxes(3, 4), Bounds()),
                                   (Affine(3, 4), Bounds()),
                                   (AffineBoxes(3, 4), AveragedBounds())):
            with th.no_grad():
                for p in relations.parameters():
                    p.add_(0.5 * th.randn_like(p))
            g = ComposedGeometry(BoxClasses(3, 4), relations, measure, Objective(0.0))
            incl, chain = th.tensor([[0, 1]]), th.tensor([[0, 1, 2]])
            opt = th.optim.Adam(g.parameters(), lr=0.02)
            before = float(g(incl, "role_inclusion") + g(chain, "role_chain"))
            for _ in range(400):
                opt.zero_grad()
                loss = g(incl, "role_inclusion").sum() + g(chain, "role_chain").sum()
                loss.backward()
                opt.step()
            self.assertLess(float(loss), 0.1 * before, type(relations).__name__)

    def test_bottom_as_class_and_image_reading_are_options(self):
        g_empty = compose(5, 2, embed_dim=3, classes="box", relations="affine map")
        g_class = ComposedGeometry(BoxClasses(5, 3), Affine(2, 3), Bounds(),
                                   Objective(bottom="class", gci2_reading="image"))
        rows = th.tensor([[0, 1, 2]])
        self.assertEqual(g_class(rows, "gci3_bot").shape, (1, 1))
        self.assertEqual(g_class(rows, "gci2").shape, (1, 1))
        self.assertEqual(g_empty(th.tensor([[0]]), "gci0_bot").shape, (1, 1))


class TestComposedELWithTheParts(TestCase):
    """Parts of one method inside another: a few epochs of training lower the loss."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def assertTrains(self, epochs=40, **kwargs):
        th.manual_seed(0)
        model = ComposedEL(self.dataset, embed_dim=8, **kwargs)

        def total():
            with th.no_grad():
                return sum(float(model.module(d[:], f).mean())
                           for f, d in model.training_datasets.items() if len(d))
        before = total()
        model.train(epochs=epochs)
        self.assertLess(total(), before, kwargs)
        return model

    def test_relation_models_with_box_classes(self):
        for relations, measure in (("box of translations", None),
                                   ("affine map", None),
                                   ("affine map with range and error box",
                                    "averaged bounds"),
                                   ("affine map", "volume share")):
            self.assertTrains(classes="box", relations=relations, measure=measure,
                              objective=Objective(0.02))

    def test_class_model_and_measure_with_other_relations(self):
        self.assertTrains(classes="box with lower corner", relations="translation",
                          measure="volume share")
        self.assertTrains(classes="box with lower corner", relations="box of translations",
                          measure="bounds per coordinate", objective=Objective(0.02))
        self.assertTrains(classes="box with fixed classes", relations="translation",
                          measure="averaged bounds", objective=Objective(0.02))
        self.assertTrains(classes="box", relations="translation", measure="masked bounds",
                          objective=Objective(0.02))

    def test_balls_take_the_box_of_translations_but_not_the_affine_maps(self):
        model = ComposedEL(self.dataset, classes="ball", relations="box of translations",
                           embed_dim=4)
        model.train(epochs=1)
        model = ComposedEL(self.dataset, classes="ball", relations="affine map", embed_dim=4)
        with self.assertRaises(ConstructorUnavailable):
            model.train(epochs=1)

    def test_published_combinations(self):
        for method in ("transbox", "boxel", "boxsel", "elkem"):
            model = self.assertTrains(method=method, margin=0.02)
            self.assertEqual(type(model.as_geometry), type(model.module))
        self.assertTrains(method="transbox", formulation="reference", margin=0.02)
        self.assertTrains(method="transbox", enhanced=False, margin=0.02)
        self.assertTrains(method="elkem", variant="reference", margin=0.02)
