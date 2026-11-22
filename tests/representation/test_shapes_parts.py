"""The parts of the shapes group on their own, and combined with parts of other methods."""
from unittest import TestCase

import torch as th

from mowl.models import ComposedEL
from mowl.representation import (
    Objective, ComposedGeometry, BallClasses, BoxClasses, Bounds, CentreDistance, Translation,
    CLASS_MODELS, MEASURES, METHODS)
from mowl.representation.library.shapes_classes import (
    Points, OctagonClasses, PrismClasses, EllipsoidClasses, SoftBoxClasses, RelationsByRegion,
    ContextualSoftBoxClasses, ContextTransform)
from mowl.representation.library.shapes_measures import (
    OctagonBounds, PrismBounds, EllipsoidMeasure, GumbelVolumeRatio, PointToRegion, ByRegion,
    LinkedMeasure, CosineOrder, PointDistance)
from mowl.representation.library.shapes_objective import ShapeObjective
from mowl.representation.library.shapes_regions import (
    Point, OctagonProduct, Prism, Ellipsoid, Vector, Linked, SoftBox)
from mowl.representation.regions import Box, ConstructorUnavailable
from tests.datasetFactory import FamilyDataset


def _boxes(n, d, seed):
    g = th.Generator().manual_seed(seed)
    return Box(th.randn(n, d, generator=g), th.rand(n, d, generator=g) * 0.8)


class TestRegistration(TestCase):

    def test_parts_are_registered_and_named_by_the_vocabulary(self):
        for key in ("point", "octagon product", "triangle x box", "ellipsoid", "soft box"):
            self.assertIn(key, CLASS_MODELS)
        for key in ("gumbel volume ratio", "point-to-region distance"):
            self.assertIn(key, MEASURES)
        for key in ("octagon", "prism", "eike", "concept2box"):
            self.assertIn(key, METHODS)
        self.assertEqual(OctagonClasses.part, "octagon product")
        self.assertEqual(PrismClasses.part, "triangle x box")
        self.assertEqual(EllipsoidClasses.part, "ellipsoid")
        self.assertEqual(SoftBoxClasses.part, "soft box")
        self.assertEqual(GumbelVolumeRatio.part, "smoothed volume ratio")
        self.assertEqual(PointToRegion.part, "point-to-region distance")

    def test_compose_by_name(self):
        from mowl.representation import compose
        for classes in ("octagon product", "triangle x box", "ellipsoid", "soft box"):
            g = compose(5, 2, 3, embed_dim=6, classes=classes)
            self.assertIsInstance(g, ComposedGeometry)
            g(th.tensor([[0, 1], [2, 3]]), "gci0")


class TestPoints(TestCase):

    def test_points_in_each_family_have_no_extent(self):
        c = th.randn(4, 6)
        self.assertEqual(float(Points(4, 6, "ball")(th.arange(4)).radius.abs().max()), 0.0)
        self.assertEqual(float(Points(4, 6, "box")(th.arange(4)).offset.abs().max()), 0.0)
        oct_ = OctagonProduct.of_point(c)
        # An octagon of no extent has width zero on every axis.
        from mowl.nn.el.octagon import geometry as G
        self.assertTrue(th.allclose(G.widths(oct_.bounds), th.zeros(4, 3, 4), atol=1e-6))
        prism = Prism.of_point(c)
        self.assertTrue(th.allclose(prism.supports.sum(-1), th.zeros(4), atol=1e-6))

    def test_point_in_a_ball_class_is_the_distance_condition(self):
        balls = BallClasses(5, 4)
        points = Points(3, 4, "ball")
        ids = th.tensor([0, 1, 2])
        got = CentreDistance().inside(points(ids), balls(ids))
        c, r = balls.centres(ids), th.abs(balls.radii(ids))
        want = th.relu(th.linalg.norm(points.centres(ids) - c, dim=1, keepdim=True) - r)
        self.assertTrue(th.allclose(got, want, atol=1e-6))

    def test_unknown_family(self):
        with self.assertRaises(ValueError):
            Points(3, 4, "blob")


class TestOctagon(TestCase):

    def test_boxes_as_octagons_agree_on_what_is_inside(self):
        m = OctagonBounds()
        for seed in range(20):
            a, b = _boxes(8, 4, seed), _boxes(8, 4, seed + 100)
            elbe = Bounds().inside(a, b)
            octa = m.inside(OctagonProduct.of_box(a), OctagonProduct.of_box(b))
            self.assertTrue(bool((elbe <= octa + 1e-5).all()))
            self.assertTrue(bool((octa <= 5 ** 0.5 * elbe + 1e-5).all()))
            self.assertTrue(bool(((elbe == 0) == (octa < 1e-6)).all()))

    def test_intersection_of_boxes_is_the_box_intersection(self):
        from mowl.nn.el.octagon import geometry as G
        a, b = _boxes(8, 4, 1), _boxes(8, 4, 2)
        region, gap = OctagonProduct.of_box(a).intersect(OctagonProduct.of_box(b))
        inter, box_gap = a.intersect(b)
        nonempty = (box_gap[:, 0] == 0) & (gap[:, 0] == 0)
        self.assertTrue(bool(nonempty.any()))
        want = G.from_box(inter.centre.reshape(8, 2, 2), inter.offset.reshape(8, 2, 2))
        got = G.tighten(region.bounds)
        self.assertTrue(th.allclose(got[nonempty], want[nonempty], atol=1e-5))
        self.assertEqual((gap[:, 0] == 0).tolist(), (box_gap[:, 0] == 0).tolist())

    def test_translation_closure_and_apart(self):
        m = OctagonBounds()
        a = OctagonClasses(3, 4)(th.arange(3))
        shifted = a.translate(th.full((3, 4), 100.0))
        self.assertTrue(bool((m.apart(a, shifted) == 0).all()))
        self.assertTrue(bool((m.apart(a, a) > 0).all()))
        self.assertTrue(bool((m.inside(a, a) < 1e-6).all()))
        self.assertTrue(bool((m.meets(a, a) == 0).all()))
        with self.assertRaises(ConstructorUnavailable):
            a.scale(th.ones(3, 4))

    def test_semantic_objective_is_zero_only_for_true_axioms(self):
        # gci3 as inclusion: the exists-region must lie inside D.
        g = ComposedGeometry(OctagonClasses(6, 4), Translation(2, 4), OctagonBounds(),
                             ShapeObjective(0.0))
        with th.no_grad():
            g.class_model.radii.weight.fill_(1.0)
            g.class_model.centres.weight.zero_()
            g.relation_model.vectors.weight.zero_()
        rows = th.tensor([[0, 1, 2]])
        self.assertAlmostEqual(float(g(rows, "gci3")), 0.0, places=5)
        with th.no_grad():
            g.class_model.centres.weight[2] = 50.0
        self.assertGreater(float(g(rows, "gci3")), 1.0)


class TestPrism(TestCase):

    def test_intersection_and_membership(self):
        th.manual_seed(0)
        classes = PrismClasses(6, 5)
        with th.no_grad():
            classes.inradii.weight.fill_(1.5)
            classes.box_offsets.weight.fill_(1.5)
        a, b = classes(th.zeros(200, dtype=th.long)), classes(th.ones(200, dtype=th.long))
        inter, gap = a.intersect(b)
        points = Prism.of_point(th.randn(200, 5) * 1.5)
        m = PrismBounds()
        in_a, in_b = m.inside(points, a)[:, 0] < 1e-6, m.inside(points, b)[:, 0] < 1e-6
        in_both = m.inside(points, inter)[:, 0] < 1e-6
        self.assertTrue(bool(in_both.eq(in_a & in_b).all()))
        self.assertTrue(bool((in_a & in_b).any()))

    def test_translation_is_a_shift_of_triangle_and_box(self):
        a = PrismClasses(3, 5)(th.arange(3))
        v = th.randn(3, 5)
        moved = a.translate(v)
        self.assertTrue(th.allclose(moved.centre, a.centre + v[:, 2:]))
        self.assertTrue(th.allclose(moved.supports.sum(-1), a.supports.sum(-1), atol=1e-6))

    def test_empty_options(self):
        a = PrismClasses(3, 5)(th.arange(3))
        small = th.zeros(3, 1)
        self.assertTrue(th.allclose(PrismBounds("triangle").empty(a, 0.2),
                                    a.supports.sum(-1, keepdim=True) + 0.2))
        self.assertTrue(bool((PrismBounds("whole").empty(a) >= PrismBounds("triangle").empty(
            a) - 1e-6).all()))
        self.assertEqual(small.shape, (3, 1))


class TestEllipsoid(TestCase):

    @staticmethod
    def _sample(e, n=400):
        u = th.randn(n, e.centre.shape[1])
        u = u / u.norm(dim=1, keepdim=True) * th.rand(n, 1) ** (1 / e.centre.shape[1])
        return e.centre + e.radius * e.axes * u

    def _pair(self, seed):
        g = th.Generator().manual_seed(seed)
        mk = lambda: Ellipsoid(th.randn(1, 3, generator=g) * 0.3,  # noqa: E731
                               th.rand(1, 3, generator=g) + 0.5, th.rand(1, 1, generator=g) + 0.3)
        return mk(), mk()

    def test_bounding_containment_is_sound(self):
        m = EllipsoidMeasure()
        checked = 0
        for seed in range(400):
            a, b = self._pair(seed)
            if float(m.inside(a, b)) == 0.0:
                checked += 1
                pts = Point(self._sample(a))
                self.assertTrue(bool((m.member(pts, b) < 1e-5).all()), seed)
        self.assertGreater(checked, 3)

    def test_published_containment_is_not_sound(self):
        m = EllipsoidMeasure("published")
        for seed in range(2000):
            a, b = self._pair(seed)
            if float(m.inside(a, b)) == 0.0:
                if bool((m.member(Point(self._sample(a)), b) > 1e-3).any()):
                    return
        self.fail("no counterexample found")

    def test_apart_is_sound(self):
        m = EllipsoidMeasure()
        for seed in range(300):
            a, b = self._pair(seed)
            if float(m.apart(a, b)) == 0.0:
                self.assertTrue(bool((m.member(Point(self._sample(a)), b) > 0).all()))

    def test_points(self):
        m = EllipsoidMeasure()
        e = Ellipsoid(th.zeros(1, 2), th.ones(1, 2), th.ones(1, 1))
        self.assertEqual(float(m.member(Point(th.tensor([[0.5, 0.5]])), e)), 0.0)
        self.assertAlmostEqual(float(m.member(Point(th.tensor([[2.0, 0.0]])), e)), 3.0, 5)
        with self.assertRaises(ConstructorUnavailable):
            e.intersect(e)


class TestVectorsAndLinks(TestCase):

    def test_cosine_order(self):
        m = CosineOrder()
        v = Vector(th.tensor([[1.0, 0.0]]))
        w = Vector(th.tensor([[2.0, 0.0]]))
        self.assertAlmostEqual(float(m.inside(v, w)), 0.0, 6)
        self.assertAlmostEqual(float(m.inside(w, v)), 1.0, 6)  # longer vector not below
        self.assertLess(float(CosineOrder("signed").inside(v, w)), 0)
        self.assertAlmostEqual(float(m.member(w, v)), 0.0, 6)

    def test_a_region_built_by_a_relation_has_no_second_part(self):
        measure = LinkedMeasure(EllipsoidMeasure(), CosineOrder(), 2.0)
        e = Ellipsoid(th.zeros(1, 2), th.ones(1, 2), th.ones(1, 1))
        a = Linked(Point(th.zeros(1, 2)), Vector(th.ones(1, 2)))
        b = Linked(e, Vector(th.tensor([[1.0, -1.0]])))
        with_second = measure.member(a, b)
        without = measure.member(a, b.translate(th.zeros(1, 2)))
        self.assertGreater(float(with_second), float(without))
        self.assertIsNone(b.translate(th.zeros(1, 2)).second)


class TestSoftBoxes(TestCase):

    def test_ratio_is_the_hard_ratio_for_small_beta(self):
        m = GumbelVolumeRatio(beta=0.01)
        a = Box(th.zeros(1, 2), th.ones(1, 2))
        b = Box(th.tensor([[1.0, 0.0]]), th.ones(1, 2))
        self.assertAlmostEqual(float(m.ratio(a, b)), 0.5, delta=0.02)
        self.assertAlmostEqual(float(m.ratio(a, a)), 1.0, delta=0.02)
        far = Box(th.full((1, 2), 10.0), th.ones(1, 2))
        self.assertLess(float(m.ratio(a, far)), 1e-3)
        # Smoothed: positive even for disjoint boxes (at a moderate scale).
        self.assertGreater(float(GumbelVolumeRatio(beta=1.0).ratio(a, far)), 0)

    def test_costs_and_power(self):
        a = Box(th.zeros(1, 2), th.ones(1, 2))
        b = Box(th.tensor([[1.0, 0.0]]), th.ones(1, 2))
        phi = float(GumbelVolumeRatio(0.01).ratio(a, b))
        self.assertAlmostEqual(float(GumbelVolumeRatio(0.01, 2).inside(a, b)), (1 - phi) ** 2, 5)
        self.assertAlmostEqual(float(GumbelVolumeRatio(0.01, 1).apart(a, b)), phi, 5)

    def test_soft_intersection_is_symmetric_and_carries_beta(self):
        a, b = _boxes(4, 3, 1), _boxes(4, 3, 2)
        sa, sb = SoftBox(a.centre, a.offset, 0.5), SoftBox(b.centre, b.offset, 0.5)
        r1, _ = sa.intersect(sb)
        r2, _ = sb.intersect(sa)
        self.assertTrue(th.allclose(r1.centre, r2.centre, atol=1e-6))
        self.assertEqual(r1.beta, 0.5)

    def test_point_to_region_is_zero_iff_inside(self):
        m = PointToRegion()
        box = Box(th.zeros(2, 2), th.ones(2, 2))
        self.assertEqual(float(m.member(Point(th.tensor([[0.5, -0.5], [2.0, 0.0]])), box)[0]),
                         0.0)
        self.assertEqual(float(m.member(Point(th.tensor([[0.5, -0.5], [2.0, 0.0]])), box)[1]),
                         1.0)

    def test_by_region_needs_a_rule(self):
        m = ByRegion([(Box, Box, GumbelVolumeRatio()), (Point, Point, PointDistance())])
        with self.assertRaises(TypeError):
            m.inside(Point(th.zeros(1, 2)), Box(th.zeros(1, 2), th.ones(1, 2)))

    def test_context_transform_needs_a_named_class(self):
        classes = ContextualSoftBoxClasses(4, 3, num_relations=2)
        relations = RelationsByRegion([(SoftBox, ContextTransform(classes))])
        box = classes(th.arange(4))
        in_r = relations.preimage(th.zeros(4, dtype=th.long), box)
        self.assertEqual(in_r.centre.shape, (4, 3))
        with self.assertRaises(ConstructorUnavailable):
            relations.preimage(th.zeros(4, dtype=th.long), box.translate(th.zeros(4, 3)))


class _RankingEL(ComposedEL):
    """A ComposedEL whose module comes from ``self.build`` and whose loss is the ranking loss
    of EIKE for negatives (the core ComposedEL adds positive and negative costs)."""

    def __init__(self, dataset, build, ranking=False, without=(), **kwargs):
        self.build, self.ranking, self.without = build, ranking, tuple(without)
        super().__init__(dataset, embed_dim=8, learning_rate=0.01, **kwargs)

    @property
    def training_datasets(self):
        # Geometries without a conjunction cannot read gci1 (ComposedEL does not know it).
        return {k: v for k, v in super().training_datasets.items() if k not in self.without}

    def init_module(self):
        sizes = (len(self.class_index_dict), len(self.object_property_index_dict),
                 len(self.individual_index_dict))
        self.module = self.build(*sizes)

    def compute_loss(self, pos_scores, neg_scores=None):
        if not self.ranking or neg_scores is None:
            return super().compute_loss(pos_scores, neg_scores)
        k = neg_scores.numel() // pos_scores.numel()
        return th.relu(pos_scores.view(1, -1) - neg_scores.view(k, -1)).mean()


class TestCombinations(TestCase):
    """Parts of the group with parts of other methods; a model trained for a few epochs has
    a lower loss."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def assertTrains(self, build, epochs=40, ranking=False, without=()):
        th.manual_seed(0)
        model = _RankingEL(self.dataset, build, ranking=ranking, without=without)

        def total():
            with th.no_grad():
                return sum(float(model.module(d[:], f).mean())
                           for f, d in model.training_datasets.items() if len(d))
        before = total()
        model.train(epochs=epochs)
        after = total()
        self.assertLess(after, before)
        return model

    def test_points_with_balls_and_boxes(self):
        d = 8
        self.assertTrains(lambda n, r, i: ComposedGeometry(
            BallClasses(n, d), Translation(r, d), CentreDistance(), Objective(0.05),
            individuals=Points(i, d, "ball")))
        self.assertTrains(lambda n, r, i: ComposedGeometry(
            BoxClasses(n, d), Translation(r, d), Bounds(), Objective(0.05),
            individuals=Points(i, d, "box")))

    def test_soft_boxes_with_the_hard_measure_and_hard_boxes_with_the_soft_measure(self):
        d = 8
        self.assertTrains(lambda n, r, i: ComposedGeometry(
            SoftBoxClasses(n, d, beta=0.5), Translation(r, d), Bounds(), Objective(0.05),
            individuals=SoftBoxClasses(i, d)))
        self.assertTrains(lambda n, r, i: ComposedGeometry(
            BoxClasses(n, d), Translation(r, d), GumbelVolumeRatio(0.5), Objective(),
            individuals=BoxClasses(i, d)))

    def test_octagons_and_prisms_with_the_semantic_objective_and_translations(self):
        d = 8
        self.assertTrains(lambda n, r, i: ComposedGeometry(
            OctagonClasses(n, d), Translation(r, d), OctagonBounds(), ShapeObjective(0.05),
            individuals=Points(i, d, "octagon")))
        self.assertTrains(lambda n, r, i: ComposedGeometry(
            PrismClasses(n, d), Translation(r, d), PrismBounds(), ShapeObjective(0.05),
            individuals=Points(i, d, "prism")))

    def test_ellipsoids_with_translations_and_point_individuals(self):
        d = 8
        self.assertTrains(lambda n, r, i: ComposedGeometry(
            EllipsoidClasses(n, d), Translation(r, d), EllipsoidMeasure(),
            ShapeObjective(0.0), individuals=Points(i, d, "point")), without=("gci1",))

    def test_the_methods_from_their_functions(self):
        for key in ("octagon", "prism", "concept2box"):
            self.assertTrains(lambda n, r, i, key=key: METHODS[key](n, r, i, embed_dim=8))
        self.assertTrains(lambda n, r, i: METHODS["eike"](n, r, i, embed_dim=8), ranking=True,
                          without=("gci1", "gci2", "gci3", "gci1_bot", "gci0_bot", "gci3_bot"))

    def test_the_methods_through_composed_el(self):
        for key in ("octagon", "prism", "concept2box", "eike"):
            model = ComposedEL(self.dataset, method=key, embed_dim=8)
            self.assertIsInstance(model.module, ComposedGeometry)
            model.train(epochs=2)
        # EIKE reads the subsumptions and the assertions only: the other forms of the ontology
        # are left out of its training data (it has no conjunction).
        model = ComposedEL(self.dataset, method="eike", embed_dim=8)
        self.assertEqual(set(model.training_datasets) - {"gci0"}, set())

    def test_abox_losses_decrease_with_point_individuals(self):
        # The family dataset has no assertions: train the individual forms directly.
        th.manual_seed(1)
        rows = {"class_assertion": th.tensor([[0, 1], [1, 2], [2, 0], [3, 1]]),
                "object_property_assertion": th.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3]])}
        for key in ("octagon", "prism", "eike", "concept2box"):
            g = METHODS[key](4, 2, 5, embed_dim=8)
            opt = th.optim.Adam(g.parameters(), lr=0.02)

            def total():
                return sum(g(r, f).mean() for f, r in rows.items())
            before = float(total())
            for _ in range(60):
                opt.zero_grad()
                total().backward()
                opt.step()
            self.assertLess(float(total()), before, key)
