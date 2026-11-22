"""The volume-based multibox as a geometry and as parts, and the sampled volume measure."""
from unittest import TestCase

import numpy as np
import torch as th

from mowl.models import ComposedEL
from mowl.nn import VolumeMultiBoxELModule
from mowl.representation import (
    Ball, Box, BallClasses, BoxClasses, ConstructorUnavailable, ComposedGeometry, Objective,
    Translation, geometry_of, METHODS)
from mowl.representation.library.multibox_methods import mcmultibox
from mowl.representation.library.multibox_parts import (
    DomainProjection, MembershipIntersection, PackedUnionClasses, PairUnions, SampledVolume,
    UnionClasses, UnionOfBoxes, membership_sampler)
from tests.datasetFactory import FamilyDataset
from tests.representation.test_adapters import N, R, I, _rows

DIM, BOXES, POINTS = 3, 3, 400


def _module(**kwargs):
    options = dict(embed_dim=DIM, num_boxes=BOXES, num_points=POINTS, point_chunk_size=128)
    options.update(kwargs)
    return VolumeMultiBoxELModule(N, R, I, **options)


def _chain_rows(form, n=12, seed=2):
    g = th.Generator().manual_seed(seed)
    return th.randint(0, R, (n, 2 if form == "role_inclusion" else 3), generator=g)


FORMS = ("gci0", "gci1", "gci2", "gci3", "role_inclusion", "role_chain")


def _rows_for(form):
    if form.startswith("role"):
        return _chain_rows(form)
    return _rows(form, n=12)


class TestVolumeGeometry(TestCase):

    def test_losses_equal_the_module(self):
        module = _module()
        geometry = geometry_of(module)
        self.assertIsInstance(geometry, ComposedGeometry)
        for form in FORMS:
            rows = _rows_for(form)
            expected = module(rows, form).reshape(-1, 1)
            got = geometry(rows, form)
            self.assertEqual(got.shape, expected.shape)
            self.assertTrue(th.allclose(got, expected, atol=1e-5), form)
            self.assertGreater(float(expected.std()), 0.0)

    def test_options_of_the_module(self):
        for kwargs in ({"aggregate": "mean"}, {"temperature": 0.3}, {"bound": 2.0},
                       {"num_boxes": 1}):
            module = _module(**kwargs)
            geometry = geometry_of(module)
            self.assertEqual(geometry.measure.aggregate, module.aggregate)
            for form in FORMS:
                rows = _rows_for(form)
                self.assertTrue(th.allclose(geometry(rows, form),
                                            module(rows, form).reshape(-1, 1), atol=1e-5),
                                (kwargs, form))

    def test_parameters_are_shared(self):
        module = _module()
        geometry = geometry_of(module)
        self.assertIs(geometry.class_model.centres, module.class_embed)
        self.assertIs(geometry.relation_model.vectors, module.rel_embed)
        rows = _rows("gci2", n=8)
        geometry(rows, "gci2").sum().backward()
        self.assertIsNotNone(module.class_embed.grad)
        self.assertIsNotNone(module.rel_embed.grad)
        with th.no_grad():
            module.class_embed.add_(0.1)
        self.assertTrue(th.allclose(geometry(rows, "gci2"),
                                    module(rows, "gci2").reshape(-1, 1), atol=1e-5))

    def test_fresh_tables_from_the_same_seed_equal_the_module(self):
        module = _module(seed=7)
        geometry = mcmultibox(N, R, I, embed_dim=DIM, num_boxes=BOXES, num_points=POINTS,
                              point_chunk_size=128, seed=7)
        for form in FORMS:
            rows = _rows_for(form)
            self.assertTrue(th.allclose(geometry(rows, form),
                                        module(rows, form).reshape(-1, 1), atol=1e-5), form)

    def test_regions_and_costs_of_the_common_interface(self):
        geometry = geometry_of(_module())
        ids = th.arange(4)
        region = geometry.classes(ids)
        self.assertIsInstance(region, UnionOfBoxes)
        self.assertEqual(region.centre.shape, (4, BOXES, DIM))
        self.assertEqual(geometry.individuals(th.arange(2)).centre.shape, (2, 1, DIM))
        exists = geometry.exists(th.zeros(4, dtype=th.long), region)
        self.assertIsInstance(exists, DomainProjection)
        both, gap = geometry.conjunction(region, geometry.classes(ids.flip(0)))
        self.assertIsInstance(both, MembershipIntersection)
        self.assertEqual(gap.shape, (4, 1))
        inside = geometry.inside(region, region)
        self.assertTrue(th.allclose(inside, th.zeros(4, 1), atol=1e-6))  # A in A
        apart = geometry.apart(region, region)
        self.assertTrue(th.allclose(apart, th.ones(4, 1), atol=1e-4))  # A meets A fully
        self.assertEqual(geometry.member(th.arange(2), geometry.classes(th.arange(2))).shape,
                         (2, 1))
        with self.assertRaises(ConstructorUnavailable):
            geometry.negation(region)

    def test_what_the_module_does_not_define_raises(self):
        geometry = geometry_of(_module())
        for form in ("gci0_bot", "gci3_bot"):
            with self.assertRaises((NotImplementedError, ConstructorUnavailable)):
                geometry(_rows(form, n=4), form)
        # extensions (the module raises): the measure has `apart`, and the individuals are
        # boxes, so these forms are costs on them
        for form in ("gci1_bot", "class_assertion", "object_property_assertion"):
            self.assertEqual(geometry(_rows(form, n=4), form).shape, (4, 1))
        self.assertEqual(geometry.neg_capable_gcis, frozenset())
        self.assertTrue(geometry.role_axiom_capable)

    def test_regularisation_is_zero(self):
        self.assertEqual(float(geometry_of(_module()).regularization_loss()), 0.0)

    def test_registered(self):
        self.assertIs(METHODS["mcmultibox"], mcmultibox)


class TestSampledVolume(TestCase):
    """The measure works for every region with a membership function."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def test_membership_of_balls_and_boxes_is_positive_iff_inside(self):
        points = th.tensor([[0.0, 0.0], [0.5, 0.0], [3.0, 3.0]])
        ball = Ball(th.zeros(1, 2), th.ones(1, 1))
        box = Box(th.zeros(1, 2), th.ones(1, 2))
        for region in (ball, box):
            m = membership_sampler(region)(points)
            self.assertEqual(m.shape, (1, 3))
            self.assertTrue(bool((m[0, :2] > 0).all()) and float(m[0, 2]) == 0.0)

    def test_inside_is_a_volume_ratio_for_balls_and_boxes(self):
        measure = SampledVolume(num_points=4000)
        small = Ball(th.zeros(1, 2), th.full((1, 1), 0.3))
        big = Ball(th.zeros(1, 2), th.full((1, 1), 0.9))
        self.assertLess(float(measure.inside(small, big)), 0.02)
        self.assertGreater(float(measure.inside(big, small)), 0.5)
        sb = Box(th.zeros(1, 2), th.full((1, 2), 0.3))
        bb = Box(th.zeros(1, 2), th.full((1, 2), 0.9))
        self.assertLess(float(measure.inside(sb, bb)), 0.02)
        self.assertGreater(float(measure.inside(bb, sb)), 0.5)
        # a ball inside a box, mixed families
        self.assertLess(float(measure.inside(small, bb)), 0.02)

    def test_conjunction_of_balls_works_with_this_measure(self):
        measure = SampledVolume(num_points=2000)
        a = Ball(th.tensor([[-0.2, 0.0]]), th.full((1, 1), 0.6))
        b = Ball(th.tensor([[0.2, 0.0]]), th.full((1, 1), 0.6))
        c = Ball(th.zeros(1, 2), th.full((1, 1), 0.9))
        self.assertLess(float(measure.intersection_inside(a, b, c)), 0.05)
        self.assertGreater(float(measure.intersection_apart(a, b, c)), 0.5)

    def test_with_ball_and_box_classes_and_translations(self):
        for classes in ("ball", "box"):
            th.manual_seed(0)
            model = ComposedEL(self.dataset, classes=classes, relations="translation",
                               measure=SampledVolume(num_points=300, chunk_size=128),
                               objective=Objective(0.0), embed_dim=4)
            data = {f: d[:] for f, d in model.training_datasets.items()
                    if len(d) and f in ("gci0", "gci1", "gci2", "gci3")}

            def total():
                return sum(float(model.module(rows, f).mean()) for f, rows in data.items())

            before = total()
            model.train(epochs=30)
            self.assertLess(total(), before, classes)

    def test_with_union_classes_and_head_tail_relations(self):
        th.manual_seed(0)
        np.random.seed(0)      # the negatives of the model are drawn with numpy
        model = ComposedEL(self.dataset, classes="union of boxes",
                           relations="head and tail boxes",
                           measure=SampledVolume(num_points=300, chunk_size=128),
                           objective=Objective(0.0), embed_dim=4)
        data = {f: d[:] for f, d in model.training_datasets.items()
                if len(d) and f in ("gci0", "gci1", "gci2", "gci3")}
        before = sum(float(model.module(r, f).mean()) for f, r in data.items())
        model.train(epochs=30)
        after = sum(float(model.module(r, f).mean()) for f, r in data.items())
        self.assertLess(after, before)

    def test_the_method_trains_as_a_composed_model(self):
        th.manual_seed(0)
        model = ComposedEL(self.dataset, method="mcmultibox", embed_dim=3, num_boxes=2,
                           num_points=300, point_chunk_size=128)
        data = {f: d[:] for f, d in model.training_datasets.items()
                if len(d) and f in ("gci0", "gci1", "gci2", "gci3")}
        before = sum(float(model.module(r, f).mean()) for f, r in data.items())
        model.train(epochs=30)
        after = sum(float(model.module(r, f).mean()) for f, r in data.items())
        self.assertLess(after, before)
        self.assertIsInstance(model.module, ComposedGeometry)

    def test_the_parts_accept_each_other(self):
        # the packed class model and the pair relation model with a plain objective
        classes = PackedUnionClasses(4, 3, components=2)
        relations = PairUnions(2, 3, components=2)
        geometry = ComposedGeometry(classes, relations, SampledVolume(num_points=200),
                                    Objective(0.0))
        rows = th.tensor([[0, 1, 2], [1, 0, 3]])
        self.assertEqual(geometry(rows, "gci2").shape, (2, 1))
        self.assertIsInstance(UnionClasses(3, 2).forward(th.tensor([0])), UnionOfBoxes)
        self.assertIsNotNone(Translation(2, 3))
        self.assertIsNotNone((BallClasses(2, 2), BoxClasses(2, 2)))
