"""Box2EL, MultiBoxEL and MultiELBE as combinations of parts: the losses equal those of the
existing modules, and the parts work outside their papers."""
from unittest import TestCase

import torch as th

from mowl.models import ComposedEL
from mowl.nn import ELBEModule
from mowl.baselines.nn.el.boxsquaredel.module import BoxSquaredELModule
from mowl.baselines.nn.el.multiboxel.module import MultiBoxELModule
from mowl.baselines.nn.el.multielbe.module import MultiELBEModule
from mowl.representation import (
    Ball, BallClasses, Bounds, Box, BoxClasses, CentreDistance, ComposedGeometry,
    Objective, Translation, compose, geometry_of, METHODS)
from mowl.representation.library.multibox_methods import box2el, multiboxel, multielbe
from mowl.representation.library.multibox_objective import (
    ComponentSpread, bump_penalty)
from mowl.representation.library.multibox_parts import (
    BumpedBox, HeadTailRelations, UnionClasses, UnionMeasure, UnionOfBoxes)
from tests.datasetFactory import FamilyDataset
from tests.representation.test_adapters import GCIS, _rows, D, I, N, R


def _role_rows(form, n=30, seed=3):
    g = th.Generator().manual_seed(seed)
    count = 2 if form == "role_inclusion" else 3
    return th.randint(0, R, (n, count), generator=g)


FORMS = GCIS + ("class_assertion", "object_property_assertion")


def _randomise(module, seed=5):
    """Move every parameter away from the initialisation, so that equality is not an accident
    of symmetric starting values."""
    g = th.Generator().manual_seed(seed)
    with th.no_grad():
        for p in module.parameters():
            p.add_(0.3 * th.randn(p.shape, generator=g))
    return module


class EqualityCase:
    """Shared checks; subclasses give ``make(**kwargs)`` and ``forms``."""

    forms = FORMS
    negatives = ("gci2", "object_property_assertion")

    def assertSame(self, module, forms, negs=(False,)):
        geometry = geometry_of(module)
        self.assertIsInstance(geometry, ComposedGeometry)
        for form in forms:
            rows = _role_rows(form) if form.startswith("role") else _rows(form)
            for neg in negs:
                expected = module(rows, form, neg=neg).reshape(-1, 1)
                got = geometry(rows, form, neg=neg)
                self.assertEqual(got.shape, expected.shape, (form, neg))
                self.assertTrue(th.allclose(got, expected, atol=1e-6, rtol=1e-5),
                                (form, neg, float((got - expected).abs().max())))
        reg = th.as_tensor(module.regularization_loss(), dtype=th.float)
        self.assertTrue(th.allclose(geometry.regularization_loss(), reg, atol=1e-6))
        return geometry

    def test_positive_losses(self):
        self.assertSame(self.make(), self.forms)

    def test_original_negatives(self):
        self.assertSame(self.make(), self.negatives, negs=(True,))

    def test_parameters_are_shared(self):
        module = self.make()
        geometry = geometry_of(module)
        before = module.class_embed.weight.detach().clone() \
            if module.class_embed is not None else None
        geometry(_rows("gci0"), "gci0").sum().backward()
        self.assertIsNotNone(module.class_embed.weight.grad)
        self.assertTrue(th.equal(module.class_embed.weight.detach(), before))


# ---------------------------------------------------------------------------------------
# Box2EL
# ---------------------------------------------------------------------------------------

class TestBox2EL(EqualityCase, TestCase):
    forms = FORMS + ("role_inclusion", "role_chain")

    def make(self, **kwargs):
        th.manual_seed(1)
        module = BoxSquaredELModule(N, R, I, embed_dim=D, gamma=0.07, delta=1.5,
                                    reg_factor=0.3, **kwargs)
        # the class attribute names of Box2EL differ from the common ones
        module.class_embed = module.class_center
        return _randomise(module)

    def test_dele_negatives(self):
        module = self.make(all_negative_losses=True, epsilon=0.05)
        self.assertSame(module, GCIS, negs=(True, False))

    def test_slack_zero_and_default_gamma(self):
        module = BoxSquaredELModule(N, R, I, embed_dim=D)
        module.class_embed = module.class_center
        self.assertSame(_randomise(module), FORMS + ("role_inclusion", "role_chain"))

    def test_without_individuals(self):
        th.manual_seed(2)
        module = _randomise(BoxSquaredELModule(N, R, embed_dim=D))
        geometry = geometry_of(module)
        self.assertEqual(geometry.num_individuals, 0)
        for form in GCIS:
            self.assertTrue(th.allclose(geometry(_rows(form), form),
                                        module(_rows(form), form).reshape(-1, 1)))

    def test_geometry_regions(self):
        module = self.make()
        g = geometry_of(module)
        ids = th.arange(4)
        region = g.classes(ids)
        self.assertIsInstance(region, BumpedBox)
        self.assertIs(g.class_model.centres, module.class_center)
        self.assertIs(g.class_model.bumps, module.bump_classes)
        # exists r.C is the head box moved by minus the bump of C
        r = th.tensor([0, 1, 2, 0])
        e = g.exists(r, region)
        self.assertTrue(th.allclose(e.centre, module.head_center(r) - module.bump_classes(ids)))
        # conjunction, individuals, member
        box, gap = g.conjunction(g.classes(ids), g.classes(ids.flip(0)))
        self.assertEqual(gap.shape, (4, 1))
        self.assertEqual(g.member(th.arange(2), g.classes(th.arange(2))).shape, (2, 1))

    def test_role_axiom_capable(self):
        self.assertTrue(geometry_of(self.make()).role_axiom_capable)

    def test_catalogue_function_makes_fresh_tables(self):
        g = box2el(N, R, I, embed_dim=D, margin=0.1)
        self.assertEqual(g.parts["relations"], "head and tail boxes")
        self.assertEqual(g(_rows("gci2"), "gci2").shape, (40, 1))
        self.assertIs(METHODS["box2el"], box2el)

    def test_bump_penalty_is_the_only_regulariser(self):
        g = box2el(N, R, I, embed_dim=D, reg_factor=0.5)
        classes = th.linalg.norm(g.class_model.bumps.weight, dim=1).mean()
        individuals = th.linalg.norm(g.individual_model.bumps.weight, dim=1).mean()
        expected = 0.5 * (classes + individuals)
        self.assertTrue(th.allclose(g.regularization_loss(), expected))


# ---------------------------------------------------------------------------------------
# MultiBoxEL
# ---------------------------------------------------------------------------------------

class TestMultiBoxEL(EqualityCase, TestCase):

    def make(self, k=2, **kwargs):
        th.manual_seed(1)
        module = MultiBoxELModule(N, R, I, embed_dim=D, n_components=k, gamma=0.05,
                                  delta=1.5, reg_factor=0.3, **kwargs)
        module.class_embed = module.class_center
        return _randomise(module)

    def test_one_component_equals_box2el(self):
        th.manual_seed(1)
        box = _randomise(BoxSquaredELModule(N, R, I, embed_dim=D, gamma=0.05, delta=1.5))
        pairs = [(n, n) for n in ("class_center", "class_offset", "head_center", "head_offset",
                                  "tail_center", "tail_offset", "bump_classes", "ind_center",
                                  "ind_offset", "bump_individuals")]
        # copy the Box2EL weights into a fresh multi-box module, then compare through parts
        target = self.make(k=1)
        with th.no_grad():
            for src, dst in pairs:
                getattr(target, dst).weight.copy_(getattr(box, src).weight)
        geometry = geometry_of(target)
        for form in FORMS:
            rows = _rows(form)
            for neg in ((False, True) if form in self.negatives else (False,)):
                a = box(rows, form, neg=neg).reshape(-1, 1)
                self.assertTrue(th.allclose(geometry(rows, form, neg=neg), a, atol=1e-6),
                                (form, neg))

    def test_two_and_three_components(self):
        for k in (2, 3):
            module = self.make(k=k)
            self.assertSame(module, FORMS)
            self.assertSame(module, self.negatives, negs=(True,))

    def test_soft_aggregation(self):
        for k in (1, 2, 3):
            module = self.make(k=k, component_aggregation="soft", tau=0.4)
            self.assertSame(module, FORMS)
            self.assertSame(module, self.negatives, negs=(True,))

    def test_soft_temperature_is_followed(self):
        module = self.make(k=2, component_aggregation="soft", tau=0.4)
        geometry = geometry_of(module)
        rows = _rows("gci0")
        before = geometry(rows, "gci0").clone()
        module.tau = 3.0  # an in-place update of the buffer
        after = geometry(rows, "gci0")
        self.assertFalse(th.allclose(before, after))
        self.assertTrue(th.allclose(after, module(rows, "gci0").reshape(-1, 1), atol=1e-6))

    def test_regularisers(self):
        for k in (1, 2, 3):
            module = self.make(k=k, component_reg=0.7, diversity_reg=0.2)
            self.assertSame(module, ("gci0",))
            module = self.make(k=k, component_reg=0.7)
            self.assertSame(module, ("gci1",))

    def test_geometry_regions_and_conjunction(self):
        geometry = geometry_of(self.make(k=3))
        region = geometry.classes(th.arange(5))
        self.assertIsInstance(region, UnionOfBoxes)
        self.assertEqual(region.centre.shape, (5, 3, D))
        both, gap = geometry.conjunction(region, geometry.classes(th.arange(5).flip(0)))
        self.assertEqual(both.centre.shape, (5, 9, D))
        self.assertEqual(gap.shape, (5, 1))
        self.assertEqual(geometry.exists(th.zeros(5, dtype=th.long), region).centre.shape,
                         (5, 1, D))

    def test_overlapping_components(self):
        # wide boxes: the intersections are not empty, so the gated branch is used
        for aggregation in ("hard", "soft"):
            module = self.make(k=3, component_aggregation=aggregation, tau=0.4)
            with th.no_grad():
                module.class_offset.weight.mul_(4.0)
                module.class_center.weight.mul_(0.2)
            self.assertSame(module, ("gci0", "gci1", "gci2", "gci3", "gci1_bot"))

    def test_zero_loss_means_inclusion(self):
        # at gamma 0 a class inside a union component has zero loss; a class outside has not
        geometry = multiboxel(2, 1, embed_dim=2, n_components=2, margin=0.0)
        with th.no_grad():
            geometry.class_model.centres.weight.copy_(th.tensor(
                [[0., 0., 5., 5.], [0., 0., 5., 5.]]))
            geometry.class_model.offsets.weight.copy_(th.tensor(
                [[1., 1., 1., 1.], [2., 2., 2., 2.]]))
        rows = th.tensor([[0, 1]])
        self.assertEqual(float(geometry(rows, "gci0")), 0.0)
        self.assertGreater(float(geometry(rows.flip(1), "gci0")), 0.0)


# ---------------------------------------------------------------------------------------
# MultiELBE
# ---------------------------------------------------------------------------------------

class TestMultiELBE(EqualityCase, TestCase):

    def make(self, k=2, **kwargs):
        th.manual_seed(1)
        return _randomise(MultiELBEModule(N, R, I, embed_dim=D, n_components=k, margin=0.07,
                                          **kwargs))

    def test_two_and_three_components(self):
        for k in (2, 3):
            module = self.make(k=k)
            self.assertSame(module, FORMS)
            self.assertSame(module, self.negatives, negs=(True,))

    def test_soft_aggregation(self):
        for k in (1, 2, 3):
            module = self.make(k=k, component_aggregation="soft", tau=0.4)
            self.assertSame(module, FORMS)
            self.assertSame(module, self.negatives, negs=(True,))

    def test_regularisers(self):
        for k in (1, 3):
            self.assertSame(self.make(k=k, component_reg=0.7, diversity_reg=0.2), ("gci0",))

    def test_one_component_equals_elbe(self):
        th.manual_seed(4)
        elbe = _randomise(ELBEModule(N, R, I, embed_dim=D, margin=0.07))
        target = self.make(k=1)
        with th.no_grad():
            for name in ("class_embed", "class_offset", "rel_embed", "ind_embed",
                         "ind_offset"):
                getattr(target, name).weight.copy_(getattr(elbe, name).weight)
        geometry = geometry_of(target)
        for form in FORMS:
            rows = _rows(form)
            for neg in ((False, True) if form in self.negatives else (False,)):
                a = elbe(rows, form, neg=neg).reshape(-1, 1)
                self.assertTrue(th.allclose(geometry(rows, form, neg=neg), a, atol=1e-6),
                                (form, neg))

    def test_overlapping_components(self):
        for aggregation in ("hard", "soft"):
            module = self.make(k=3, component_aggregation=aggregation, tau=0.4)
            with th.no_grad():
                module.class_offset.weight.mul_(4.0)
                module.class_embed.weight.mul_(0.2)
            self.assertSame(module, ("gci0", "gci1", "gci2", "gci3", "gci1_bot"))

    def test_gci3_inclusion_option(self):
        module = self.make(k=2)
        overlap = geometry_of(module)(_rows("gci3"), "gci3")
        inclusion = multielbe(None, None, embed_dim=D, margin=0.07, n_components=2,
                              gci3="inclusion", tables=module)(_rows("gci3"), "gci3")
        self.assertTrue(bool((inclusion >= overlap - 1e-6).all()))
        self.assertFalse(th.allclose(inclusion, overlap))


# ---------------------------------------------------------------------------------------
# the parts outside their papers
# ---------------------------------------------------------------------------------------

def _loss(model):
    return sum(float(model.module(d[:], f).mean())
               for f, d in model.training_datasets.items() if len(d))


class TestPartsOutsideTheirPapers(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def _trains(self, **kwargs):
        th.manual_seed(0)
        model = ComposedEL(self.dataset, embed_dim=8, **kwargs)
        before = _loss(model)
        model.train(epochs=40)
        self.assertLess(_loss(model), before, kwargs)
        return model

    def test_head_tail_relations_with_plain_boxes_and_a_semantic_objective(self):
        model = self._trains(classes="box", relations="head and tail boxes",
                             objective=Objective(0.02))
        self.assertEqual(model.module.parts["relations"], "head and tail boxes")

    def test_head_tail_relations_with_bumped_boxes(self):
        model = self._trains(classes="box with bump", relations="head and tail boxes",
                             objective=Objective(0.02),
                             regularizers=(bump_penalty(0.1),))
        self.assertIsInstance(model.module.classes(th.arange(2)), BumpedBox)

    def test_head_tail_relations_with_balls(self):
        model = self._trains(classes="ball",
                             relations=lambda n, d: HeadTailRelations(n, d, shape="ball"),
                             objective=Objective(0.02))
        region = model.module.exists(th.tensor([0]), model.module.classes(th.tensor([0])))
        self.assertIsInstance(region, Ball)

    def test_head_tail_relations_with_unions(self):
        self._trains(classes="union of boxes", relations="head and tail boxes",
                     objective=Objective(0.02))

    def test_union_classes_with_translations_and_the_default_objective(self):
        model = self._trains(classes="union of boxes", relations="translation",
                             objective=Objective(0.02))
        self.assertEqual(model.module.parts["classes"], "union of boxes")

    def test_union_classes_with_elbe_objective_and_soft_minimum(self):
        measure = UnionMeasure(Bounds("every"), "soft", 0.5)
        self._trains(classes="union of boxes", relations="translation", measure=measure,
                     objective=Objective(0.02, gci3="overlap", gci2_negative="overlap",
                                         negatives="gci2"),
                     regularizers=(ComponentSpread(0.1),))

    def test_union_measure_on_boxes_equals_the_box_measure(self):
        # a box is a union of one: the lifted measure is the base measure
        g = compose(N, R, embed_dim=D, classes="box", relations="head and tail boxes",
                    measure="union of bounds", objective=Objective(0.03))
        h = compose(N, R, embed_dim=D, classes="box", relations="head and tail boxes",
                    objective=Objective(0.03))
        h.load_state_dict(g.state_dict())
        for form in ("gci0", "gci1", "gci2", "gci3"):
            self.assertTrue(th.allclose(g(_rows(form), form), h(_rows(form), form),
                                        atol=1e-6), form)

    def test_new_methods_are_registered(self):
        for key in ("box2el", "multiboxel", "multielbe", "mcmultibox"):
            self.assertIn(key, METHODS)
        model = ComposedEL(self.dataset, method="multiboxel", embed_dim=6, margin=0.0,
                           n_components=2)
        model.train(epochs=2)
        self.assertIsInstance(model.module, ComposedGeometry)
        model = ComposedEL(self.dataset, method="box2el", embed_dim=6)
        model.train(epochs=2)
        model = ComposedEL(self.dataset, method="multielbe", embed_dim=6, margin=0.05,
                           n_components=2)
        model.train(epochs=2)

    def test_classes_type_checks(self):
        with self.assertRaises(TypeError):
            ComposedGeometry(BallClasses(N, D), Translation(R, D), UnionMeasure())
        # a union measure compares boxes
        ComposedGeometry(BoxClasses(N, D), Translation(R, D), UnionMeasure())
        with self.assertRaises(TypeError):
            ComposedGeometry(UnionClasses(N, D), Translation(R, D), CentreDistance())
        self.assertIs(Box, BoxClasses.region)
