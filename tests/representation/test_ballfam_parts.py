"""EmEL++, EmEL(var) and RegD as parts: equal losses, and the parts outside their paper."""
from unittest import TestCase

import torch as th

from mowl.models import ComposedEL
from mowl.nn.el.emelpp.module import EmELPPModule, EmELVarModule
from mowl.nn.el.regd.module import RegDELBEModule, RegDELEmModule
from mowl.representation import (geometry_of, ComposedGeometry, METHODS, MEASURES,
                                 RELATION_MODELS, BoxClasses, BallClasses, Bounds,
                                 Translation, ConstructorUnavailable, Objective, Relations)
from mowl.representation.library.ballfam_parts import (
    BoundaryDepth, RegDObjective, ToleranceTranslation)
from tests.datasetFactory import FamilyDataset
from tests.representation.test_adapters import _rows, GCIS, N, R, I, D

ALL = GCIS + ("class_assertion", "object_property_assertion")


def _same(test, module, forms, negs=(False,), geometry=None):
    geometry = geometry if geometry is not None else geometry_of(module)
    test.assertIsInstance(geometry, ComposedGeometry)
    for form in forms:
        rows = _rows(form)
        for neg in negs:
            expected = module(rows, form, neg=neg).reshape(-1, 1)
            got = geometry(rows, form, neg=neg)
            test.assertEqual(got.shape, expected.shape, (form, neg))
            test.assertTrue(th.allclose(got, expected, atol=1e-5), (form, neg))


def _role_rows(n=30):
    g = th.Generator().manual_seed(3)
    ri = th.stack([th.randint(0, R, (n,), generator=g) for _ in range(2)], dim=1)
    rc = th.stack([th.randint(0, R, (n,), generator=g) for _ in range(3)], dim=1)
    return ri, rc


class TestEmELPP(TestCase):
    cls = EmELPPModule

    def make(self, **kw):
        th.manual_seed(1)
        module = self.cls(N, R, I, embed_dim=D, margin=0.07, **kw)
        # radii small enough that hinges are active in both directions
        return module

    def test_positive_and_negative_losses(self):
        m = self.make()
        _same(self, m, ALL)
        _same(self, m, ("gci2", "object_property_assertion"), negs=(True,))
        # forms without a negative loss return the positive one, as in the module
        _same(self, m, GCIS, negs=(True,))

    def test_options(self):
        m = self.make()
        m.intersection_radius_term = False
        _same(self, m, ("gci1",))
        m = self.make()
        m.negative_loss_form = "linear"
        _same(self, m, ("gci2", "object_property_assertion"), negs=(True,))
        m = self.make()
        m.radius_form = "relu"
        with th.no_grad():
            m.class_rad.weight.mul_(-1)
        _same(self, m, ALL)
        m = self.make()
        m.subclass_relation = th.nn.Parameter(th.randn(D))
        _same(self, m, ("gci0", "class_assertion"))

    def test_role_axioms(self):
        ri, rc = _role_rows()
        for direction in ("row", "batch"):
            m = self.make()
            m.role_direction = direction
            geometry = geometry_of(m)
            self.assertTrue(geometry.role_axiom_capable)
            for rows, form in ((ri, "role_inclusion"), (rc, "role_chain")):
                self.assertTrue(th.allclose(geometry(rows, form),
                                            m(rows, form).reshape(-1, 1), atol=1e-6),
                                (form, direction))

    def test_regularization_and_sharing(self):
        m = self.make()
        g = geometry_of(m)
        self.assertEqual(float(g.regularization_loss()), 0.0)
        self.assertIs(g.class_embed, m.class_embed)
        g(_rows("gci2"), "gci2").sum().backward()
        self.assertIsNotNone(m.rel_embed.weight.grad)

    def test_from_new_tables(self):
        th.manual_seed(0)
        g = METHODS["emelpp"](N, R, I, embed_dim=D, margin=0.1)
        self.assertIsInstance(g, ComposedGeometry)
        for form in ALL:
            self.assertEqual(g(_rows(form), form).shape[1], 1)
        self.assertEqual(g.parts["relations"], "translation")


class TestEmELVar(TestEmELPP):

    cls = EmELVarModule

    def make(self, **kw):
        th.manual_seed(1)
        m = EmELVarModule(N, R, I, embed_dim=D, margin=0.07,
                          sigma_reg_weight=kw.pop("sigma_reg_weight", 1.0), **kw)
        with th.no_grad():
            m.rel_sigma.weight.copy_(th.rand(R, 1) * 0.3 - 0.1)
        return m

    def test_sigma_weight(self):
        _same(self, self.make(sigma_reg_weight=0.3), ("gci2", "gci3", "object_property_assertion"))

    def test_parts(self):
        g = geometry_of(self.make())
        self.assertEqual(g.parts["relations"], "translation with tolerance")

    def test_tolerance_is_shared(self):
        m = self.make()
        g = geometry_of(m)
        self.assertIs(g.relation_model.tolerances, m.rel_sigma)

    def test_exists_is_fuzzy(self):
        g = geometry_of(self.make())
        ids = th.arange(3)
        region = g.classes(ids)
        sharp = g.relation_model.sharp_preimage(ids, region)
        fuzzy = g.exists(ids, region)
        self.assertTrue(th.allclose(fuzzy.radius - sharp.radius,
                                    g.relation_model.tolerance(ids)))


class TestRegD(TestCase):

    def make_box(self, **kw):
        th.manual_seed(2)
        m = RegDELBEModule(N, R, I, embed_dim=D, margin=0.07, **kw)
        with th.no_grad():
            m.class_offset.weight[0, 0] = 0.0   # an offset below eps
        return m

    def make_ball(self, **kw):
        th.manual_seed(3)
        return RegDELEmModule(N, R, I, embed_dim=D, margin=0.07, **kw)

    def check(self, make, **kw):
        m = make(**kw)
        _same(self, m, ALL)
        _same(self, m, ("gci0", "gci2", "object_property_assertion"), negs=(True,))
        return m

    def test_box_losses(self):
        self.check(self.make_box)

    def test_box_options(self):
        for kw in ({"lamda": 0.5, "p": 2}, {"depth_on_containment": False},
                   {"gamma1": 0.1, "gamma2": 3.0}, {"flip_gci2": True}):
            self.check(self.make_box, **kw)
        m = self.make_box()
        m.enable_all_negative_losses()
        _same(self, m, ("gci0_bot", "gci1_bot", "gci3_bot"), negs=(True,))

    def test_ball_losses(self):
        self.check(self.make_ball)
        m = self.make_ball()
        self.assertTrue(th.allclose(geometry_of(m).regularization_loss(),
                                    m.regularization_loss()))

    def test_ball_options(self):
        for kw in ({"p": 1}, {"depth_on_containment": False}, {"flip_gci2": True},
                   {"gci1_radius_term": False}, {"gamma1": 0.1, "gamma2": 3.0}):
            self.check(self.make_ball, **kw)
        m = self.make_ball()
        m.enable_all_negative_losses()
        _same(self, m, ("gci1", "gci0_bot", "gci1_bot", "gci3_bot"), negs=(True,))

    def test_ball_softplus(self):
        # by default every form reads the softplus radius, the bottom forms, class assertions
        # and the negatives of the conjunctions included
        m = self.make_ball(radius="softplus")
        self.assertTrue(geometry_of(m).objective.consistent_radius)
        _same(self, m, ALL)
        _same(self, m, ALL, negs=(True,))
        m.enable_all_negative_losses()
        _same(self, m, ("gci1", "gci0_bot", "gci1_bot", "gci3_bot"), negs=(True,))

    def test_ball_softplus_with_the_earlier_mixed_radii(self):
        # consistent_radius=False: those forms read |radius|; the adapter selects the option
        m = self.make_ball(radius="softplus", consistent_radius=False)
        g = geometry_of(m)
        self.assertFalse(g.objective.consistent_radius)
        _same(self, m, ALL)
        _same(self, m, ("gci0", "gci2", "gci1", "gci1_bot"), negs=(True,))
        m.enable_all_negative_losses()
        _same(self, m, ("gci1", "gci0_bot", "gci1_bot", "gci3_bot"), negs=(True,))
        # and the two readings differ
        fixed = self.make_ball(radius="softplus")
        rows = _rows("gci0_bot")
        self.assertFalse(th.allclose(geometry_of(fixed)(rows, "gci0_bot"),
                                     g(rows, "gci0_bot")))

    def test_energy_is_the_modules(self):
        m = self.make_box()
        g = geometry_of(m)
        rows = _rows("gci0")
        with m.scoring():
            expected = m(rows, "gci0").reshape(-1, 1)
        got = g.measure.energy(g.classes(rows[:, 0]), g.classes(rows[:, 1]))
        self.assertTrue(th.allclose(got, expected, atol=1e-6))

    def test_parts(self):
        for m, classes, conj in ((self.make_box(), "box", "exact"),
                                 (self.make_ball(), "ball", "approximate")):
            p = geometry_of(m).parts
            self.assertEqual((p["classes"], p["relations"], p["measure"], p["conjunction"]),
                             (classes, "translation", "boundary and depth", conj))

    def test_from_new_tables(self):
        th.manual_seed(0)
        for key in ("regd", "regd_elem"):
            g = METHODS[key](N, R, I, embed_dim=D)
            for form in ALL:
                self.assertEqual(g(_rows(form), form).shape[1], 1)
            self.assertTrue(isinstance(g.objective, RegDObjective))


class TestPartsOutsideTheirPaper(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def _train(self, **kw):
        th.manual_seed(0)
        model = ComposedEL(self.dataset, embed_dim=8, margin=0.05, **kw)

        def total():
            return sum(float(model.module(d[:], f).mean())
                       for f, d in model.training_datasets.items() if len(d))
        before = total()
        model.train(epochs=40)
        self.assertLess(total(), before, kw)
        return model

    def test_registered(self):
        self.assertIn("boundary and depth", MEASURES)
        self.assertIn("translation with tolerance", RELATION_MODELS)
        for key in ("emelpp", "emelvar", "regd", "regd_elem"):
            self.assertIn(key, METHODS)

    def test_tolerance_with_balls_and_boxes(self):
        for classes in ("ball", "box"):
            objective = Objective(0.05, slack=True, tolerance_weight=0.5)
            model = self._train(classes=classes, relations="translation with tolerance",
                                objective=objective)
            self.assertEqual(model.module.parts["relations"], "translation with tolerance")

    def test_regd_measure_with_both_regions_and_other_relations(self):
        for classes in ("ball", "box"):
            for relations in ("translation", "translation with tolerance"):
                self._train(classes=classes, relations=relations,
                            measure="boundary and depth")

    def test_regd_measure_with_elbe_objective(self):
        self._train(method="regd")
        self._train(method="regd_elem")

    def test_emelpp_and_emelvar_train(self):
        self._train(method="emelpp")
        self._train(method="emelvar")

    def test_role_axioms_of_a_relation_model(self):
        # role axioms with the tolerance model on boxes: loss decreases under training
        th.manual_seed(0)
        g = ComposedGeometry(BoxClasses(6, 4), ToleranceTranslation(4, 4), Bounds(),
                             Objective(0.0))
        rows = th.tensor([[0, 1, 2], [1, 1, 3]])
        opt = th.optim.Adam(g.parameters(), lr=0.05)
        before = float(g(rows, "role_chain").mean())
        for _ in range(60):
            opt.zero_grad()
            g(rows, "role_chain").mean().backward()
            opt.step()
        self.assertLess(float(g(rows, "role_chain").mean()), before)
        inc = th.tensor([[0, 1]])
        self.assertEqual(g(inc, "role_inclusion").shape, (1, 1))

    def test_plain_translation_has_no_role_axioms(self):
        class NoRoles(Relations):  # a relation model that does not define role axioms
            def preimage(self, ids, region):
                return region

        g = ComposedGeometry(BallClasses(4, 3), NoRoles(), MEASURES["centre distance"](),
                             Objective())
        self.assertFalse(g.role_axiom_capable)
        with self.assertRaises(ConstructorUnavailable):
            g.cost("role_inclusion", th.tensor([[0, 1]]))
        g = ComposedGeometry(BallClasses(4, 3), Translation(2, 3, direction="row"),
                             MEASURES["centre distance"](), Objective())
        self.assertEqual(g.cost("role_inclusion", th.tensor([[0, 1]])).shape, (1, 1))
        self.assertTrue(g.role_axiom_capable)

    def test_boundary_depth_is_an_inclusion(self):
        for classes in (BallClasses(5, 3), BoxClasses(5, 3)):
            measure = BoundaryDepth()
            a, b = classes(th.arange(5)), classes(th.arange(5))
            self.assertTrue(th.all(measure.inside(a, b) >= 0))
            # a region inside a larger copy of itself costs nothing, the reverse does
            big = measure.regularise(b)
            big = big.grow(th.tensor(2.0)) if hasattr(big, "grow") else \
                type(big)(big.centre, big.radius + 2.0, None)
            self.assertTrue(th.allclose(measure.inside(a, big), th.zeros(5, 1)))
            self.assertTrue(th.all(measure.inside(big, a) > 0))
