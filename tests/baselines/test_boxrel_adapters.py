"""The geometry of the direct TransBox, BoxEL, BoxSEL and ELK-Em modules (the baselines) gives
the modules' numbers, for every form and every option that changes a loss, with the modules'
parameters shared. (The modules users get are geometries themselves; ``geometry_of`` describes
the direct modules, and the modules of earlier releases.)"""
from unittest import TestCase

import torch as th

from mowl.baselines.nn.el.boxel.module import BoxELModule
from mowl.baselines.nn.el.boxsel.module import BoxSELModule
from mowl.baselines.nn.el.elkem.module import ELKEmModule
from mowl.baselines.nn.el.transbox.module import TransBoxModule
from mowl.representation import geometry_of, ComposedGeometry
from mowl.training import Regularization

N, R, I, D = 12, 3, 5, 6
FORM_COLUMNS = {"gci0": "cc", "gci0_bot": "cc", "gci1": "ccc", "gci1_bot": "ccc",
                "gci2": "crc", "gci3": "rcc", "gci3_bot": "rcc", "class_assertion": "ic",
                "object_property_assertion": "iri", "role_inclusion": "rr",
                "role_chain": "rrr"}
HIGH = {"c": N, "r": R, "i": I}
GCIS = ("gci0", "gci0_bot", "gci1", "gci1_bot", "gci2", "gci3", "gci3_bot")
ASSERTIONS = ("class_assertion", "object_property_assertion")
ROLES = ("role_inclusion", "role_chain")


def rows(form, n=40, seed=0):
    g = th.Generator().manual_seed(seed)
    return th.stack([th.randint(0, HIGH[k], (n,), generator=g)
                     for k in FORM_COLUMNS[form]], dim=1)


class Case:
    """Subclasses give ``make(**options)`` (a module) and ``forms``."""

    forms = GCIS + ASSERTIONS + ROLES

    def assertSame(self, module, forms=None, negs=(False, True), atol=1e-5):
        geometry = geometry_of(module)
        self.assertIsInstance(geometry, ComposedGeometry)
        for form in forms or self.forms:
            for neg in negs:
                data = rows(form)
                try:
                    expected = module(data, form, neg=neg).reshape(-1, 1)
                except NotImplementedError:
                    # a module that refuses a negative loss it does not have: so does the
                    # geometry
                    with self.assertRaises(NotImplementedError):
                        geometry(data, form, neg=neg)
                    continue
                got = geometry(data, form, neg=neg)
                self.assertEqual(got.shape, expected.shape, (form, neg))
                self.assertTrue(th.allclose(got, expected, atol=atol, rtol=1e-5),
                                (form, neg, (got - expected).abs().max().item()))

    def test_every_form(self):
        self.assertSame(self.make())

    def test_parameters_are_shared(self):
        module = self.make()
        geometry = geometry_of(module)
        self.assertIs(geometry.class_embed, module_table(module, "class"))
        geometry(rows("gci0"), "gci0").sum().backward()
        self.assertIsNotNone(module_table(module, "class").weight.grad)

    def test_regularisation(self):
        module = self.make()
        geometry = geometry_of(module)
        expected = th.as_tensor(module.regularization_loss(), dtype=th.float)
        self.assertTrue(th.allclose(geometry.regularization_loss(), expected, atol=1e-6),
                        (geometry.regularization_loss(), expected))


def module_table(module, what):
    for name in ("class_center_embedding", "min_embedding", "class_center"):
        if hasattr(module, name):
            return getattr(module, name)


def scrambled(module, seed=0):
    """Make the parameters generic: offsets of both signs, scales that are not all near
    one."""
    g = th.Generator().manual_seed(seed)
    with th.no_grad():
        for p in module.parameters():
            p.add_(0.3 * th.randn(p.shape, generator=g))
    return module


class TestTransBox(Case, TestCase):

    # no role-axiom negatives, assertions ignore neg in this module
    def make(self, **kwargs):
        th.manual_seed(1)
        return scrambled(TransBoxModule(N, R, I, embed_dim=D, margin=0.07, **kwargs))

    def test_options(self):
        for kwargs in ({"use_enhancement": False}, {"formulation": "reference"},
                       {"formulation": "reference", "use_enhancement": False},
                       {"reg_factor": 0.5}):
            self.assertSame(self.make(**kwargs))

    def test_old_options(self):
        # the earlier behaviour: a negative of a form without one is the positive loss, and
        # the reference formulation's bottom forms are those of the mowl formulation
        for kwargs in ({"strict_negatives": False},
                       {"formulation": "reference", "reference_bottom_losses": "mowl"},
                       {"formulation": "reference", "reference_bottom_losses": "mowl",
                        "strict_negatives": False}):
            module = self.make(**kwargs)
            self.assertSame(module)
            geometry = geometry_of(module)
            self.assertEqual(geometry.objective.strict_negatives, module.strict_negatives)
            self.assertEqual(geometry.objective.reference_bottom_losses,
                             module.reference_bottom_losses)

    def test_defaults_are_the_corrected_readings(self):
        objective = geometry_of(self.make(formulation="reference")).objective
        self.assertTrue(objective.strict_negatives)
        self.assertEqual(objective.reference_bottom_losses, "code")
        # the bottom forms of the code carry its centre-norm term, those of mowl do not
        code = geometry_of(self.make(formulation="reference"))
        old = geometry_of(self.make(formulation="reference", reference_bottom_losses="mowl"))
        data = rows("gci0_bot")
        self.assertFalse(th.allclose(code(data, "gci0_bot"), old(data, "gci0_bot")))

    def test_enhancement_clamped_differs_from_reference(self):
        # The reference code leaves o_r - o_D unclamped; with a filler wider than the
        # relation the two readings of C subsumed by exists r.D differ.
        data = rows("gci2")
        mowl = geometry_of(self.make())
        reference = geometry_of(self.make(formulation="reference"))
        self.assertFalse(th.allclose(mowl(data, "gci2"), reference(data, "gci2")))

    def test_parts(self):
        parts = geometry_of(self.make()).parts
        self.assertEqual((parts["classes"], parts["relations"], parts["conjunction"]),
                         ("box", "box of translations", "exact"))


class TestBoxEL(Case, TestCase):
    forms = tuple(f for f in Case.forms if f not in ROLES)

    def make(self, **kwargs):
        th.manual_seed(2)
        return scrambled(BoxELModule(N, R, I, embed_dim=D, **kwargs), 1)

    def test_temperature(self):
        self.assertSame(self.make(temperature=2.5), self.forms)

    def test_negative_scale(self):
        # A factor below zero: by default the module uses |s| (the box is never inverted);
        # the adapter does the same. ``positive_scale=False`` is the signed scale of the
        # released BoxEL code, which inverts the box; the adapter selects it.
        for kwargs, scale in (({}, "positive"), ({"positive_scale": False}, "signed")):
            module = self.make(**kwargs)
            with th.no_grad():
                module.scaling_embedding.weight.mul_(-1)
            self.assertSame(module, ("gci2", "gci3", "object_property_assertion"))
            self.assertEqual(geometry_of(module).relation_model.scale, scale)

    def test_role_axioms_and_chain_composition(self):
        for kwargs in ({}, {"positive_scale": False}, {"role_chain_composition": "as_released"},
                       {"positive_scale": False, "role_chain_composition": "as_released"}):
            module = self.make(**kwargs)
            self.assertSame(module, ROLES, negs=(False,))
            self.assertEqual(geometry_of(module).relation_model.role_chain_composition,
                             module.role_chain_composition)

    def test_builder_defaults_to_the_corrected_readings(self):
        from mowl.representation import METHODS
        relations = METHODS["boxel"](4, 2, embed_dim=3).relation_model
        self.assertEqual((relations.scale, relations.role_chain_composition),
                         ("positive", "in_order"))


class TestBoxSEL(TestBoxEL):

    def make(self, **kwargs):
        th.manual_seed(3)
        return scrambled(BoxSELModule(N, R, I, embed_dim=D, **kwargs), 2)

    def test_weights_and_beta(self):
        self.test_regularisation()
        for kwargs in ({"beta": 0.7, "location_weight": 0.5, "volume_weight": 0.3},
                       {"volume_weight": 1.0}):
            module = self.make(**kwargs)
            geometry = geometry_of(module)
            self.assertTrue(th.allclose(geometry.regularization_loss(),
                                        module.regularization_loss(), atol=1e-6))

    def test_conditionals(self):
        from mowl.baselines.nn.el.boxsel.losses import compile_expression
        module = self.make(temperature=1.5)
        geometry = geometry_of(module)
        index = {f"c{i}": i for i in range(N)}
        roles = {f"r{i}": i for i in range(R)}
        for conclusion, premise in ((("c1"), ("c2")),
                                    (("and", "c1", "c3"), ("some", "r1", "c2")),
                                    (("some", "r0", ("and", "c4", "c5")), ("c0"))):
            s_d, l_d = compile_expression(conclusion, index, roles)
            s_c, l_c = compile_expression(premise, index, roles)
            ld = th.tensor([l_d] * 7)
            lc = th.tensor([l_c] * 7)
            lo = th.rand(7)
            hi = lo + th.rand(7)
            expected = module.conditional_loss(s_d, s_c, ld, lc, lo, hi)
            got = geometry.conditional_loss(s_d, s_c, ld, lc, lo, hi)
            self.assertTrue(th.allclose(got.reshape(-1), expected, atol=1e-6))
            self.assertTrue(th.allclose(geometry.probability(s_d, s_c, ld, lc).reshape(-1),
                                        module.probability(s_d, s_c, ld, lc),
                                        equal_nan=True, atol=1e-6))


class TestELKEm(Case, TestCase):
    forms = GCIS + ASSERTIONS + ROLES

    def make(self, **kwargs):
        th.manual_seed(4)
        module = ELKEmModule(N, R, I, embed_dim=D, margin=0.07, **kwargs)
        scrambled(module, 3)
        return module

    def fixed(self, **kwargs):
        # Class 0 is bottom, class 1 top.
        module = self.make(bot_ids=[0], top_ids=[1], **kwargs)
        return module

    def test_options(self):
        for kwargs in ({"variant": "reference"}, {"abox_threshold": 0.7},
                       {"epsilon": 1e-3}):
            self.assertSame(self.make(**kwargs))

    def test_fixed_classes(self):
        # Rows are random, so they include the fixed classes. A class fixed to the full
        # box differs for gci0_bot: the module uses its raw offset, the region has the full box.
        for variant in ("paper", "reference"):
            self.assertSame(self.fixed(variant=variant),
                            tuple(f for f in self.forms if f != "gci0_bot"))
        module = self.fixed()
        data = rows("gci0_bot")
        data = data[data[:, 0] != 1]
        self.assertTrue(th.allclose(geometry_of(module)(data, "gci0_bot"),
                                    module(data, "gci0_bot").reshape(-1, 1), atol=1e-6))

    def test_disjointness_with_an_empty_class_default_and_old_option(self):
        # variant="paper" (the default of module and builder): a disjointness row with a class
        # fixed to the empty box costs 0; variant="reference" (the authors' code) charges the
        # margin. Module and parts agree in both, and the two differ from each other.
        gci1_bot = th.tensor([[0, 1, 5], [1, 0, 5], [4, 3, 5]])    # class 0 is bottom, 1 top
        gci3_bot = th.tensor([[1, 0, 1], [1, 2, 1]])
        default = self.fixed()
        self.assertEqual(default.variant, "paper")
        for variant, trivial_cost in (("paper", 0.0), (None, 0.0), ("reference", 0.07)):
            # (the cost of a disjointness with an empty filler depends on the range box)
            module = default if variant is None else self.fixed(variant=variant)
            geometry = geometry_of(module)
            for form, data, n_trivial in (("gci1_bot", gci1_bot, 2), ("gci3_bot", gci3_bot, 1)):
                got = geometry(data, form)
                self.assertTrue(th.allclose(got, module(data, form).reshape(-1, 1), atol=1e-6))
                if form == "gci1_bot" or variant != "reference":
                    self.assertTrue(th.allclose(got[:n_trivial],
                                                th.full((n_trivial, 1), trivial_cost),
                                                atol=1e-6), (variant, form, got))
            # the margin-free loss is a column, in the module as in the geometry's mode
            self.assertEqual(module.zero_margin_loss(gci1_bot, "gci1_bot").shape, (3, 1))
        from mowl.representation.library.boxrel_methods import elkem
        self.assertTrue(elkem(N, R).objective.trivial_disjointness)
        self.assertFalse(elkem(N, R, variant="reference").objective.trivial_disjointness)

    def test_evaluation_mode(self):
        module = self.make().eval()
        self.assertSame(module)

    def test_regularisation(self):
        # The module has no regularisation term of its own (the size penalty is separate).
        self.assertEqual(float(geometry_of(self.make()).regularization_loss()), 0.0)

    def test_size_penalty(self):
        module = self.fixed()
        geometry = geometry_of(module)
        from mowl.representation.library.boxrel_regularizers import SizeCapPenalty
        got = SizeCapPenalty().penalty(geometry)
        self.assertTrue(th.allclose(got, module.size_regularization_loss(), atol=1e-7))

    def test_parts(self):
        parts = geometry_of(self.make()).parts
        self.assertEqual((parts["classes"], parts["relations"], parts["conjunction"]),
                         ("box", "affine map with range and error box", "exact"))


class TestRegularizationRegime(TestCase):
    """The penalties are ordinary regularizers: the regime accepts them."""

    def test_regime_attaches(self):
        from mowl.representation.library.boxrel_regularizers import LocationPenalty
        geometry = geometry_of(BoxELModule(N, R, I, embed_dim=D))
        regime = Regularization(LocationPenalty(0.5), keep_default=False)
        self.assertEqual(regime.overridden_hooks(), frozenset({"regularization"}))
        self.assertIsNotNone(LocationPenalty(0.5).penalty(geometry))
