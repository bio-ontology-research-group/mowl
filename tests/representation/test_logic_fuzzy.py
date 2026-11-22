"""Geometries and parts of DF-ALC, FALCON and Al-cone: the module's own numbers, and the parts
used outside their paper."""
import itertools
from unittest import TestCase

import torch as th
from java.util import HashSet

from mowl.datasets import Dataset
from mowl.models import ComposedEL
from mowl.baselines.nn.alc.alcone.module import ALConeModule
from mowl.baselines.nn.alc.dfalc.module import DFALCModule
from mowl.baselines.nn.alc.falcon.module import FALCONModule
from mowl.owlapi import OWLAPIAdapter
from mowl.representation import geometry_of, ComposedGeometry, METHODS
from mowl.representation.library import logic_fuzzy as F
from mowl.representation.library.logic_expressions import expression_region

NS = "http://mowl/logic#"
AD = OWLAPIAdapter()
NAMES = ["A", "B", "C", "D", "Thing", "Nothing"]
CLS = {n: AD.create_class(NS + n) for n in NAMES}
R = AD.create_object_property(NS + "r")
S = AD.create_object_property(NS + "s")
CLASS_IDS = {NS + n: i for i, n in enumerate(NAMES)}
PROP_IDS = {NS + "r": 0, NS + "s": 1}
IND_IDS = {NS + "x": 0, NS + "y": 1, NS + "z": 2}
IND = {n: AD.create_individual(NS + n) for n in "xyz"}
THING, NOTHING = 4, 5


def columns(e):
    """The ids of the names of ``e`` in the order in which ``forward_fs`` reads them."""
    t = str(e.getClassExpressionType())
    if t == "Class":
        return [CLASS_IDS[str(e.asOWLClass().toStringID())]]
    if t in ("ObjectSomeValuesFrom", "ObjectAllValuesFrom"):
        return [PROP_IDS[str(e.getProperty().asOWLObjectProperty().toStringID())]] \
            + columns(e.getFiller())
    if t in ("ObjectIntersectionOf", "ObjectUnionOf"):
        return [c for o in e.getOperandsAsList() for c in columns(o)]
    return columns(e.getOperand())


def x_of(*expressions):
    return th.tensor([[c for e in expressions for c in columns(e)]])


def rows(n, *highs, seed=0):
    g = th.Generator().manual_seed(seed)
    return th.stack([th.randint(0, h, (n,), generator=g) for h in highs], dim=1)


A, B, C, D = (CLS[n] for n in "ABCD")
EXPRESSIONS = [
    AD.create_object_intersection_of(A, AD.create_object_some_values_from(R, B)),
    AD.create_complement_of(AD.create_object_all_values_from(
        R, AD.create_object_union_of(B, AD.create_complement_of(C)))),
    AD.create_object_union_of(AD.create_object_some_values_from(
        S, AD.create_object_some_values_from(R, D)), C),
]


def small_dataset():
    """A ⊑ B, B ⊓ C ⊑ D, A ⊑ ∃r.B, ∃r.C ⊑ D, B ⊓ C ⊑ ⊥, x : A, r(x, y)."""
    ad = OWLAPIAdapter()
    ont = ad.create_ontology("http://mowl/logic/small")
    c = {n: ad.create_class(NS + n) for n in "ABCD"}
    r = ad.create_object_property(NS + "r")
    axs = HashSet()
    axs.add(ad.create_subclass_of(c["A"], c["B"]))
    axs.add(ad.create_subclass_of(ad.create_object_intersection_of(c["B"], c["C"]), c["D"]))
    axs.add(ad.create_subclass_of(c["A"], ad.create_object_some_values_from(r, c["B"])))
    axs.add(ad.create_subclass_of(ad.create_object_some_values_from(r, c["C"]), c["D"]))
    axs.add(ad.create_disjoint_classes(c["B"], c["C"]))
    x, y = ad.create_individual(NS + "x"), ad.create_individual(NS + "y")
    axs.add(ad.create_class_assertion(c["A"], x))
    axs.add(ad.create_object_property_assertion(r, x, y))
    ad.owl_manager.addAxioms(ont, axs)
    return Dataset(ont)


class Shared:
    """Checks that hold for the three geometries."""

    def test_is_a_geometry_of_the_module(self):
        module = self.make()
        geometry = geometry_of(module)
        self.assertIsInstance(geometry, ComposedGeometry)
        rows_ = rows(5, 4, 4)
        geometry(rows_, "gci0").sum().backward()
        self.assertIsNotNone(module.c_embedding.weight.grad)        # parameters are shared

    def test_classes_negation_disjunction_conjunction(self):
        module = self.make()
        g = geometry_of(module)
        ids = th.tensor([0, 1, 2])
        a, b = g.classes(ids), g.classes(ids.flip(0))
        f = a.family
        self.assertTrue(th.allclose(g.negation(a).values, f.not_(a.values)))
        self.assertTrue(th.allclose(g.disjunction(a, b).values, f.or_(a.values, b.values)))
        region, gap = g.conjunction(a, b)
        self.assertTrue(th.allclose(region.values, f.and_(a.values, b.values)))
        self.assertEqual(gap.shape, (3, 1))


class TestDFALC(Shared, TestCase):

    def make(self, loss_type="hierarchical"):
        th.manual_seed(2)
        return DFALCModule(6, 2, 5, top_index=THING, bottom_index=NOTHING,
                           loss_type=loss_type, threshold=0.7)

    def test_regions_equal_forward_fs(self):
        module = self.make()
        g = geometry_of(module)
        for e in EXPRESSIONS:
            expected, _ = module.forward_fs(e, x_of(e), None)
            got = expression_region(g, e, CLASS_IDS, PROP_IDS, IND_IDS).values
            self.assertTrue(th.allclose(got, expected, atol=1e-6))

    def axioms(self):
        sub = AD.create_subclass_of
        ex = AD.create_object_some_values_from
        return {
            "gci0": (sub(A, B), lambda x: x, (4, 4)),
            "gci2": (sub(A, ex(R, B)), lambda x: x, (4, 2, 4)),
            "gci3": (sub(ex(R, C), D), lambda x: x, (2, 4, 4)),
            "gci1": (sub(AD.create_object_intersection_of(B, C), D), lambda x: x, (4, 4, 4)),
            "gci0_bot": (sub(A, CLS["Nothing"]), lambda x: th.cat(
                [x, th.full((len(x), 1), NOTHING)], dim=1), (4,)),
            "gci1_bot": (AD.create_disjoint_classes(B, C), lambda x: x, (4, 4)),
            "class_assertion": (AD.create_class_assertion(A, IND["x"]), lambda x: x, (3, 4)),
            "object_property_assertion": (
                AD.create_object_property_assertion(R, IND["x"], IND["y"]),
                lambda x: x[:, [2, 1, 0]], (3, 2, 3)),
        }

    def check_losses(self, module):
        g = geometry_of(module)
        for form, (axiom, to_module, highs) in self.axioms().items():
            data = rows(30, *highs)
            module_x = to_module(data)
            expected = module(axiom, module_x).reshape(-1, 1)
            got = g(data, form)
            self.assertEqual(got.shape, expected.shape, form)
            self.assertTrue(th.allclose(got, expected, atol=1e-6), form)

    def test_losses_hierarchical(self):
        self.check_losses(self.make())

    def test_losses_rule_based(self):
        self.check_losses(self.make("rule"))
        data = rows(30, 4, 4)
        rule, plain = geometry_of(self.make("rule")), geometry_of(self.make())
        self.assertFalse(th.allclose(rule(data, "gci0"), plain(data, "gci0")))

    def test_role_assertion_both_readings(self):
        module = self.make()
        g = geometry_of(module)
        data = rows(20, 3, 2, 3)
        image = g(data, "object_property_assertion")
        g.objective.role_assertion = "preimage"
        self.assertTrue(th.allclose(g(data, "object_property_assertion"), image, atol=1e-6))

    def test_swap_the_operator_family(self):
        module = self.make()
        e = EXPRESSIONS[0]
        x = x_of(e)
        godel = expression_region(geometry_of(module), e, CLASS_IDS, PROP_IDS).values
        for name in ("product", "lukasiewicz"):
            g = METHODS["dfalc"](None, None, tables=module, family=name)
            got = expression_region(g, e, CLASS_IDS, PROP_IDS).values
            c_a = module.concept_truth(x[:, 0])
            c_b = module.concept_truth(x[:, 2])
            r = module.role_truth(x[:, 1])
            fam = F.get_family(name)
            expected = fam.and_(c_a, fam.sup(fam.and_(r, c_b.unsqueeze(1)), -1))
            self.assertTrue(th.allclose(got, expected, atol=1e-6), name)
            self.assertFalse(th.allclose(got, godel, atol=1e-6), name)
            self.assertEqual(g.parts["family"], type(fam).__name__)

    def test_families_are_those_of_falcon(self):
        th.manual_seed(3)
        x, y, r = (th.rand(1, 5) for _ in range(3))
        pairs = th.rand(1, 5, 5)           # r(x, y) for every pair of the domain
        for name, falcon_name in (("product", "product"), ("godel", "minmax"),
                                  ("lukasiewicz", "Łukasiewicz")):
            module = FALCONModule(5, 4, 2, {}, {}, embed_dim=4, t_norm=falcon_name)
            fam = F.get_family(name)
            self.assertTrue(th.allclose(fam.and_(x, y), module._logical_and(x, y)))
            self.assertTrue(th.allclose(fam.or_(x, y), module._logical_or(x, y)))
            self.assertTrue(th.allclose(fam.not_(x), module._logical_not(x)))
            # all pairs (the default): one value per element
            self.assertTrue(th.allclose(fam.exists(pairs, y), module._logical_exist(pairs, y)))
            self.assertTrue(th.allclose(fam.forall(pairs, y),
                                        module._logical_forall(pairs, y)))
            # the earlier reading: the pair of an element with itself, one value for all
            module = FALCONModule(5, 4, 2, {}, {}, embed_dim=4, t_norm=falcon_name,
                                  role_pairs="self")
            self.assertTrue(th.allclose(fam.sup(fam.and_(r, y), -1),
                                        module._logical_exist(r, y)[:, 0]))
            self.assertTrue(th.allclose(fam.inf(fam.implies(r, y), -1),
                                        module._logical_forall(r, y)[:, 0]))

    def test_trains_from_parts_also_with_other_families(self):
        for options in ({}, {"family": "product"}, {"family": "lukasiewicz"},
                        {"loss_type": "rule"}):
            self.assertLessEqual(*_train(ComposedEL(
                small_dataset(), method="dfalc", embed_dim=6, **options), 60), 0)


class TestFALCON(Shared, TestCase):

    def make(self, t_norm="product", max_measure="max", **kwargs):
        th.manual_seed(4)
        return FALCONModule(6, 3, 2, {}, {}, embed_dim=8, t_norm=t_norm,
                            max_measure=max_measure, **kwargs)

    #: the readings of FALCON: the default (published) one and the earlier module's
    READINGS = ({}, {"role_pairs": "self"},
                {"bottom_index": NOTHING}, {"bottom_index": NOTHING, "nothing": "learned"},
                {"top_index": THING, "thing": "all"},
                {"role_pairs": "self", "bottom_index": NOTHING, "nothing": "learned"})

    def test_regions_equal_forward_fs(self):
        for t_norm in ("product", "minmax", "Łukasiewicz"):
            for reading in self.READINGS:
                module = self.make(t_norm, **reading)
                g = geometry_of(module)
                e_emb = module.e_embedding.weight
                for e in EXPRESSIONS:
                    expected, _ = module.forward_fs(e, x_of(e), e_emb)
                    got = expression_region(g, e, CLASS_IDS, PROP_IDS, IND_IDS).values
                    self.assertTrue(th.allclose(got, expected, atol=1e-6),
                                    (t_norm, reading, str(e)))

    def test_adapter_reads_the_module_options(self):
        for reading in self.READINGS:
            module = self.make(**reading)
            g = geometry_of(module)
            self.assertEqual(g.relation_model.role_pairs, module.role_pairs)
            self.assertEqual(g.class_model.bottom_index, module.bottom_index)
            self.assertEqual(g.class_model.top_index, module.top_index)

    def test_builder_defaults_are_the_published_readings(self):
        g = METHODS["falcon"](4, 2, 3, embed_dim=4, bottom_index=1, top_index=0)
        self.assertEqual(g.relation_model.role_pairs, "all")
        self.assertEqual((g.class_model.bottom_index, g.class_model.top_index), (1, None))

    def test_tbox_losses(self):
        sub = AD.create_subclass_of
        ex = AD.create_object_some_values_from
        cases = {
            "gci0": (sub(A, B), (4, 4)),
            "gci2": (sub(A, ex(R, B)), (4, 2, 4)),
            "gci3": (sub(ex(R, C), D), (2, 4, 4)),
            "gci1": (sub(AD.create_object_intersection_of(B, C), D), (4, 4, 4)),
            "gci1_bot": (AD.create_disjoint_classes(B, C), (4, 4)),
        }
        for max_measure, reading in itertools.product(("max", "pmean2"), self.READINGS[:2]):
            module = self.make("product", max_measure, **reading)
            g = geometry_of(module)
            e_emb = module.e_embedding.weight
            for form, (axiom, highs) in cases.items():
                data = rows(25, *highs)
                module_x = data      # conjunction is commutative: any order of C, D
                expected = module(axiom, module_x, e_emb)
                got = g(data, form).mean()
                self.assertTrue(th.allclose(got, expected, atol=1e-6),
                                (max_measure, reading, form))

    def test_abox_losses(self):
        module = self.make()
        g = geometry_of(module)
        e_emb = module.e_embedding.weight
        eps = 1e-10
        data = rows(20, 3, 4)
        degree = module._mem(module.c_embedding(data[:, 1]), e_emb[data[:, 0]]).reshape(-1)
        self.assertTrue(th.allclose(g(data, "class_assertion").reshape(-1),
                                    -th.log(degree + eps), atol=1e-6))
        self.assertTrue(th.allclose(g(data, "class_assertion", neg=True).reshape(-1),
                                    -th.log(1 - degree + eps), atol=1e-6))
        data = rows(20, 3, 2, 3)
        pair = module._mem(e_emb[data[:, 0]] + module.r_embedding(data[:, 1]),
                           e_emb[data[:, 2]]).reshape(-1)
        self.assertTrue(th.allclose(g(data, "object_property_assertion").reshape(-1),
                                    -th.log(pair + eps), atol=1e-6))
        self.assertTrue(th.allclose(
            g(data, "object_property_assertion", neg=True).reshape(-1),
            -th.log(1 - pair + eps), atol=1e-6))

    def test_pairwise_exists_is_the_option_that_uses_the_asked_element(self):
        module = self.make(role_pairs="self")
        earlier = geometry_of(module)
        r_ids = th.tensor([0])
        filler = earlier.classes(th.tensor([1]))
        constant = earlier.exists(r_ids, filler).values
        self.assertTrue(th.allclose(constant, constant[:, :1].expand_as(constant)))
        pairwise = geometry_of(self.make())
        module = self.make()
        varied = pairwise.exists(r_ids, pairwise.classes(th.tensor([1]))).values
        self.assertGreater(float(varied.detach().max() - varied.detach().min()), 1e-4)
        # it is the degree the module gives to the assertions r(e_a, e_b)
        e = module.e_embedding.weight
        pair = module._mem(e[1:2] + module.r_embedding(r_ids), e[2:3])
        image = pairwise.relation_model.image(
            r_ids, pairwise.individuals(th.tensor([1]))).values[0, 2]
        self.assertTrue(th.allclose(pair.reshape(()), image, atol=1e-6))

    def test_trains_from_parts_with_other_t_norms(self):
        for options in ({}, {"t_norm": "minmax"}, {"t_norm": "lukasiewicz"},
                        {"role_pairs": "self"}):
            self.assertLessEqual(*_train(ComposedEL(
                small_dataset(), method="falcon", embed_dim=8, **options), 60), 0)

    def test_conflict_measure_options(self):
        th.manual_seed(0)
        fs = F.FuzzySet(th.rand(3, 5), F.Product())
        add = F.ConflictMeasure("max", "add").empty(fs)
        clamp = F.ConflictMeasure("max", "clamp").empty(fs)
        self.assertTrue(th.allclose(add, clamp, atol=1e-6))
        zero = F.FuzzySet(th.zeros(1, 5), F.Product())
        self.assertLessEqual(float(F.ConflictMeasure(stabiliser="add").empty(zero)), 0)
        self.assertEqual(float(F.ConflictMeasure().empty(zero)), 0)
        with self.assertRaises(ValueError):
            F.ConflictMeasure().inside(fs, fs, margin=0.1)


class TestALCone(Shared, TestCase):

    def make(self, temperature=0.0, **kwargs):
        th.manual_seed(5)
        module = ALConeModule(6, 3, 2, dim=3, margin=0.1, top_index=THING,
                              bottom_index=NOTHING, **kwargs)
        module.temperature = temperature
        return module

    def test_regions_equal_forward_fs(self):
        for temperature in (0.0, 0.3):
            module = self.make(temperature)
            g = geometry_of(module)
            for e in EXPRESSIONS:
                expected, _ = module.forward_fs(e, x_of(e), None)
                got = expression_region(g, e, CLASS_IDS, PROP_IDS, IND_IDS).values
                self.assertTrue(th.allclose(got, expected, atol=1e-5), (temperature, str(e)))

    def test_losses_through_thing_and_nothing(self):
        # the constant logits: the default (1) and the 1e4 of the earlier module
        sub = AD.create_subclass_of
        for constant_logit in (1.0, 5.0, 1e4):
            module = self.make(constant_logit=constant_logit)
            g = geometry_of(module)
            self.assertEqual(g.class_model.constant_logit, constant_logit)
            for pair in ((0, NOTHING), (NOTHING, 1), (THING, 2), (3, THING)):
                data = th.tensor([pair] * 4)
                expected = module(sub(A, B), data).reshape(-1, 1)
                self.assertTrue(th.allclose(g(data, "gci0"), expected, rtol=1e-5, atol=1e-5),
                                (constant_logit, pair))

    def test_the_default_constant_logit_is_the_modules(self):
        self.assertEqual(METHODS["alcone"](4, 1, 1, embed_dim=2).class_model.constant_logit,
                         ALConeModule(4, 1, 1).constant_logit)

    def test_temperature_is_read_live(self):
        module = self.make()
        g = geometry_of(module)
        e = EXPRESSIONS[0]
        before = expression_region(g, e, CLASS_IDS, PROP_IDS).values
        module.temperature = 0.5
        after = expression_region(g, e, CLASS_IDS, PROP_IDS).values
        self.assertFalse(th.allclose(before, after))

    def test_losses(self):
        sub = AD.create_subclass_of
        ex = AD.create_object_some_values_from
        cases = {
            "gci0": (sub(A, B), (4, 4), lambda x: x),
            "gci2": (sub(A, ex(R, B)), (4, 2, 4), lambda x: x),
            "gci3": (sub(ex(R, C), D), (2, 4, 4), lambda x: x),
            "gci1_bot": (AD.create_disjoint_classes(B, C), (4, 4), lambda x: x),
            "class_assertion": (AD.create_class_assertion(A, IND["x"]), (3, 4), lambda x: x),
            "object_property_assertion": (
                AD.create_object_property_assertion(R, IND["x"], IND["y"]), (3, 2, 3),
                lambda x: x[:, [2, 1, 0]]),
        }
        for temperature in (0.0, 0.4):
            module = self.make(temperature)
            g = geometry_of(module)
            for form, (axiom, highs, to_module) in cases.items():
                data = rows(30, *highs)
                expected = module(axiom, to_module(data)).reshape(-1, 1)
                self.assertTrue(th.allclose(g(data, form), expected, atol=1e-5),
                                (temperature, form))

    def test_zero_loss_means_the_axiom_holds_after_discretising(self):
        g = METHODS["alcone"](4, 1, 1, embed_dim=2, margin=0.1)
        with th.no_grad():
            g.class_model.centres.weight.copy_(th.tensor(
                [[1., 1, -1, -1], [1, 1, 1, -1], [-1, -1, 1, 1], [-1, -1, -1, 1]]))
        self.assertEqual(float(g(th.tensor([[0, 1]]), "gci0")), 0)          # A in B
        self.assertGreater(float(g(th.tensor([[1, 0]]), "gci0")), 0)        # B not in A
        self.assertEqual(float(g(th.tensor([[0, 2]]), "gci1_bot")), 0)      # A, C disjoint

    def test_trains_from_parts_and_with_parts_of_other_methods(self):
        # the annealed relaxation of the method (its training regime, with parameter noise
        # meant for the learning rate of Al-cone) is taken by ComposedEL too: switched off
        # here, where the parts alone are trained
        with_schedule = ComposedEL(small_dataset(), method="alcone", embed_dim=3, margin=0.1)
        self.assertIs(with_schedule.training_regime, with_schedule.module.schedule)
        self.assertLessEqual(*_train(ComposedEL(
            small_dataset(), method="alcone", embed_dim=3, margin=0.1, anneal=None), 60), 0)
        self.assertLessEqual(*_train(ComposedEL(
            small_dataset(), classes="half-axis cone", relations="incidence matrix",
            measure="half-axis hinge", embed_dim=3, margin=0.1), 60), 0)


class TestPartsAcrossMethods(TestCase):

    def test_dfalc_tables_with_falcon_measure_and_product_family(self):
        # classes and relations of DF-ALC, the loss of FALCON, the product t-norm
        model = ComposedEL(small_dataset(), classes="fuzzy set (product)",
                           relations="fuzzy relation", measure="conflict", embed_dim=6)
        self.assertIsInstance(model.module.measure, F.ConflictMeasure)
        self.assertLessEqual(*_train(model, 80), 0)

    def test_a_composed_fuzzy_geometry_has_negation_and_disjunction(self):
        # the region family gives a composed geometry negation and disjunction
        g = METHODS["dfalc"](4, 1, 1, embed_dim=3)
        self.assertIsInstance(g, ComposedGeometry)
        self.assertEqual(g.parts["classes"], "fuzzy set")
        a, b = g.classes(th.tensor([0, 1])), g.classes(th.tensor([2, 3]))
        self.assertTrue(th.allclose(g.negation(a).values, 1 - a.values))
        self.assertTrue(th.allclose(g.disjunction(a, b).values,
                                    th.maximum(a.values, b.values)))


def _train(model, epochs):
    """``(loss after - loss before, 0)`` of a short training run on the mean axiom loss."""
    def total():
        with th.no_grad():
            return sum(float(model.module(d[:], f).mean())
                       for f, d in model.training_datasets.items() if len(d))
    th.manual_seed(0)
    before = total()
    model.train(epochs=epochs)
    after = total()
    assert after < before, (before, after)
    return after - before, 0
