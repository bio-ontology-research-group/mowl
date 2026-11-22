"""Class expressions on any geometry: regions, subsumption cost, rankings."""
from unittest import TestCase

import torch as th

from mowl.models import ELBE, ELEmbeddings
from mowl.models.e2r.model import E2R
from mowl.representation import geometry_of, ConstructorUnavailable, METHODS
from mowl.representation.library.logic_expressions import (
    expression_region, subsumption_cost, rank_classes, rank_individuals, id_maps)
from mowl.representation.library.logic_subspace import SubspaceGeometry
from tests.representation.test_logic_fuzzy import (
    AD, CLS, R, IND, CLASS_IDS, PROP_IDS, IND_IDS, NS, small_dataset)

A, B, C, D = (CLS[n] for n in "ABCD")
ex = AD.create_object_some_values_from
inter = AD.create_object_intersection_of
union = AD.create_object_union_of
neg = AD.create_complement_of


class TestBoxesAndBalls(TestCase):

    @classmethod
    def setUpClass(cls):
        th.manual_seed(0)
        cls.elbe = ELBE(small_dataset(), embed_dim=4, margin=0.05)
        cls.elem = ELEmbeddings(small_dataset(), embed_dim=4, margin=0.05)
        cls.maps = id_maps(cls.elbe)
        cls.cls, cls.props, cls.inds = cls.maps

    def cid(self, name):
        return th.tensor([self.cls[NS + name]])

    def test_region_of_an_existential_conjunction(self):
        g = geometry_of(self.elbe)
        region = expression_region(g, inter(A, ex(R, B)), *self.maps)
        r = g.exists(th.tensor([self.props[NS + "r"]]), g.classes(self.cid("B")))
        expected, _ = g.conjunction(g.classes(self.cid("A")), r)
        self.assertTrue(th.allclose(region.centre, expected.centre))
        self.assertTrue(th.allclose(region.offset, expected.offset))

    def test_cost_follows_the_losses_of_the_geometry(self):
        g = geometry_of(self.elbe)
        plain = subsumption_cost(g, A, B, *self.maps)
        cost = plain
        self.assertAlmostEqual(cost, float(g.inside(g.classes(self.cid("A")),
                                                    g.classes(self.cid("B")))), places=6)
        self.assertAlmostEqual(cost, subsumption_cost(g, NS + "A", NS + "B", *self.maps))
        rows = th.tensor([[self.cls[NS + n] for n in "BCD"]])
        both = subsumption_cost(g, inter(B, C), D, *self.maps, require_nonempty=True)
        self.assertAlmostEqual(both, float(g(rows, "gci1")), places=6)
        rows = th.tensor([[self.cls[NS + "A"], self.props[NS + "r"], self.cls[NS + "B"]]])
        self.assertAlmostEqual(subsumption_cost(g, A, ex(R, B), *self.maps),
                               float(g(rows, "gci2")), places=6)
        # the model is accepted as well as its geometry
        self.assertAlmostEqual(subsumption_cost(self.elbe, A, B, *self.maps), cost)

    def test_rankings(self):
        g = geometry_of(self.elbe)
        ranked = rank_classes(g, ex(R, B), *self.maps)
        self.assertEqual(len(ranked), len(self.cls))
        self.assertEqual([c for _, c in ranked], sorted(c for _, c in ranked))
        for name, cost in ranked:
            self.assertAlmostEqual(cost, subsumption_cost(g, name, ex(R, B), *self.maps),
                                   places=5)
        self.assertEqual(len(rank_classes(g, B, *self.maps, top_k=2)), 2)
        members = rank_individuals(g, A, *self.maps)
        self.assertEqual(sorted(n for n, _ in members), sorted(self.inds))
        for name, cost in members:
            expected = g.member(th.tensor([self.inds[name]]), g.classes(self.cid("A")))
            self.assertAlmostEqual(cost, float(expected), places=5)

    def test_nominals_and_has_value(self):
        g = geometry_of(self.elbe)
        one_of = AD.data_factory.getOWLObjectOneOf(IND["x"])
        region = expression_region(g, one_of, *self.maps)
        point = g.individuals(th.tensor([self.inds[NS + "x"]]))
        self.assertTrue(th.allclose(region.centre, point.centre))
        has_value = AD.data_factory.getOWLObjectHasValue(R, IND["y"])
        expected = g.exists(th.tensor([self.props[NS + "r"]]),
                            g.individuals(th.tensor([self.inds[NS + "y"]])))
        self.assertTrue(th.allclose(expression_region(g, has_value, *self.maps).centre,
                                    expected.centre))
        self.assertEqual(len(rank_individuals(g, ex(R, one_of), *self.maps)), 2)

    def test_balls_have_no_conjunction(self):
        g = geometry_of(self.elem)
        self.assertEqual(len(expression_region(g, ex(R, A), *self.maps).centre), 1)
        with self.assertRaises(ConstructorUnavailable) as caught:
            subsumption_cost(g, inter(A, B), C, *self.maps)
        self.assertIn("intersection", str(caught.exception))
        self.assertIn("ComposedGeometry", str(caught.exception))
        self.assertIn("cannot be evaluated", str(caught.exception))

    def test_missing_constructors_are_named(self):
        g = geometry_of(self.elbe)
        for expression, word in ((neg(A), "complement"), (union(A, B), "union"),
                                 (AD.create_object_all_values_from(R, A), "universal")):
            with self.assertRaises(ConstructorUnavailable) as caught:
                expression_region(g, expression, *self.maps)
            self.assertIn(word, str(caught.exception))
        no_thing = {k: v for k, v in self.cls.items() if "Thing" not in k}
        with self.assertRaises(ConstructorUnavailable):
            expression_region(g, AD.create_class("http://www.w3.org/2002/07/owl#Thing"),
                              no_thing, self.props)
        with self.assertRaises(KeyError):
            expression_region(g, AD.create_class(NS + "Unknown"), *self.maps)


class TestFuzzyAndSubspace(TestCase):

    def setUp(self):
        th.manual_seed(1)
        self.dfalc = METHODS["dfalc"](6, 2, 3, embed_dim=5, top_index=4, bottom_index=5)
        self.maps = (CLASS_IDS, PROP_IDS, IND_IDS)

    def test_negation_disjunction_universal(self):
        g = self.dfalc
        ids = th.tensor([0])
        a = g.classes(ids).values
        self.assertTrue(th.allclose(
            expression_region(g, neg(neg(A)), *self.maps).values, a, atol=1e-6))
        self.assertAlmostEqual(subsumption_cost(g, A, union(A, B), *self.maps), 0, places=6)
        self.assertAlmostEqual(subsumption_cost(g, A, CLS["Thing"], *self.maps), 0, places=6)
        self.assertAlmostEqual(subsumption_cost(g, CLS["Nothing"], A, *self.maps), 0,
                               places=6)
        forall = AD.create_object_all_values_from(R, B)
        manual = expression_region(g, neg(ex(R, neg(B))), *self.maps).values
        self.assertTrue(th.allclose(
            expression_region(g, forall, *self.maps).values, manual, atol=1e-6))

    def test_nothing_without_a_class_is_the_complement_of_thing(self):
        thing = "http://www.w3.org/2002/07/owl#Thing"
        nothing = AD.create_class("http://www.w3.org/2002/07/owl#Nothing")
        only_thing = {NS + n: i for i, n in enumerate("ABCD")}
        only_thing[thing] = 4
        region = expression_region(self.dfalc, nothing, only_thing, PROP_IDS)
        self.assertTrue(th.allclose(region.values, th.zeros_like(region.values)))

    def test_rankings_of_individuals_use_member(self):
        g = self.dfalc
        ranked = rank_individuals(g, ex(R, A), *self.maps)
        self.assertEqual(len(ranked), 3)
        costs = [c for _, c in ranked]
        self.assertEqual(costs, sorted(costs))
        top = rank_classes(g, union(A, B), *self.maps, top_k=3)
        self.assertEqual(len(top), 3)

    def test_e2r_geometry_ranks_like_the_model(self):
        th.manual_seed(2)
        model = E2R(small_dataset(), embed_dim=4)
        maps = id_maps(model)
        g = geometry_of(model)
        self.assertIsInstance(g, SubspaceGeometry)      # the geometry of E2R is its module
        for expression in (inter(A, B), union(A, neg(C)), neg(inter(B, C))):
            region = expression_region(model, expression, *maps)
            self.assertTrue(th.allclose(region.indicator[0],
                                        model.concept_indicator(expression)))
            ranked = [name for name, _ in rank_individuals(model, expression, *maps)]
            self.assertEqual(ranked, model.rank_members(expression))
            self.assertAlmostEqual(subsumption_cost(model, expression, B, *maps),
                                   model.score_subsumption(expression, B), places=5)
        with self.assertRaises(ConstructorUnavailable) as caught:
            expression_region(model, ex(R, A), *maps)
        self.assertIn("existential", str(caught.exception))
