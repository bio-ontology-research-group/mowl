"""The consolidated core: region additions, measure and relation-model interfaces, the
options of Objective, ComposedGeometry and ComposedEL."""
from unittest import TestCase

import torch as th
from jpype import java
from org.semanticweb.owlapi.model import IRI

from mowl.datasets import Dataset
from mowl.models import ComposedEL
from mowl.owlapi import OWLAPIAdapter
from mowl.representation import (
    Ball, BallClasses, Bounds, Box, BoxClasses, CentreDistance, ComposedGeometry,
    ConstructorUnavailable, Objective, Relations, Translation, METHODS, METHOD_KINDS,
    FORMS, ROLE_FORMS, geometry_of)
from mowl.representation.regions import Region
from tests.datasetFactory import FamilyDataset


def role_dataset():
    """C0 ⊑ C1, C1 ⊑ ∃R0.C2, ∃R0.C3 ⊑ C4, R0 ⊑ R1, R1 ∘ R2 ⊑ R0."""
    adapter = OWLAPIAdapter()
    f = adapter.data_factory
    c = [f.getOWLClass(IRI.create(f"http://test/C{i}")) for i in range(5)]
    r = [f.getOWLObjectProperty(IRI.create(f"http://test/R{i}")) for i in range(3)]
    ont = adapter.owl_manager.createOntology(IRI.create("http://test/core_roles"))
    ont.addAxiom(f.getOWLSubClassOfAxiom(c[0], c[1]))
    ont.addAxiom(f.getOWLSubClassOfAxiom(c[1], f.getOWLObjectSomeValuesFrom(r[0], c[2])))
    ont.addAxiom(f.getOWLSubClassOfAxiom(f.getOWLObjectSomeValuesFrom(r[0], c[3]), c[4]))
    ont.addAxiom(f.getOWLSubObjectPropertyOfAxiom(r[0], r[1]))
    chain = java.util.ArrayList()
    chain.add(r[1])
    chain.add(r[2])
    ont.addAxiom(f.getOWLSubPropertyChainOfAxiom(chain, r[0]))
    return Dataset(ont)


class TestRegions(TestCase):

    def test_len_does_not_assume_a_centre(self):
        class Bag(Region):
            def __init__(self, x):
                self.x = x
        self.assertEqual(len(Bag(th.zeros(7, 2))), 7)
        self.assertEqual(len(Ball(th.zeros(3, 2), th.ones(3, 1))), 3)

    def test_grow(self):
        ball = Ball(th.zeros(2, 3), th.ones(2, 1), th.ones(2, 1))
        grown = ball.grow(th.full((2, 1), 0.5))
        self.assertTrue(th.allclose(grown.radius, th.full((2, 1), 1.5)))
        self.assertIsNone(grown.raw)
        box = Box(th.zeros(2, 3), th.ones(2, 3)).grow(0.5)
        self.assertTrue(th.allclose(box.offset, th.full((2, 3), 1.5)))
        with self.assertRaises(ConstructorUnavailable):
            Region().grow(1.0)

    def test_signed_intersection(self):
        a = Box(th.zeros(1, 2), th.ones(1, 2))
        b = Box(th.tensor([[3.0, 0.0]]), th.ones(1, 2))      # apart in the first coordinate
        plain, gap = a.intersect(b, signed=False)      # the earlier default
        signed, gap_signed = a.intersect(b, signed=True)
        default, _ = a.intersect(b)                     # exact: the empty box
        self.assertTrue(th.equal(default.offset, signed.offset))
        self.assertTrue(th.allclose(gap, gap_signed) and float(gap) > 0)
        self.assertTrue((plain.offset >= 0).all())
        self.assertTrue(float(signed.offset[0, 0]) < 0)
        self.assertTrue(th.allclose(signed.offset.abs(), plain.offset))


class TestMeasures(TestCase):

    def test_defaults(self):
        measure = Bounds()
        a, b = BoxClasses(3, 2)(th.arange(3)), BoxClasses(3, 2)(th.arange(3))
        self.assertTrue(th.equal(measure.member(a, b, 0.1), measure.inside(a, b, 0.1)))
        self.assertTrue(th.equal(measure.negative(a, b, 0.1), measure.apart(a, b, 0.1)))
        self.assertTrue(th.equal(measure.meets_all(a, b), measure.meets(a, b)))
        self.assertIs(measure.bind(None), measure)
        # non-inclusion has no generic definition: a measure without its own refuses it
        from mowl.representation.measures import RegionMeasure

        class Bare(RegionMeasure):
            accepts = (Box,)
        with self.assertRaises(ConstructorUnavailable):
            Bare().outside(a, b)
        # boxes have one: zero where a box sticks out of the other
        far = Box(a.centre + 10, a.offset)
        self.assertTrue(bool((measure.outside(far, b) == 0).all()))

    def test_empty_takes_a_margin(self):
        a = BoxClasses(3, 2)(th.arange(3))
        self.assertTrue(th.allclose(Bounds().empty(a, 0.0), Bounds().empty(a)))
        self.assertTrue((Bounds().empty(a, 0.5) > Bounds().empty(a)).all())
        balls = BallClasses(3, 2)(th.arange(3))
        self.assertTrue(th.allclose(CentreDistance().empty(balls, 0.5),
                                    balls.radius + 0.5))


class TestRelationModels(TestCase):

    def test_translation_reads_role_axioms(self):
        r = Translation(3, 4)
        self.assertTrue(r.reads_role_axioms)
        with th.no_grad():
            r.vectors.weight[1] = r.vectors.weight[0]
            r.vectors.weight[2] = r.vectors.weight[0] + r.vectors.weight[1]
        ids = th.tensor([0])
        self.assertAlmostEqual(float(r.role_inclusion(ids, th.tensor([1]), None)), 0.0, 5)
        self.assertAlmostEqual(float(r.role_chain(ids, th.tensor([1]), th.tensor([2]), None)),
                               0.0, 5)
        self.assertGreater(float(r.role_inclusion(ids, th.tensor([2]), None)), 0.0)
        # equality has no strict margin; a slack is read
        self.assertEqual(float(r.role_inclusion(ids, th.tensor([2]), None, -100.0)), 0.0)

    def test_a_relation_model_without_role_axioms(self):
        class Plain(Relations):
            def preimage(self, ids, region):
                return region
        self.assertFalse(Plain().reads_role_axioms)
        with self.assertRaises(ConstructorUnavailable):
            Plain().role_inclusion(th.tensor([0]), th.tensor([1]), Bounds())


class TestObjective(TestCase):

    def setUp(self):
        th.manual_seed(0)

    def test_option_validation(self):
        for bad in ({"gci2_negative": "x"}, {"bottom": "x"}, {"gci2_reading": "x"},
                    {"negatives": "x"}, {"negative_margin": "x"}):
            with self.assertRaises(ValueError):
                Objective(**bad)

    def test_negatives_collection_and_neg_forms(self):
        self.assertEqual(Objective(negatives="all").neg_forms, frozenset(FORMS))
        self.assertEqual(Objective(negatives=("gci0", "class_assertion")).neg_forms,
                         frozenset({"gci0", "class_assertion"}))
        self.assertEqual(Objective.hierarchy().neg_forms, frozenset({"gci0"}))

    def test_slack_negatives_use_the_slack_sense_by_default(self):
        balls = BallClasses(4, 2)
        rows = th.tensor([[0, 1], [2, 3]])
        slack = ComposedGeometry(balls, None, CentreDistance(),
                                 Objective(0.3, slack=True, negatives="all"))
        strict = ComposedGeometry(balls, None, CentreDistance(),
                                  Objective(0.3, slack=True, negatives="all",
                                            negative_margin="strict"))
        a, b = balls(rows[:, 0]), balls(rows[:, 1])
        self.assertTrue(th.allclose(slack(rows, "gci0", neg=True),
                                    CentreDistance().apart(a, b, -0.3)))
        self.assertTrue(th.allclose(strict(rows, "gci0", neg=True),
                                    CentreDistance().apart(a, b, 0.3)))

    def test_squared_weights_and_ranking(self):
        g = ComposedGeometry(BoxClasses(5, 3), Translation(2, 3), Bounds(),
                             Objective(0.1, squared=True, weights={"gci0": 2.0},
                                       ranking={"gci2": 0.4}, negatives="all"))
        plain = ComposedGeometry(BoxClasses(5, 3), Translation(2, 3), Bounds(), Objective(0.1))
        plain.load_state_dict(g.state_dict())
        rows = th.tensor([[0, 1], [2, 3]])
        self.assertTrue(th.allclose(g(rows, "gci0"), 2.0 * plain(rows, "gci0") ** 2))
        triples = th.tensor([[0, 1, 2], [3, 0, 4]])
        self.assertTrue(th.allclose(g(triples, "gci2", neg=True),
                                    plain(triples, "gci2") - 0.4))

    def test_the_role_forms_go_through_the_relation_model(self):
        g = ComposedGeometry(BoxClasses(3, 2), Translation(3, 2), Bounds(), Objective(),
                             )
        self.assertTrue(g.role_axiom_capable)
        self.assertEqual(g.supported_forms, frozenset(FORMS + ROLE_FORMS))
        self.assertEqual(g.role_inclusion_loss(th.tensor([[0, 1]])).shape, (1, 1))
        self.assertEqual(g.role_chain_loss(th.tensor([[0, 1, 2]])).shape, (1, 1))
        self.assertEqual(g(th.tensor([[0, 1]]), "role_inclusion").shape, (1, 1))
        # no negative reading: an error by default, the positive loss as an option
        rows = th.tensor([[0, 1]])
        with self.assertRaises(NotImplementedError):
            g(rows, "role_inclusion", neg=True)
        lenient = ComposedGeometry(BoxClasses(3, 2), Translation(3, 2), Bounds(),
                                   Objective(strict_negatives=False))
        self.assertTrue(th.equal(lenient(rows, "role_inclusion", neg=True),
                                 lenient(rows, "role_inclusion")))

    def test_strict_negatives_by_default_in_the_objective(self):
        rows = th.tensor([[0, 1]])
        for negatives, form in (("gci2", "gci0"), ("gci0", "gci2"), ("none", "gci0")):
            rows = th.tensor([[0, 1]]) if form == "gci0" else th.tensor([[0, 0, 1]])
            strict = ComposedGeometry(BoxClasses(3, 2), Translation(3, 2), Bounds(),
                                      Objective(negatives=negatives))
            with self.assertRaises(NotImplementedError):
                strict(rows, form, neg=True)
            lenient = ComposedGeometry(BoxClasses(3, 2), Translation(3, 2), Bounds(),
                                       Objective(negatives=negatives, strict_negatives=False))
            self.assertTrue(th.equal(lenient(rows, form, neg=True), lenient(rows, form)))
        # the published ELEmbeddings and ELBE losses give the positive loss
        self.assertFalse(Objective.elem().strict_negatives)
        self.assertFalse(Objective.elbe().strict_negatives)
        self.assertTrue(Objective().strict_negatives)

    def test_unit_norm_terms(self):
        plain = ComposedGeometry(BallClasses(4, 3), Translation(2, 3), CentreDistance(),
                                 Objective(0.1))
        normed = ComposedGeometry(BallClasses(4, 3), Translation(2, 3), CentreDistance(),
                                  Objective(0.1, unit_norm=True))
        normed.load_state_dict(plain.state_dict())
        rows = th.tensor([[0, 1], [2, 3]])
        with th.no_grad():
            normed.class_model.centres.weight.mul_(3.0)
            plain.class_model.centres.weight.mul_(3.0)
        expected = sum((th.linalg.norm(normed.class_model.centres(rows[:, i]), dim=1,
                                       keepdim=True) - 1).abs() for i in range(2))
        self.assertTrue(th.allclose(normed(rows, "gci0") - plain(rows, "gci0"), expected))


class TestComposedGeometry(TestCase):

    def test_supported_forms_and_neg_capable_arguments(self):
        g = ComposedGeometry(BallClasses(3, 2), None, CentreDistance(), Objective(),
                             supported_forms=("gci0",), neg_capable_gcis={"gci0"})
        self.assertEqual(g.supported_forms, frozenset({"gci0"}))
        self.assertEqual(g.neg_capable_gcis, frozenset({"gci0"}))
        self.assertEqual(g.scored_forms, ("gci0",))
        self.assertFalse(g.role_axiom_capable)
        with self.assertRaises(ValueError):
            ComposedGeometry(BallClasses(3, 2), None, CentreDistance(), Objective(),
                             supported_forms=("gci9",))

    def test_negation_and_disjunction_delegate_to_the_regions(self):
        g = ComposedGeometry(BallClasses(3, 2), None, CentreDistance(), Objective())
        a = g.classes(th.arange(2))
        with self.assertRaises(ConstructorUnavailable):
            g.negation(a)
        with self.assertRaises(ConstructorUnavailable):
            g.disjunction(a, a)

    def test_the_measure_is_bound_to_the_relation_model(self):
        class Needy(Bounds):
            bound_to = None

            def bind(self, relations):
                self.bound_to = relations
                return self

        relations = Translation(2, 3)
        g = ComposedGeometry(BoxClasses(3, 3), relations, Needy(), Objective())
        self.assertIs(g.measure.bound_to, relations)

    def test_regularisation_loss_is_device_safe_without_a_table(self):
        g = ComposedGeometry(BallClasses(3, 2), None, CentreDistance(), Objective())
        g.class_embed = None   # packed parameters have no ``class_embed.weight``
        self.assertEqual(float(g.regularization_loss()), 0.0)
        self.assertEqual(g.device, th.device("cpu"))

    def test_conjunction_parts_come_from_the_class_model(self):
        self.assertEqual(ComposedGeometry(BoxClasses(3, 2), None, Bounds()).parts[
            "conjunction"], "exact")
        self.assertEqual(ComposedGeometry(BallClasses(3, 2), None, CentreDistance()).parts[
            "conjunction"], "approximate")

    def test_project_calls_the_parts_that_have_it(self):
        calls = []

        class Projecting(BoxClasses):
            def project(self):
                calls.append(1)

        ComposedGeometry(Projecting(3, 2), None, Bounds()).project()
        self.assertEqual(calls, [1])
        ComposedGeometry(BoxClasses(3, 2), None, Bounds()).project()   # no-op


class TestComposedEL(TestCase):

    def test_query_methods_are_refused_with_a_clear_error(self):
        for key in ("acone", "cone_qe"):
            self.assertEqual(METHOD_KINDS[key], "queries")
            with self.assertRaisesRegex(TypeError, "queries.*ComposedEL cannot train"):
                ComposedEL(FamilyDataset(), method=key, embed_dim=8)
        self.assertEqual(METHOD_KINDS["elbe"], "axioms")

    def test_only_the_supported_forms_are_trained(self):
        model = ComposedEL(FamilyDataset(), method="order_embeddings", embed_dim=8)
        self.assertEqual(set(model.training_datasets), {"gci0"})
        before = float(model.module(model.training_datasets["gci0"][:], "gci0").mean())
        th.manual_seed(0)
        model.train(epochs=30)
        after = float(model.module(model.training_datasets["gci0"][:], "gci0").mean())
        self.assertLess(after, before)

    def test_projection_after_each_step(self):
        model = ComposedEL(FamilyDataset(), method="box_lattice", embed_dim=3,
                           learning_rate=0.5)
        model.train(epochs=3)
        weight = model.module.class_embed.weight
        self.assertTrue(bool(((weight >= 0) & (weight <= 1)).all()))

    def test_role_axioms_are_loaded_when_the_geometry_reads_them(self):
        model = ComposedEL(role_dataset(), classes="box", relations="translation",
                           embed_dim=6, margin=0.0)
        self.assertTrue(model.load_role_axioms)
        datasets = model.training_datasets
        self.assertEqual(len(datasets["role_inclusion"]), 1)
        self.assertGreaterEqual(len(datasets["role_chain"]), 1)
        th.manual_seed(0)
        model.train(epochs=5)
        # a geometry that cannot read them leaves them out
        model = ComposedEL(role_dataset(), classes="box", relations="affine map",
                           embed_dim=6)
        self.assertTrue(model.load_role_axioms)       # the affine map has role axioms
        model = ComposedEL(role_dataset(), method="order_embeddings", embed_dim=6)
        self.assertFalse(model.load_role_axioms)
        self.assertNotIn("role_inclusion", model.training_datasets)

    def test_a_ranking_loss(self):
        objective = Objective(0.0, ranking={"gci2": 0.2}, negatives="all")
        model = ComposedEL(FamilyDataset(), classes="box", relations="translation",
                           objective=objective, embed_dim=6)
        model.module.last_form = "gci2"
        pos = th.tensor([[1.0], [2.0]])
        neg = th.tensor([[0.5], [3.0], [1.5], [1.0]])       # two negatives per positive
        expected = th.relu(pos.reshape(1, -1) - neg.reshape(2, 2)).mean()
        self.assertTrue(th.allclose(model.compute_loss(pos, neg), expected))
        model.module.last_form = "gci0"
        self.assertTrue(th.allclose(model.compute_loss(pos, neg),
                                    pos.mean() + neg.mean()))
        model.train(epochs=3)

    def test_geometry_of_looks_up_the_model_class_first(self):
        from mowl.representation.adapters import ADAPTERS, register_adapter

        class Marker:
            module = object()      # not a usable module

        @register_adapter("Marker")
        def _marker(model):
            return "found by the model's own class"

        try:
            self.assertEqual(geometry_of(Marker()), "found by the model's own class")
        finally:
            del ADAPTERS["Marker"]
        with self.assertRaises(NotImplementedError):
            geometry_of(Marker())
        self.assertIn("elbe", METHODS)
