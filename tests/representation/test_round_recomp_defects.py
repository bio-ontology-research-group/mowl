"""Defects of the round "recomp", each with its default and the option that keeps the old
behaviour: FaithEL's unbounded negative term, FALCON's direction of role assertions, and the
role axioms of BoxEL / BoxSEL."""
from unittest import TestCase

import mowl
mowl.init_jvm("1g")

import torch as th

from mowl.models import BoxEL, BoxSEL, ComposedALC, FaithEL
from mowl.representation import METHODS
from mowl.representation.library import logic_fuzzy as F
from mowl.representation.library import solved_faithel as sf
from tests.baselines.pairs_boxrel import tiny_ontology
from tests.baselines.pairs_logic import NS, _ontology


# -- FaithEL ---------------------------------------------------------------------------------
class FaithELNegativeTerm(TestCase):

    def geometry(self, **options):
        th.manual_seed(0)
        g = METHODS["faithel"](4, 1, 6, 5, **options)
        with th.no_grad():
            # individuals that have drifted far apart, as the unbounded term makes them
            g.individual_model.centres.weight.copy_(th.randn(6, 5) * 1000.0)
        return g

    def rows(self):
        return th.tensor([[0, 1], [1, 2], [2, 0], [3, 3], [4, 1], [5, 2]])

    def loss(self, g):
        return g.cost("class_assertion", self.rows()).mean()

    def test_the_default_is_bounded_below(self):
        g = self.geometry()
        self.assertEqual(g.objective.negative_term, "margin")
        self.assertGreaterEqual(float(self.loss(g).detach()), 0.0)
        for _ in range(3):          # whatever negatives are drawn
            self.assertGreaterEqual(float(self.loss(g).detach()), 0.0)

    def test_the_published_form_is_unbounded_below(self):
        g = self.geometry(negative_term="unbounded", psi=10.0)
        self.assertLess(float(self.loss(g).detach()), -1e6)
        near, far = th.zeros(1, 5), th.full((1, 5), 1e5)
        self.assertLess(float(g.objective._push_apart(near, far)), -1e9)

    def test_the_default_is_bounded_where_the_published_form_diverges(self):
        g = self.geometry(psi=10.0)
        self.assertGreaterEqual(float(self.loss(g).detach()), 0.0)
        near = th.zeros(1, 5)
        for scale in (1.0, 1e2, 1e5):
            self.assertGreaterEqual(float(g.objective._push_apart(near, near + scale)), 0.0)

    def test_the_margin_form_is_the_hinge_of_the_distance(self):
        g = self.geometry(neg_margin=1.0)
        o = g.objective
        near, far = th.zeros(2, 5), th.tensor([[0.0] * 5, [2.0] * 5])
        pushed = o._push_apart(near, far).flatten()
        self.assertAlmostEqual(float(pushed[0]), o.psi * 1.0, places=6)    # coincident: m
        self.assertAlmostEqual(float(pushed[1]), 0.0, places=6)            # beyond m: nothing

    def test_the_negative_term_is_zero_when_switched_off(self):
        g = self.geometry(neg_sampling=False)
        self.assertGreaterEqual(float(self.loss(g).detach()), 0.0)

    def test_an_unknown_form_is_refused(self):
        with self.assertRaises(ValueError):
            sf.AnchoredObjective(negative_term="soft")

    def test_the_named_method_selects_the_published_form(self):
        from inspect import signature
        # the named class and the general model have the same default; the authors' loss is
        # the explicit option
        self.assertEqual(signature(FaithEL.__init__).parameters["negative_term"].default,
                         "margin")
        self.assertEqual(signature(sf.AnchoredObjective.__init__)
                         .parameters["negative_term"].default, "margin")
        self.assertEqual(signature(METHODS["faithel"]).parameters["negative_term"].default,
                         "margin")


# -- FALCON ----------------------------------------------------------------------------------
def pair_dataset(n=8):
    """``r(a_i, b_i)``: ``a_i`` is the subject of the assertion, ``b_i`` its object."""
    def build(ad, axioms):
        A, B = (ad.create_class(NS + c) for c in "AB")
        r = ad.create_object_property(NS + "r")
        axioms.add(ad.create_subclass_of(A, B))
        for i in range(n):
            a, b = ad.create_individual(NS + f"a{i}"), ad.create_individual(NS + f"b{i}")
            axioms.add(ad.create_object_property_assertion(r, a, b))
            axioms.add(ad.create_class_assertion(A, a))
            axioms.add(ad.create_class_assertion(B, b))
    return _ontology(build)


class FalconDirection(TestCase):
    """The rows of the ALC dataset for ``r(a, b)`` are ``[b, r, a]``: object first. FALCON's
    direct code takes the first column as the head, so it trains ``mem(vector(b) + r,
    vector(a))``, the opposite of the reading ``a in exists r.{b}``."""

    @classmethod
    def setUpClass(cls):
        from mowl.datasets import ALCDataset
        cls.data = pair_dataset()
        cls.alc = ALCDataset(cls.data.ontology, cls.data)
        cls.ids = {str(k.toStringID()): v for k, v in cls.alc.individual_to_id.items()}

    def test_the_row_layout_of_the_alc_dataset_is_object_role_subject(self):
        for ax in next(a for p, a in self.alc.grouped_axioms.items()
                       if "ObjectPropertyAssertion" in str(p) and "Class" not in str(p)):
            row = self.alc.get_axiom_vector(ax)
            self.assertEqual(row[0], self.ids[str(ax.getObject().toStringID())])
            self.assertEqual(row[2], self.ids[str(ax.getSubject().toStringID())])
            self.assertNotEqual(row[0], row[2])

    def trained(self, **options):
        th.manual_seed(0)
        model = ComposedALC(self.data, method="falcon", embed_dim=8, anon_e=2, **options)
        model.train(epochs=300)
        g = model.module
        a = th.tensor([self.ids[NS + f"a{i}"] for i in range(8)])
        b = th.tensor([self.ids[NS + f"b{i}"] for i in range(8)])
        role = th.zeros(8, dtype=th.long)
        with th.no_grad():
            forward = g.relation_model.degree_at(role, a, b).mean()   # a + r -> b (exists)
            backward = g.relation_model.degree_at(role, b, a).mean()
        return float(forward), float(backward)

    def test_the_part_default_reads_the_subject_as_the_head(self):
        self.assertEqual(F.FuzzyObjective().abox_head, "subject")
        forward, backward = self.trained(abox_head="subject")
        self.assertGreater(forward, backward + 0.2)

    def test_the_named_method_keeps_the_direction_of_the_code(self):
        forward, backward = self.trained()                 # builder default: abox_head="object"
        self.assertGreater(backward, forward + 0.2)


# -- BoxEL / BoxSEL: role axioms -------------------------------------------------------------
class RoleAxioms(TestCase):

    def check(self, cls):
        off = cls(tiny_ontology(), embed_dim=4, batch_size=64)
        self.assertNotIn("role_inclusion", off.training_datasets)
        self.assertNotIn("role_chain", off.training_datasets)
        model = cls(tiny_ontology(), embed_dim=4, batch_size=64, load_role_axioms=True)
        data = model.training_datasets
        self.assertGreater(len(data["role_inclusion"]), 0)
        self.assertGreater(len(data["role_chain"]), 0)
        model.train(epochs=1)
        for p in model.module.parameters():
            self.assertTrue(bool(th.isfinite(p).all()))

    def test_boxel(self):
        self.check(BoxEL)

    def test_boxsel(self):
        self.check(BoxSEL)
