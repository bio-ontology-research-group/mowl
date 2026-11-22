"""The geometry of E2R gives the module's own numbers."""
from unittest import TestCase

import torch as th

from mowl.baselines.nn.alc.e2r.module import E2RModule
from mowl.representation import geometry_of, ConstructorUnavailable
from mowl.representation.library.logic_subspace import E2RGeometry, Subspace


def rows(n, *highs, seed=0):
    g = th.Generator().manual_seed(seed)
    return th.stack([th.randint(0, h, (n,), generator=g) for h in highs], dim=1)


class TestE2RGeometry(TestCase):

    def setUp(self):
        th.manual_seed(0)
        self.module = E2RModule(4, 6, 2, dim=5, top_index=4, bottom_index=5)
        self.g = geometry_of(self.module)

    def test_adapter(self):
        self.assertIsInstance(self.g, E2RGeometry)
        self.assertEqual((self.g.num_classes, self.g.num_individuals), (6, 4))

    def test_inside_apart_member(self):
        m, g = self.module, self.g
        data = rows(20, 6, 6)
        a, b = g.classes(data[:, 0]), g.classes(data[:, 1])
        self.assertTrue(th.allclose(g.inside(a, b).reshape(-1), m.concept_inclusion_loss(data)))
        self.assertTrue(th.allclose(g.apart(a, b).reshape(-1),
                                    m.concept_disjointness_loss(data)))
        data = rows(20, 4, 6)
        region = g.classes(data[:, 1])
        residual = m.membership_residual(m.concept(data[:, 1]), m.entity(data[:, 0]))
        self.assertTrue(th.allclose(g.member(data[:, 0], region).reshape(-1) ** 0.5,
                                    residual, atol=1e-6))
        with_unit = E2RGeometry(m, unit_length=True)
        self.assertTrue(th.allclose(
            with_unit.member(data[:, 0], region).reshape(-1), m.class_assertion_loss(data)))
        roles = rows(20, 4, 2, 4)
        self.assertTrue(th.allclose(
            with_unit.relation_member(roles[:, 0], roles[:, 1], roles[:, 2]).reshape(-1),
            m.role_assertion_loss(roles)))

    def test_constructors_are_those_of_the_defining_losses(self):
        m, g = self.module, self.g
        with th.no_grad():
            m.c_embedding.weight.copy_((m.c_embedding.weight > 0.5).float())
        ids = th.tensor([0, 1, 2])
        a, b = g.classes(ids), g.classes(ids.flip(0))
        meet, gap = g.conjunction(a, b)
        self.assertTrue(th.equal(meet.indicator, a.indicator * b.indicator))
        self.assertTrue(th.equal(g.disjunction(a, b).indicator,
                                 th.maximum(a.indicator, b.indicator)))
        # the complement: eq. (8) on the real half is zero for a class and its complement
        not_a = g.negation(a)
        d = m.dim
        self.assertTrue(th.equal(not_a.indicator[:, :d], 1 - a.indicator[:, :d]))
        self.assertTrue(th.equal(not_a.indicator[:, d:], th.zeros(3, d)))
        from mowl.baselines.nn.alc.e2r import losses as L
        self.assertEqual(float(L.negation_loss(a.indicator, not_a.indicator, dim=d).sum()), 0)
        self.assertIsInstance(meet, Subspace)

    def test_no_existential(self):
        with self.assertRaises(ConstructorUnavailable):
            self.g.exists(th.tensor([0]), self.g.classes(th.tensor([0])))
