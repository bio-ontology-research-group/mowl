"""Loss semantics of EmEL++ [mondal2021] and EmEL(var) [mohapatra2021] on hand-built balls."""

import math
from unittest import TestCase

import torch as th

import mowl.nn.el.emelpp.losses as L
from mowl.nn.el.elem.module import ELEmModule
from mowl.nn.el.emelpp.module import EmELPPModule, EmELVarModule

E1 = [1.0, 0.0, 0.0]
E2 = [0.0, 1.0, 0.0]
E3 = [0.0, 0.0, 1.0]
HALF = math.sqrt(3) / 2


def set_class(module, idx, center, radius):
    module.class_embed.weight.data[idx] = th.tensor(center)
    module.class_rad.weight.data[idx] = th.tensor([radius])


def set_rel(module, idx, vec):
    module.rel_embed.weight.data[idx] = th.tensor(vec)


def idx(*rows):
    return th.tensor([list(rows)])


def value(t):
    return t.sum().item()


class TestNormalForms(TestCase):
    """Zero loss iff the geometric condition holds (margin 0, unit-norm centres)."""

    def setUp(self):
        self.m = EmELPPModule(6, 2, 0, embed_dim=3, margin=0.0)

    def test_gci0(self):
        set_class(self.m, 0, E1, 0.2)
        set_class(self.m, 1, E1, 0.5)
        self.assertAlmostEqual(value(self.m.gci0_loss(idx(0, 1))), 0.0, places=6)
        # sub-class radius larger than the super-class radius
        self.assertAlmostEqual(value(self.m.gci0_loss(idx(1, 0))), 0.3, places=6)
        # centres apart: ||e1 - e2|| = sqrt 2, radii cancel
        set_class(self.m, 2, E2, 0.5)
        self.assertAlmostEqual(value(self.m.gci0_loss(idx(1, 2))), math.sqrt(2), places=5)

    def test_gci0_normalisation_term(self):
        set_class(self.m, 0, [2.0, 0.0, 0.0], 0.2)
        set_class(self.m, 1, E1, 0.5)
        # ||c - d|| = 1, rc - rd = -0.3, plus | ||c|| - 1 | = 1
        self.assertAlmostEqual(value(self.m.gci0_loss(idx(0, 1))), 0.7 + 1.0, places=5)

    def test_gci1(self):
        for i, r in enumerate([0.4, 0.4, 0.4]):
            set_class(self.m, i, E1, r)
        self.assertAlmostEqual(value(self.m.gci1_loss(idx(0, 1, 2))), 0.0, places=6)
        # fourth term: the radius of E must reach the smaller radius of C and D
        set_class(self.m, 2, E1, 0.1)
        self.assertAlmostEqual(value(self.m.gci1_loss(idx(0, 1, 2))), 0.3, places=6)
        # C and D disjoint: first term ||c - d|| - rc - rd = sqrt 2 - 0.8
        set_class(self.m, 1, E2, 0.4)
        set_class(self.m, 2, E1, 0.4)
        expected = (math.sqrt(2) - 0.8) + (math.sqrt(2) - 0.4)
        self.assertAlmostEqual(value(self.m.gci1_loss(idx(0, 1, 2))), expected, places=5)

    def test_gci2(self):
        # c + r = d, so translating the ball of C lands inside the ball of D
        set_class(self.m, 0, E1, 0.2)
        set_class(self.m, 1, E2, 0.5)
        set_rel(self.m, 0, [-1.0, 1.0, 0.0])
        self.assertAlmostEqual(value(self.m.gci2_loss(idx(0, 0, 1))), 0.0, places=6)
        # radius of C larger than the radius of D
        set_class(self.m, 0, E1, 0.8)
        self.assertAlmostEqual(value(self.m.gci2_loss(idx(0, 0, 1))), 0.3, places=6)
        # wrong translation: c + r = e3, ||c + r - d|| = sqrt 2
        set_class(self.m, 0, E1, 0.2)
        set_rel(self.m, 0, [-1.0, 0.0, 1.0])
        self.assertAlmostEqual(value(self.m.gci2_loss(idx(0, 0, 1))),
                               math.sqrt(2) + 0.2 - 0.5, places=5)

    def test_gci3(self):
        # c - r = d
        set_class(self.m, 0, E1, 0.2)
        set_class(self.m, 1, E2, 0.3)
        set_rel(self.m, 0, [1.0, -1.0, 0.0])
        self.assertAlmostEqual(value(self.m.gci3_loss(idx(0, 0, 1))), 0.0, places=6)
        set_rel(self.m, 0, [-1.0, 1.0, 0.0])   # wrong direction: ||c - r - d|| = 2 sqrt 2... > 0
        self.assertGreater(value(self.m.gci3_loss(idx(0, 0, 1))), 1.0)

    def test_bottom_losses(self):
        set_class(self.m, 0, E1, 0.3)
        set_class(self.m, 1, E1, -0.7)     # radii are used through their absolute value
        self.assertAlmostEqual(value(self.m.gci0_bot_loss(idx(0, 5))), 0.3, places=6)
        self.assertAlmostEqual(value(self.m.gci0_bot_loss(idx(1, 5))), 0.7, places=6)
        self.assertAlmostEqual(value(self.m.gci3_bot_loss(idx(0, 0, 5))), 0.3, places=6)
        # disjointness: zero when the balls are apart, positive when they overlap
        set_class(self.m, 2, E2, 0.5)
        set_class(self.m, 3, E3, 0.5)
        self.assertAlmostEqual(value(self.m.gci1_bot_loss(idx(2, 3, 5))), 0.0, places=6)
        set_class(self.m, 3, E2, 0.5)
        self.assertAlmostEqual(value(self.m.gci1_bot_loss(idx(2, 3, 5))), 1.0, places=6)

    def test_negative_gci2(self):
        """Eq. 10: zero when the translated ball and D are apart, positive when C ⊑ ∃R.D
        would hold."""
        set_class(self.m, 0, E1, 0.2)
        set_class(self.m, 1, E2, 0.3)
        set_rel(self.m, 0, [-1.0, 1.0, 0.0])      # c + r = d
        self.assertAlmostEqual(value(self.m.gci2_loss(idx(0, 0, 1), neg=True)), 0.5, places=6)
        set_rel(self.m, 0, [0.0, 0.0, 0.0])       # ||c + r - d|| = sqrt 2 > 0.5
        self.assertAlmostEqual(value(self.m.gci2_loss(idx(0, 0, 1), neg=True)), 0.0, places=6)

    def test_neg_capability_is_truthful(self):
        """Only gci2 (and the role assertion, the same normal form) have a negative loss."""
        self.assertEqual(self.m.neg_capable_gcis, {"gci2", "object_property_assertion"})
        self.assertTrue(self.m.role_axiom_capable)
        set_class(self.m, 0, E1, 0.2)
        set_class(self.m, 1, E2, 0.3)
        for name, data in [("gci0", idx(0, 1)), ("gci1", idx(0, 1, 0)), ("gci3", idx(0, 0, 1)),
                           ("gci1_bot", idx(0, 1, 5))]:
            self.assertTrue(th.equal(self.m(data, name), self.m(data, name, neg=True)),
                            name)

    def test_one_value_per_row(self):
        data = th.tensor([[0, 1], [1, 0], [2, 3]])
        self.assertEqual(self.m.gci0_loss(data).shape, (3, 1))
        self.assertEqual(self.m.role_inclusion_loss(th.tensor([[0, 1], [1, 0]])).shape, (2, 1))
        self.assertTrue((self.m.gci0_loss(data) >= 0).all())

    def test_no_separate_regularisation(self):
        self.assertEqual(self.m.regularization_loss().item(), 0.0)


class TestAgainstELEm(TestCase):
    """The concept losses of EmEL++ are those of ELEmbeddings (same embeddings and
    initialisation) up to the three differences documented on the module."""

    @classmethod
    def setUpClass(cls):
        th.manual_seed(7)
        cls.elem = ELEmModule(12, 3, 0, embed_dim=8, margin=0.1)
        th.manual_seed(7)
        cls.emel = EmELPPModule(12, 3, 0, embed_dim=8, margin=0.1)
        radii = th.rand(12, 1, generator=th.Generator().manual_seed(1))
        cls.elem.class_rad.weight.data.copy_(radii)
        cls.emel.class_rad.weight.data.copy_(radii)
        g = th.Generator().manual_seed(0)
        cls.g2 = th.randint(0, 12, (20, 2), generator=g)
        cls.g3 = th.cat([th.randint(0, 12, (20, 1), generator=g),
                         th.randint(0, 3, (20, 1), generator=g),
                         th.randint(0, 12, (20, 1), generator=g)], dim=1)
        cls.g3r = th.cat([cls.g3[:, 1:2], cls.g3[:, :1], cls.g3[:, 2:]], dim=1)
        cls.g111 = th.randint(0, 12, (20, 3), generator=g)

    def test_same_embeddings(self):
        for a, b in zip(self.elem.parameters(), self.emel.parameters()):
            self.assertTrue(th.equal(a, b))

    def test_gci0_gci3_gci1_bot_equal(self):
        for name, data in [("gci0", self.g2), ("gci3", self.g3r),
                           ("gci1_bot", self.g111[:, :2])]:
            a = getattr(self.elem, f"{name}_loss")(data)
            b = getattr(self.emel, f"{name}_loss")(data)
            self.assertTrue(th.allclose(a, b, atol=1e-6), name)

    def test_gci2_equal_up_to_elem_constant(self):
        a = self.elem.gci2_loss(self.g3)
        b = self.emel.gci2_loss(self.g3)
        self.assertTrue(th.allclose(a - 10e-6, b, atol=1e-6))

    def test_gci2_negative_equal(self):
        a = self.elem.gci2_loss(self.g3, neg=True)
        b = self.emel.gci2_loss(self.g3, neg=True)
        self.assertTrue(th.allclose(a, b, atol=1e-6))

    def test_gci1_differs_by_the_radius_term(self):
        d = self.g111
        rc, rd, re = (th.abs(self.emel.class_rad(d[:, i])) for i in range(3))
        extra = th.relu(th.minimum(rc, rd) - re - 0.1)
        a = self.elem.gci1_loss(d)
        b = self.emel.gci1_loss(d)
        self.assertTrue(th.allclose(a + extra, b, atol=1e-6))
        self.assertGreater(extra.sum().item(), 0)

    def test_bottom_losses_use_absolute_radius(self):
        d = self.g3
        self.assertTrue(th.allclose(th.abs(self.elem.gci0_bot_loss(d)),
                                    self.emel.gci0_bot_loss(d)))
        self.assertTrue((self.emel.gci0_bot_loss(d) >= 0).all())


class TestRoleAxioms(TestCase):

    def setUp(self):
        self.m = EmELPPModule(2, 4, 0, embed_dim=3, margin=0.0)

    def test_role_inclusion(self):
        set_rel(self.m, 0, E1)
        set_rel(self.m, 1, E1)
        self.assertAlmostEqual(value(self.m.role_inclusion_loss(idx(0, 1))), 0.0, places=6)
        # orthogonal unit vectors: distance sqrt 2 plus direction term 1
        set_rel(self.m, 1, E2)
        self.assertAlmostEqual(value(self.m.role_inclusion_loss(idx(0, 1))),
                               math.sqrt(2) + 1.0, places=5)
        # same direction, different length: distance 1, direction 0, norm term 1
        set_rel(self.m, 1, [2.0, 0.0, 0.0])
        self.assertAlmostEqual(value(self.m.role_inclusion_loss(idx(0, 1))), 1.0 + 1.0,
                               places=5)

    def test_role_inclusion_margin(self):
        m = EmELPPModule(2, 2, 0, embed_dim=3, margin=0.5)
        set_rel(m, 0, E1)
        set_rel(m, 1, [HALF, 0.5, 0.0])      # unit vector at 30 degrees
        dist = math.sqrt((1 - HALF) ** 2 + 0.25)
        expected = max(0.0, dist - 0.5) + (1 - HALF)
        self.assertAlmostEqual(value(m.role_inclusion_loss(idx(0, 1))), expected, places=5)

    def test_role_chain(self):
        r1 = E1
        r2 = [-0.5, HALF, 0.0]
        s = [0.5, HALF, 0.0]               # unit vector, r1 + r2 = s
        set_rel(self.m, 0, r1)
        set_rel(self.m, 1, r2)
        set_rel(self.m, 2, s)
        self.assertAlmostEqual(value(self.m.role_chain_loss(idx(0, 1, 2))), 0.0, places=5)
        # a different target: s = E3
        set_rel(self.m, 3, E3)
        self.assertGreater(value(self.m.role_chain_loss(idx(0, 1, 3))), 1.0)
        # same direction as r1 + r2 but shorter
        set_rel(self.m, 3, [0.25, HALF / 2, 0.0])
        loss = value(self.m.role_chain_loss(idx(0, 1, 3)))
        self.assertAlmostEqual(loss, 0.5 + 0.0 + 0.5, places=5)   # distance 0.5, norm term 0.5

    def test_role_inclusion_loss_is_symmetric(self):
        """LIMITATION (chen2025, bourgaux2024): L(r ⊑ s) == L(s ⊑ r) for every r, s, so
        r ⊑ s also enforces s ⊑ r."""
        th.manual_seed(0)
        m = EmELPPModule(2, 10, 0, embed_dim=5, margin=0.1)
        for sign in (1, -1):
            m.rel_embed.weight.data.mul_(sign * 1.7)
            fwd = m.role_inclusion_loss(th.tensor([[i, j] for i in range(10)
                                                   for j in range(10)]))
            bwd = m.role_inclusion_loss(th.tensor([[j, i] for i in range(10)
                                                   for j in range(10)]))
            self.assertTrue(th.allclose(fwd, bwd, atol=1e-6))

    def test_role_inclusion_training_moves_the_super_role(self):
        """Training r ⊑ s with r frozen pulls s onto r, i.e. the 'super-role' is changed
        exactly like the 'sub-role' would be; afterwards s ⊑ r holds as well."""
        th.manual_seed(1)
        m = EmELPPModule(2, 2, 0, embed_dim=3, margin=0.0)
        set_rel(m, 0, E1)
        set_rel(m, 1, E2)
        opt = th.optim.Adam(m.parameters(), lr=0.05)
        for _ in range(400):
            opt.zero_grad()
            loss = m.role_inclusion_loss(idx(0, 1)).sum()
            loss.backward()
            m.rel_embed.weight.grad[0] = 0          # r frozen: only s can move
            opt.step()
        s = m.rel_embed.weight.data[1]
        self.assertLess((s - th.tensor(E1)).norm().item(), 0.05)
        self.assertLess(value(m.role_inclusion_loss(idx(0, 1))), 0.05)
        self.assertLess(value(m.role_inclusion_loss(idx(1, 0))), 0.05)   # s ⊑ r as well

    def test_role_chain_loss_is_commutative(self):
        """LIMITATION (bourgaux2024): the chain loss only sees r1 + r2."""
        th.manual_seed(2)
        m = EmELPPModule(2, 8, 0, embed_dim=4, margin=0.1)
        rows = th.tensor([[i, j, k] for i in range(4) for j in range(4, 8) for k in range(8)])
        swapped = rows[:, [1, 0, 2]]
        self.assertTrue(th.allclose(m.role_chain_loss(rows), m.role_chain_loss(swapped),
                                    atol=1e-6))

    def test_forward_dispatch(self):
        for name, data in [("role_inclusion", idx(0, 1)), ("role_chain", idx(0, 1, 2))]:
            self.assertEqual(self.m(data, name).shape, (1, 1))


class TestEmELVarLosses(TestCase):

    def setUp(self):
        self.m = EmELVarModule(4, 2, 0, embed_dim=3, margin=0.0, sigma_init=0.0)

    def set_sigma(self, value_):
        self.m.rel_sigma.weight.data[:] = value_

    def test_sigma_zero_is_emelpp(self):
        th.manual_seed(3)
        a = EmELPPModule(6, 2, 0, embed_dim=4, margin=0.1)
        th.manual_seed(3)
        b = EmELVarModule(6, 2, 0, embed_dim=4, margin=0.1, sigma_init=0.0)
        data = th.tensor([[0, 0, 1], [2, 1, 3], [4, 0, 5]])      # (C, R, D) rows
        data3 = data[:, [1, 0, 2]]                                # (R, C, D) rows
        self.assertTrue(th.allclose(a(data, "gci2"), b(data, "gci2"), atol=1e-6))
        self.assertTrue(th.allclose(a(data3, "gci3"), b(data3, "gci3"), atol=1e-6))
        self.assertTrue(th.allclose(a(data, "gci2", neg=True), b(data, "gci2", neg=True)))
        self.assertTrue(th.allclose(a(data[:, [0, 2]], "gci0"), b(data[:, [0, 2]], "gci0")))
        pairs = th.tensor([[0, 1], [1, 0]])
        self.assertTrue(th.allclose(a(pairs, "role_inclusion"), b(pairs, "role_inclusion")))
        chains = th.tensor([[0, 1, 1]])
        self.assertTrue(th.allclose(a(chains, "role_chain"), b(chains, "role_chain")))

    def test_sigma_enters_gci2_and_gci3_as_in_eq_7_and_8(self):
        set_class(self.m, 0, E1, 0.2)
        set_class(self.m, 1, E2, 0.3)
        set_rel(self.m, 0, [-1.0, 0.0, 1.0])           # c + r = (0, 0, 1): ||. - d|| = sqrt 2
        self.set_sigma(0.0)
        hinge = math.sqrt(2) + 0.2 - 0.3
        self.assertAlmostEqual(value(self.m.gci2_loss(idx(0, 0, 1))), hinge, places=5)
        self.set_sigma(0.5)
        # Eq. 7: max(0, x - sigma) + sigma
        self.assertAlmostEqual(value(self.m.gci2_loss(idx(0, 0, 1))), hinge - 0.5 + 0.5,
                               places=5)
        self.set_sigma(2.0)
        self.assertAlmostEqual(value(self.m.gci2_loss(idx(0, 0, 1))), 2.0, places=5)
        # Eq. 8 for gci3: ||c - r - d|| - rc - rd - sigma
        set_rel(self.m, 0, [1.0, 0.0, 0.0])            # c - r = 0, ||0 - d|| = 1
        self.set_sigma(0.0)
        self.assertAlmostEqual(value(self.m.gci3_loss(idx(0, 0, 1))), 1.0 - 0.5, places=5)
        self.set_sigma(0.1)
        self.assertAlmostEqual(value(self.m.gci3_loss(idx(0, 0, 1))), 1.0 - 0.5, places=5)

    def test_sigma_is_used_through_its_absolute_value(self):
        set_class(self.m, 0, E1, 0.2)
        set_class(self.m, 1, E2, 0.3)
        set_rel(self.m, 0, [-1.0, 0.0, 1.0])
        self.set_sigma(2.0)
        pos = self.m.gci2_loss(idx(0, 0, 1))
        self.set_sigma(-2.0)
        self.assertTrue(th.equal(pos, self.m.gci2_loss(idx(0, 0, 1))))

    def test_negative_loss_ignores_sigma(self):
        set_class(self.m, 0, E1, 0.2)
        set_class(self.m, 1, E2, 0.3)
        set_rel(self.m, 0, [-1.0, 1.0, 0.0])
        self.set_sigma(0.0)
        a = self.m.gci2_loss(idx(0, 0, 1), neg=True)
        self.set_sigma(3.0)
        self.assertTrue(th.equal(a, self.m.gci2_loss(idx(0, 0, 1), neg=True)))


class TestOneToMany(TestCase):
    """C ⊑ ∃r.D1 and C ⊑ ∃r.D2 with D1, D2 disjoint (the pattern of mohapatra2021).

    D1 and D2 are balls of radius 0.5 around (1, 0, 0) and (-1, 0, 0), a gap of 1 apart.
    With margin 0 the two hinges of Eq. 3 add up to at least ||d1 - d2|| - r1 - r2 + 2 rc = 1
    for any translated centre, so plain translation cannot satisfy both.
    """

    def build(self, module_cls, **kwargs):
        m = module_cls(3, 1, 0, embed_dim=3, margin=0.0, **kwargs)
        set_class(m, 0, E2, 0.1)
        set_class(m, 1, E1, 0.5)
        set_class(m, 2, [-1.0, 0.0, 0.0], 0.5)
        set_rel(m, 0, [0.0, -1.0, 0.0])                 # c + r = origin, between D1 and D2
        return m

    DATA = th.tensor([[0, 0, 1], [0, 0, 2]])

    def optimise(self, m, steps=600):
        """Optimise C, r, sigma (the balls D1, D2 are fixed) on the loss of the pattern."""
        opt = th.optim.Adam(m.parameters(), lr=0.02)
        for _ in range(steps):
            opt.zero_grad()
            m.gci2_loss(self.DATA).sum().backward()
            m.class_embed.weight.grad[1:] = 0
            m.class_rad.weight.grad[1:] = 0
            opt.step()
        return m.gci2_loss(self.DATA).sum().item()

    def hinge_only(self, m):
        """Sum of the hinge terms (loss without normalisation terms and without +sigma)."""
        c = m.class_embed(self.DATA[:, 0])
        d = m.class_embed(self.DATA[:, 2])
        dist = th.linalg.norm(c + m.rel_embed(self.DATA[:, 1]) - d, dim=1, keepdim=True)
        slack = m._sigma(self.DATA[:, 1])
        slack = 0 if slack is None else slack
        return th.relu(dist + m._rad(self.DATA[:, 0]) - m._rad(self.DATA[:, 2]) - slack).sum()

    def test_translation_cannot_satisfy_one_to_many(self):
        m = self.build(EmELPPModule)
        self.assertAlmostEqual(self.hinge_only(m).item(), 1.2, places=5)  # 2 x (1 + 0.1 - 0.5)
        total = self.optimise(m)
        self.assertGreaterEqual(self.hinge_only(m).item(), 1.0 - 1e-3)
        self.assertGreaterEqual(total, 1.0 - 1e-3)

    def test_variance_satisfies_the_hinge_of_one_to_many(self):
        m = self.build(EmELVarModule, sigma_init=0.6)
        self.assertAlmostEqual(self.hinge_only(m).item(), 0.0, places=6)

    def test_published_loss_does_not_profit_from_sigma(self):
        """max(0, x - sigma) + sigma >= max(0, x): with the weight 1 of Eq. 7 the total loss
        of EmEL(var) is never below that of EmEL++ on the same embeddings, and the optimum
        on the one-to-many pattern is no better than for plain translation."""
        th.manual_seed(5)
        plain = EmELPPModule(8, 3, 0, embed_dim=4, margin=0.1)
        th.manual_seed(5)
        var = EmELVarModule(8, 3, 0, embed_dim=4, margin=0.1)
        data = th.randint(0, 8, (200, 3))
        data[:, 1] = data[:, 1] % 3                              # (C, R, D) rows
        data3 = data[:, [1, 0, 2]]                                # (R, C, D) rows
        for sigma in (0.0, 0.05, 0.5, 3.0):
            var.rel_sigma.weight.data[:] = sigma
            self.assertTrue((var(data, "gci2") >= plain(data, "gci2") - 1e-6).all(), sigma)
            self.assertTrue((var(data3, "gci3") >= plain(data3, "gci3") - 1e-6).all(), sigma)
        m = self.build(EmELVarModule, sigma_init=0.6)
        total = self.optimise(m)
        self.assertGreaterEqual(total, 1.0 - 1e-3)

    def test_variance_pays_off_with_a_smaller_sigma_weight(self):
        """NOT the paper's loss (sigma_reg_weight < 1): the variance lets the pattern be
        fitted with a total loss far below the plain-translation minimum of 1."""
        m = self.build(EmELVarModule, sigma_init=0.6, sigma_reg_weight=0.1)
        total = self.optimise(m)
        self.assertLess(total, 0.5)
        self.assertLess(self.hinge_only(m).item(), 0.05)

    def test_sigma_weight_default_is_the_paper(self):
        self.assertEqual(EmELVarModule(2, 1).sigma_reg_weight, 1.0)


class TestLossFunctions(TestCase):

    def test_losses_are_exported_at_module_level(self):
        for name in ("gci0_loss", "gci1_loss", "gci2_loss", "gci3_loss", "gci2_neg_loss",
                     "gci1_bot_loss", "bot_loss", "role_inclusion_loss", "role_chain_loss"):
            self.assertTrue(callable(getattr(L, name)))
