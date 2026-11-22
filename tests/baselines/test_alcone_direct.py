"""Loss semantics of the learnable al-cone module (mowl.baselines.nn.alc.alcone).

The relaxation is ours (the al-cone papers give no learner). The tests check, against the
exact checker of mowl.geometric.cones, that (i) a positive Gödel value means "half-axis in
the discretised cone" and a negative one "not in", (ii) every axiom loss is zero only if
the discretised interpretation satisfies the axiom, and is zero when it does and all
logits are bounded away from 0 by the margin.
"""
import random

import numpy as np
import pytest
import torch as th

from mowl.baselines.geometric.core.cones import (ALCone, ALConeChecker, ALConeInterpretation, And,
                                  Atomic, BOTTOM_CONCEPT, ConceptAssertion, Exists, Forall,
                                  GCI, Not, Or, RoleAssertion, TOP_CONCEPT)
from mowl.baselines.nn.alc.alcone import losses as L
from mowl.baselines.nn.alc.alcone.module import ALConeModule
from mowl.owlapi import OWLAPIAdapter

GAMMA = 0.1
NAMES = ["A", "B", "C"]
ROLES = ["r", "s"]


def _random_concept(rng, depth):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice([Atomic(n) for n in NAMES] + [TOP_CONCEPT, BOTTOM_CONCEPT])
    k = rng.randrange(5)
    if k == 0:
        return Not(_random_concept(rng, depth - 1))
    if k in (1, 2):
        args = (_random_concept(rng, depth - 1), _random_concept(rng, depth - 1))
        return And(args) if k == 1 else Or(args)
    f = _random_concept(rng, depth - 1)
    r = rng.choice(ROLES)
    return Exists(r, f) if k == 3 else Forall(r, f)


def _soft(concept, P, tau=0.0):
    """Evaluate a concept on logits with the module's operators."""
    if concept == TOP_CONCEPT:
        return th.full_like(next(iter(P["c"].values())), 1e4)
    if concept == BOTTOM_CONCEPT:
        return th.full_like(next(iter(P["c"].values())), -1e4)
    if isinstance(concept, Atomic):
        return P["c"][concept.name]
    if isinstance(concept, Not):
        return L.godel_not(_soft(concept.arg, P, tau))
    if isinstance(concept, (And, Or)):
        out = _soft(concept.args[0], P, tau)
        for a in concept.args[1:]:
            v = _soft(a, P, tau)
            out = L.godel_and(out, v, tau) if isinstance(concept, And) \
                else L.godel_or(out, v, tau)
        return out
    op = L.exists if isinstance(concept, Exists) else L.forall
    return op(P["r"][concept.role], _soft(concept.filler, P, tau), tau)


def _random_params(seed, dim=3, separated=False):
    g = th.Generator().manual_seed(seed)

    def draw(*shape):
        x = th.randn(*shape, generator=g)
        if separated:
            x = th.sign(x) * (GAMMA + x.abs())
        return x
    m = 2 * dim
    return {"c": {n: draw(1, m) for n in NAMES},
            "r": {r: draw(1, m, m) for r in ROLES},
            "e": {i: draw(1, m) for i in ("a", "b")}}


def _discretise(P, dim=3):
    concepts = {n: ALCone(v[0].numpy() > 0) for n, v in P["c"].items()}
    roles = {r: v[0].numpy() > 0 for r, v in P["r"].items()}
    inds = {}
    for i, u in P["e"].items():
        u = u[0].numpy().reshape(-1, 2)
        pt = np.zeros(dim)
        for k, (p, q) in enumerate(u):
            if p > 0 and (q <= 0 or p >= q):
                pt[k] = 1.0
            elif q > 0:
                pt[k] = -1.0
        inds[i] = pt
    return ALConeInterpretation(dim, concepts, roles, inds)


class TestExactnessLemma:

    @pytest.mark.parametrize("seed", range(5))
    def test_sign_of_value_decides_membership(self, seed):
        """value > 0 ⇒ half-axis in the discretised cone; value < 0 ⇒ not in."""
        rng = random.Random(seed)
        P = _random_params(seed)
        chk = ALConeChecker(_discretise(P))
        for _ in range(100):
            c = _random_concept(rng, 4)
            v = _soft(c, P)[0].numpy()
            bits = chk.cone(c).bits
            assert np.all(bits[v > 0])
            assert not np.any(bits[v < 0])

    def test_smooth_operators_converge(self):
        P = _random_params(0)
        rng = random.Random(0)
        for _ in range(20):
            c = _random_concept(rng, 3)
            exact = _soft(c, P)
            assert th.allclose(_soft(c, P, tau=1e-4), exact, atol=1e-2)


class TestLossSemantics:

    @pytest.mark.parametrize("seed", range(8))
    def test_zero_loss_iff_satisfied(self, seed):
        """For margin-separated logits, each loss is zero iff the exact checker says the
        discretised interpretation satisfies the axiom. For arbitrary logits, zero loss
        still implies satisfaction."""
        rng = random.Random(seed)
        for separated in (True, False):
            P = _random_params(seed, separated=separated)
            chk = ALConeChecker(_discretise(P))
            for _ in range(60):
                c, d = _random_concept(rng, 3), _random_concept(rng, 3)
                vc, vd = _soft(c, P), _soft(d, P)
                cases = [
                    (L.inclusion_loss(vc, vd, GAMMA), GCI(c, d)),
                    (L.disjointness_loss(vc, vd, GAMMA), GCI(And((c, d)), BOTTOM_CONCEPT)),
                ]
                u = P["e"]["a"]
                memb = L.membership_loss(u, vc, GAMMA) + L.point_validity_loss(u, GAMMA)
                cases.append((memb, ConceptAssertion(c, "a")))
                r = rng.choice(ROLES)
                ua, ub = P["e"]["a"], P["e"]["b"]
                role = L.role_assertion_loss(ua, P["r"][r], ub, GAMMA)
                role = role + L.point_validity_loss(ua, GAMMA) + L.point_validity_loss(ub, GAMMA)
                cases.append((role, RoleAssertion(r, "a", "b")))
                for loss, ax in cases:
                    zero = float(loss) == 0.0
                    sat = chk.satisfies(ax)
                    if zero:
                        assert sat, ax
                    if separated and not isinstance(ax, (ConceptAssertion, RoleAssertion)):
                        assert zero == sat, ax

    def test_assertion_losses_on_constructed_points(self):
        """Hand-built configuration: a on half-axis 0, b on half-axis 3, r: 0 -> 3."""
        m = 4
        u_a = th.full((1, m), -1.0)
        u_a[0, 0] = 1.0
        u_b = th.full((1, m), -1.0)
        u_b[0, 3] = 1.0
        r = th.full((1, m, m), -1.0)
        r[0, 3, 0] = 1.0
        assert float(L.role_assertion_loss(u_a, r, u_b, GAMMA)) == 0.0
        assert float(L.role_assertion_loss(u_b, r, u_a, GAMMA)) > 0.0
        c = th.tensor([[1.0, -1.0, -1.0, -1.0]])
        assert float(L.membership_loss(u_a, c, GAMMA)) == 0.0
        assert float(L.membership_loss(u_b, c, GAMMA)) > 0.0
        assert float(L.point_validity_loss(u_a, GAMMA)) == 0.0
        assert float(L.point_validity_loss(th.full((1, m), -1.0), GAMMA)) > 0.0  # origin
        both = th.tensor([[1.0, 1.0, -1.0, -1.0]])
        assert float(L.point_validity_loss(both, GAMMA)) > 0.0  # +e1 and -e1 at once

    def test_wrong_direction_is_penalised(self):
        """A ⊑ B with A = R+ × {0}, B = R × {0}: zero; the converse: positive."""
        a = th.tensor([[1.0, -1.0, -1.0, -1.0]])
        b = th.tensor([[1.0, 1.0, -1.0, -1.0]])
        assert float(L.inclusion_loss(a, b, GAMMA)) == 0.0
        assert float(L.inclusion_loss(b, a, GAMMA)) > 0.0
        assert float(L.disjointness_loss(a, -b, GAMMA)) == 0.0  # A ⊓ ¬B = {0}


@pytest.fixture(scope="module")
def patterns():
    ad = OWLAPIAdapter()
    c0, c1 = ad.create_class("http://x#P0"), ad.create_class("http://x#P1")
    r = ad.create_object_property("http://x#r")
    ind = ad.create_individual("http://x#i")
    return {
        "sub": ad.create_subclass_of(c0, ad.create_object_some_values_from(r, c1)),
        "forall": ad.create_subclass_of(
            ad.create_object_union_of(c0, ad.create_complement_of(c1)),
            ad.create_object_all_values_from(r, c1)),
        "assert": ad.create_class_assertion(ad.create_object_intersection_of(c0, c1),
                                            ind),
        "role": ad.create_object_property_assertion(r, ind, ind),
        "disj": ad.create_disjoint_classes(c0, c1),
    }


class TestModuleForward:

    def _module(self):
        th.manual_seed(0)
        return ALConeModule(5, 2, 1, dim=2, margin=GAMMA, top_index=3, bottom_index=4)

    def test_forward_matches_losses(self, patterns):
        mod = self._module()
        x = th.tensor([[0, 0, 1], [1, 0, 2]])
        c = mod.class_logits(x[:, 0])
        rr = mod.role_logits(x[:, 1])
        d = mod.class_logits(x[:, 2])
        expected = L.inclusion_loss(c, L.exists(rr, d), GAMMA)
        assert th.allclose(mod(patterns["sub"], x), expected)
        # columns of SubClassOf(P0 ⊔ ¬P1, ∀r.P1): [c, c', r, c'']
        x2 = th.tensor([[0, 1, 0, 1]])
        sub = L.godel_or(mod.class_logits(x2[:, 0]), -mod.class_logits(x2[:, 1]))
        sup = L.forall(mod.role_logits(x2[:, 2]), mod.class_logits(x2[:, 3]))
        assert th.allclose(mod(patterns["forall"], x2), L.inclusion_loss(sub, sup, GAMMA))

    def test_top_and_bottom(self, patterns):
        mod = self._module()
        # A ⊑ ∃r.⊤ and ⊥ ⊑ anything
        x = th.tensor([[4, 0, 0]])  # ⊥ ⊑ ∃r.P0
        assert float(mod(patterns["sub"], x).detach()) == 0.0
        assert th.all(mod.class_logits(th.tensor([3])) > 0)
        assert th.all(mod.class_logits(th.tensor([4])) < 0)

    def test_role_assertion_column_order(self, patterns):
        """ALCDataset encodes R(a, b) as [b, R, a]."""
        mod = self._module()
        m = 4
        with th.no_grad():
            mod.e_embedding.weight[:] = -1.0
            mod.e_embedding.weight[0, 0] = 1.0  # a on half-axis 0
            mod.e_embedding.weight[1, 3] = 1.0  # b on half-axis 3
            mod.r_embedding.weight[:] = -1.0
            w = mod.r_embedding.weight.view(1, m, m)
            w[0, 3, 0] = 1.0  # 0 -> 3
        good = th.tensor([[1, 0, 0]])  # r(a, b)
        bad = th.tensor([[0, 0, 1]])  # r(b, a)
        assert float(mod(patterns["role"], good).detach()) == 0.0
        assert float(mod(patterns["role"], bad).detach()) > 0.0

    def test_zero_loss_implies_checker(self, patterns):
        """Random modules: whenever a pattern loss is zero, the discretised
        interpretation satisfies the corresponding axiom (exact checker)."""
        for seed in range(20):
            th.manual_seed(seed)
            mod = ALConeModule(3, 2, 1, dim=2, margin=GAMMA, init_std=2.0)
            names = ["P0", "P1", "P2"]
            interp = mod.discretise(names, ["r"], ["a", "b"])
            chk = ALConeChecker(interp)
            P0, P1 = Atomic("P0"), Atomic("P1")
            checks = [
                (patterns["sub"], th.tensor([[0, 0, 1]]), GCI(P0, Exists("r", P1))),
                (patterns["forall"], th.tensor([[0, 1, 0, 1]]),
                 GCI(Or((P0, Not(P1))), Forall("r", P1))),
                (patterns["disj"], th.tensor([[0, 1]]), GCI(And((P0, P1)), BOTTOM_CONCEPT)),
                (patterns["assert"], th.tensor([[0, 0, 1]]),
                 ConceptAssertion(And((P0, P1)), "a")),
                (patterns["role"], th.tensor([[1, 0, 0]]), RoleAssertion("r", "a", "b")),
            ]
            for pat, x, ax in checks:
                if float(mod(pat, x).detach()) == 0.0:
                    assert chk.satisfies(ax), (seed, ax)

    def test_invalid_margin(self):
        with pytest.raises(ValueError):
            ALConeModule(2, 1, 1, dim=2, margin=0.0)


class TestConstantLogit:
    """The logits of owl:Thing / owl:Nothing are on the scale of learned logits by default;
    ``constant_logit=1e4`` reproduces the numbers of earlier versions."""

    def _setup(self, **kw):
        th.manual_seed(0)
        mod = ALConeModule(5, 2, 1, dim=2, margin=GAMMA, top_index=3, bottom_index=4, **kw)
        return mod

    def test_default_scale(self, patterns):
        mod = self._setup()
        assert th.all(mod.class_logits(th.tensor([3])) == 1.0)
        assert th.all(mod.class_logits(th.tensor([4])) == -1.0)
        # P0 ⊑ ⊥-like axiom: the loss stays on the scale of the learned logits
        x = th.tensor([[0, 0, 4]])  # P0 ⊑ ∃r.⊥
        assert float(mod(patterns["sub"], x).detach()) < 10.0

    def test_old_value_reproduces_old_numbers(self, patterns):
        mod = self._setup(constant_logit=1e4)
        assert th.all(mod.class_logits(th.tensor([4])) == -1e4)
        x = th.tensor([[0, 0, 4]])
        c = mod.class_logits(x[:, 0])
        d = mod.class_logits(x[:, 2])
        expected = L.inclusion_loss(c, L.exists(mod.role_logits(x[:, 1]), d), GAMMA)
        assert th.allclose(mod(patterns["sub"], x), expected)
        assert float(mod(patterns["sub"], x).detach()) > 100.0

    def test_must_cover_margin(self):
        with pytest.raises(ValueError):
            self._setup(constant_logit=0.01)
