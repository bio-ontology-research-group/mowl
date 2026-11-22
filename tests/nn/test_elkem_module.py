"""Tests for the ELK-Em module [singh2026]: loss primitives, the box of ∃r.C, role inclusion
(Proposition 2), role composition (Propositions 4 and 5), ⊤/⊥ handling and the lexical
regulariser.

The geometric checks use point sampling of the relational semantics
``r^E = {(x, y) | x ∈ T_r(y) ⊕ E_r, y ∈ R_r}`` written here independently of the loss code.
"""

from mowl import init_jvm
init_jvm("1g")

from unittest import TestCase

import torch as th

import mowl.nn.el.elkem.losses as L
from mowl.nn.el.elkem.module import ELKEmModule

EPS = 1e-5


# ---------------------------------------------------------------- independent semantics

def in_box(x, lower, upper, tol=1e-6):
    return bool(((lower - tol <= x) & (x <= upper + tol)).all())


def in_role(x, y, role, tol=1e-6):
    """(x, y) ∈ r^E, directly from the definition."""
    rl, ru, el, eu, m, c = role
    return in_box(y, rl, ru, tol) and in_box(x, m * y + c + el, m * y + c + eu, tol)


def has_witness(x, role, cl, cu, tol=1e-6):
    """Is there y ∈ C ∩ R_r with (x, y) ∈ r^E?  Per dimension, x ∈ m y + c + [el, eu] iff
    y ∈ [(x - c - eu) / m, (x - c - el) / m]; the conditions are separable."""
    rl, ru, el, eu, m, c = role
    lo = th.maximum(th.maximum(cl, rl), (x - c - eu) / m)
    hi = th.minimum(th.minimum(cu, ru), (x - c - el) / m)
    return bool((lo <= hi + tol).all())


def sample_box(lower, upper, k, gen):
    u = th.rand(k, lower.shape[-1], generator=gen, dtype=lower.dtype)
    return lower + u * (upper - lower)


def corners_and_samples(lower, upper, k, gen):
    pts = [lower.unsqueeze(0), upper.unsqueeze(0), sample_box(lower, upper, k, gen)]
    return th.cat(pts)


def random_box(n, gen, scale=1.0, width=0.5):
    c = (th.rand(n, generator=gen, dtype=th.float64) - 0.5) * 2 * scale
    o = th.rand(n, generator=gen, dtype=th.float64) * width + 0.01
    return c - o, c + o


def random_role(n, gen):
    rl, ru = random_box(n, gen)
    e = th.rand(n, generator=gen, dtype=th.float64) * 0.3
    m = th.rand(n, generator=gen, dtype=th.float64) * 1.5 + 0.2
    c = (th.rand(n, generator=gen, dtype=th.float64) - 0.5)
    return rl, ru, -e, e, m, c


def sample_role_pairs(role, k, gen):
    """Pairs (x, y) of r^E, including the corners of R_r and E_r."""
    rl, ru, el, eu, m, c = role
    ys = corners_and_samples(rl, ru, k, gen)
    pairs = []
    for y in ys:
        for x in corners_and_samples(m * y + c + el, m * y + c + eu, 4, gen):
            pairs.append((x, y))
    return pairs


def batched(role):
    return tuple(t.unsqueeze(0) for t in role)


# ---------------------------------------------------------------- module helpers

def set_class(module, idx, lower, upper):
    lower = th.as_tensor(lower, dtype=th.float32)
    upper = th.as_tensor(upper, dtype=th.float32)
    module.class_center.weight.data[idx] = (lower + upper) / 2
    module.class_offset.weight.data[idx] = (upper - lower) / 2 - module.epsilon


def set_role(module, idx, range_lower, range_upper, error, slope, shift):
    rl = th.as_tensor(range_lower, dtype=th.float32)
    ru = th.as_tensor(range_upper, dtype=th.float32)
    module.range_center.weight.data[idx] = (rl + ru) / 2
    module.range_offset.weight.data[idx] = (ru - rl) / 2
    module.error_offset.weight.data[idx] = th.as_tensor(error, dtype=th.float32)
    module.slope.weight.data[idx] = th.as_tensor(slope, dtype=th.float32) - module.epsilon
    module.shift.weight.data[idx] = th.as_tensor(shift, dtype=th.float32)


def t(*rows):
    return th.tensor(rows, dtype=th.long)


class TestLossPrimitives(TestCase):
    """Eqs. (1)-(3) of the paper on hand-built boxes."""

    def box(self, lower, upper):
        return (th.tensor([lower], dtype=th.float64), th.tensor([upper], dtype=th.float64))

    def test_inclusion(self):
        inner = self.box([0.2, 0.2], [0.4, 0.4])
        outer = self.box([0.0, 0.0], [1.0, 1.0])
        self.assertEqual(L.inclusion_loss(*inner, *outer, 0.0).item(), 0.0)
        self.assertGreater(L.inclusion_loss(*outer, *inner, 0.0).item(), 0.0)
        # touching the boundary is still inclusion at margin 0, not at margin 0.1
        edge = self.box([0.0, 0.0], [0.5, 0.5])
        self.assertEqual(L.inclusion_loss(*edge, *outer, 0.0).item(), 0.0)
        self.assertGreater(L.inclusion_loss(*edge, *outer, 0.1).item(), 0.0)
        # value: one dimension sticks out by 0.5, mean over n = 2 dimensions
        out = self.box([0.0, 0.0], [1.5, 1.0])
        self.assertAlmostEqual(L.inclusion_loss(*out, *outer, 0.0).item(), 0.25)

    def test_inclusion_of_empty_box_is_zero(self):
        empty = self.box([1.0, 0.0], [0.0, 1.0])
        far = self.box([5.0, 5.0], [6.0, 6.0])
        self.assertEqual(L.inclusion_loss(*empty, *far, 0.1).item(), 0.0)

    def test_overlap(self):
        a = self.box([0.0, 0.0], [1.0, 1.0])
        b = self.box([0.5, 0.5], [2.0, 2.0])
        c = self.box([1.5, 0.0], [2.0, 1.0])
        self.assertEqual(L.overlap_loss(*a, *b, 0.0).item(), 0.0)
        self.assertGreater(L.overlap_loss(*a, *c, 0.0).item(), 0.0)

    def test_disjointness(self):
        a = self.box([0.0, 0.0], [1.0, 1.0])
        b = self.box([0.5, 0.5], [2.0, 2.0])
        c = self.box([1.5, 0.0], [2.0, 1.0])
        self.assertGreater(L.disjoint_loss(*a, *b, 0.1).item(), 0.0)
        self.assertEqual(L.disjoint_loss(*a, *c, 0.1).item(), 0.0)
        # separated by 0.5 in one dimension only: zero up to margin 0.5
        self.assertEqual(L.disjoint_loss(*a, *c, 0.5).item(), 0.0)
        self.assertGreater(L.disjoint_loss(*a, *c, 0.6).item(), 0.0)

    def test_pearson(self):
        good = L.pearson_contrastive_loss(th.tensor([0.9, 1.0]), th.tensor([0.1, 0.0, 0.2]))
        bad = L.pearson_contrastive_loss(th.tensor([0.1, 0.0]), th.tensor([0.9, 1.0, 0.8]))
        self.assertLess(good.item(), 0.1)
        self.assertGreater(bad.item(), 1.9)


class TestExistentialBox(TestCase):
    """(∃r.C)^E = T_r(C ∩ R_r) ⊕ E_r against the relational definition."""

    def test_box_matches_relational_definition(self):
        gen = th.Generator().manual_seed(0)
        n = 3
        for trial in range(40):
            role = random_role(n, gen)
            cl, cu = random_box(n, gen)
            el, eu = (v[0] for v in L.existential_box(batched(role), cl[None], cu[None]))
            il, iu = th.maximum(cl, role[0]), th.minimum(cu, role[1])
            if (il > iu).any():
                # C ∩ R_r empty: no point has a witness, and the box is empty
                self.assertTrue(bool((el > eu).any()))
                for x in sample_box(el.clamp(max=eu) - 1, eu + 1, 50, gen):
                    self.assertFalse(has_witness(x, role, cl, cu))
                continue
            # every r-predecessor of a point of C ∩ R_r lies in the box
            for y in corners_and_samples(il, iu, 20, gen):
                for x in corners_and_samples(role[4] * y + role[5] + role[2],
                                             role[4] * y + role[5] + role[3], 3, gen):
                    self.assertTrue(in_box(x, el, eu))
            # every point of the box has a witness
            for x in corners_and_samples(el, eu, 100, gen):
                self.assertTrue(has_witness(x, role, cl, cu))
            # points just outside the box have none
            for d in range(n):
                for side, delta in ((eu, 1e-3), (el, -1e-3)):
                    x = (el + eu) / 2
                    x[d] = side[d] + delta
                    self.assertFalse(has_witness(x, role, cl, cu))


class TestRoleInclusion(TestCase):
    """Proposition 2: r^E ⊆ s^E iff the RI0 margin-free loss is zero."""

    def test_proposition_2_by_sampling(self):
        gen = th.Generator().manual_seed(1)
        n = 2
        seen = {True: 0, False: 0}
        for trial in range(300):
            r = random_role(n, gen)
            if trial % 2:
                # s obtained by enlarging r slightly, so that inclusion often holds
                rl, ru, el, eu, m, c = r
                grow = th.rand(n, generator=gen, dtype=th.float64) * 0.3
                s = (rl - grow, ru + grow, el - grow, eu + grow,
                     m + (th.rand(n, generator=gen, dtype=th.float64) - 0.5) * 0.1,
                     c + (th.rand(n, generator=gen, dtype=th.float64) - 0.5) * 0.1)
            else:
                s = random_role(n, gen)
            loss_zero = L.role_inclusion_loss(batched(r), batched(s), 0.0).item() == 0.0
            contained = all(in_role(x, y, s) for x, y in sample_role_pairs(r, 30, gen))
            self.assertEqual(loss_zero, contained, f"trial {trial}")
            seen[contained] += 1
        self.assertGreater(seen[True], 20)
        self.assertGreater(seen[False], 20)


def sample_composite(r1, r2, k, gen):
    """Pairs (x, z) of r1^E ∘ r2^E with their intermediate y."""
    out = []
    rl2, ru2, el2, eu2, m2, c2 = r2
    rl1, ru1, el1, eu1, m1, c1 = r1
    for z in corners_and_samples(rl2, ru2, k, gen):
        lo = th.maximum(m2 * z + c2 + el2, rl1)
        hi = th.minimum(m2 * z + c2 + eu2, ru1)
        if (lo > hi).any():
            continue
        for y in corners_and_samples(lo, hi, 3, gen):
            for x in corners_and_samples(m1 * y + c1 + el1, m1 * y + c1 + eu1, 2, gen):
                out.append((x, z))
    return out


def in_composite(x, z, r1, r2, tol=1e-6):
    """∃y: (x, y) ∈ r1^E and (y, z) ∈ r2^E (per-dimension interval check)."""
    rl2, ru2, el2, eu2, m2, c2 = r2
    rl1, ru1, el1, eu1, m1, c1 = r1
    if not in_box(z, rl2, ru2, tol):
        return False
    lo = th.maximum(th.maximum(m2 * z + c2 + el2, rl1), (x - c1 - eu1) / m1)
    hi = th.minimum(th.minimum(m2 * z + c2 + eu2, ru1), (x - c1 - el1) / m1)
    return bool((lo <= hi + tol).all())


class TestRoleComposition(TestCase):
    """Propositions 4 and 5 of the paper, by sampling."""

    def test_proposition_4_over_approximation_and_equality(self):
        gen = th.Generator().manual_seed(2)
        n = 2
        compatible_seen = incompatible_strict = 0
        for trial in range(200):
            r1, r2 = random_role(n, gen), random_role(n, gen)
            if trial % 2:
                # make the compatibility condition T_r2(R_r2) ⊕ E_r2 ⊆ R_r1 hold
                dl = r2[4] * r2[0] + r2[5] + r2[2]
                du = r2[4] * r2[1] + r2[5] + r2[3]
                r1 = (dl - 0.1, du + 0.1) + r1[2:]
            mx = L.compose_roles(r1, r2)
            # (1) r1 ∘ r2 ⊆ max(r1 ∘ r2)
            for x, z in sample_composite(r1, r2, 20, gen):
                self.assertTrue(in_role(x, z, mx))
            dl = r2[4] * r2[0] + r2[5] + r2[2]
            du = r2[4] * r2[1] + r2[5] + r2[3]
            compatible = bool(((r1[0] <= dl) & (du <= r1[1])).all())
            max_pairs = sample_role_pairs(mx, 30, gen)
            all_in = all(in_composite(x, z, r1, r2) for x, z in max_pairs)
            if compatible:
                # (2) equality under the compatibility condition
                self.assertTrue(all_in)
                compatible_seen += 1
            elif not all_in:
                incompatible_strict += 1
        self.assertGreater(compatible_seen, 20)
        self.assertGreater(incompatible_strict, 20)

    def test_proposition_5_chain_condition_is_sufficient(self):
        gen = th.Generator().manual_seed(3)
        n = 2
        checked = 0
        for trial in range(200):
            r1, r2 = random_role(n, gen), random_role(n, gen)
            rl, ru, el, eu, m, c = L.compose_roles(r1, r2)
            grow = th.rand(n, generator=gen, dtype=th.float64) * 0.2
            s = (rl - grow, ru + grow, el - grow, eu + grow, m, c) if trial % 2 \
                else random_role(n, gen)
            if L.role_chain_loss(batched(r1), batched(r2), batched(s), 0.0).item() != 0.0:
                continue
            checked += 1
            for x, z in sample_composite(r1, r2, 20, gen):
                self.assertTrue(in_role(x, z, s))
        self.assertGreater(checked, 50)


class TestModuleLosses(TestCase):
    """Per-axiom losses of Table 2 on hand-built boxes, with the column orders of mOWL's
    ELDataset."""

    def setUp(self):
        th.manual_seed(0)
        # classes 0..5, 6 = owl:Nothing, 7 = owl:Thing
        self.m = ELKEmModule(8, 3, nb_inds=3, embed_dim=2, margin=0.05,
                             bot_ids=[6], top_ids=[7])
        self.m.train()
        set_class(self.m, 0, [0.0, 0.0], [1.0, 1.0])
        set_class(self.m, 1, [0.2, 0.2], [0.4, 0.4])
        set_class(self.m, 2, [0.5, 0.5], [2.0, 2.0])
        set_class(self.m, 3, [3.0, 3.0], [4.0, 4.0])
        set_class(self.m, 4, [-1.0, -1.0], [5.0, 5.0])
        set_class(self.m, 5, [0.6, 0.6], [0.9, 0.9])
        # role 0: identity map, range [0, 1]^2, error ±0.2
        set_role(self.m, 0, [0.0, 0.0], [1.0, 1.0], [0.2, 0.2], [1.0, 1.0], [0.0, 0.0])

    def test_flags(self):
        self.assertTrue(ELKEmModule.role_axiom_capable)
        self.assertEqual(ELKEmModule.neg_capable_gcis,
                         frozenset({"class_assertion", "object_property_assertion"}))

    def test_gci0_column_order(self):
        self.assertEqual(self.m.gci0_loss(t([1, 0])).item(), 0.0)  # B_1 ⊆ B_0
        self.assertGreater(self.m.gci0_loss(t([0, 1])).item(), 0.0)

    def test_gci1(self):
        # B_0 ∩ B_2 = [0.5, 1]^2 ⊆ B_4 and the two overlap
        self.assertEqual(self.m.gci1_loss(t([0, 2, 4])).item(), 0.0)
        # B_0 ∩ B_2 ⊄ B_5
        self.assertGreater(self.m.gci1_loss(t([0, 2, 5])).item(), 0.0)
        # disjoint conjuncts: the inclusion holds trivially but the overlap term is positive
        self.assertGreater(self.m.gci1_loss(t([0, 3, 5])).item(), 0.0)

    def test_gci2_column_order(self):
        # rows are [C, r, D]: ∃r.B_5 = [0.6, 0.9] ⊕ [-0.2, 0.2] = [0.4, 1.1]^2 ⊇ [0.5, 1]^2
        set_class(self.m, 1, [0.5, 0.5], [1.0, 1.0])
        self.assertEqual(self.m.gci2_loss(t([1, 0, 5])).item(), 0.0)
        self.assertGreater(self.m.gci2_loss(t([3, 0, 5])).item(), 0.0)
        # B_3 ∩ R_r = ∅, so ∃r.B_3 is empty, even though the gap between them is narrower
        # than E_r: C ⊑ ∃r.B_3 must fail for every non-empty C
        set_class(self.m, 3, [1.3, 1.3], [2.0, 2.0])   # gap 0.3 < width 0.4 of E_r
        self.assertGreater(self.m.gci2_loss(t([5, 0, 3])).item(), 0.0)
        self.m.eval()
        self.assertGreater(self.m.gci2_loss(t([5, 0, 3])).item(), 0.0)

    def test_gci3_column_order(self):
        # ∃r.B_5 = [0.4, 1.1]^2 ⊆ B_4, R_r overlaps B_5
        self.assertEqual(self.m.gci3_loss(t([0, 5, 4])).item(), 0.0)
        self.assertGreater(self.m.gci3_loss(t([0, 5, 1])).item(), 0.0)

    def test_gci3_bot_does_not_force_filler_empty(self):
        """∃r.C ⊑ ⊥ is satisfied by R_r ∩ B_C = ∅ with a non-empty B_C."""
        self.assertEqual(self.m.gci3_bot_loss(t([0, 3, 6])).item(), 0.0)
        lower, upper = self.m.class_box(t(3))
        self.assertFalse(bool((lower > upper).any()))
        el, eu = self.m.existential_box(t(0), t(3))
        self.assertTrue(bool((el > eu).any()))
        # a filler meeting the range violates it
        self.assertGreater(self.m.gci3_bot_loss(t([0, 5, 6])).item(), 0.0)

    def test_gci1_bot(self):
        self.assertEqual(self.m.gci1_bot_loss(t([0, 3, 6])).item(), 0.0)
        self.assertGreater(self.m.gci1_bot_loss(t([0, 2, 6])).item(), 0.0)

    def test_top_is_full_box(self):
        lower, upper = self.m.class_box(t(7))
        self.assertTrue(bool((lower <= -1e5).all() and (upper >= 1e5).all()))
        self.assertEqual(self.m.gci0_loss(t([3, 7])).item(), 0.0)      # C ⊑ ⊤
        self.assertGreater(self.m.gci0_loss(t([7, 4])).item(), 1e3)    # ⊤ ⋢ bounded box
        # ∃r.⊤ = T_r(R_r) ⊕ E_r = [-0.2, 1.2]^2
        el, eu = self.m.existential_box(t(0), t(7))
        self.assertTrue(th.allclose(el, th.tensor([-0.2, -0.2])))
        self.assertTrue(th.allclose(eu, th.tensor([1.2, 1.2])))
        self.assertEqual(self.m.gci3_loss(t([0, 7, 4])).item(), 0.0)

    def test_bottom_is_empty_box(self):
        lower, upper = self.m.class_box(t(6))
        self.assertTrue(bool((lower > upper).all()))
        self.assertEqual(self.m.gci0_loss(t([6, 3])).item(), 0.0)      # ⊥ ⊑ C
        self.assertEqual(self.m.gci0_bot_loss(t([6, 6])).item(), 0.0)
        # a class not fixed empty cannot satisfy C ⊑ ⊥
        self.assertGreater(self.m.gci0_bot_loss(t([3, 6])).item(), 0.0)
        # ∃r.⊥ is empty, so ∃r.⊥ ⊑ D holds; the overlap term is dropped for it
        self.assertEqual(self.m.gci3_loss(t([0, 6, 3])).item(), 0.0)
        # C ⊓ ⊥ ⊑ D holds
        self.assertEqual(self.m.gci1_loss(t([0, 6, 3])).item(), 0.0)
        # C ⊑ ∃r.⊥ fails for non-empty C
        self.assertGreater(self.m.gci2_loss(t([0, 0, 6])).item(), 0.0)

    def test_fixing_a_class_empty(self):
        self.m.set_fixed_classes(bot_ids=[6, 3], top_ids=[7])
        self.assertEqual(self.m.gci0_bot_loss(t([3, 6])).item(), 0.0)
        self.assertEqual(self.m.gci0_loss(t([3, 1])).item(), 0.0)
        with self.assertRaises(ValueError):
            self.m.set_fixed_classes(bot_ids=[7], top_ids=[7])

    def test_role_axioms_column_order(self):
        set_role(self.m, 1, [-1.0, -1.0], [2.0, 2.0], [0.5, 0.5], [1.0, 1.0], [0.0, 0.0])
        self.assertEqual(self.m.role_inclusion_loss(t([0, 1])).item(), 0.0)
        self.assertGreater(self.m.role_inclusion_loss(t([1, 0])).item(), 0.0)
        # r0 ∘ r0: range [0,1], error ±0.4, identity map ⊆ r1 but not ⊆ r0
        self.assertEqual(self.m.role_chain_loss(t([0, 0, 1])).item(), 0.0)
        self.assertGreater(self.m.role_chain_loss(t([0, 0, 0])).item(), 0.0)

    def test_assertions(self):
        self.m.ind_embedding.weight.data[0] = th.tensor([0.3, 0.3])
        self.m.ind_embedding.weight.data[1] = th.tensor([0.5, 0.5])
        self.m.ind_embedding.weight.data[2] = th.tensor([3.0, 3.0])
        # class assertion rows are [individual, class]
        self.assertEqual(self.m.class_assertion_loss(t([0, 1])).item(), 0.0)
        self.assertGreater(self.m.class_assertion_loss(t([2, 1])).item(), 0.0)
        # negatives: ReLU(-log τ - ℓ)
        self.assertAlmostEqual(self.m.class_assertion_loss(t([0, 1]), neg=True).item(),
                               -th.log(th.tensor(0.4)).item(), places=5)
        self.assertEqual(self.m.class_assertion_loss(t([2, 0]), neg=True).item(), 0.0)
        # role assertion rows are [subject, property, object]; r0(a0, a1): a1 ∈ R_r,
        # a0 ∈ a1 ± 0.2
        self.m.ind_embedding.weight.data[0] = th.tensor([0.45, 0.45])
        self.assertEqual(self.m.object_property_assertion_loss(t([0, 0, 1])).item(), 0.0)
        self.assertGreater(self.m.object_property_assertion_loss(t([2, 0, 1])).item(), 0.0)
        self.assertGreater(self.m.object_property_assertion_loss(t([0, 0, 2])).item(), 0.0)

    def test_margin_only_in_training_mode(self):
        # B_1 ⊆ B_0 with distance 0.2 to the boundary: zero with margin 0.1 and 0.
        set_class(self.m, 1, [0.0, 0.0], [0.5, 0.5])   # touches the boundary of B_0
        self.m.train()
        self.assertGreater(self.m.gci0_loss(t([1, 0])).item(), 0.0)
        self.m.eval()
        self.assertEqual(self.m.gci0_loss(t([1, 0])).item(), 0.0)
        self.assertEqual(self.m.score(t([1, 0]), "gci0").item(), 1.0)
        self.m.train()
        s = self.m.score(t([0, 1]), "gci0").item()
        self.assertAlmostEqual(s, th.exp(-self.m.zero_margin_loss(t([0, 1]), "gci0")).item())
        self.assertTrue(0 < s < 1)

    def test_lexical_regulariser_decreases(self):
        th.manual_seed(1)
        module = ELKEmModule(4, 1, nb_inds=40, embed_dim=8)
        z = th.randn(40, 16)
        idx = th.arange(40)
        opt = th.optim.Adam([module.ind_embedding.weight], lr=0.05)
        start = module.lexical_loss(idx, z).item()
        for _ in range(200):
            opt.zero_grad()
            loss = module.lexical_loss(idx, z)
            loss.backward()
            opt.step()
        self.assertLess(module.lexical_loss(idx, z).item(), 0.5 * start)

    def test_lexical_loss_value(self):
        # identical directions and identical points: s_ext = s_geo = 1 -> zero loss
        points = th.zeros(3, 2)
        feats = th.ones(3, 4)
        self.assertEqual(L.lexical_loss(points, feats, 1.0).item(), 0.0)
        # opposite directions (s_ext = 0) and identical points (s_geo = 1) -> loss 1
        feats = th.tensor([[1.0, 0.0], [-1.0, 0.0]])
        self.assertAlmostEqual(L.lexical_loss(th.zeros(2, 2), feats, 1.0).item(), 1.0)


class TestTrivialDisjointnessAndShape(TestCase):

    def build(self, variant="paper"):
        th.manual_seed(0)
        m = ELKEmModule(6, 1, 2, embed_dim=3, margin=0.1, bot_ids=[0], top_ids=[1],
                        variant=variant)
        m.train()
        return m

    def test_bottom_and_top_disjointness_is_free(self):
        rows = th.tensor([[0, 1], [1, 0], [0, 2], [2, 3]])
        m = self.build()
        loss = m.gci1_bot_loss(rows).reshape(-1)
        self.assertTrue(th.equal(loss[:3], th.zeros(3)))
        self.assertGreaterEqual(float(loss[3]), 0.0)
        gb = th.tensor([[0, 0], [0, 2]])
        self.assertTrue(th.equal(m.gci3_bot_loss(gb).reshape(-1)[:1], th.zeros(1)))
        # the reference variant keeps the primitive's value (the margin)
        ref = self.build("reference")
        self.assertAlmostEqual(float(ref.gci1_bot_loss(rows)[0]), 0.1, places=5)
        self.assertAlmostEqual(float(ref.gci1_bot_loss(rows)[1]), 0.1, places=5)

    def test_zero_margin_loss_is_a_column(self):
        m = self.build()
        rows = th.tensor([[2, 3], [3, 4]])
        for name in ("gci0", "gci1_bot"):
            self.assertEqual(m.zero_margin_loss(rows, name).shape, (2, 1))
        m.per_axiom_column = False
        self.assertEqual(m.zero_margin_loss(rows, "gci0").shape, (2,))
