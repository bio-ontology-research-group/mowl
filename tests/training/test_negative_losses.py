"""DELE negative losses (Mashkova et al., DELE, Section 4): each is zero exactly when the
geometric condition the paper gives for "the axiom is not satisfied" holds, and the
modules' default behaviour is unchanged."""
from unittest import TestCase

import numpy as np
import torch as th

from mowl.nn import ELEmModule, ELBEModule, BoxSquaredELModule

N, R, ROWS = 12, 3, 4000
EPS = 0.3


def _rows(columns):
    """Random index rows; ``columns`` lists 'c' (class) or 'r' (role) per column."""
    g = th.Generator().manual_seed(0)
    cols = [th.randint(0, N if c == "c" else R, (ROWS,), generator=g) for c in columns]
    return th.stack(cols, dim=1)


def _randomise(module):
    g = th.Generator().manual_seed(1)
    for name, p in module.named_parameters():
        if "offset" in name or "rad" in name:
            p.data = th.rand(p.shape, generator=g) * 0.6
        else:
            p.data = (th.rand(p.shape, generator=g) * 2 - 1) * 2


def _check_iff(test, loss, condition):
    loss = loss.detach().flatten().numpy()
    condition = np.asarray(condition).flatten()
    test.assertEqual(loss.shape, condition.shape)
    test.assertTrue((loss >= 0).all())
    zero = loss <= 1e-6
    test.assertTrue(zero.any() and (~zero).any(), "degenerate sample, adjust the test")
    np.testing.assert_array_equal(zero, condition)


class TestDefaultsUnchanged(TestCase):

    def test_neg_capable_gcis_unchanged_by_default(self):
        for m in (ELEmModule(N, R, 0), ELBEModule(N, R), BoxSquaredELModule(N, R)):
            self.assertEqual(m.neg_capable_gcis, frozenset({"gci2", "object_property_assertion"}))
            self.assertFalse(m.all_negative_losses)

    def test_neg_flag_ignored_by_default(self):
        th.manual_seed(0)
        for m in (ELEmModule(N, R, 0), ELBEModule(N, R), BoxSquaredELModule(N, R)):
            for name, cols in (("gci0", "cc"), ("gci1", "ccc"), ("gci3", "rcc")):
                data = _rows(cols)[:20]
                self.assertTrue(th.equal(m(data, name), m(data, name, neg=True)))

    def test_enabling_is_truthful(self):
        m = ELEmModule(N, R, 0, all_negative_losses=True, epsilon=0.2)
        dele = {"gci0", "gci1", "gci3", "gci0_bot", "gci1_bot", "gci3_bot"}
        self.assertTrue(dele <= m.neg_capable_gcis)
        self.assertEqual(m.epsilon, 0.2)
        # the class attribute is untouched
        self.assertEqual(ELEmModule.neg_capable_gcis,
                         frozenset({"gci2", "object_property_assertion"}))
        data = _rows("cc")[:20]
        self.assertFalse(th.equal(m(data, "gci0"), m(data, "gci0", neg=True)))


class TestELEmNegativeLosses(TestCase):

    def setUp(self):
        self.m = ELEmModule(N, R, 0, embed_dim=2, margin=0.1, all_negative_losses=True,
                            epsilon=EPS)
        _randomise(self.m)
        self.c = self.m.class_embed.weight.detach().numpy()
        self.r = np.abs(self.m.class_rad.weight.detach().numpy()[:, 0])
        self.rel = self.m.rel_embed.weight.detach().numpy()
        self.g = self.m.margin

    def _d(self, x, y):
        return np.linalg.norm(x - y, axis=1)

    def test_gci0(self):  # Eq. 1: balls separated by the margin
        d = _rows("cc").numpy()
        a, b = d[:, 0], d[:, 1]
        cond = self._d(self.c[a], self.c[b]) >= self.r[a] + self.r[b] + self.g
        _check_iff(self, self.m(th.tensor(d), "gci0", neg=True), cond)

    def test_gci1(self):  # Eq. 3: A, B overlap and centre of E outside both balls
        d = _rows("ccc").numpy()
        a, b, e = d[:, 0], d[:, 1], d[:, 2]
        overlap = self._d(self.c[a], self.c[b]) <= self.r[a] + self.r[b] + self.g
        out_a = self._d(self.c[a], self.c[e]) >= self.r[a] + self.g
        out_b = self._d(self.c[b], self.c[e]) >= self.r[b] + self.g
        cond = overlap & out_a & out_b
        _check_iff(self, self.m(th.tensor(d), "gci1", neg=True), cond)

    def test_gci3(self):  # Eq. 2: translated ball of A separated from ball of B
        d = _rows("rcc").numpy()
        r, a, b = d[:, 0], d[:, 1], d[:, 2]
        cond = self._d(self.c[a] - self.rel[r], self.c[b]) >= \
            self.r[a] + self.r[b] + self.g
        _check_iff(self, self.m(th.tensor(d), "gci3", neg=True), cond)

    def test_gci1_bot(self):  # Eq. 4: balls overlap
        d = _rows("ccc").numpy()
        a, b = d[:, 0], d[:, 1]
        cond = self._d(self.c[a], self.c[b]) <= self.r[a] + self.r[b] + self.g
        _check_iff(self, self.m(th.tensor(d), "gci1_bot", neg=True), cond)

    def test_bot_unary(self):  # Eqs. 5, 6: radius at least epsilon
        d = _rows("cc").numpy()
        _check_iff(self, self.m(th.tensor(d), "gci0_bot", neg=True), self.r[d[:, 0]] >= EPS)
        d = _rows("rcc").numpy()
        _check_iff(self, self.m(th.tensor(d), "gci3_bot", neg=True), self.r[d[:, 1]] >= EPS)


def _box_inter(c1, o1, c2, o2):
    lo = np.maximum(c1 - o1, c2 - o2)
    up = np.minimum(c1 + o1, c2 + o2)
    return (lo + up) / 2, np.abs(up - lo) / 2


class TestELBENegativeLosses(TestCase):

    def setUp(self):
        self.m = ELBEModule(N, R, embed_dim=2, margin=0.05, all_negative_losses=True,
                            epsilon=EPS)
        _randomise(self.m)
        self.c = self.m.class_embed.weight.detach().numpy()
        self.o = np.abs(self.m.class_offset.weight.detach().numpy())
        self.rel = self.m.rel_embed.weight.detach().numpy()
        self.g = self.m.margin

    def _sep(self, c1, o1, c2, o2):
        """Eq. 7 zero set: separated by the margin in every dimension."""
        return (np.abs(c1 - c2) >= o1 + o2 + self.g).all(axis=1)

    def test_gci0(self):
        d = _rows("cc").numpy()
        a, b = d[:, 0], d[:, 1]
        _check_iff(self, self.m(th.tensor(d), "gci0", neg=True),
                   self._sep(self.c[a], self.o[a], self.c[b], self.o[b]))

    def test_gci1(self):  # Eq. 8 plus the reference code's non-empty-intersection term
        d = _rows("ccc").numpy()
        a, b, e = d[:, 0], d[:, 1], d[:, 2]
        cn, on = _box_inter(self.c[a], self.o[a], self.c[b], self.o[b])
        lo = np.maximum(self.c[a] - self.o[a], self.c[b] - self.o[b])
        up = np.minimum(self.c[a] + self.o[a], self.c[b] + self.o[b])
        _check_iff(self, self.m(th.tensor(d), "gci1", neg=True),
                   self._sep(cn, on, self.c[e], self.o[e]) & (lo <= up).all(axis=1))

    def test_gci3(self):  # Eq. 10
        d = _rows("rcc").numpy()
        r, a, b = d[:, 0], d[:, 1], d[:, 2]
        _check_iff(self, self.m(th.tensor(d), "gci3", neg=True),
                   self._sep(self.c[a] - self.rel[r], self.o[a], self.c[b], self.o[b]))

    def test_gci1_bot(self):  # Eq. 9
        d = _rows("ccc").numpy()
        a, b = d[:, 0], d[:, 1]
        _, on = _box_inter(self.c[a], self.o[a], self.c[b], self.o[b])
        _check_iff(self, self.m(th.tensor(d), "gci1_bot", neg=True),
                   np.linalg.norm(on, axis=1) >= EPS)

    def test_bot_unary(self):  # Eqs. 11, 12
        d = _rows("cc").numpy()
        _check_iff(self, self.m(th.tensor(d), "gci0_bot", neg=True),
                   np.linalg.norm(self.o[d[:, 0]], axis=1) >= EPS)
        d = _rows("rcc").numpy()
        _check_iff(self, self.m(th.tensor(d), "gci3_bot", neg=True),
                   np.linalg.norm(self.o[d[:, 1]], axis=1) >= EPS)


class TestBoxSquaredELNegativeLosses(TestCase):

    def setUp(self):
        self.m = BoxSquaredELModule(N, R, embed_dim=2, gamma=0.05, delta=1.5,
                                    all_negative_losses=True,
                                    epsilon=EPS)
        _randomise(self.m)
        self.c = self.m.class_center.weight.detach().numpy()
        self.o = np.abs(self.m.class_offset.weight.detach().numpy())
        self.g = self.m.gamma

    def _apart(self, c1, o1, c2, o2):
        """Eq. 13 zero set: d(X, Y) = |c1 - c2| - o1 - o2 >= -γ in every dimension."""
        return (np.abs(c1 - c2) - o1 - o2 >= -self.g).all(axis=1)

    def test_gci0(self):
        d = _rows("cc").numpy()
        a, b = d[:, 0], d[:, 1]
        _check_iff(self, self.m(th.tensor(d), "gci0", neg=True),
                   self._apart(self.c[a], self.o[a], self.c[b], self.o[b]))

    def test_gci1(self):  # Eq. 14 plus the reference code's non-empty-intersection term
        d = _rows("ccc").numpy()
        a, b, e = d[:, 0], d[:, 1], d[:, 2]
        cn, on = _box_inter(self.c[a], self.o[a], self.c[b], self.o[b])
        lo = np.maximum(self.c[a] - self.o[a], self.c[b] - self.o[b])
        up = np.minimum(self.c[a] + self.o[a], self.c[b] + self.o[b])
        _check_iff(self, self.m(th.tensor(d), "gci1", neg=True),
                   self._apart(cn, on, self.c[e], self.o[e]) & (lo <= up).all(axis=1))

    def test_gci1_bot(self):  # Eq. 16
        d = _rows("ccc").numpy()
        a, b = d[:, 0], d[:, 1]
        _, on = _box_inter(self.c[a], self.o[a], self.c[b], self.o[b])
        _check_iff(self, self.m(th.tensor(d), "gci1_bot", neg=True),
                   np.linalg.norm(on, axis=1) >= EPS)

    def test_bot_unary(self):  # Eqs. 15, 17
        d = _rows("cc").numpy()
        _check_iff(self, self.m(th.tensor(d), "gci0_bot", neg=True),
                   np.linalg.norm(self.o[d[:, 0]], axis=1) >= EPS)
        d = _rows("rcc").numpy()
        _check_iff(self, self.m(th.tensor(d), "gci3_bot", neg=True),
                   np.linalg.norm(self.o[d[:, 1]], axis=1) >= EPS)

    def test_gci3(self):  # Eq. 18: zero iff the minimal distance equals delta
        d = th.tensor([[0, 1, 2]])
        m = self.m
        with th.no_grad():
            m.head_center.weight[0] = 0.0
            m.head_offset.weight[0] = 0.5
            m.bump_classes.weight[1] = 0.0
            m.class_offset.weight[2] = 0.5
            for x, expect_zero in ((0.0, False), (2.5 - m.gamma, True)):
                m.class_center.weight[2] = 0.0
                m.class_center.weight[2, 0] = x
                # with one separated dimension: µ = relu(|x| - 1 + γ) = 1.5 = δ for x = 2.5 - γ
                loss = float(m(d, "gci3", neg=True))
                self.assertEqual(loss <= 1e-9, expect_zero, (x, loss))
