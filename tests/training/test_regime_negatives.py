"""Negative sampling regimes: which columns are corrupted, from which pools, how many, and
what the filters reject."""
from unittest import TestCase

import numpy as np
import torch as th

from mowl.training import ComposedRegime
from mowl.training.negatives import (FixedColumnCorruption, EitherSideCorruption,
                                     PerColumnCorruption, CorruptionNegatives,
                                     TypeConstrainedCorruption, KnownPositiveFilter,
                                     SiblingHardNegatives, ContradictionNegatives,
                                     corrupted_columns)
from mowl.training.positives import _TensorDataset
from tests.training.stubs import StubModel


GCI0 = [[0, 1], [1, 2], [2, 3], [4, 3], [5, 3]]
GCI2 = [[0, 0, 1], [1, 1, 2], [2, 2, 3]]


def _model(**kw):
    return StubModel({"gci0": GCI0, "gci2": GCI2}, **kw)


def _ds(rows):
    return _TensorDataset(th.as_tensor(rows, dtype=th.long))


class TestCorruption(TestCase):

    def test_fixed_corrupts_only_right_hand_side(self):
        r = FixedColumnCorruption(num_negatives=3, seed=0)
        r.attach(_model())
        pos = _ds(GCI2)
        neg = r.negatives("gci2", pos)
        self.assertEqual(neg.shape, (9, 3))
        cols = corrupted_columns(pos[:], neg)
        self.assertTrue(all(set(c) <= {2} for c in cols))
        # the head and relation columns are those of positive i mod n
        self.assertTrue(th.equal(neg[:, :2], pos[:].repeat(3, 1)[:, :2]))

    def test_defaults_restricted_to_negative_capable_forms(self):
        r = FixedColumnCorruption()
        r.attach(_model(neg_capable=("gci2",)))
        self.assertEqual(set(r.columns), {"gci2"})
        self.assertIs(r.negatives("gci0", _ds(GCI0)), NotImplemented)

    def test_explicit_column_without_negative_loss_is_refused(self):
        r = FixedColumnCorruption(columns={"gci0": 1})
        with self.assertRaises(TypeError):
            r.attach(_model(neg_capable=("gci2",)))
        r = FixedColumnCorruption(columns={"gci0": 1}, require_negative_loss=False)
        r.attach(_model(neg_capable=("gci2",)))

    def test_either_side_one_column_per_row_both_used(self):
        r = EitherSideCorruption(num_negatives=200, seed=1, different=True)
        r.attach(_model())
        pos = _ds(GCI2)
        neg = r.negatives("gci2", pos)
        cols = corrupted_columns(pos[:], neg)
        self.assertTrue(all(len(c) == 1 and c[0] in (0, 2) for c in cols))
        used = {c[0] for c in cols}
        self.assertEqual(used, {0, 2})

    def test_per_column_blocks(self):
        r = PerColumnCorruption(num_negatives=2, seed=0, different=True)
        r.attach(_model())
        pos = _ds(GCI2)
        neg = r.negatives("gci2", pos)
        self.assertEqual(len(neg), 2 * 2 * len(GCI2))
        cols = corrupted_columns(pos[:], neg)
        n = len(GCI2)
        # blocks: col 0, col 0, col 2, col 2
        for b, col in enumerate([0, 0, 2, 2]):
            for c in cols[b * n:(b + 1) * n]:
                self.assertEqual(c, [col])

    def test_all_mode_corrupts_both(self):
        r = CorruptionNegatives(columns={"gci2": (0, 2)}, mode="all", different=True,
                                seed=0)
        r.attach(_model())
        pos = _ds(GCI2)
        cols = corrupted_columns(pos[:], r.negatives("gci2", pos))
        self.assertTrue(all(c == [0, 2] for c in cols))

    def test_pools(self):
        r = FixedColumnCorruption(columns={"gci2": 2}, pools={"gci2": {2: [7, 8]}},
                                  num_negatives=50, seed=0)
        r.attach(_model())
        neg = r.negatives("gci2", _ds(GCI2))
        self.assertTrue(set(neg[:, 2].tolist()) <= {7, 8})
        # relation column of gci2 would be drawn from roles, never from classes
        r = FixedColumnCorruption(columns={"gci2": 1}, num_negatives=50, seed=0)
        r.attach(_model())
        self.assertTrue(set(r.negatives("gci2", _ds(GCI2))[:, 1].tolist()) <= {0, 1, 2})

    def test_type_constrained(self):
        proteins, functions = list(range(0, 5)), list(range(10, 20))
        r = TypeConstrainedCorruption(types=[proteins, functions],
                                      columns={"gci2": (0, 2)}, mode="each",
                                      num_negatives=20, seed=0)
        r.attach(_model())
        pos = _ds([[1, 0, 12], [3, 1, 15]])
        neg = r.negatives("gci2", pos)
        n = 2
        heads = neg[:20 * n, 0].tolist()
        tails = neg[20 * n:, 2].tolist()
        self.assertTrue(set(heads) <= set(proteins))
        self.assertTrue(set(tails) <= set(functions))

    def _train(self, model_cls, regime, **kw):
        from tests.datasetFactory import FamilyDataset
        th.manual_seed(0)
        model = model_cls(FamilyDataset(), embed_dim=4, batch_size=64, **kw)
        if regime is not None:
            model.training_regime = regime
        np.random.seed(0)
        model.train(epochs=3, validate_every=100)
        return [p.detach().clone() for p in model.module.parameters()]

    def test_reproduces_default_sampler_numerically(self):
        """FixedColumnCorruption() on ELEmbeddings gives exactly the parameters of the
        model's own sampler (same NumPy draws)."""
        from mowl.models import ELEmbeddings
        a = self._train(ELEmbeddings, None)
        b = self._train(ELEmbeddings, FixedColumnCorruption())
        for x, y in zip(a, b):
            self.assertTrue(th.equal(x, y))

    def test_reproduces_transbox_two_sided_negatives(self):
        """PerColumnCorruption on gci2 (0, 2) reproduces TransBox's built-in sampler."""
        from mowl.models import TransBox
        a = self._train(TransBox, None)
        b = self._train(TransBox, PerColumnCorruption(columns={"gci2": (0, 2)}))
        for x, y in zip(a, b):
            self.assertTrue(th.equal(x, y))
        # and differs from the one-sided sampler
        c = self._train(TransBox, FixedColumnCorruption(columns={"gci2": 2}))
        self.assertFalse(all(th.equal(x, y) for x, y in zip(a, c)))


class TestFilters(TestCase):

    def test_known_positive_filter_resamples(self):
        m = _model()
        f = KnownPositiveFilter(pools={"gci0": {1: [1, 2, 3, 9]}}, seed=0)
        f.attach(m)
        pos = m.training_datasets["gci0"]
        neg = th.tensor([[0, 1], [1, 9], [2, 3], [4, 9], [5, 3]])  # rows 0, 2, 4 known
        out = f.filter_negatives("gci0", pos, neg)
        known = set(map(tuple, GCI0))
        for row in out.tolist():
            self.assertNotIn(tuple(row), known)
        self.assertEqual(f.num_filtered, 3)
        # nothing could be dropped: every row has three admissible replacements
        self.assertEqual(out[:, 0].tolist(), [0, 1, 2, 4, 5])

    def test_known_positive_filter_drops_when_impossible(self):
        m = _model()
        f = KnownPositiveFilter(pools={"gci0": {1: [1]}}, max_resample=3, seed=0)
        f.attach(m)
        out = f.filter_negatives("gci0", m.training_datasets["gci0"],
                                 th.tensor([[0, 1], [1, 1], [2, 1], [4, 1], [5, 1]]))
        self.assertEqual(out.tolist(), [[1, 1], [2, 1], [4, 1], [5, 1]])
        self.assertEqual(f.num_dropped, 1)

    def test_composition_generator_then_filter(self):
        m = _model()
        regime = ComposedRegime(
            FixedColumnCorruption(columns={"gci0": 1}, pools={"gci0": {1: [1, 3, 11]}},
                                  num_negatives=20, seed=0),
            KnownPositiveFilter(pools={"gci0": {1: [1, 3, 11]}}, seed=1))
        regime.attach(m)
        pos = m.training_datasets["gci0"]
        neg = regime.filter_negatives("gci0", pos, regime.negatives("gci0", pos))
        self.assertTrue(set(map(tuple, neg.tolist())).isdisjoint(set(map(tuple, GCI0))))


class TestHardAndContradictionNegatives(TestCase):

    def test_siblings(self):
        # 1, 2 and 6 are children of 0; 3 is a child of 1; 2 is also below 1? no.
        gci0 = [[1, 0], [2, 0], [6, 0], [3, 1], [2, 7]]
        m = StubModel({"gci0": gci0})
        r = SiblingHardNegatives(num_negatives=30, seed=0)
        r.attach(m)
        neg = r.negatives("gci0", m.training_datasets["gci0"])
        n = len(gci0)
        for i, (c, s) in enumerate(neg.tolist()):
            self.assertEqual(c, gci0[i % n][0])
            if c == 1:
                self.assertIn(s, {2, 6})
            if c == 2:
                self.assertIn(s, {1, 6})        # siblings below 0; 7 has no other child
        # 3 has no sibling: random fallback, never one of its superclasses 1 and 0
        threes = [s for c, s in neg.tolist() if c == 3]
        self.assertTrue(all(s not in (0, 1, 3) for s in threes))
        self.assertGreater(r.num_random_fallback, 0)

    def test_contradiction_negatives_are_disjoint(self):
        bot = 20
        gci0 = [[1, 0], [2, 1], [4, 3], [5, 4]]
        gci1 = [[0, 3, bot]]            # 0 and 3 are disjoint
        m = StubModel({"gci0": gci0, "gci1": gci1,
                       "class_assertion": [[0, 2]]},
                      neg_capable=("gci0", "class_assertion"))
        r = ContradictionNegatives(p_bot=1.0, num_negatives=20,
                                   normal_forms=("gci0", "class_assertion"), seed=0)
        r.attach(m)
        self.assertEqual(set(r.contradictory_classes((2,)).tolist()), {3, 4, 5})
        neg = r.negatives("gci0", m.training_datasets["gci0"])
        for c, d in neg.tolist():
            if c in (1, 2):                 # below 0: contradictions are 3, 4, 5
                self.assertIn(d, {3, 4, 5})
            if c in (4, 5):                 # below 3: contradiction is 0, 1, 2
                self.assertIn(d, {0, 1, 2})
        ca = r.negatives("class_assertion", m.training_datasets["class_assertion"])
        self.assertTrue(set(ca[:, 1].tolist()) <= {3, 4, 5})

    def test_contradiction_mixture(self):
        bot = 20
        m = StubModel({"gci0": [[1, 0]] * 200, "gci1": [[0, 3, bot]]})
        r = ContradictionNegatives(p_bot=0.0, seed=0)
        r.attach(m)
        r.negatives("gci0", m.training_datasets["gci0"])
        self.assertEqual(r.num_contradictory, 0)
        r = ContradictionNegatives(p_bot=0.5, seed=0)
        r.attach(m)
        r.negatives("gci0", m.training_datasets["gci0"])
        self.assertTrue(50 < r.num_contradictory < 150)


class TestEndToEnd(TestCase):

    def test_train_elembeddings_with_composed_negatives(self):
        from mowl.models import ELEmbeddings
        from tests.datasetFactory import FamilyDataset
        th.manual_seed(0)
        np.random.seed(0)
        model = ELEmbeddings(FamilyDataset(), embed_dim=4, batch_size=64)
        regime = ComposedRegime(EitherSideCorruption(num_negatives=2, seed=0),
                                KnownPositiveFilter(seed=0))
        model.training_regime = regime
        before = [p.detach().clone() for p in model.module.parameters()]
        model.train(epochs=2, validate_every=100)
        after = list(model.module.parameters())
        self.assertTrue(any(not th.equal(a, b) for a, b in zip(before, after)))
