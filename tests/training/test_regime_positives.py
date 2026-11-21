"""Transitive reduction and closure of the told hierarchy as training positives."""
from unittest import TestCase

import torch as th

from mowl.training import ComposedRegime
from mowl.training.positives import (TransitiveReductionPositives,
                                     TransitiveClosurePositives)
from tests.training.stubs import StubModel

# 0 < 1 < 2, plus the shortcut 0 < 2 and a cycle 3 <-> 4 with 3 < 5, 4 < 5
GCI0 = [[0, 1], [1, 2], [0, 2], [3, 4], [4, 3], [3, 5], [4, 5]]


class TestPositives(TestCase):

    def test_reduction_drops_shortcuts_only(self):
        m = StubModel({"gci0": GCI0})
        r = TransitiveReductionPositives()
        r.attach(m)
        out = r.training_datasets(m.training_datasets)["gci0"][:].tolist()
        self.assertNotIn([0, 2], out)
        for edge in ([0, 1], [1, 2], [3, 4], [4, 3]):
            self.assertIn(edge, out)
        # 3 < 5 follows from 3 < 4 < 5 (and vice versa): both lie on the cycle, at least
        # one direct edge to 5 is kept
        self.assertTrue([3, 5] in out or [4, 5] in out)
        self.assertEqual(r.num_before, len(GCI0))

    def test_closure_adds_paths(self):
        m = StubModel({"gci0": [[0, 1], [1, 2], [2, 3]]})
        r = TransitiveClosurePositives()
        r.attach(m)
        out = set(map(tuple, r.training_datasets(m.training_datasets)["gci0"][:].tolist()))
        self.assertEqual(out, {(0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3)})

    def test_reduction_of_closure_is_reduction(self):
        m = StubModel({"gci0": [[0, 1], [1, 2], [2, 3]]})
        regime = ComposedRegime(TransitiveClosurePositives(), TransitiveReductionPositives())
        regime.attach(m)
        out = regime.training_datasets(m.training_datasets)["gci0"][:].tolist()
        self.assertEqual(sorted(map(tuple, out)), [(0, 1), (1, 2), (2, 3)])

    def test_other_datasets_untouched(self):
        m = StubModel({"gci0": [[0, 1]], "gci2": [[0, 0, 1]]})
        r = TransitiveClosurePositives()
        r.attach(m)
        out = r.training_datasets(m.training_datasets)
        self.assertIs(out["gci2"], m.training_datasets["gci2"])

    def test_end_to_end(self):
        from mowl.models import ELEmbeddings
        from tests.datasetFactory import FamilyDataset
        th.manual_seed(0)
        model = ELEmbeddings(FamilyDataset(), embed_dim=4, batch_size=64)
        regime = TransitiveClosurePositives()
        model.training_regime = regime
        model.train(epochs=2, validate_every=100)
        self.assertGreaterEqual(regime.num_after, regime.num_before)
