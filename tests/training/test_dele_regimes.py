"""DELE training regimes: filtered negatives are never entailed, all-normal-form
negatives, closure positives, and short end-to-end runs."""
from unittest import TestCase

import numpy as np
import torch as th

from mowl.models import ELEmbeddings, ELBE, BoxSquaredEL
from mowl.training import (AllNormalFormNegatives, ClosureFilteredNegatives,
                           ClosurePositives, ComposedRegime, DeductiveClosure, dele_regime)
from tests.datasetFactory import FamilyDataset


class TestClosureFilteredNegatives(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()
        th.manual_seed(0)
        cls.model = ELEmbeddings(cls.dataset, embed_dim=8, batch_size=64)
        cls.closure = DeductiveClosure.from_model(cls.model)

    def test_entailed_negatives_are_replaced(self):
        filt = ClosureFilteredNegatives(self.closure)
        self.model.training_regime = filt
        for gci in ("gci0", "gci2", "gci1", "gci3"):
            ds = self.model.training_datasets[gci]
            if len(ds) == 0:
                continue
            # The positives themselves are entailed: worst case for the filter.
            out = filt.filter_negatives(gci, ds, ds[:].clone())
            if out is not None:
                self.assertFalse(self.closure.contains(gci, out).any(), gci)
            # and so are random negatives after filtering
            np.random.seed(0)
            negs = self.model.generate_negatives(gci, ds) if gci == "gci2" else None
            if negs is not None:
                out = filt.filter_negatives(gci, ds, negs)
                self.assertFalse(self.closure.contains(gci, out).any())
        self.assertGreater(filt.num_entailed, 0)
        self.model.training_regime = None

    def test_without_resampling_entailed_rows_are_dropped(self):
        filt = ClosureFilteredNegatives(self.closure, max_resample=0)
        self.model.training_regime = filt
        ds = self.model.training_datasets["gci0"]
        self.assertIsNone(filt.filter_negatives("gci0", ds, ds[:].clone()))
        self.model.training_regime = None

    def test_assertions_pass_through(self):
        filt = ClosureFilteredNegatives(self.closure)
        x = th.zeros((3, 2), dtype=th.long)
        self.assertIs(filt.filter_negatives("class_assertion", None, x), x)
        self.assertIsNone(filt.filter_negatives("gci0", None, None))


class TestAllNormalFormNegatives(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def test_enables_module_and_samples_bot_forms(self):
        model = ELEmbeddings(self.dataset, embed_dim=8, batch_size=64)
        self.assertNotIn("gci0", model.get_negative_sampling_config())
        regime = AllNormalFormNegatives(epsilon=0.05)
        model.training_regime = regime
        self.assertTrue(model.module.all_negative_losses)
        self.assertEqual(model.module.epsilon, 0.05)
        # gci0/gci1/gci3 now come from the model's own sampler, with DELE's columns
        cfg = model.get_negative_sampling_config()
        self.assertEqual(cfg["gci0"]["corrupt_column"], 1)
        self.assertEqual(cfg["gci1"]["corrupt_column"], 2)
        self.assertIs(regime.negatives("gci0", model.training_datasets["gci0"]),
                      NotImplemented)
        # bot forms are sampled by the regime, corrupting the subclass / filler
        ds = model.training_datasets["gci1_bot"]
        if len(ds) > 0:
            neg = regime.negatives("gci1_bot", ds)
            self.assertTrue(th.equal(neg[:, 1:], ds[:][:, 1:]))

    def test_rejects_incapable_module(self):
        from mowl.models import BoxEL
        model = BoxEL(self.dataset, embed_dim=4)
        with self.assertRaises(TypeError):
            model.training_regime = AllNormalFormNegatives()


class TestEndToEnd(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def _run(self, cls_, **kw):
        th.manual_seed(0)
        np.random.seed(0)
        model = cls_(self.dataset, embed_dim=8, batch_size=64, **kw)
        model.training_regime = dele_regime(epsilon=0.01)
        model.train(epochs=3)
        for p in model.module.parameters():
            self.assertTrue(th.isfinite(p).all())
        return model

    def test_elem(self):
        model = self._run(ELEmbeddings)
        filt = model.training_regime.regimes[1]
        self.assertIsInstance(filt, ClosureFilteredNegatives)

    def test_elbe(self):
        self._run(ELBE)

    def test_box2el(self):
        self._run(BoxSquaredEL)

    def test_closure_positives(self):
        th.manual_seed(0)
        model = ELEmbeddings(self.dataset, embed_dim=8, batch_size=64)
        pos = ClosurePositives()
        model.training_regime = ComposedRegime(pos, dele_regime())
        before = len(model.training_datasets["gci0"])
        datasets = pos.training_datasets(model.training_datasets)
        self.assertGreater(pos.num_added["gci0"], 0)
        self.assertEqual(len(datasets["gci0"]), before + pos.num_added["gci0"])
        rows = datasets["gci0"][:]
        self.assertTrue(pos.closure.contains("gci0", rows[before:]).all())
        model.train(epochs=2)
