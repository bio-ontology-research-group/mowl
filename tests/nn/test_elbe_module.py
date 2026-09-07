from tests.nn.fixtures import ELAxioms
from unittest import TestCase
from mowl.nn import ELBEModule
from tests.datasetFactory import FamilyDataset
import torch as th

class TestELBEModule(TestCase):

    @classmethod
    def setUpClass(self):
        ds = FamilyDataset()
        nb_classes = len(ds.classes)
        nb_relations = len(ds.object_properties)
        nb_individuals = len(ds.individuals)
        self.module = ELBEModule(nb_classes, nb_relations, nb_individuals)
        self.axioms = ELAxioms()
        
    def _assert_loss(self, result):
        """Loss tensor must be finite and contain one value per sample in the batch."""
        self.assertIsInstance(result, th.Tensor)
        self.assertEqual(result.numel(), 1)
        self.assertTrue(th.isfinite(result).all(), "Loss contains NaN or Inf")

    def test_gci_0(self):
        result = self.module(self.axioms.gci0_data, "gci0")
        self._assert_loss(result)

    def test_gci_1(self):
        result = self.module(self.axioms.gci1_data, "gci1")
        self._assert_loss(result)

    def test_gci_2(self):
        result = self.module(self.axioms.gci2_data, "gci2")
        self._assert_loss(result)

    def test_gci_3(self):
        result = self.module(self.axioms.gci3_data, "gci3")
        self._assert_loss(result)

    def test_gci_0_bot(self):
        result = self.module(self.axioms.gci0_bot_data, "gci0_bot")
        self._assert_loss(result)

    def test_gci_1_bot(self):
        result = self.module(self.axioms.gci1_bot_data, "gci1_bot")
        self._assert_loss(result)

    def test_gci_3_bot(self):
        result = self.module(self.axioms.gci3_bot_data, "gci3_bot")
        self._assert_loss(result)

        




    def test_loss_has_one_value_per_sample(self):
        """A batch of n samples must produce n loss values.

        ``gci1_loss`` added a column of shape ``(n, 1)`` to a flat vector of shape ``(n,)``,
        which broadcasts to an ``(n, n)`` matrix rather than summing the two scores of each
        sample. Every other fixture here holds a single row, where the mistake is invisible
        because ``(1, 1)`` and ``(1,)`` broadcast to ``(1, 1)``.
        """
        batches = {
            "gci0": self.axioms.gci0_data,
            "gci1": self.axioms.gci1_data,
            "gci2": self.axioms.gci2_data,
            "gci3": self.axioms.gci3_data,
            "gci0_bot": self.axioms.gci0_bot_data,
            "gci1_bot": self.axioms.gci1_bot_data,
            "gci3_bot": self.axioms.gci3_bot_data,
        }

        for gci_name, single_row in batches.items():
            with self.subTest(gci=gci_name):
                batch = th.cat([single_row] * 3, dim=0)
                result = self.module(batch, gci_name)
                self.assertEqual(result.numel(), 3)
