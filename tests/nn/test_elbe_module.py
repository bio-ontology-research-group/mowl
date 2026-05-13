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

        



