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
        
    def test_gci_0(self):
        """This should test the correct behavior of the module when the input is a GCI0"""

        gci0 = self.axioms.gci0_data
        result = self.module(gci0, "gci0")
        self.assertIsInstance(result, th.Tensor)


    def test_gci_1(self):
        """This should test the correct behavior of the module when the input is a GCI1"""

        gci1 = self.axioms.gci1_data
        result = self.module(gci1, "gci1")
        self.assertIsInstance(result, th.Tensor)


    def test_gci_2(self):
        """This should test the correct behavior of the module when the input is a GCI2"""

        gci2 = self.axioms.gci2_data
        result = self.module(gci2, "gci2")
        self.assertIsInstance(result, th.Tensor)

    def test_gci_3(self):
        """This should test the correct behavior of the module when the input is a GCI3"""

        gci3 = self.axioms.gci3_data
        result = self.module(gci3, "gci3")
        self.assertIsInstance(result, th.Tensor)

    def test_gci_0_bot(self):
        """This should test the correct behavior of the module when the input is a GCI0 with a bottom element"""

        gci0_bot = self.axioms.gci0_bot_data
        result = self.module(gci0_bot, "gci0_bot")
        self.assertIsInstance(result, th.Tensor)

    def test_gci_1_bot(self):
        """This should test the correct behavior of the module when the input is a GCI1 with a bottom element"""

        gci1_bot = self.axioms.gci1_bot_data
        result = self.module(gci1_bot, "gci1_bot")
        self.assertIsInstance(result, th.Tensor)

    def test_gci_3_bot(self):
        """This should test the correct behavior of the module when the input is a GCI3 with a bottom element"""

        gci3_bot = self.axioms.gci3_bot_data
        result = self.module(gci3_bot, "gci3_bot")
        self.assertIsInstance(result, th.Tensor)

        



