import pytest
from tests.nn.fixtures import ELAxioms
from unittest import TestCase
from mowl.nn import ELEmModule
from tests.datasetFactory import FamilyDataset
import torch as th

class TestELEmModule(TestCase):

    @classmethod
    def setUpClass(self):
        ds = FamilyDataset()
        nb_classes = len(ds.classes)
        nb_relations = len(ds.object_properties)
        nb_individuals = len(ds.individuals)
        self.module = ELEmModule(nb_classes, nb_relations, nb_individuals)
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

        





def _gci1_module(radius_term, re):
    import torch as th
    from mowl.nn import ELEmModule
    module = ELEmModule(3, 1, None, embed_dim=2, margin=0.0, gci1_radius_term=radius_term)
    with th.no_grad():
        # C and D are the same ball of radius 1; E has the same centre and radius ``re``.
        module.class_embed.weight.copy_(th.tensor([[1.0, 0.0]] * 3))
        module.class_rad.weight.copy_(th.tensor([[1.0], [1.0], [re]]))
    return module(th.tensor([[0, 1, 2]]), "gci1")


def test_gci1_radius_term_is_default_and_penalises_a_too_small_superclass():
    """C ⊓ D ⊑ E with E a smaller concentric ball does not hold: the default loss is
    positive, the released-code loss (no radius term) is zero."""
    assert _gci1_module(True, 0.25).item() == pytest.approx(0.75)
    assert _gci1_module(False, 0.25).item() == pytest.approx(0.0)
    from mowl.nn import ELEmModule
    assert ELEmModule(3, 1, None).gci1_radius_term is True


def test_gci1_radius_term_is_zero_when_the_superclass_contains_the_intersection():
    assert _gci1_module(True, 1.5).item() == pytest.approx(0.0)
