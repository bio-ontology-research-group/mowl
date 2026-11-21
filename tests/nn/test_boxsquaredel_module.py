import pytest
from tests.nn.fixtures import ELAxioms
from unittest import TestCase
from mowl.nn import BoxSquaredELModule
from tests.datasetFactory import FamilyDataset
import torch as th

class TestBoxSquaredELModule(TestCase):

    @classmethod
    def setUpClass(self):
        ds = FamilyDataset()
        nb_classes = len(ds.classes)
        nb_rels = len(ds.object_properties)
        nb_inds = len(ds.individuals)
        self.module = BoxSquaredELModule(nb_classes, nb_rels, nb_inds)
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

        





def _role_module():
    """Three roles in two dimensions: head and tail boxes of r (0) lie inside those of s
    (1); t (2) is far away."""
    module = BoxSquaredELModule(2, 3, None, embed_dim=2, gamma=0.0)
    with th.no_grad():
        module.head_center.weight.copy_(th.tensor([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]]))
        module.head_offset.weight.copy_(th.tensor([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]]))
        module.tail_center.weight.copy_(th.tensor([[1.0, 0.0], [1.0, 0.0], [-5.0, 5.0]]))
        module.tail_offset.weight.copy_(th.tensor([[0.5, 0.5], [1.0, 1.0], [1.0, 1.0]]))
    return module


def test_role_inclusion_loss_is_zero_iff_head_and_tail_boxes_are_included():
    module = _role_module()
    assert BoxSquaredELModule.role_axiom_capable
    loss = module(th.tensor([[0, 1], [1, 0], [0, 2]]), "role_inclusion")
    assert loss.shape[0] == 3
    assert loss[0].item() == 0.0          # r ⊑ s holds
    assert loss[1].item() > 0.0           # s ⊑ r does not: the boxes of s are larger
    assert loss[2].item() > 0.0


def test_role_chain_loss_uses_the_head_of_the_first_and_the_tail_of_the_second_role():
    module = _role_module()
    loss = module(th.tensor([[0, 0, 1], [0, 2, 1], [2, 0, 1]]), "role_chain")
    assert loss[0].item() == 0.0          # head(r) ⊆ head(s) and tail(r) ⊆ tail(s)
    assert loss[1].item() > 0.0           # tail(t) is not inside tail(s)
    assert loss[2].item() > 0.0           # head(t) is not inside head(s)
    # Values of the reference code: mean of the two inclusion losses.
    expected = (0.0 + th.linalg.norm(th.relu(th.tensor([6.0, 5.0]) + 1.0 - 1.0))) / 2
    assert loss[1].item() == pytest.approx(expected.item())
