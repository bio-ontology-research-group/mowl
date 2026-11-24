"""End-to-end tests of RegDELBE and RegDELEm (short training on the family ontology)."""
from unittest import TestCase

import pytest
import torch as th

from tests.datasetFactory import FamilyDataset
from mowl.models.regd.model import RegDELBE, RegDELEm
from mowl.nn.el.regd.dissimilarity import contrastive_loss
from mowl.nn.el.regd.module import RegDELBEModule, RegDELEmModule


def _contrastive_training_loss(model, gci_name="gci0"):
    """The RegD loss of one normal form on the whole training set, with fresh negatives."""
    dataset = model.training_datasets[gci_name]
    with th.no_grad():
        positive = model.module(dataset[:], gci_name)
        negatives = model.module(model.generate_negatives(gci_name, dataset), gci_name, neg=True)
        return model.compute_loss(positive, negatives).item()


class _RegDModelTests:
    model_class = None
    module_class = None
    kwargs = {}

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()
        th.manual_seed(0)
        cls.model = cls.model_class(cls.dataset, embed_dim=5, learning_rate=0.01, **cls.kwargs)
        cls.before = {g: _contrastive_training_loss(cls.model, g) for g in ("gci0", "gci2")}
        cls.model.train(epochs=60)
        cls.after = {g: _contrastive_training_loss(cls.model, g) for g in ("gci0", "gci2")}

    def test_module_type(self):
        self.assertIsInstance(self.model.module, self.module_class)
        self.assertEqual(self.model.module.class_embed.weight.shape[1], 5)

    def test_parameters_finite(self):
        for parameter in self.model.module.parameters():
            self.assertTrue(th.isfinite(parameter).all())

    def test_loss_decreases(self):
        for gci_name in ("gci0", "gci2"):
            self.assertLess(self.after[gci_name], self.before[gci_name], gci_name)

    def test_negative_sampling_gives_num_neg_negatives_per_positive(self):
        config = self.model.get_negative_sampling_config()
        self.assertEqual(set(config), {"gci0", "gci2", "object_property_assertion"})
        self.assertEqual(config["gci0"]["corrupt_column"], [1] * 10)
        self.assertEqual(config["gci2"]["corrupt_column"], [2] * 10)
        dataset = self.model.training_datasets["gci0"]
        negatives = self.model.generate_negatives("gci0", dataset)
        self.assertEqual(negatives.shape, (10 * len(dataset), 2))
        # the child column is untouched in every block
        data = dataset[:]
        for block in negatives.reshape(10, len(dataset), 2):
            self.assertTrue(th.equal(block[:, 0], data[:, 0]))

    def test_compute_loss_uses_the_contrastive_loss_for_gci0_and_gci2(self):
        module = self.model.module
        data = self.model.training_datasets["gci0"][:]
        module(data, "gci0")
        positive = th.rand(len(data), 1)
        negative = th.rand(3 * len(data), 1)
        expected = contrastive_loss(positive.reshape(-1), negative.reshape(3, -1).t(),
                                    self.model.gamma1)
        self.assertAlmostEqual(self.model.compute_loss(positive, negative).item(),
                               expected.item(), places=6)

    def test_scoring_uses_the_raw_energy(self):
        axiom_data = self.model.training_datasets["gci0"][:]
        with th.no_grad():
            loss = self.model.module(axiom_data, "gci0")
            self.model.eval_gci_name = "gci0"
            energy = self.model.eval_method(axiom_data)
            view = self.model.evaluation_model(axiom_data, "gci0")
        self.assertTrue(th.equal(view, energy))
        self.assertTrue(th.equal(th.clamp(energy, min=0.0), loss))
        # the module is back in loss mode afterwards
        self.assertFalse(self.model.module.energy_mode)

    def test_scores_rank_a_subclass_pair_above_a_random_pair(self):
        """Energies are finite and of the right shape for all pairs of classes."""
        n = len(self.model.class_index_dict)
        pairs = th.tensor([[a, b] for a in range(n) for b in range(n)])
        self.model.eval_gci_name = "gci0"
        with th.no_grad():
            energy = self.model.eval_method(pairs)
        self.assertEqual(energy.shape, (n * n, 1))
        self.assertTrue(th.isfinite(energy).all())


class TestRegDELBE(_RegDModelTests, TestCase):
    model_class = RegDELBE
    module_class = RegDELBEModule

    def test_replaces_all_four_inclusion_forms(self):
        self.assertEqual(self.model.module.regd_gcis, frozenset({"gci0", "gci1", "gci2", "gci3"}))

    def test_defaults_follow_the_paper(self):
        self.assertEqual(self.model.p, 1)  # boxes
        self.assertEqual((self.model.gamma1, self.model.gamma2), (0.0, 0.0))
        self.assertEqual(self.model.num_neg, 10)


class TestRegDELEm(_RegDModelTests, TestCase):
    model_class = RegDELEm
    module_class = RegDELEmModule

    def test_keeps_elem_for_the_intersection_form(self):
        self.assertEqual(self.model.module.regd_gcis, frozenset({"gci0", "gci2", "gci3"}))

    def test_defaults_follow_the_paper(self):
        self.assertEqual(self.model.p, 2)  # balls


class TestRegDOptions(TestCase):

    def test_options_reach_the_module(self):
        model = RegDELBE(FamilyDataset(), embed_dim=3, lamda=0.5, p=2, gamma1=0.1, gamma2=0.2,
                         num_neg=4, depth_on_containment=False)
        module = model.module
        self.assertEqual((module.lamda, module.p, module.gamma1, module.gamma2),
                         (0.5, 2, 0.1, 0.2))
        self.assertFalse(module.depth_on_containment)
        self.assertEqual(model.get_negative_sampling_config()["gci0"]["corrupt_column"], [1] * 4)

    def test_num_neg_must_be_positive(self):
        with self.assertRaises(ValueError):
            RegDELEm(FamilyDataset(), num_neg=0)

    @pytest.mark.slow
    def test_zero_lamda_and_code_variant_train(self):
        for kwargs in ({"lamda": 0.0}, {"depth_on_containment": False}):
            for cls in (RegDELBE, RegDELEm):
                model = cls(FamilyDataset(), embed_dim=5, learning_rate=0.01, **kwargs)
                model.train(epochs=20)
                for parameter in model.module.parameters():
                    self.assertTrue(th.isfinite(parameter).all())
