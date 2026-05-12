from unittest import TestCase
import torch as th
from tests.datasetFactory import PPIYeastSlimDataset, FamilyDataset, GDAHumanELDataset
from mowl.models import ELBE, ELBEPPI, ELBEGDA, ELEmGDA, BoxSquaredEL


def _assert_trained(test_case, model, embed_dim):
    """After any training run: parameters must be finite and have the right embedding dimension."""
    first_param = next(model.module.parameters())
    test_case.assertEqual(first_param.shape[-1], embed_dim)
    test_case.assertTrue(
        th.isfinite(first_param).all(),
        "Model parameters contain NaN or Inf after training"
    )


class TestELBE(TestCase):

    @classmethod
    def setUpClass(self):
        self.family_dataset = FamilyDataset()
        self.ppi_dataset = PPIYeastSlimDataset()

    def test_elbe_initialization(self):
        model = ELBE(self.family_dataset, embed_dim=30)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.module)

    def test_elbe_train_family_dataset(self):
        model = ELBE(self.family_dataset, embed_dim=30)
        model.train(epochs=1)
        _assert_trained(self, model, embed_dim=30)

    def test_elbe_train_ppi_dataset(self):
        model = ELBE(self.ppi_dataset, embed_dim=30)
        model.eval_gci_name = "gci2"
        model.train(epochs=1, validate_every=1)
        _assert_trained(self, model, embed_dim=30)


class TestELBEPPI(TestCase):

    @classmethod
    def setUpClass(self):
        self.ppi_dataset = PPIYeastSlimDataset()

    def test_elbeppi_initialization(self):
        model = ELBEPPI(self.ppi_dataset, embed_dim=30)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.module)

    def test_elbeppi_train(self):
        model = ELBEPPI(self.ppi_dataset, embed_dim=30)
        model.train(epochs=1, validate_every=1)
        _assert_trained(self, model, embed_dim=30)

    def test_elbeppi_with_evaluator(self):
        model = ELBEPPI(self.ppi_dataset, embed_dim=30)
        model.eval_gci_name = "gci2"
        model.train(epochs=1, validate_every=1)
        model.evaluate(
            self.ppi_dataset.testing, filter_ontologies=[self.ppi_dataset.ontology]
        )
        self.assertIn("mr", model.metrics)
        self.assertIn("f_mr", model.metrics)


class TestELBEGDA(TestCase):

    @classmethod
    def setUpClass(self):
        self.gda_dataset = GDAHumanELDataset()

    def test_elbegda_initialization(self):
        model = ELBEGDA(self.gda_dataset, embed_dim=30)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.module)

    def test_elbegda_train(self):
        model = ELBEGDA(self.gda_dataset, embed_dim=30, batch_size=32)
        model.train(epochs=1)
        _assert_trained(self, model, embed_dim=30)


class TestELEmGDA(TestCase):

    @classmethod
    def setUpClass(self):
        self.gda_dataset = GDAHumanELDataset()

    def test_elemgda_initialization(self):
        model = ELEmGDA(self.gda_dataset, embed_dim=30)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.module)

    def test_elemgda_train(self):
        model = ELEmGDA(self.gda_dataset, embed_dim=30)
        model.train(epochs=1)
        _assert_trained(self, model, embed_dim=30)


class TestBoxSquaredEL(TestCase):

    @classmethod
    def setUpClass(self):
        self.family_dataset = FamilyDataset()
        self.ppi_dataset = PPIYeastSlimDataset()

    def test_boxsquaredel_initialization(self):
        model = BoxSquaredEL(self.family_dataset, embed_dim=30)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.module)

    def test_boxsquaredel_train_family_dataset(self):
        model = BoxSquaredEL(self.family_dataset, embed_dim=30)
        model.train(epochs=1)
        _assert_trained(self, model, embed_dim=30)

    def test_boxsquaredel_train_ppi_dataset(self):
        model = BoxSquaredEL(self.ppi_dataset, embed_dim=30)
        model.eval_gci_name = "gci2"
        model.train(epochs=1, validate_every=1)
        _assert_trained(self, model, embed_dim=30)
