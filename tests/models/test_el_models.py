from unittest import TestCase
from tests.datasetFactory import PPIYeastSlimDataset, FamilyDataset, GDAHumanELDataset
from mowl.models import ELBE, ELBEPPI, ELBEGDA, ELEmGDA, BoxSquaredEL
from mowl.evaluation import PPIEvaluator


class TestELBE(TestCase):
    """Test the ELBE base model"""

    @classmethod
    def setUpClass(self):
        self.family_dataset = FamilyDataset()
        self.ppi_dataset = PPIYeastSlimDataset()

    def test_elbe_initialization(self):
        """Test ELBE model can be initialized"""
        model = ELBE(self.family_dataset, embed_dim=30, epochs=1)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.module)

    def test_elbe_train_family_dataset(self):
        """Test ELBE model can train on family dataset"""
        model = ELBE(self.family_dataset, embed_dim=30, epochs=1)
        model.train(epochs=1)
        self.assertTrue(True)

    def test_elbe_train_ppi_dataset(self):
        """Test ELBE model can train on PPI dataset with validation"""
        model = ELBE(self.ppi_dataset, embed_dim=30, epochs=1)
        model.train(epochs=1, validate_every=1)
        self.assertTrue(True)


class TestELBEPPI(TestCase):
    """Test the ELBEPPI model for protein-protein interaction prediction"""

    @classmethod
    def setUpClass(self):
        self.ppi_dataset = PPIYeastSlimDataset()

    def test_elbeppi_initialization(self):
        """Test ELBEPPI model can be initialized"""
        model = ELBEPPI(self.ppi_dataset, embed_dim=30, epochs=1)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.module)

    def test_elbeppi_train(self):
        """Test ELBEPPI model can train"""
        model = ELBEPPI(self.ppi_dataset, embed_dim=30, epochs=1)
        model.train(validate_every=1)
        self.assertTrue(True)

    def test_elbeppi_with_evaluator(self):
        """Test ELBEPPI model with PPIEvaluator"""
        model = ELBEPPI(self.ppi_dataset, embed_dim=30, epochs=1)
        model.set_evaluator(PPIEvaluator)
        model.train(validate_every=1)
        model.evaluate(
            self.ppi_dataset.testing, filter_ontologies=[self.ppi_dataset.ontology]
        )

        # Check that metrics are computed
        self.assertIn("mr", model.metrics)
        self.assertIn("f_mr", model.metrics)


class TestELBEGDA(TestCase):
    """Test the ELBEGDA model for gene-disease association prediction"""

    @classmethod
    def setUpClass(self):
        self.gda_dataset = GDAHumanELDataset()

    def test_elbegda_initialization(self):
        """Test ELBEGDA model can be initialized"""
        model = ELBEGDA(self.gda_dataset, embed_dim=30, epochs=1)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.module)

    def test_elbegda_train(self):
        """Test ELBEGDA model can train.
        Note: ELBEGDA.train() uses self.epochs, so we set epochs=1 in constructor.
        """
        model = ELBEGDA(self.gda_dataset, embed_dim=30, epochs=1, batch_size=32)
        model.train()
        self.assertTrue(True)


class TestELEmGDA(TestCase):
    """Test the ELEmGDA model for gene-disease association prediction"""

    @classmethod
    def setUpClass(self):
        self.gda_dataset = GDAHumanELDataset()

    def test_elemgda_initialization(self):
        """Test ELEmGDA model can be initialized"""
        model = ELEmGDA(self.gda_dataset, embed_dim=30, epochs=1)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.module)

    def test_elemgda_train(self):
        """Test ELEmGDA model can train.
        Note: ELEmGDA.train() uses self.epochs, so we set epochs=1 in constructor.
        """
        model = ELEmGDA(self.gda_dataset, embed_dim=30, epochs=1)
        model.train()
        self.assertTrue(True)


class TestBoxSquaredEL(TestCase):
    """Test the BoxSquaredEL model"""

    @classmethod
    def setUpClass(self):
        self.family_dataset = FamilyDataset()
        self.ppi_dataset = PPIYeastSlimDataset()

    def test_boxsquaredel_initialization(self):
        """Test BoxSquaredEL model can be initialized"""
        model = BoxSquaredEL(self.family_dataset, embed_dim=30, epochs=1)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.module)

    def test_boxsquaredel_train_family_dataset(self):
        """Test BoxSquaredEL model can train on family dataset"""
        model = BoxSquaredEL(self.family_dataset, embed_dim=30, epochs=1)
        model.train(epochs=1)
        self.assertTrue(True)

    def test_boxsquaredel_train_ppi_dataset(self):
        """Test BoxSquaredEL model can train on PPI dataset with validation"""
        model = BoxSquaredEL(self.ppi_dataset, embed_dim=30, epochs=1)
        model.train(epochs=1, validate_every=1)
        self.assertTrue(True)
