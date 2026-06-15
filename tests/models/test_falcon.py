from unittest import TestCase
import torch as th
from tests.datasetFactory import FamilyDataset
from mowl.models import FALCONModel
from mowl.nn import ALCModule, FALCONModule
from mowl.visualization import FALCONVisualizer


class TestFALCON(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()
        cls.model = FALCONModel(cls.dataset, embed_dim=10, anon_e=4)
        cls.model.train(epochs=1, validate_every=1)

    def test_initialization(self):
        """FALCONModel builds a FALCONModule that respects the ALCModule interface."""
        self.assertIsNotNone(self.model.module)
        self.assertIsInstance(self.model.module, FALCONModule)
        self.assertIsInstance(self.model.module, ALCModule)

    def test_parameters_finite(self):
        """Parameters remain finite after training."""
        for param in self.model.module.parameters():
            self.assertTrue(th.isfinite(param).all(),
                            "Model parameters contain NaN or Inf after training")

    def test_embedding_shapes(self):
        """Embedding tables match the ontology vocabulary sizes."""
        nb_classes = len(self.model.class_index_dict)
        nb_rels = len(self.model.object_property_index_dict)
        self.assertEqual(self.model.module.c_embedding.weight.shape[0], nb_classes)
        self.assertEqual(self.model.module.r_embedding.weight.shape[0], nb_rels)
        self.assertEqual(self.model.module.c_embedding.weight.shape[1], 10)

    def test_visualizer_membership_matrix(self):
        """The membership heatmap has consistent dimensions and is in [0, 1]."""
        viz = FALCONVisualizer(self.model)
        fs, concept_labels, entity_labels = viz.membership_matrix(n_anon=4)
        self.assertEqual(fs.shape, (len(concept_labels), len(entity_labels)))
        self.assertTrue((fs >= 0).all() and (fs <= 1).all())


class TestFALCONValidation(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def test_alpha_beta_sum_exceeds_one(self):
        with self.assertRaises(ValueError):
            FALCONModel(self.dataset, alpha=0.7, beta=0.5)

    def test_alpha_out_of_range(self):
        with self.assertRaises(ValueError):
            FALCONModel(self.dataset, alpha=-0.1)
