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


class TestFALCONWithABox(TestCase):
    """Exercises the concept-assertion (ABox-EC) training path and the heatmap over
    named individuals."""

    @classmethod
    def setUpClass(cls):
        from java.util import HashSet
        from mowl.datasets import Dataset
        from mowl.owlapi.adapter import OWLAPIAdapter

        adapter = OWLAPIAdapter()
        onto = adapter.create_ontology("http://mowl/abox")
        male = adapter.create_class("http://Male")
        person = adapter.create_class("http://Person")
        axioms = HashSet()
        axioms.add(adapter.create_subclass_of(male, person))
        axioms.add(adapter.create_class_assertion(
            male, adapter.create_individual("http://john")))
        axioms.add(adapter.create_class_assertion(
            person, adapter.create_individual("http://mary")))
        adapter.owl_manager.addAxioms(onto, axioms)
        cls.dataset = Dataset(onto, validation=onto, testing=onto)
        cls.model = FALCONModel(cls.dataset, embed_dim=10, anon_e=2)
        cls.model.train(epochs=1, validate_every=1)

    def test_trains_with_individuals(self):
        self.assertEqual(len(self.dataset.individuals), 2)
        for param in self.model.module.parameters():
            self.assertTrue(th.isfinite(param).all())

    def test_heatmap_includes_named_individuals(self):
        viz = FALCONVisualizer(self.model)
        fs, concepts, entities = viz.membership_matrix(n_anon=2)
        # two named individuals + two anonymous entities
        self.assertEqual(len(entities), 4)
        self.assertEqual(fs.shape, (len(concepts), 4))


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
