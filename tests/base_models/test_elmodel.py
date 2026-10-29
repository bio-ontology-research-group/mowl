from unittest import TestCase
from mowl.base_models.elmodel import EmbeddingELModel
from mowl.base_models.model import Model
from mowl.datasets import Dataset
from tests.datasetFactory import FamilyDataset, PPIYeastSlimDataset
from mowl.datasets.el import ELDataset
from mowl.models import ELEmbeddings
import random
import torch as th
import numpy as np

class TestEmbeddingElModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.family_dataset = FamilyDataset()
        self.ppi_dataset = PPIYeastSlimDataset()

    def test_instance_of_embedding_model(self):
        """This checks if the class EmbeddingELModel is a subclass of Model"""
        model = EmbeddingELModel(self.family_dataset, 1, False)
        self.assertTrue(isinstance(model, Model))

    def test_constructor_param_types(self):
        """This checks if the constructor parameters are of the correct type"""
        with self.assertRaisesRegex(TypeError, "Parameter dataset must be a mOWL Dataset."):
            EmbeddingELModel(True, 1, 1)


        # embedding size
        with self.assertRaisesRegex(TypeError, "Parameter 'embed_dim' must be of type int."):
            EmbeddingELModel(self.family_dataset, "1", 1)
            
        # batch size
        with self.assertRaisesRegex(TypeError, "Parameter batch_size must be of type int."):
            EmbeddingELModel(self.family_dataset, 1, "1")

        # optional extended
        with self.assertRaisesRegex(TypeError, "Optional parameter extended must be of type \
bool."):
            EmbeddingELModel(self.family_dataset, 1, 1, "True")

        # optional model_filepath
        with self.assertRaisesRegex(TypeError, "Optional parameter model_filepath must be of \
type str."):
            EmbeddingELModel(self.family_dataset, 1, 1, True, 1)

        # optional load_normalized
        with self.assertRaisesRegex(TypeError, "Optional parameter load_normalized must be of \
type bool."):
            EmbeddingELModel(self.family_dataset, 1, 1, True, "model_filepath", 1)
            
        # optional device
        with self.assertRaisesRegex(TypeError, "Optional parameter device must be of type str."):
            EmbeddingELModel(self.family_dataset, 1, 1, True, "model_filepath", False, 1)

    def test_class_attribute_training_dataset(self):
        """This should check that the attribute training_datasets is a dictionary of \
str -> ELDataset"""

        model = EmbeddingELModel(self.family_dataset, 1, False)

        training_datasets = model.training_datasets
 
        self.assertTrue(isinstance(training_datasets, dict))

        idx = random.randrange(0, len(training_datasets))
        random_item = list(training_datasets.items())[idx]
        self.assertTrue(isinstance(random_item[0], str))
        self.assertTrue(isinstance(random_item[1], th.utils.data.Dataset))

    def test_class_attribute_validation_dataset(self):
        """This should check that the attribute validation_datasets is a dictionary of \
str -> ELDataset"""

        model = EmbeddingELModel(self.ppi_dataset, 1, 1,  False)

        validation_datasets = model.validation_datasets
        self.assertTrue(isinstance(validation_datasets, dict))
        idx = random.randrange(0, len(validation_datasets))
        random_item = list(validation_datasets.items())[idx]
        self.assertTrue(isinstance(random_item[0], str))
        self.assertTrue(isinstance(random_item[1], th.utils.data.Dataset))

    def test_class_attribute_testing_dataset(self):
        """This should check that the attribute testing_datasets is a dictionary of \
str -> ELDataset"""

        model = EmbeddingELModel(self.ppi_dataset, 1, 1, False)

        testing_datasets = model.testing_datasets

        self.assertTrue(isinstance(testing_datasets, dict))
        idx = random.randrange(0, len(testing_datasets))
        random_item = list(testing_datasets.items())[idx]
        self.assertTrue(isinstance(random_item[0], str))
        self.assertTrue(isinstance(random_item[1], th.utils.data.Dataset))

    def test_class_attribute_training_dataloaders(self):
        """This should check that the attribute training_dataloaders is a dictionary of \
str -> DataLoader"""

        model = EmbeddingELModel(self.ppi_dataset, 1, 1, False)

        training_dataloaders = model.training_dataloaders
   
        self.assertTrue(isinstance(training_dataloaders, dict))
        idx = random.randrange(0, len(training_dataloaders))
        random_item = list(training_dataloaders.items())[idx]
        self.assertTrue(isinstance(random_item[0], str))
        self.assertTrue(isinstance(random_item[1], th.utils.data.DataLoader))

    def test_class_attribute_validation_dataloaders(self):
        """This should check that the attribute validation_dataloaders is a dictionary of \
str -> DataLoader"""

        model = EmbeddingELModel(self.ppi_dataset, 1, 1, False)

        validation_dataloaders = model.validation_dataloaders

        self.assertTrue(isinstance(validation_dataloaders, dict))
        idx = random.randrange(0, len(validation_dataloaders))
        random_item = list(validation_dataloaders.items())[idx]
        self.assertTrue(isinstance(random_item[0], str))
        self.assertTrue(isinstance(random_item[1], th.utils.data.DataLoader))

    def test_class_attribute_testing_dataloaders(self):
        """This should check that the attribute testing_dataloaders is a dictionary \
of str -> DataLoader"""

        model = EmbeddingELModel(self.ppi_dataset, 1, 1, False)

        testing_dataloaders = model.testing_dataloaders

        self.assertTrue(isinstance(testing_dataloaders, dict))
        idx = random.randrange(0, len(testing_dataloaders))
        random_item = list(testing_dataloaders.items())[idx]
        self.assertTrue(isinstance(random_item[0], str))
        self.assertTrue(isinstance(random_item[1], th.utils.data.DataLoader))

    def test_extended_attribute(self):
        """This should check if the parameter extended works as intended"""

        # Without extended
        model = EmbeddingELModel(self.family_dataset, 1, 1, extended=False)
        training_datasets = model.training_datasets
        self.assertTrue(len(training_datasets) == 4)
        self.assertIn("gci0", training_datasets)
        self.assertIn("gci1", training_datasets)
        self.assertIn("gci2", training_datasets)
        self.assertIn("gci3", training_datasets)

        # With extended
        model = EmbeddingELModel(self.family_dataset, 1, 1, extended=True)
        training_datasets = model.training_datasets
        self.assertTrue(len(training_datasets) == 7)
        self.assertIn("gci0", training_datasets)
        self.assertIn("gci1", training_datasets)
        self.assertIn("gci2", training_datasets)
        self.assertIn("gci3", training_datasets)
        self.assertIn("gci0_bot", training_datasets)
        self.assertIn("gci1_bot", training_datasets)
        self.assertIn("gci3_bot", training_datasets)

    def test_accesing_non_attributes(self):
        """This should check if the model raises an error when trying to access \
non-existing attributes"""

        model = EmbeddingELModel(self.family_dataset, 1, 1, False)

        with self.assertRaisesRegex(AttributeError, "Validation dataset is None"):
            model.validation_datasets

        with self.assertRaisesRegex(AttributeError, "Testing dataset is None"):
            model.testing_datasets

        with self.assertRaisesRegex(AttributeError, "Validation dataloader is None"):
            model.validation_dataloaders

        with self.assertRaisesRegex(AttributeError, "Testing dataloader is None"):
            model.testing_dataloaders

    def test_accessing_embeddings_attributes(self):
        """This should check if the model returns the correct embeddings attributes"""
        embed_dim = random.randrange(1, 100)
        model = ELEmbeddings(self.family_dataset, embed_dim = embed_dim)

        num_classes = len(model.dataset.classes)
        num_relations = len(model.dataset.object_properties)
        num_individuals = len(model.dataset.individuals)

        class_embeddings = model.class_embeddings
        self.assertIsInstance(class_embeddings, dict)
        self.assertTrue(len(class_embeddings) == num_classes)
        for key, value in class_embeddings.items():
            with self.subTest(key=key):
                self.assertIsInstance(key, str)
                self.assertIsInstance(value, np.ndarray)
                self.assertEqual(value.shape, (embed_dim,))

        object_property_embeddings = model.object_property_embeddings
        self.assertIsInstance(object_property_embeddings, dict)
        self.assertTrue(len(object_property_embeddings) == num_relations)
        for key, value in object_property_embeddings.items():
            with self.subTest(key=key):
                self.assertIsInstance(key, str)
                self.assertIsInstance(value, np.ndarray)
                self.assertEqual(value.shape, (embed_dim,))

        individual_embeddings = model.individual_embeddings
        self.assertIsInstance(individual_embeddings, dict)
        self.assertTrue(len(individual_embeddings) == num_individuals)
        for key, value in individual_embeddings.items():
            with self.subTest(key=key):
                self.assertIsInstance(key, str)
                self.assertIsInstance(value, np.ndarray)
                self.assertEqual(value.shape, (embed_dim,))


class TestEmbeddingElModelAuxiliaryEntities(TestCase):
    """Auxiliary classes introduced by the normalizer are indexed by EL models."""

    @classmethod
    def setUpClass(cls):
        from mowl.owlapi import OWLAPIAdapter
        ad = OWLAPIAdapter()
        ns = "http://test.org/elmodel_aux#"
        cls.ont = ad.create_ontology("http://test.org/elmodel_aux")
        A, B, C, D = [ad.create_class(ns + x) for x in "ABCD"]
        r = ad.create_object_property(ns + "r")
        ad.owl_manager.addAxiom(cls.ont, ad.create_subclass_of(
            ad.create_object_intersection_of(B, ad.create_object_some_values_from(r, C)), A))
        ad.owl_manager.addAxiom(cls.ont, ad.create_subclass_of(
            D, ad.create_object_some_values_from(r, ad.create_object_intersection_of(B, C))))
        cls.dataset = Dataset(cls.ont)

    def test_auxiliary_classes_are_appended_to_index(self):
        from mowl.ontology.normalize import is_auxiliary_iri
        model = EmbeddingELModel(self.dataset, 4, 4)
        dataset_classes = list(self.dataset.classes.as_str)
        index = model.class_index_dict
        # dataset classes keep their indices, auxiliaries come after them
        for i, name in enumerate(dataset_classes):
            self.assertEqual(index[name], i)
        aux = [name for name in index if is_auxiliary_iri(name)]
        self.assertGreater(len(aux), 0)
        self.assertEqual(sorted(index[a] for a in aux),
                         list(range(len(dataset_classes), len(index))))
        # the dataset itself is untouched
        self.assertFalse(any(is_auxiliary_iri(c) for c in self.dataset.classes.as_str))

    def test_training_axioms_are_all_indexed(self):
        for dataset in (self.dataset, FamilyDataset()):
            with self.subTest(dataset=type(dataset).__name__):
                model = EmbeddingELModel(dataset, 4, 4)
                el_dataset = ELDataset(dataset.ontology, model.class_index_dict,
                                       model.object_property_index_dict,
                                       normalized_gcis=model._training_normalized_gcis)
                el_dataset.get_gci_datasets()
                self.assertEqual(el_dataset.num_skipped_axioms, 0)

    def test_training_ontology_normalized_once(self):
        from unittest import mock
        from mowl.ontology.normalize import ELNormalizer
        original = ELNormalizer.normalize
        with mock.patch.object(ELNormalizer, "normalize", autospec=True,
                               side_effect=original) as patched:
            model = EmbeddingELModel(self.dataset, 4, 4)
            model.training_datasets
        self.assertEqual(patched.call_count, 1)

    def test_module_covers_auxiliary_classes(self):
        model = ELEmbeddings(self.dataset, embed_dim=4, batch_size=4)
        self.assertEqual(model.module.class_embed.weight.shape[0], len(model.class_index_dict))
        model.train(epochs=1)
