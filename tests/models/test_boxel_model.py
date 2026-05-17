from unittest import TestCase
import torch as th
from tests.datasetFactory import PPIYeastSlimDataset, FamilyDataset, GDAHumanELDataset
from mowl.models import BoxEL, BoxELPPI, BoxELGDA


def _assert_trained(test_case, model, embed_dim):
    first_param = next(model.module.parameters())
    test_case.assertEqual(first_param.shape[-1], embed_dim)
    test_case.assertTrue(
        th.isfinite(first_param).all(),
        "Model parameters contain NaN or Inf after training"
    )


class TestBoxEL(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()
        cls.model = BoxEL(cls.dataset, embed_dim=30)
        cls.model.train(epochs=1)

    def test_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.module)

    def test_parameters_finite_and_correct_dim(self):
        _assert_trained(self, self.model, embed_dim=30)

    def test_embedding_shapes(self):
        embed_dim = 30
        nb_classes = len(self.model.class_index_dict)
        nb_rels = len(self.model.object_property_index_dict)
        self.assertEqual(self.model.module.min_embedding.weight.shape, (nb_classes, embed_dim))
        self.assertEqual(self.model.module.delta_embedding.weight.shape, (nb_classes, embed_dim))
        self.assertEqual(self.model.module.relation_embedding.weight.shape, (nb_rels, embed_dim))
        self.assertEqual(self.model.module.scaling_embedding.weight.shape, (nb_rels, embed_dim))


class TestBoxELPPI(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = PPIYeastSlimDataset()
        cls.model = BoxELPPI(cls.dataset, embed_dim=30)
        cls.model.train(epochs=1, validate_every=1)

    def test_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.module)

    def test_parameters_finite_and_correct_dim(self):
        _assert_trained(self, self.model, embed_dim=30)

    def test_evaluate(self):
        self.model.evaluate(
            self.dataset.testing, filter_ontologies=[self.dataset.ontology]
        )
        self.assertIn("mr", self.model.metrics)
        self.assertIn("f_mr", self.model.metrics)


class TestBoxELGDA(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = GDAHumanELDataset()
        cls.model = BoxELGDA(cls.dataset, embed_dim=30, batch_size=32)
        cls.model.train(epochs=1)

    def test_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.module)

    def test_parameters_finite_and_correct_dim(self):
        _assert_trained(self, self.model, embed_dim=30)
