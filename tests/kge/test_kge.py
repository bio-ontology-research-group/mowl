import shutil
import os
import mowl.error as err
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam
from mowl.projection import DL2VecProjector, Edge
from tests.datasetFactory import PPIYeastSlimDataset
from pykeen.triples.triples_factory import TriplesFactory
from pykeen.models import TransE
from mowl.kge import KGEModel
from unittest import TestCase


class TestKge(TestCase):

    @classmethod
    def setUpClass(self):
        dataset = PPIYeastSlimDataset()
        graph = DL2VecProjector(True).project(dataset.ontology)
        self.triples = Edge.as_pykeen(graph)

    def test_constructor_parameter_types(self):
        """This checks if the constructor parameters have the correct type."""

        graph = [("a", "b", "c"), ("d", "e", "f")]
        pykeen_model = TransE(triples_factory=self.triples)
        with self.assertRaisesRegex(
            TypeError,
            "Parameter triples_factory must be of type or subtype of \
pykeen.triples.triples_factory.TriplesFactory."):
            KGEModel(graph, pykeen_model, 32)

        with self.assertRaisesRegex(
                TypeError,
                "Parameter pykeen_model must be of type or subtype of pykeen.models.ERModel."):
            KGEModel(self.triples, TransE, 32)

        with self.assertRaisesRegex(TypeError, "Parameter epochs must be of type int."):
            KGEModel(self.triples, pykeen_model, "32")

        with self.assertRaisesRegex(
                TypeError,
                "Optional parameter batch_size must be of type int."):
            KGEModel(self.triples, pykeen_model, 32, batch_size="32")

        with self.assertRaisesRegex(
                TypeError,
                "Optional parameter optimizer must be a subtype of torch.optim.Optimizer."):
            KGEModel(self.triples, pykeen_model, 32, optimizer=ExponentialLR)

        with self.assertRaisesRegex(TypeError, "Optional parameter lr must be of type float."):
            KGEModel(self.triples, pykeen_model, 32, lr="32")

        with self.assertRaisesRegex(TypeError, "Optional parameter device must be of type str."):
            KGEModel(self.triples, pykeen_model, 32, device=32)

        with self.assertRaisesRegex(
                TypeError, "Optional parameter model_filepath must be of type str."):
            KGEModel(self.triples, pykeen_model, 32, model_filepath=32)

    def test_raise_value_error_if_model_not_trained(self):
        """This checks if the model raises a ValueError if it is not trained."""

        graph = [Edge("a", "b", "c"), Edge("d", "e", "f")]
        graph = Edge.as_pykeen(graph)
        pykeen_model = TransE(triples_factory=graph)
        kge_model = KGEModel(graph, pykeen_model, 32)
        with self.assertRaisesRegex(AttributeError, err.EMBEDDINGS_NOT_FOUND_MODEL_NOT_TRAINED):
            _ = kge_model.class_embeddings_dict

        with self.assertRaisesRegex(AttributeError, err.EMBEDDINGS_NOT_FOUND_MODEL_NOT_TRAINED):
            _ = kge_model.object_property_embeddings_dict

    def test_attributes_class_index_dict_type(self):
        """This checks if the class_index_dict attribute has the correct type."""

        graph = [Edge("a", "b", "c"), Edge("d", "e", "f")]
        graph = Edge.as_pykeen(graph)
        pykeen_model = TransE(triples_factory=graph, random_seed=0)
        kge_model = KGEModel(graph, pykeen_model, 32)
        self.assertIsInstance(kge_model.class_index_dict, dict)
        self.assertIsInstance(list(kge_model.class_index_dict.keys())[0], str)
