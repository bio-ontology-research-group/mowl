from unittest import TestCase
from tests.datasetFactory import PPIYeastSlimDataset
from mowl.models.elembeddings.examples.model_ppi import ELEmPPI
from mowl.owlapi import OWLAPIAdapter
from mowl.evaluation import PPIEvaluator
from copy import deepcopy
import mowl.error.messages as msg
import os

from utils import auc_from_mr, allowed_diff

class TestELModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.ppi_dataset = PPIYeastSlimDataset()


    def test_evaluation_without_instantiating_evaluation_class(self):
        model = ELEmPPI(self.ppi_dataset, epochs=3)
        model.set_evaluator(PPIEvaluator)
        model.train(validate_every=1)
        model.evaluate(self.ppi_dataset.testing, filter_ontologies=[self.ppi_dataset.ontology])
        mr = model.metrics["mr"]
        fmr = model.metrics["f_mr"]

        self.assertTrue(mr > fmr)
        
    def test_evaluation_instantiating_evaluation_class(self):
        model = ELEmPPI(self.ppi_dataset, epochs=3)
        evaluator = PPIEvaluator(self.ppi_dataset)
        model.set_evaluator(evaluator)
        model.train(validate_every=1)
        model.evaluate(self.ppi_dataset.testing, filter_ontologies=[self.ppi_dataset.ontology])

        mr = model.metrics["mr"]
        fmr = model.metrics["f_mr"]
        auc = model.metrics["auc"]

        true_auc = auc_from_mr(mr, len(self.ppi_dataset.evaluation_classes[0]))
        diff_auc = abs(true_auc - auc)
        
        self.assertTrue(mr > fmr)
        self.assertLess(diff_auc, allowed_diff)

