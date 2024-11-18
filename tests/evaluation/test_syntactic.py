from unittest import TestCase
from tests.datasetFactory import PPIYeastSlimDataset
from mowl.models import SyntacticPlusW2VModel
from mowl.evaluation import PPIEvaluator
import mowl.error.messages as msg
from utils import auc_from_mr, allowed_diff


class TestSyntacticModel(TestCase):

    @classmethod
    def setUpClass(self):
        self.ppi_dataset = PPIYeastSlimDataset()

    def init_model(self):
        model = SyntacticPlusW2VModel(self.ppi_dataset)
        model.set_w2v_model(min_count=1)
        model.generate_corpus(save=True, with_annotations=True)
        return model
        
    def test_evaluation_without_instantiating_evaluation_class(self):
        model = self.init_model()
        model.set_evaluator(PPIEvaluator)
        model.train(epochs=1)
        model.evaluate(self.ppi_dataset.testing, filter_ontologies = [self.ppi_dataset.ontology])
        
        mr = model.metrics["mr"]
        fmr = model.metrics["f_mr"]

        auc = model.metrics["auc"]
        true_auc = auc_from_mr(mr, len(self.ppi_dataset.evaluation_classes[0]))
        diff_auc = abs(true_auc - auc)
        
        self.assertTrue(mr > fmr)
        self.assertLess(diff_auc, allowed_diff)
        
    def test_evaluation_instantiating_evaluation_class(self):
        model = self.init_model()
        evaluator = PPIEvaluator(self.ppi_dataset)
        model.set_evaluator(evaluator)
        model.train(epochs=1)
        model.evaluate(self.ppi_dataset.testing, filter_ontologies = [self.ppi_dataset.ontology])

        mr = model.metrics["mr"]
        fmr = model.metrics["f_mr"]

        auc = model.metrics["auc"]
        true_auc = auc_from_mr(mr, len(self.ppi_dataset.evaluation_classes[0]))
        diff_auc = abs(true_auc - auc)
        
        self.assertTrue(mr > fmr)
        self.assertLess(diff_auc, allowed_diff)
        
    def test_not_set_evaluator(self):
        model = self.init_model()
        model.train(epochs=1)

        with self.assertRaisesRegex(AttributeError, msg.EVALUATOR_NOT_SET):
            model.evaluate()
