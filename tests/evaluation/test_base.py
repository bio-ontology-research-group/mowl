import tests
from unittest import TestCase
from mowl.evaluation import BaseRankingEvaluator
import torch as th
from utils import auc_from_mr

allowed_diff = 1e-6

class PredictionModel(th.nn.Module):
    def __init__(self):
        super().__init__()
        
        scores_matrix = {"a": [1, 2, 3, 4],
                         "b": [5, 6, 7, 8],
                         "c": [9, 10, 11, 12],
                         "d": [13, 14, 15, 16]}

        scores_tensor = th.tensor(list(scores_matrix.values()))
        self.scores_tensor = th.nn.Parameter(scores_tensor, requires_grad=False)

        
    def forward(self, x):
        assert x.shape[1] == 2
        head, tail = x[:, 0], x[:, 1]
        return self.scores_tensor[head, tail]


class TestBaseRankingEvaluator(TestCase):

    def setUp(self):

        self.entities = ["a", "b", "c", "d"]
        self.train_set = [("a", "a"), ("a", "b"),  ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")]
        self.valid_set = [("c", "c"), ("c", "d")]
        self.test_set = [("d", "c"), ("d", "d")]
        self.extra_set = [("b", "a"),  ("b", "b"), ("c", "a"), ("c", "b"), ("d", "a"), ("d", "b")]

        self.entity_to_id = {entity: i for i, entity in enumerate(self.entities)}
        self.entities_tensor = th.tensor(list(self.entity_to_id.values()), dtype=th.long)
        
        self.train_set_tensor = th.tensor([(self.entity_to_id[head], self.entity_to_id[tail]) for head, tail in self.train_set])
        self.valid_set_tensor = th.tensor([(self.entity_to_id[head], self.entity_to_id[tail]) for head, tail in self.valid_set])
        self.test_set_tensor = th.tensor([(self.entity_to_id[head], self.entity_to_id[tail]) for head, tail in self.test_set])
        self.extra_set_tensor = th.tensor([(self.entity_to_id[head], self.entity_to_id[tail]) for head, tail in self.extra_set])

        self.evaluation_model = PredictionModel()
        
    def test_ranking_evaluation_head(self):
        evaluator = BaseRankingEvaluator(self.entities_tensor, self.entities_tensor, 2, "cpu")
        
        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.test_set_tensor, mode="head_centric")
        mr = metrics["mr"]
        mrr = metrics["mrr"]
        auc = metrics["auc"]
        
        rank_d_c = 3
        rank_d_d = 4

        mrr_d_c = 1 / (rank_d_c)
        mrr_d_d = 1 / (rank_d_d)
        
        true_mr = (rank_d_c + rank_d_d) / 2
        true_mrr = (mrr_d_c + mrr_d_d) / 2

        true_auc = auc_from_mr(true_mr, len(self.entities))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(mrr, true_mrr)
        print(auc, true_auc)
        diff_auc = abs(auc - true_auc)
        self.assertLess(diff_auc, allowed_diff)
        
    def test_base_evaluator_tail(self):
        evaluator = BaseRankingEvaluator(self.entities_tensor, self.entities_tensor, 2, "cpu")
        
        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.test_set_tensor, mode="tail_centric")
        mr = metrics["mr"]
        mrr = metrics["mrr"]
        auc = metrics["auc"]
        
        rank_d_c = 4
        rank_d_d = 4

        mrr_d_c = 1 / (rank_d_c)
        mrr_d_d = 1 / (rank_d_d)
        
        true_mr = (rank_d_c + rank_d_d) / 2
        true_mrr = (mrr_d_c + mrr_d_d) / 2

        true_auc = auc_from_mr(true_mr, len(self.entities))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(mrr, true_mrr)

        diff_auc = abs(auc - true_auc)
        self.assertLess(diff_auc, allowed_diff)

    def test_base_evaluator_both(self):
        evaluator = BaseRankingEvaluator(self.entities_tensor, self.entities_tensor, 2, "cpu")
        
        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.test_set_tensor, mode="both")
        mr = metrics["mr"]
        mrr = metrics["mrr"]
        auc = metrics["auc"]
        
        rank_d_c_head = 3
        rank_d_d_head = 4
        rank_d_c_tail = 4
        rank_d_d_tail = 4

        mrr_d_c_head = 1 / (rank_d_c_head)
        mrr_d_d_head = 1 / (rank_d_d_head)
        mrr_d_c_tail = 1 / (rank_d_c_tail)
        mrr_d_d_tail = 1 / (rank_d_d_tail)
        
        true_mr = (rank_d_c_head + rank_d_d_head + rank_d_c_tail + rank_d_d_tail) / 4
        true_mrr = (mrr_d_c_head + mrr_d_d_head + mrr_d_c_tail + mrr_d_d_tail) / 4

        true_auc = auc_from_mr(true_mr, len(self.entities))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(mrr, true_mrr)

        diff_auc = abs(auc - true_auc)
        self.assertLess(diff_auc, allowed_diff)


    def test_filtering_head(self):
        evaluator = BaseRankingEvaluator(self.entities_tensor, self.entities_tensor, 2, "cpu")

        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.valid_set_tensor, mode="head_centric", filter_data=self.extra_set_tensor)
        mr = metrics["mr"]
        fmr = metrics["f_mr"]
        mrr = metrics["mrr"]
        fmrr = metrics["f_mrr"]
        fauc = metrics["f_auc"]
        
        rank_c_c = 3
        rank_c_d = 4
        f_rank_c_c = 1
        f_rank_c_d = 2

        mrr_c_c = 1 / (rank_c_c)
        mrr_c_d = 1 / (rank_c_d)
        f_mrr_c_c = 1 / (f_rank_c_c)
        f_mrr_c_d = 1 / (f_rank_c_d)

        true_mr = (rank_c_c + rank_c_d) / 2
        true_fmr = (f_rank_c_c + f_rank_c_d) / 2
        true_mrr = (mrr_c_c + mrr_c_d) / 2
        true_fmrr = (f_mrr_c_c + f_mrr_c_d) / 2

        true_fauc = auc_from_mr(true_fmr, len(self.entities))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(fmr, true_fmr)
        self.assertEqual(mrr, true_mrr)
        self.assertEqual(fmrr, true_fmrr)

        diff_fauc = abs(fauc - true_fauc)
        self.assertLess(diff_fauc, allowed_diff)
        

    def test_filtering_tail(self):
        evaluator = BaseRankingEvaluator(self.entities_tensor, self.entities_tensor, 2, "cpu")

        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.test_set_tensor, mode="tail_centric", filter_data=self.train_set_tensor)
        mr = metrics["mr"]
        fmr = metrics["f_mr"]
        mrr = metrics["mrr"]
        fmrr = metrics["f_mrr"]
        fauc = metrics["f_auc"]
        
        rank_d_c = 4
        rank_d_d = 4
        f_rank_d_c = 2
        f_rank_d_d = 2

        mrr_d_c = 1 / (rank_d_c)
        mrr_d_d = 1 / (rank_d_d)
        f_mrr_d_c = 1 / (f_rank_d_c)
        f_mrr_d_d = 1 / (f_rank_d_d)

        true_mr = (rank_d_c + rank_d_d) / 2
        true_fmr = (f_rank_d_c + f_rank_d_d) / 2
        true_mrr = (mrr_d_c + mrr_d_d) / 2
        true_fmrr = (f_mrr_d_c + f_mrr_d_d) / 2

        true_fauc = auc_from_mr(true_fmr, len(self.entities))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(fmr, true_fmr)
        self.assertEqual(mrr, true_mrr)
        self.assertEqual(fmrr, true_fmrr)

        diff_fauc = abs(fauc - true_fauc)
        self.assertLess(diff_fauc, allowed_diff)
        

    def test_filtering_both(self):
        evaluator = BaseRankingEvaluator(self.entities_tensor, self.entities_tensor, 2, "cpu")

        filter_data = th.cat([self.train_set_tensor, self.extra_set_tensor], dim=0)
        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.valid_set_tensor, mode="both", filter_data=filter_data)
        mr = metrics["mr"]
        fmr = metrics["f_mr"]
        mrr = metrics["mrr"]
        fmrr = metrics["f_mrr"]
        auc = metrics["auc"]
        fauc = metrics["f_auc"]

        
        rank_c_c_head = 3
        rank_c_d_head = 4
        rank_c_c_tail = 3
        rank_c_d_tail = 3

        f_rank_c_c_head = 1
        f_rank_c_d_head = 2
        f_rank_c_c_tail = 1
        f_rank_c_d_tail = 1

        mrr_c_c_head = 1 / (rank_c_c_head)
        mrr_c_d_head = 1 / (rank_c_d_head)
        mrr_c_c_tail = 1 / (rank_c_c_tail)
        mrr_c_d_tail = 1 / (rank_c_d_tail)

        f_mrr_c_c_head = 1 / (f_rank_c_c_head)
        f_mrr_c_d_head = 1 / (f_rank_c_d_head)
        f_mrr_c_c_tail = 1 / (f_rank_c_c_tail)
        f_mrr_c_d_tail = 1 / (f_rank_c_d_tail)
        
        true_mr = (rank_c_c_head + rank_c_d_head + rank_c_c_tail + rank_c_d_tail) / 4
        true_fmr = (f_rank_c_c_head + f_rank_c_d_head + f_rank_c_c_tail + f_rank_c_d_tail) / 4
        true_mrr = (mrr_c_c_head + mrr_c_d_head + mrr_c_c_tail + mrr_c_d_tail) / 4
        true_fmrr = (f_mrr_c_c_head + f_mrr_c_d_head + f_mrr_c_c_tail + f_mrr_c_d_tail) / 4

        true_fauc = auc_from_mr(true_fmr, len(self.entities))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(fmr, true_fmr)
        self.assertEqual(mrr, true_mrr)
        self.assertEqual(fmrr, true_fmrr)

        diff_fauc = abs(fauc - true_fauc)
        self.assertLess(diff_fauc, allowed_diff)


class TestBaseRankingEvaluatorSomeEntitiesOnly(TestCase):

    def setUp(self):

        self.entities = ["a", "b", "c", "d"]
        self.entities_of_interest = ["b", "c", "d"]
        self.train_set = [("a", "a"), ("a", "b"),  ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")]
        self.valid_set = [("c", "c"), ("c", "d")]
        self.test_set = [("d", "c"), ("d", "d")]
        self.extra_set = [("b", "a"),  ("b", "b"), ("c", "a"), ("c", "b"), ("d", "a"), ("d", "b")]

        self.entity_to_id = {entity: i for i, entity in enumerate(self.entities)}
        self.entities_tensor = th.tensor(list(self.entity_to_id.values()), dtype=th.long)
        self.entities_of_interest_tensor = th.tensor([self.entity_to_id[entity] for entity in self.entities_of_interest], dtype=th.long)
        
        self.train_set_tensor = th.tensor([(self.entity_to_id[head], self.entity_to_id[tail]) for head, tail in self.train_set])
        self.valid_set_tensor = th.tensor([(self.entity_to_id[head], self.entity_to_id[tail]) for head, tail in self.valid_set])
        self.test_set_tensor = th.tensor([(self.entity_to_id[head], self.entity_to_id[tail]) for head, tail in self.test_set])
        self.extra_set_tensor = th.tensor([(self.entity_to_id[head], self.entity_to_id[tail]) for head, tail in self.extra_set])

        self.evaluation_model = PredictionModel()
        
    def test_ranking_evaluation_head(self):
        evaluator = BaseRankingEvaluator(self.entities_of_interest_tensor, self.entities_of_interest_tensor, 2, "cpu")
        
        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.test_set_tensor, mode="head_centric")
        mr = metrics["mr"]
        mrr = metrics["mrr"]
        auc = metrics["auc"]

        
        rank_d_c = 2
        rank_d_d = 3

        mrr_d_c = 1 / (rank_d_c)
        mrr_d_d = 1 / (rank_d_d)
        
        true_mr = (rank_d_c + rank_d_d) / 2
        true_mrr = (mrr_d_c + mrr_d_d) / 2

        true_auc = auc_from_mr(true_mr, len(self.entities_of_interest))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(mrr, true_mrr)

        diff_auc = abs(auc - true_auc)
        self.assertLess(diff_auc, allowed_diff)

    def test_base_evaluator_tail(self):
        evaluator = BaseRankingEvaluator(self.entities_of_interest_tensor, self.entities_of_interest_tensor, 2, "cpu")
        
        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.test_set_tensor, mode="tail_centric")
        mr = metrics["mr"]
        mrr = metrics["mrr"]
        auc = metrics["auc"]
        
        rank_d_c = 3
        rank_d_d = 3

        mrr_d_c = 1 / (rank_d_c)
        mrr_d_d = 1 / (rank_d_d)
        
        true_mr = (rank_d_c + rank_d_d) / 2
        true_mrr = (mrr_d_c + mrr_d_d) / 2

        true_auc = auc_from_mr(true_mr, len(self.entities_of_interest))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(mrr, true_mrr)

        diff_auc = abs(auc - true_auc)
        self.assertLess(diff_auc, allowed_diff)

        
    def test_base_evaluator_both(self):
        evaluator = BaseRankingEvaluator(self.entities_of_interest_tensor, self.entities_of_interest_tensor, 2, "cpu")
        
        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.test_set_tensor, mode="both")
        mr = metrics["mr"]
        mrr = metrics["mrr"]
        auc = metrics["auc"]
        
        rank_d_c_head = 2
        rank_d_d_head = 3
        rank_d_c_tail = 3
        rank_d_d_tail = 3

        mrr_d_c_head = 1 / (rank_d_c_head)
        mrr_d_d_head = 1 / (rank_d_d_head)
        mrr_d_c_tail = 1 / (rank_d_c_tail)
        mrr_d_d_tail = 1 / (rank_d_d_tail)

        true_mr = (rank_d_c_head + rank_d_d_head + rank_d_c_tail + rank_d_d_tail) / 4
        true_mrr = (mrr_d_c_head + mrr_d_d_head + mrr_d_c_tail + mrr_d_d_tail) / 4

        true_auc = auc_from_mr(true_mr, len(self.entities_of_interest))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(mrr, true_mrr)

        diff_auc = abs(auc - true_auc)
        self.assertLess(diff_auc, allowed_diff)


    def test_filtering_head(self):
        evaluator = BaseRankingEvaluator(self.entities_of_interest_tensor, self.entities_of_interest_tensor, 2, "cpu")

        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.valid_set_tensor, mode="head_centric", filter_data=self.extra_set_tensor)
        mr = metrics["mr"]
        fmr = metrics["f_mr"]
        mrr = metrics["mrr"]
        fmrr = metrics["f_mrr"]
        fauc = metrics["f_auc"]
        
        rank_c_c = 2
        rank_c_d = 3
        f_rank_c_c = 1
        f_rank_c_d = 2

        mrr_c_c = 1 / (rank_c_c)
        mrr_c_d = 1 / (rank_c_d)
        f_mrr_c_c = 1 / (f_rank_c_c)
        f_mrr_c_d = 1 / (f_rank_c_d)
        
        true_mr = (rank_c_c + rank_c_d) / 2
        true_fmr = (f_rank_c_c + f_rank_c_d) / 2
        true_mrr = (mrr_c_c + mrr_c_d) / 2
        true_fmrr = (f_mrr_c_c + f_mrr_c_d) / 2

        true_fauc = auc_from_mr(true_fmr, len(self.entities_of_interest))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(fmr, true_fmr)
        self.assertEqual(mrr, true_mrr)
        self.assertEqual(fmrr, true_fmrr)

        diff_fauc = abs(fauc - true_fauc)
        self.assertLess(diff_fauc, allowed_diff)

    def test_filtering_tail(self):
        evaluator = BaseRankingEvaluator(self.entities_of_interest_tensor, self.entities_of_interest_tensor, 2, "cpu")

        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.test_set_tensor, mode="tail_centric", filter_data=self.train_set_tensor)
        mr = metrics["mr"]
        fmr = metrics["f_mr"]
        mrr = metrics["mrr"]
        fmrr = metrics["f_mrr"]
        fauc = metrics["f_auc"]
        
        rank_d_c = 3
        rank_d_d = 3
        f_rank_d_c = 2
        f_rank_d_d = 2

        mrr_d_c = 1 / (rank_d_c)
        mrr_d_d = 1 / (rank_d_d)
        f_mrr_d_c = 1 / (f_rank_d_c)
        f_mrr_d_d = 1 / (f_rank_d_d)
        
        true_mr = (rank_d_c + rank_d_d) / 2
        true_fmr = (f_rank_d_c + f_rank_d_d) / 2
        true_mrr = (mrr_d_c + mrr_d_d) / 2
        true_fmrr = (f_mrr_d_c + f_mrr_d_d) / 2

        true_fauc = auc_from_mr(true_fmr, len(self.entities_of_interest))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(fmr, true_fmr)
        self.assertEqual(mrr, true_mrr)
        self.assertEqual(fmrr, true_fmrr)

        diff_auc = abs(fauc - true_fauc)
        self.assertLess(diff_auc, allowed_diff)

        
    def test_filtering_both(self):
        evaluator = BaseRankingEvaluator(self.entities_of_interest_tensor, self.entities_of_interest_tensor, 2, "cpu")

        filter_data = th.cat([self.train_set_tensor, self.extra_set_tensor], dim=0)
        metrics = evaluator.compute_ranking_metrics(self.evaluation_model, self.valid_set_tensor, mode="both", filter_data=filter_data)
        mr = metrics["mr"]
        fmr = metrics["f_mr"]
        mrr = metrics["mrr"]
        fmrr = metrics["f_mrr"]
        fauc = metrics["f_auc"]
        
        rank_c_c_head = 2
        rank_c_d_head = 3
        rank_c_c_tail = 2
        rank_c_d_tail = 2

        f_rank_c_c_head = 1
        f_rank_c_d_head = 2
        f_rank_c_c_tail = 1
        f_rank_c_d_tail = 1

        mrr_c_c_head = 1 / (rank_c_c_head)
        mrr_c_d_head = 1 / (rank_c_d_head)
        mrr_c_c_tail = 1 / (rank_c_c_tail)
        mrr_c_d_tail = 1 / (rank_c_d_tail)

        f_mrr_c_c_head = 1 / (f_rank_c_c_head)
        f_mrr_c_d_head = 1 / (f_rank_c_d_head)
        f_mrr_c_c_tail = 1 / (f_rank_c_c_tail)
        f_mrr_c_d_tail = 1 / (f_rank_c_d_tail)

        true_mr = (rank_c_c_head + rank_c_d_head + rank_c_c_tail + rank_c_d_tail) / 4
        true_fmr = (f_rank_c_c_head + f_rank_c_d_head + f_rank_c_c_tail + f_rank_c_d_tail) / 4
        true_mrr = (mrr_c_c_head + mrr_c_d_head + mrr_c_c_tail + mrr_c_d_tail) / 4
        true_fmrr = (f_mrr_c_c_head + f_mrr_c_d_head + f_mrr_c_c_tail + f_mrr_c_d_tail) / 4

        true_fauc = auc_from_mr(true_fmr, len(self.entities_of_interest))
        
        self.assertEqual(mr, true_mr)
        self.assertEqual(fmr, true_fmr)
        self.assertEqual(mrr, true_mrr)
        self.assertEqual(fmrr, true_fmrr)

        diff_fauc = abs(fauc - true_fauc)
        self.assertLess(diff_fauc, allowed_diff)
