"""DELE evaluation regime (mowl.evaluation.deductive)."""
from unittest import TestCase

import numpy as np
import torch as th
from java.util import HashSet

from mowl.datasets import Dataset
from mowl.evaluation.base import compute_rank_roc
from mowl.evaluation.deductive import (DELEEvaluation, DeductiveRankingEvaluator, rank_auc,
                                       summarise_ranks, tie_rank)
from mowl.evaluation.regime import EvaluationRegime
from mowl.owlapi import OWLAPIAdapter

NS = "http://test.deductive/"


def _ont(iri, pairs):
    ad = OWLAPIAdapter()
    ont = ad.create_ontology(iri)
    axioms = HashSet()
    for a, b in pairs:
        axioms.add(ad.create_subclass_of(ad.create_class(NS + a), ad.create_class(NS + b)))
    ad.owl_manager.addAxioms(ont, axioms)
    return ont


class MatrixScorer(th.nn.Module):
    """Scores (lower is better) read from a fixed class x class matrix."""

    def __init__(self, matrix):
        super().__init__()
        self.matrix = th.as_tensor(matrix, dtype=th.float)

    def forward(self, data, gci_name):
        assert gci_name == "gci0"
        return self.matrix[data[:, 0], data[:, -1]].unsqueeze(1)


class TestHelpers(TestCase):

    def test_tie_rank(self):
        s = th.tensor([0.0, 0.0, 1.0, 0.0, 2.0])
        self.assertEqual(tie_rank(s, 0), 2.0)       # three-way tie at the top
        self.assertEqual(tie_rank(s, 2), 4.0)
        self.assertEqual(tie_rank(s, 0, tie_method="min"), 1.0)
        self.assertEqual(tie_rank(s, 0, tie_method="max"), 3.0)
        valid = th.tensor([True, False, True, False, True])
        self.assertEqual(tie_rank(s, 2, valid=valid), 2.0)
        self.assertEqual(tie_rank(s, 1, valid=valid), 1.5)  # the target always counts

    def test_rank_auc_matches_compute_rank_roc(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            ranks = rng.integers(1, 50, size=30)
            counts = {}
            for r in ranks.tolist():
                counts[r] = counts.get(r, 0) + 1
            self.assertAlmostEqual(rank_auc(ranks, 50), compute_rank_roc(counts, 50))

    def test_macro_micro(self):
        m = summarise_ranks([1, 3, 10], heads=[0, 0, 1], num_candidates=11, hits_at=(1, 10))
        self.assertAlmostEqual(m["macro_mr"], 14 / 3)
        self.assertAlmostEqual(m["micro_mr"], (2 + 10) / 2)
        self.assertAlmostEqual(m["hits@1"], 1 / 3)
        self.assertAlmostEqual(m["hits@10"], 1.0)
        self.assertAlmostEqual(m["macro_auc"], np.mean([(11 - r) / 10 for r in (1, 3, 10)]))
        self.assertAlmostEqual(m["micro_auc"], np.mean([np.mean([1.0, 0.8]), 0.1]))


class TestDELEEvaluation(TestCase):

    @classmethod
    def setUpClass(cls):
        # train: A ⊑ B ⊑ C; test: A ⊑ D (novel) and A ⊑ C (entailed by train)
        train = _ont(NS + "train", [("A", "B"), ("B", "C"), ("D", "E")])
        test = _ont(NS + "test", [("A", "D"), ("A", "C")])
        cls.dataset = Dataset(train, testing=test)
        cls.classes = [NS + c for c in "ABCDE"]
        ids = {c: i for i, c in enumerate(cls.dataset.classes.as_str)}
        cls.ids = ids
        n = len(ids)
        mat = np.full((n, n), 5.0)
        a = ids[NS + "A"]
        # For A: B and C (entailed) score best, then D (the test answer), then the rest
        mat[a, ids[NS + "B"]] = 0.0
        mat[a, ids[NS + "C"]] = 0.0
        mat[a, ids[NS + "A"]] = 0.5
        mat[a, ids[NS + "D"]] = 1.0
        cls.scorer = MatrixScorer(mat)

    def _regime(self, **kw):
        return DELEEvaluation(gci_name="gci0", heads=self.classes, tails=self.classes, **kw)

    def test_is_evaluation_regime(self):
        self.assertIsInstance(self._regime(), EvaluationRegime)

    def test_entailed_test_axioms_removed(self):
        regime = self._regime()
        out = regime.evaluate(self.scorer, self.dataset)["all"]
        self.assertEqual(out["num_test"], 1)  # A ⊑ C was entailed and removed
        a, d = self.ids[NS + "A"], self.ids[NS + "D"]
        self.assertEqual(regime.last_test_data.tolist(), [[a, d]])
        # NF: B, C, A score better than D -> rank 4; F: A, B, C are entailed -> rank 1;
        # E is not entailed (D ⊑ E does not make A ⊑ E) and scores worse.
        self.assertEqual(out["macro_mr"], 4.0)
        self.assertEqual(out["f_macro_mr"], 1.0)
        self.assertEqual(out["nf_minus_f_macro_mr"], 3.0)
        self.assertEqual(out["f_hits@1"], 1.0)
        self.assertEqual(out["hits@1"], 0.0)
        self.assertAlmostEqual(out["macro_auc"], (5 - 4) / 4)

    def test_keep_entailed_and_no_filter(self):
        out = self._regime(remove_entailed_test=False, filter_with=None).evaluate(
            self.scorer, self.dataset)["all"]
        self.assertEqual(out["num_test"], 2)
        self.assertEqual(out["macro_mr"], out["f_macro_mr"])
        # A ⊑ C ties with A ⊑ B at the top: average rank 1.5; A ⊑ D rank 4
        self.assertAlmostEqual(out["macro_mr"], (1.5 + 4) / 2)

    def test_filter_with_train(self):
        out = self._regime(filter_with="train").evaluate(self.scorer, self.dataset)["all"]
        self.assertEqual(out["f_macro_mr"], 1.0)

    def test_closure_as_test_set(self):
        regime = self._regime(test_set="closure")
        out = regime.evaluate(self.scorer, self.dataset)["all"]
        a, c = self.ids[NS + "A"], self.ids[NS + "C"]
        self.assertEqual(regime.last_test_data.tolist(), [[a, c]])
        self.assertEqual(out["num_test"], 1)

    def test_evaluator_with_explicit_filter_tensor(self):
        ev = DeductiveRankingEvaluator(self.dataset, "gci0", heads=self.classes,
                                       tails=self.classes)
        a, b, c, d = (self.ids[NS + x] for x in "ABCD")
        test = th.tensor([[a, d]])
        out = ev.compute_ranking_metrics(self.scorer, test,
                                         filter_data=th.tensor([[a, b], [a, c]]))
        self.assertEqual(out["macro_mr"], 4.0)
        self.assertEqual(out["f_macro_mr"], 2.0)  # A itself is not filtered
