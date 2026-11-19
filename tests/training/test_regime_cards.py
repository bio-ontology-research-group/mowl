"""The regime cards of the training/evaluation classification load, and every implemented
regime class declares the key of its card."""
import inspect
from unittest import TestCase

from mowl.catalog.registry import list_regimes

KEYS = (
    "fixed-column-corruption", "either-side-corruption", "per-column-corruption",
    "type-constrained-corruption", "known-positive-filtering", "sibling-hard-negatives",
    "contradiction-negatives", "existential-concept-negatives",
    "transitive-reduction-positives", "transitive-closure-positives", "loss-sum",
    "mse-to-target", "pairwise-margin-ranking", "logsigmoid-negative-sampling", "bpr-loss",
    "logsumexp-contrastive", "pearson-correlation-loss", "probabilistic-bce",
    "normal-form-frequency-weighting", "step-lr-decay", "cosine-lr-decay", "cyclic-lr",
    "reduce-lr-on-plateau", "early-stopping", "per-normal-form-minibatch",
    "validation-loss-selection", "validation-rank-selection", "loss-scoring",
    "centre-distance-scoring", "margin-free-loss-scoring", "region-inclusion-accuracy",
    "intersection-equivalence-ranking", "complex-axiom-prediction",
    "paraconsistent-entailment-classification", "protein-centric-threshold-f1",
    "hierarchy-threshold-f1", "sampled-negative-auc", "kg-filtered-link-prediction",
    "query-answering")


class TestRegimeCards(TestCase):

    def test_cards_load_and_keys_match(self):
        cards = {c.key: c for c in list_regimes()}
        for key in KEYS:
            self.assertIn(key, cards)
            card = cards[key]
            if not card.implemented:
                self.assertTrue(card.not_implemented_reason)
                continue
            obj = card.load()
            if inspect.isclass(obj):
                self.assertEqual(obj.key, key, msg=card.implementation)
