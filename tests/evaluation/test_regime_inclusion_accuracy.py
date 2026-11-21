"""The geometric inclusion accuracy of EmEL++/BoxEL and the margin-free loss scorer."""
from unittest import TestCase

import torch as th

from mowl.evaluation.inclusion_accuracy import InclusionAccuracyRegime, inclusion_mask
from mowl.evaluation.scoring_rules import margin_free_loss_scores
from mowl.evaluation.el_scoring import loss_scores
from tests.datasetFactory import FamilyDataset


class TestInclusionAccuracy(TestCase):

    def test_ball_and_box_inclusion_by_hand(self):
        from mowl.models import ELEmbeddings, ELBE
        dataset = FamilyDataset()
        model = ELEmbeddings(dataset, embed_dim=2, batch_size=64)
        with th.no_grad():
            model.module.class_embed.weight.zero_()
            model.module.class_rad.weight.copy_(
                th.arange(model.module.class_rad.weight.shape[0]).float().unsqueeze(1))
        rows = th.tensor([[1, 3], [3, 1], [2, 2]])
        self.assertEqual(inclusion_mask(model, rows).tolist(), [True, False, True])

        elbe = ELBE(dataset, embed_dim=2)
        with th.no_grad():
            elbe.module.class_embed.weight.zero_()
            elbe.module.class_offset.weight.copy_(
                th.arange(elbe.module.class_offset.weight.shape[0]).float().unsqueeze(1)
                .repeat(1, 2))
        self.assertEqual(inclusion_mask(elbe, rows).tolist(), [True, False, True])

    def test_regime(self):
        from mowl.models import ELBE
        th.manual_seed(0)
        dataset = FamilyDataset()
        model = ELBE(dataset, embed_dim=4)
        model.train(epochs=2, validate_every=100)
        res = InclusionAccuracyRegime(splits=("train",)).evaluate(model, dataset)
        self.assertTrue(0.0 <= res["train"]["accuracy"] <= 1.0)
        self.assertGreater(res["train"]["num_axioms"], 0)


class TestMarginFreeScores(TestCase):

    def test_margin_restored_and_used(self):
        from mowl.models import ELEmbeddings
        model = ELEmbeddings(FamilyDataset(), embed_dim=4, batch_size=64, margin=0.5)
        rows = model.training_datasets["gci0"][:]
        before = model.module.margin
        free = margin_free_loss_scores(model, "gci0", rows)
        self.assertEqual(model.module.margin, before)
        model.module.margin = 0.0
        ref = loss_scores(model, "gci0", rows)
        model.module.margin = before
        self.assertTrue(th.allclose(free, ref))
