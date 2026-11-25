"""The regimes of the structural and point readings (mowl.training.structural): what the
models MultiBoxEL and MultiELBE did in their own ``train`` and ``additional_training_loss``,
now regimes that other geometries can take."""
import os
import random
from unittest import TestCase

import numpy as np
import torch as th

import mowl
mowl.init_jvm("1g")

from mowl.datasets import PathDataset  # noqa: E402
from mowl.models import MultiBoxEL, MultiELBE  # noqa: E402
from mowl.training import IdleComponentReseeding  # noqa: E402
from mowl.training.structural import (ExtraPenalties, SampledPointLoss,  # noqa: E402
                                      StructuralTables)

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "geometric", "fixtures")


def _seed():
    random.seed(0)
    np.random.seed(0)
    th.manual_seed(0)


def _t3():
    return PathDataset(os.path.join(FIXTURES, "t3.owl"))


class TestStructuralRegimes(TestCase):

    def model(self, **options):
        _seed()
        return MultiBoxEL(_t3(), embed_dim=4, n_components=2, objective="structural",
                          margin=0.1, max_components=8, neg_sampling_gcis=[], **options)

    def test_the_reading_attaches_its_regimes(self):
        model = self.model()
        kinds = [type(r) for r in model.training_regime.regimes]
        self.assertEqual(kinds, [StructuralTables, ExtraPenalties])
        self.assertIsNone(model.additional_training_loss(0))

    def test_tables_are_installed_when_training_starts(self):
        model = self.model()
        self.assertIsNone(model.module.definition_table)
        model.training_regime.training_datasets(model.training_datasets)
        self.assertIsNotNone(model.module.definition_table)
        self.assertGreater(model.module.class_model.n_defined, 0)
        self.assertIsNotNone(model._structural_tables)

    def test_extra_penalties_are_the_penalties_of_the_reading(self):
        model = self.model(witness_weight=0.5)
        model.training_regime.training_datasets(model.training_datasets)
        extra = ExtraPenalties()
        extra.attach(model)
        self.assertAlmostEqual(float(extra.additional_loss(0)),
                               float(model.module.structural_extra_loss()), places=6)
        self.assertGreater(float(extra.additional_loss(0)), 0.0)

    def test_an_assigned_regime_is_added_and_one_of_the_same_class_replaces(self):
        model = self.model()
        reseed = IdleComponentReseeding(every=1)
        model.training_regime = reseed
        kinds = [type(r) for r in model.training_regime.regimes]
        self.assertEqual(kinds, [StructuralTables, ExtraPenalties, IdleComponentReseeding])
        mine = StructuralTables()
        model.training_regime = mine
        self.assertIn(mine, model.training_regime.regimes)
        self.assertEqual(sum(isinstance(r, StructuralTables)
                             for r in model.training_regime.regimes), 1)
        model.training_regime = None
        self.assertIsNone(model.training_regime)

    def test_trains_with_reseeding_and_keeps_the_tables(self):
        model = self.model()
        reseed = IdleComponentReseeding(every=1)
        model.training_regime = reseed
        model.train(epochs=2, validate_every=10)
        self.assertEqual([e for e, _ in reseed.history], [0, 1])
        self.assertIsNotNone(model.module.definition_table)

    def test_refused_by_a_geometry_without_the_hooks(self):
        _seed()
        axiom = MultiBoxEL(_t3(), embed_dim=4, neg_sampling_gcis=[])
        self.assertIsNone(axiom.training_regime)
        with self.assertRaises(TypeError):
            ExtraPenalties().attach(MultiELBE(_t3(), embed_dim=4))


class TestSampledPointLoss(TestCase):

    def test_the_point_readings_take_it_and_it_matches_the_direct_loss(self):
        from mowl.baselines.models.multiboxel.model import MultiBoxEL as Direct
        for cls in (MultiBoxEL, MultiELBE):
            _seed()
            model = cls(_t3(), embed_dim=4, n_components=2, objective="point",
                        points_per_epoch=64, points_seed=1)
            regime = model.training_regime
            self.assertIsInstance(regime, SampledPointLoss)
            loss = regime.additional_loss(0)
            self.assertTrue(bool(th.isfinite(loss)))
            self.assertTrue(loss.requires_grad)
        _seed()
        old = Direct(_t3(), embed_dim=4, n_components=2, objective="point",
                     points_per_epoch=64, points_seed=1)
        _seed()
        new = MultiBoxEL(_t3(), embed_dim=4, n_components=2, objective="point",
                         points_per_epoch=64, points_seed=1)
        new.module.load_state_dict(old.module.state_dict())
        old._ensure_point_supervision()
        new._ensure_point_supervision()
        new.module.point_terms.elem_embed.weight.data.copy_(old.module.elem_embed.weight)
        a = old.additional_training_loss(3)
        b = new.training_regime.additional_loss(3)
        self.assertAlmostEqual(float(a), float(b), places=5)

    def test_axiom_reading_has_no_regime(self):
        _seed()
        self.assertIsNone(MultiELBE(_t3(), embed_dim=4).training_regime)
