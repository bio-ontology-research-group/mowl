"""The direct query models trained with parts that share their parameters (the bridge from the
direct implementation to the parts)."""
from unittest import TestCase

import numpy as np
import torch as th

from mowl.baselines.models.acone import AConE
from mowl.baselines.models.betae import BetaE
from mowl.baselines.models.cone_qe import ConE
from mowl.baselines.models.query2box import Query2Box
from mowl.representation import ConstructorUnavailable, geometry_of
from mowl.representation.library.query_methods import (
    ComposedQueryGeometry, acone, cone_qe, query2box, query_geometry)
from tests.datasetFactory import FamilyDataset


def _train_on_one_batch(model, steps=60, lr=1e-2):
    """Loss of a fixed batch of queries before and after Adam steps on it."""
    groups = model._batch()
    optimiser = th.optim.Adam(model.module.parameters(), lr=lr)
    first = None
    for _ in range(steps):
        optimiser.zero_grad()
        loss = model.loss(groups)
        first = float(loss) if first is None else first
        loss.backward()
        optimiser.step()
    return first, float(model.loss(groups))


class TestQueryModelsOnParts(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def make(self, cls, **kwargs):
        th.manual_seed(0)
        return cls(self.dataset, embed_dim=8, batch_size=24, negatives=4, seed=0, **kwargs)

    def test_geometry_of_a_model_and_query_geometry(self):
        for cls in (Query2Box, BetaE, ConE, AConE):
            model = self.make(cls)
            g = geometry_of(model.module)
            self.assertIsInstance(g, ComposedQueryGeometry)
            self.assertEqual(g.classes(th.tensor([0, 1])).centre.shape[0], 2)
            with self.assertRaisesRegex(ConstructorUnavailable, "number of classes"):
                g.individuals(th.tensor([0]))
            g = query_geometry(model)
            self.assertEqual(g.num_classes, model.num_classes)
            if model.graph.num_entities > model.num_classes:
                ids = th.tensor([0])
                state = g.classes(th.tensor([0]))
                with th.no_grad():
                    want = model.module.distance(
                        tuple(getattr(state, f) for f in state.__dataclass_fields__),
                        (ids + model.num_classes).unsqueeze(1))
                    self.assertTrue(th.allclose(g.member(ids, state), want, atol=1e-6))

    def test_a_model_trains_with_the_rebuilt_geometry(self):
        for cls in (Query2Box, BetaE, ConE, AConE):
            model = self.make(cls)
            model.module = geometry_of(model.module)
            first, last = _train_on_one_batch(model)
            self.assertLess(last, first, cls.__name__)
            self.assertTrue(all(np.isfinite(float(p.sum())) for p in model.module.parameters()))

    def test_the_rebuilt_geometry_trains_the_original_parameters(self):
        model = self.make(Query2Box)
        original = model.module
        entity_before = original.entity_embedding.detach().clone()
        model.module = geometry_of(original)
        _train_on_one_batch(model, steps=5)
        self.assertFalse(th.equal(entity_before, original.entity_embedding))

    def test_query2box_with_exact_intersection_trains(self):
        model = self.make(Query2Box)
        model.module = query2box(tables=model.module, intersection="exact")
        self.assertEqual(model.module.parts["conjunction"], "exact")
        first, last = _train_on_one_batch(model)
        self.assertLess(last, first)

    def test_cone_with_the_acone_relation_model_trains(self):
        model = self.make(ConE)
        model.module = cone_qe(tables=model.module, relations="rotation")
        first, last = _train_on_one_batch(model, lr=3e-3)
        self.assertLess(last, first)
        model = self.make(AConE)
        model.module = acone(tables=model.module)
        first, last = _train_on_one_batch(model, lr=3e-3)
        self.assertLess(last, first)
