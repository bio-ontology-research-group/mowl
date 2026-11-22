"""Query models trained with rebuilt parts, and the query parts inside ComposedEL."""
from unittest import TestCase

import numpy as np
import torch as th

from mowl.models import AConE, BetaE, ComposedEL, ConE, Query2Box
from mowl.representation import (BoxClasses, ComposedGeometry, Bounds, Objective,
                                 ConstructorUnavailable, geometry_of)
from mowl.representation.library.query_methods import (
    ComposedQueryGeometry, QueryClasses, acone, cone_qe, query2box, query_geometry)
from mowl.representation.library.boxrel_relations import TranslationBoxes
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


def _loss(model):
    return sum(float(model.module(d[:], f).mean())
               for f, d in model.training_datasets.items() if len(d))


class TestPartsInComposedEL(TestCase):
    """The parts of the query methods as parts of an EL model trained on normal-form
    axioms."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def trains(self, model):
        before = _loss(model)
        model.train(epochs=40)
        self.assertLess(_loss(model), before)

    def test_box_of_translations_with_boxes_and_with_balls(self):
        for classes in ("box", "ball"):
            th.manual_seed(0)
            model = ComposedEL(
                self.dataset, classes=classes, embed_dim=8, margin=0.05,
                relations=lambda n, d: TranslationBoxes(n, d, activation="relu",
                                                        init_range=0.1))
            self.assertEqual(model.module.parts["relations"], "box of translations")
            self.trains(model)

    def test_shift_and_widen_is_exists_as_image_under_the_inverse(self):
        relation = TranslationBoxes(2, 3, activation="relu", init_range=0.1)
        region = BoxClasses(4, 3)(th.tensor([0, 1]))
        ids = th.tensor([0, 1])
        image = relation.image(ids, region)
        self.assertTrue(th.allclose(image.offset, region.offset + relation.offsets(ids)
                                    .relu()))
        back = relation.preimage(ids, image)
        # widening is not undone: the preimage of the image contains the region
        self.assertTrue(bool((back.offset >= region.offset - 1e-6).all()))
        self.assertTrue(th.allclose(back.centre, region.centre, atol=1e-6))

    def test_sector_cones_with_the_rotation(self):
        th.manual_seed(0)
        model = ComposedEL(self.dataset, classes="sector cone", relations="angle rotation",
                           embed_dim=8, margin=0.05)
        self.assertEqual(model.module.parts["classes"], "sector cone")
        self.assertEqual(model.module.parts["measure"], "angle")
        # no cone intersection: the axioms with a conjunction cannot be trained
        with self.assertRaises(ConstructorUnavailable):
            model.module(model.training_datasets["gci1"][:], "gci1")
        for form in [f for f in model.training_datasets if "gci1" in f]:
            model._training_datasets.pop(form)
        self.trains(model)

    def test_query_geometry_as_class_and_relation_model(self):
        """The class model is the extension of an anchor and the relation model the box of
        translations of a Query2Box geometry; the EL model trains, but the classes cannot
        differ in size."""
        th.manual_seed(0)
        sizes = (len(self.dataset.classes), len(self.dataset.object_properties))
        made = []

        def classes(size, dim):
            if not made:
                made.append(query2box(sizes[0], sizes[1], 0, embed_dim=dim, gamma=2.0))
                return QueryClasses(made[0])
            return BoxClasses(size, dim)           # individuals: plain boxes

        def relations(size, dim):
            return made[0].relations

        model = ComposedEL(self.dataset, classes=classes, relations=relations,
                           objective=Objective(0.05), embed_dim=8)
        self.assertIsInstance(model.module, ComposedGeometry)
        offsets = model.module.classes(th.arange(sizes[0])).offset
        self.assertTrue(th.allclose(offsets, offsets[:1].expand_as(offsets)))   # one size
        before = [p.detach().clone() for p in made[0].entities.parameters()]
        for form in [f for f in model.training_datasets if "gci1" in f]:
            model._training_datasets.pop(form)
        self.trains(model)
        self.assertFalse(th.equal(before[0], made[0].entities.table.weight))
        offsets = model.module.classes(th.arange(sizes[0])).offset
        self.assertTrue(th.allclose(offsets, offsets[:1].expand_as(offsets)))   # still
        self.assertTrue(model.module.measure.exact_intersection)
        self.assertIsInstance(model.module.measure, Bounds)
