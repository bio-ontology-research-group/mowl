"""ComposedEL: a model assembled from parts."""
from unittest import TestCase

import torch as th

from mowl.models import ComposedEL, ELBE, ELEmbeddings
from mowl.representation import (Objective, ComposedGeometry, Bounds, CentreDistance,
                                 geometry_of)
from mowl.training import Regularization, SphereProjection
from tests.datasetFactory import FamilyDataset


class TestComposedEL(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def _same_losses(self, original, composed, tables):
        with th.no_grad():
            for name in tables:
                getattr(composed.module, name[1]).weight.copy_(
                    getattr(original.module, name[0]).weight)
        for form, data in original.training_datasets.items():
            if len(data) == 0:
                continue
            rows = data[:]
            for neg in (False, True) if form == "gci2" else (False,):
                a = original.module(rows, form, neg=neg).reshape(-1, 1)
                b = composed.module(rows, form, neg=neg)
                self.assertTrue(th.allclose(a, b, atol=1e-6), (form, neg))
        self.assertTrue(th.allclose(
            th.as_tensor(original.get_regularization_loss(), dtype=th.float),
            th.as_tensor(composed.get_regularization_loss(), dtype=th.float), atol=1e-6))

    def test_elbe_from_parts_equals_elbe(self):
        original = ELBE(self.dataset, embed_dim=8, margin=0.05)
        composed = ComposedEL(self.dataset, method="elbe", embed_dim=8, margin=0.05)
        with th.no_grad():
            composed.module.class_model.offsets.weight.copy_(
                original.module.class_offset.weight)
        self._same_losses(original, composed, [("class_embed", "class_embed"),
                                                ("rel_embed", "rel_embed")])

    def test_elem_from_parts_equals_elem(self):
        original = ELEmbeddings(self.dataset, embed_dim=8, margin=0.05)
        composed = ComposedEL(self.dataset, method="elem", embed_dim=8, margin=0.05)
        with th.no_grad():
            composed.module.class_model.radii.weight.copy_(original.module.class_rad.weight)
        self._same_losses(original, composed, [("class_embed", "class_embed"),
                                                ("rel_embed", "rel_embed")])

    def test_trains(self):
        for kwargs in ({"method": "elbe"}, {"method": "elem"},
                       {"classes": "ball"}, {"classes": "box"},
                       {"method": "elbe", "gci3": "inclusion"}):
            th.manual_seed(0)
            model = ComposedEL(self.dataset, embed_dim=8, margin=0.05, **kwargs)
            before = sum(float(model.module(d[:], f).mean())
                         for f, d in model.training_datasets.items() if len(d))
            model.train(epochs=40)
            after = sum(float(model.module(d[:], f).mean())
                        for f, d in model.training_datasets.items() if len(d))
            self.assertLess(after, before, kwargs)
            self.assertIsInstance(model.as_geometry, ComposedGeometry)

    def test_new_combination(self):
        # Boxes with the semantic objective and a slack margin: not a published method.
        model = ComposedEL(self.dataset, classes="box", measure=Bounds(),
                           objective=Objective(0.02, slack=True), embed_dim=8)
        self.assertEqual(model.module.parts["classes"], "box")
        self.assertEqual(model.module.margin_semantics, "slack")
        model.train(epochs=2)

    def test_parts_must_fit(self):
        with self.assertRaisesRegex(TypeError, "Ball"):
            ComposedEL(self.dataset, classes="box", measure=CentreDistance(), embed_dim=8)
        with self.assertRaisesRegex(ValueError, "available"):
            ComposedEL(self.dataset, classes="blob", embed_dim=8)
        with self.assertRaisesRegex(ValueError, "available"):
            ComposedEL(self.dataset, method="nope", embed_dim=8)

    def test_takes_regimes(self):
        model = ComposedEL(self.dataset, method="elem", embed_dim=8)
        model.training_regime = Regularization(SphereProjection(), keep_default=False)
        model.train(epochs=2)
        norms = model.module.class_embed.weight.norm(dim=1)
        self.assertTrue(th.allclose(norms, th.ones_like(norms), atol=1e-5))

    def test_existing_models_have_a_geometry(self):
        model = ELBE(self.dataset, embed_dim=8)
        self.assertIs(model.as_geometry.class_embed, model.module.class_embed)
        self.assertEqual(geometry_of(model).parts["measure"], "bounds per coordinate")
