"""Regularisation as a component: penalties and projections on any EL model."""
from unittest import TestCase

import torch as th

from mowl.models import ELEmbeddings, ELBE
from mowl.training import (Regularization, NormPenalty, MaxNormPenalty, PowerPenalty,
                           SphereProjection, BallProjection, IntervalProjection,
                           ComposedRegime, EarlyStopping, embedding_tables, HOOKS)
from mowl.training.regime import unsupported_hooks
from mowl.models import Query2Box
from tests.datasetFactory import FamilyDataset


class TestRegularizers(TestCase):

    def setUp(self):
        th.manual_seed(0)
        self.table = th.nn.Embedding(5, 4)
        self.holder = th.nn.Module()
        self.holder.class_embed = self.table

    def test_norm_penalty_is_elem_regularisation(self):
        from mowl.baselines.nn.el.elem.losses import regularization_loss
        expected = regularization_loss(self.table, None, 1)
        self.assertTrue(th.allclose(NormPenalty().penalty(self.holder), expected))

    def test_penalties_vanish_on_the_allowed_set(self):
        with th.no_grad():
            self.table.weight.copy_(th.nn.functional.normalize(self.table.weight, dim=1) * 0.5)
        self.assertAlmostEqual(NormPenalty(target=0.5).penalty(self.holder).item(), 0.0, places=6)
        self.assertEqual(MaxNormPenalty(max_norm=0.5 + 1e-6).penalty(self.holder).item(), 0.0)
        self.assertGreater(MaxNormPenalty(max_norm=0.1).penalty(self.holder).item(), 0.0)
        self.assertAlmostEqual(PowerPenalty(weight=2.0).penalty(self.holder).item(), 0.5,
                               places=5)

    def test_projections(self):
        with th.no_grad():
            self.table.weight[0] = 0.01
            self.table.weight[1] = 5.0
        before = self.table.weight.detach().clone()
        BallProjection(max_norm=1.0).project(self.holder)
        norms = self.table.weight.norm(dim=1)
        self.assertTrue((norms <= 1.0 + 1e-6).all())
        self.assertTrue(th.equal(self.table.weight[0], before[0]))  # inside: untouched
        SphereProjection(radius=2.0).project(self.holder)
        self.assertTrue(th.allclose(self.table.weight.norm(dim=1), th.full((5,), 2.0)))
        IntervalProjection(low=0.0, high=0.5).project(self.holder)
        self.assertTrue(((self.table.weight >= 0) & (self.table.weight <= 0.5)).all())
        self.assertIsNone(SphereProjection().penalty(self.holder))

    def test_columns_restrict_the_constraint(self):
        before = self.table.weight.detach().clone()
        SphereProjection(columns=(0, 2)).project(self.holder)
        self.assertTrue(th.allclose(self.table.weight[:, :2].norm(dim=1), th.ones(5)))
        self.assertTrue(th.equal(self.table.weight[:, 2:], before[:, 2:]))

    def test_unknown_parameter_names_the_tables(self):
        with self.assertRaisesRegex(TypeError, "class_embed"):
            NormPenalty(parameters="nope").tensors(self.holder)
        self.assertEqual(embedding_tables(self.holder), ("class_embed",))
        with self.assertRaises(ValueError):
            IntervalProjection()


class TestRegularizationRegime(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()

    def _model(self, cls=ELEmbeddings):
        th.manual_seed(0)
        return cls(self.dataset, embed_dim=8)

    def test_hooks_are_declared(self):
        self.assertIn("regularization", HOOKS)
        self.assertIn("after_step", HOOKS)
        self.assertEqual(Regularization().overridden_hooks(), frozenset())
        self.assertEqual(Regularization(keep_default=False).overridden_hooks(),
                         {"regularization"})
        self.assertEqual(Regularization(SphereProjection(), NormPenalty()).overridden_hooks(),
                         {"regularization", "after_step"})

    def test_default_term_kept_or_dropped(self):
        model = self._model()
        own = model.get_regularization_loss()
        self.assertGreater(float(own), 0.0)
        self.assertTrue(th.equal(model.step_regularization_loss(), own))
        model.training_regime = Regularization(keep_default=False)
        self.assertEqual(float(model.step_regularization_loss()), 0.0)
        model.training_regime = Regularization(NormPenalty(target=3.0), keep_default=False)
        expected = NormPenalty(target=3.0).penalty(model)
        self.assertTrue(th.allclose(model.step_regularization_loss(), expected))
        model.training_regime = Regularization(NormPenalty(target=3.0))
        self.assertTrue(th.allclose(model.step_regularization_loss(), own + expected))

    def test_sphere_projection_holds_after_training(self):
        model = self._model()
        model.training_regime = Regularization(SphereProjection(), keep_default=False)
        model.train(epochs=3)
        norms = model.module.class_embed.weight.norm(dim=1)
        self.assertTrue(th.allclose(norms, th.ones_like(norms), atol=1e-5))

    def test_without_penalty_norms_are_free(self):
        model = self._model()
        model.training_regime = Regularization(keep_default=False)
        model.train(epochs=3)
        norms = model.module.class_embed.weight.norm(dim=1)
        self.assertFalse(th.allclose(norms, th.ones_like(norms), atol=1e-5))

    def test_penalty_on_another_method(self):
        model = self._model(ELBE)
        model.training_regime = Regularization(BallProjection(max_norm=0.3))
        model.train(epochs=2)
        self.assertTrue((model.module.class_embed.weight.norm(dim=1) <= 0.3 + 1e-5).all())

    def test_composes_with_other_regimes(self):
        model = self._model()
        own = model.get_regularization_loss()
        model.training_regime = ComposedRegime(
            Regularization(NormPenalty(target=3.0)), EarlyStopping(patience=2),
            Regularization(SphereProjection(radius=2.0)))
        self.assertTrue(th.allclose(
            model.step_regularization_loss(),
            own + NormPenalty(target=3.0).penalty(model)))
        model.train(epochs=1)
        norms = model.module.class_embed.weight.norm(dim=1)
        self.assertTrue(th.allclose(norms, th.full_like(norms, 2.0), atol=1e-5))

    def test_wrong_table_fails_on_assignment(self):
        model = self._model()
        with self.assertRaisesRegex(TypeError, "embedding tables"):
            model.training_regime = Regularization(NormPenalty(parameters="no_such_table"))

    def test_refused_by_loops_that_would_ignore_it(self):
        self.assertEqual(unsupported_hooks(Regularization(SphereProjection()), Query2Box),
                         {"after_step"})
