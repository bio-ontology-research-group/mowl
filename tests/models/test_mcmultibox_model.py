"""End-to-end tests of VolumeMultiBoxEL (Aharoni 2024), kept very small."""
from unittest import TestCase

import torch as th

from tests.datasetFactory import FamilyDataset
from mowl.models.mcmultibox.model import VolumeMultiBoxEL
from mowl.nn.el.mcmultibox.module import VolumeMultiBoxELModule


class TestVolumeMultiBoxEL(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()
        cls.model = VolumeMultiBoxEL(cls.dataset, embed_dim=3, num_boxes=2, num_points=512,
                                     num_batches=2, learning_rate=0.05, seed=1)
        cls.model.train(epochs=6)

    def test_module_type_and_shapes(self):
        m = self.model.module
        self.assertIsInstance(m, VolumeMultiBoxELModule)
        self.assertEqual(m.class_embed.shape, (len(self.model.class_index_dict), 2, 6))
        self.assertEqual(m.rel_embed.shape,
                         (len(self.model.object_property_index_dict), 2, 12))

    def test_loss_decreases_and_parameters_are_finite(self):
        losses = self.model.training_losses
        self.assertEqual(len(losses), 6)
        self.assertLess(losses[-1], losses[0])
        for p in self.model.module.parameters():
            self.assertTrue(th.isfinite(p).all())

    def test_only_thesis_normal_forms_are_trained(self):
        names = set(self.model._trainable_datasets())
        self.assertTrue(names <= set(VolumeMultiBoxEL.TRAINABLE_GCIS))
        self.assertIn("gci0", names)

    def test_evaluation_distance(self):
        self.model.eval_gci_name = "gci0"
        data = self.model.training_datasets["gci0"][:][:20]
        dist = self.model.eval_method(data)
        self.assertEqual(dist.shape, (len(data), 1))
        self.assertTrue(((dist >= 0) & (dist <= 1)).all())
        same = th.stack([data[:, 0], data[:, 0]], dim=1)
        self.assertTrue((self.model.evaluation_model(same, "gci0") == 0).all())
        # deterministic
        self.assertTrue(th.equal(dist, self.model.eval_method(data)))

    def test_role_axioms_option(self):
        model = VolumeMultiBoxEL(self.dataset, embed_dim=2, num_boxes=2, num_points=128,
                                 num_batches=1, load_role_axioms=True)
        model.train(epochs=1)
        self.assertTrue(model.module.role_axiom_capable)

    def test_training_is_reproducible(self):
        def run():
            m = VolumeMultiBoxEL(self.dataset, embed_dim=2, num_boxes=2, num_points=128,
                                 num_batches=2, seed=4)
            m.train(epochs=2)
            return m.module.class_embed.detach().clone()
        self.assertTrue(th.allclose(run(), run()))

    def test_training_in_row_chunks_gives_the_same_model(self):
        """Bounded memory (a few axioms at a time) trains the same model as whole batches."""
        def run(budget):
            m = VolumeMultiBoxEL(self.dataset, embed_dim=2, num_boxes=2, num_points=128,
                                 num_batches=1, seed=4, max_chunk_elements=budget)
            self.assertEqual(m.module.max_chunk_elements, budget)
            m.train(epochs=2)
            return m.module.class_embed.detach().clone()
        # 128 points x 2 boxes x 2 coordinates: 3 axioms per chunk
        self.assertTrue(th.allclose(run(None), run(3 * 128 * 2 * 2), atol=1e-6, rtol=0))
