from unittest import TestCase
import pytest
import torch as th
from tests.datasetFactory import FamilyDataset, PPIYeastSlimDataset
from mowl.models import TransBox
from mowl.nn import TransBoxModule


def _assert_trained(test_case, model, embed_dim):
    first_param = next(model.module.parameters())
    test_case.assertEqual(first_param.shape[-1], embed_dim)
    test_case.assertTrue(
        th.isfinite(first_param).all(),
        "Model parameters contain NaN or Inf after training"
    )


class TestTransBox(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = FamilyDataset()
        cls.model = TransBox(cls.dataset, embed_dim=30)
        cls.model.train(epochs=1)

    def test_initialization(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.module)

    def test_module_declares_role_axiom_capable(self):
        """The module implements the role axiom losses, so it must declare the flag.

        Without it, a model built with ``load_role_axioms=True`` would raise
        ``NotImplementedError`` at the start of training even though the losses
        exist.
        """
        self.assertTrue(self.model.module.role_axiom_capable)

    def test_parameters_finite_and_correct_dim(self):
        _assert_trained(self, self.model, embed_dim=30)

    def test_embedding_shapes(self):
        embed_dim = 30
        nb_classes = len(self.model.class_index_dict)
        nb_rels = len(self.model.object_property_index_dict)
        module = self.model.module
        self.assertEqual(module.class_center_embedding.weight.shape, (nb_classes, embed_dim))
        self.assertEqual(module.class_offset_embedding.weight.shape, (nb_classes, embed_dim))
        self.assertEqual(module.relation_center_embedding.weight.shape, (nb_rels, embed_dim))
        self.assertEqual(module.relation_offset_embedding.weight.shape, (nb_rels, embed_dim))

    def test_negative_sampling_corrupts_both_sides_of_gci2(self):
        """The paper negates C ⊑ ∃R.D on both sides (C and D), not D only."""
        config = self.model.get_negative_sampling_config()
        self.assertEqual(
            config["gci2"], {"index_pool": "classes", "corrupt_column": [0, 2]})

        data = self.model.training_datasets["gci2"][:]
        neg = self.model.generate_negatives("gci2", self.model.training_datasets["gci2"])
        n = len(data)
        self.assertEqual(neg.shape, (2 * n, 3))
        class_ids = set(self.model.class_index_dict.values())
        block_c, block_d = neg[:n], neg[n:]
        # Block C: role and D untouched, C from the class pool
        self.assertTrue(th.equal(block_c[:, 1], data[:, 1]))
        self.assertTrue(th.equal(block_c[:, 2], data[:, 2]))
        self.assertTrue(set(block_c[:, 0].tolist()) <= class_ids)
        # Block D: class C and role untouched, D from the class pool
        self.assertTrue(th.equal(block_d[:, 0], data[:, 0]))
        self.assertTrue(th.equal(block_d[:, 1], data[:, 1]))
        self.assertTrue(set(block_d[:, 2].tolist()) <= class_ids)

    @pytest.mark.slow
    def test_trains_with_role_axioms(self):
        """TransBox can train on the EL++ role axioms (PPI yeast has 3 role
        inclusions and 6 role chains), exercising ``role_inclusion_loss`` and
        ``role_chain_loss`` end to end."""
        model = TransBox(PPIYeastSlimDataset(), embed_dim=30, load_role_axioms=True)
        self.assertTrue(model.load_role_axioms)
        self.assertIn("role_inclusion", model.training_datasets)
        self.assertIn("role_chain", model.training_datasets)
        model.train(epochs=1, validate_every=2)
        _assert_trained(self, model, embed_dim=30)


class TestTransBoxLosses(TestCase):
    """Unit tests for the TransBox loss functions on hand-crafted boxes."""

    @classmethod
    def setUpClass(cls):
        cls.module = TransBoxModule(nb_ont_classes=4, nb_rels=2, embed_dim=3,
                                    margin=0.0, use_enhancement=True)

    def _set_class_box(self, idx, center, offset):
        self.module.class_center_embedding.weight.data[idx] = th.tensor(center, dtype=th.float32)
        self.module.class_offset_embedding.weight.data[idx] = th.tensor(offset, dtype=th.float32)

    def test_gci0_identical_boxes_zero_loss(self):
        self._set_class_box(0, [1.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        self._set_class_box(1, [1.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        data = th.tensor([[0, 1]])
        self.assertAlmostEqual(self.module.gci0_loss(data).item(), 0.0, places=5)

    def test_gci0_included_box_zero_loss(self):
        # Box(1) = [0.5, 1.5] x ... is contained in Box(0) = [0, 2] x ...
        self._set_class_box(0, [1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        self._set_class_box(1, [1.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        data = th.tensor([[1, 0]])
        self.assertAlmostEqual(self.module.gci0_loss(data).item(), 0.0, places=5)

    def test_gci0_disjoint_boxes_positive_loss(self):
        self._set_class_box(0, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
        self._set_class_box(1, [10.0, 0.0, 0.0], [0.1, 0.1, 0.1])
        data = th.tensor([[0, 1]])
        self.assertGreater(self.module.gci0_loss(data).item(), 0.0)

    def test_gci1_empty_intersection_penalized(self):
        # C1 = [0, 1], C2 = [2, 3] on the first dimension: empty intersection
        self._set_class_box(0, [0.5, 0.0, 0.0], [0.5, 1.0, 1.0])
        self._set_class_box(1, [2.5, 0.0, 0.0], [0.5, 1.0, 1.0])
        self._set_class_box(2, [1.0, 0.0, 0.0], [5.0, 5.0, 5.0])
        data = th.tensor([[0, 1, 2]])
        loss_empty = self.module.gci1_loss(data).item()
        # Non-empty case: shift C2 to overlap C1
        self._set_class_box(1, [0.5, 0.0, 0.0], [0.5, 1.0, 1.0])
        loss_full = self.module.gci1_loss(data).item()
        self.assertGreater(loss_empty, loss_full)

    def test_gci2_enhanced_offset(self):
        # With use_enhancement, Box(∃rall.B) has offset max(0, o(r) - o(B)).
        # If o(r) < o(B) everywhere, the enhanced target is the point c(r)+c(B),
        # so Box(C) at that point with zero offset has zero loss.
        self.module.use_enhancement = True
        self.module.relation_center_embedding.weight.data[0] = th.tensor([1.0, 0.0, 0.0])
        self.module.relation_offset_embedding.weight.data[0] = th.tensor([0.2, 0.2, 0.2])
        self._set_class_box(0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])  # C (point)
        self._set_class_box(1, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])  # B
        data = th.tensor([[0, 0, 1]])
        self.assertAlmostEqual(self.module.gci2_loss(data).item(), 0.0, places=5)

        # Without enhancement, Box(∃r.B) has offset o(r) + o(B) and contains
        # the same point, so the loss is still zero — but a point *farther*
        # away than o(r) + o(B) is only violated in the non-enhanced case if
        # it is outside the larger box. Check the enhancement is stricter:
        # a box C' whose offset fits inside o(r) - o(B) = -0.3 (i.e. only the
        # point) but not a larger offset.
        self.module.use_enhancement = False
        self._set_class_box(0, [1.0, 0.0, 0.0], [0.2, 0.2, 0.2])
        # Non-enhanced target offset is 0.7 > 0.2 -> contained -> zero loss
        self.assertAlmostEqual(self.module.gci2_loss(data).item(), 0.0, places=5)
        # Enhanced target offset is 0 -> C' (offset 0.2) not contained
        self.module.use_enhancement = True
        self.assertGreater(self.module.gci2_loss(data).item(), 0.0)

    def test_gci2_negative_loss(self):
        # A well-separated negative should give loss close to 1 (no overlap
        # term), an overlapping one lower.
        self.module.use_enhancement = False
        self.module.relation_center_embedding.weight.data[0] = th.tensor([0.0, 0.0, 0.0])
        self.module.relation_offset_embedding.weight.data[0] = th.tensor([0.1, 0.1, 0.1])
        self._set_class_box(0, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])  # C
        self._set_class_box(1, [10.0, 10.0, 10.0], [0.1, 0.1, 0.1])  # B far away
        data = th.tensor([[0, 0, 1]])
        loss_separated = self.module.gci2_loss(data, neg=True).item()
        self.assertAlmostEqual(loss_separated, 1.0, places=5)
        # C inside Box(∃r.B): overlap > 0 -> loss < 1
        self._set_class_box(1, [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
        loss_overlap = self.module.gci2_loss(data, neg=True).item()
        self.assertLess(loss_overlap, loss_separated)

    def test_gci3_existential_inclusion(self):
        # ∃r.C ⊑ D: Box(∃r.C) = c(r) + c(C), o(r) + o(C) must be in Box(D)
        self.module.relation_center_embedding.weight.data[0] = th.tensor([1.0, 0.0, 0.0])
        self.module.relation_offset_embedding.weight.data[0] = th.tensor([0.2, 0.2, 0.2])
        self._set_class_box(0, [0.0, 0.0, 0.0], [0.2, 0.2, 0.2])   # C
        self._set_class_box(1, [1.0, 0.0, 0.0], [1.0, 1.0, 1.0])   # D
        data = th.tensor([[0, 0, 1]])
        self.assertAlmostEqual(self.module.gci3_loss(data).item(), 0.0, places=5)
        # Shrink D so it no longer contains ∃r.C
        self._set_class_box(1, [1.0, 0.0, 0.0], [0.2, 0.2, 0.2])
        self.assertGreater(self.module.gci3_loss(data).item(), 0.0)

    def test_offsets_non_negative(self):
        offsets = self.module.class_offset(th.arange(4))
        self.assertTrue((offsets >= 0).all())
        roffsets = self.module.relation_offset(th.arange(2))
        self.assertTrue((roffsets >= 0).all())
