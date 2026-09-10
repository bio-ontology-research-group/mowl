from unittest import TestCase
import numpy as np
import torch as th
from tests.datasetFactory import FamilyDataset, GDAHumanELDataset
from mowl.models import ELEmbeddings


def _legacy_generate_negatives(model, gci_name, gci_dataset):
    """Verbatim copy of the pre-#159 single-column implementation, for regression checks."""
    cfg = model.get_negative_sampling_config()[gci_name]
    index_pool = cfg["index_pool"]
    corrupt_column = cfg["corrupt_column"]

    if index_pool == "classes":
        all_ids = list(model.class_index_dict.values())
    elif index_pool == "individuals":
        all_ids = list(model.individual_index_dict.values())
    else:
        raise ValueError(f"Unknown index_pool: {index_pool}")

    data = gci_dataset[:]
    idxs_for_negs = np.random.choice(all_ids, size=len(gci_dataset), replace=True)
    rand_index = th.tensor(idxs_for_negs, dtype=th.long, device=model.device)

    neg_data = th.cat([data[:, :corrupt_column], rand_index.unsqueeze(1)], dim=1)
    if corrupt_column + 1 < data.shape[1]:
        neg_data = th.cat([neg_data, data[:, corrupt_column + 1:]], dim=1)

    return neg_data


class TestGenerateNegatives(TestCase):
    """Tests for multi-column negative sampling in EmbeddingELModel (see #159)."""

    @classmethod
    def setUpClass(cls):
        cls.model = ELEmbeddings(FamilyDataset(), embed_dim=8, batch_size=16)
        cls.class_ids = set(cls.model.class_index_dict.values())

    def _set_config(self, config):
        """Shadow the class-level default config with a per-instance one."""
        self.model._DEFAULT_NEG_SAMPLING_CONFIG = config

    def _reset_config(self):
        del self.model._DEFAULT_NEG_SAMPLING_CONFIG

    def test_single_column_unchanged(self):
        """A single int column behaves exactly as before: one negative per positive."""
        self._set_config({"gci2": {"index_pool": "classes", "corrupt_column": 2}})
        try:
            data = self.model.training_datasets["gci2"][:]
            neg = self.model.generate_negatives(
                "gci2", self.model.training_datasets["gci2"])
            self.assertEqual(neg.shape, th.Size([len(data), 3]))
            self.assertTrue(th.equal(neg[:, 0], data[:, 0]))
            self.assertTrue(th.equal(neg[:, 1], data[:, 1]))
            self.assertTrue(set(neg[:, 2].tolist()) <= self.class_ids)
        finally:
            self._reset_config()

    def test_single_column_identical_to_legacy(self):
        """With a seeded RNG, the single-column output matches the old implementation exactly."""
        self._set_config({"gci2": {"index_pool": "classes", "corrupt_column": 2}})
        try:
            gci_dataset = self.model.training_datasets["gci2"]
            np.random.seed(42)
            neg_new = self.model.generate_negatives("gci2", gci_dataset)
            np.random.seed(42)
            neg_legacy = _legacy_generate_negatives(self.model, "gci2", gci_dataset)
            self.assertTrue(th.equal(neg_new, neg_legacy))
        finally:
            self._reset_config()

    def test_multiple_columns(self):
        """A list of columns yields one negative set per column, concatenated."""
        self._set_config({"gci2": {"index_pool": "classes", "corrupt_column": [0, 2]}})
        try:
            data = self.model.training_datasets["gci2"][:]
            neg = self.model.generate_negatives(
                "gci2", self.model.training_datasets["gci2"])
            n = len(data)
            # Two blocks of n rows: [C', R, D] then [C, R, D']
            self.assertEqual(neg.shape, th.Size([2 * n, 3]))
            block_a, block_b = neg[:n], neg[n:]
            # Block A corrupts column 0 only
            self.assertTrue(th.equal(block_a[:, 1], data[:, 1]))
            self.assertTrue(th.equal(block_a[:, 2], data[:, 2]))
            self.assertTrue(set(block_a[:, 0].tolist()) <= self.class_ids)
            # Block B corrupts column 2 only
            self.assertTrue(th.equal(block_b[:, 0], data[:, 0]))
            self.assertTrue(th.equal(block_b[:, 1], data[:, 1]))
            self.assertTrue(set(block_b[:, 2].tolist()) <= self.class_ids)
        finally:
            self._reset_config()

    def test_single_column_in_list_equivalent_to_int(self):
        """[col] as a list produces the same structure as col as an int."""
        data = self.model.training_datasets["gci2"][:]
        self._set_config({"gci2": {"index_pool": "classes", "corrupt_column": 2}})
        neg_int = self.model.generate_negatives(
            "gci2", self.model.training_datasets["gci2"])
        self._set_config({"gci2": {"index_pool": "classes", "corrupt_column": [2]}})
        try:
            neg_list = self.model.generate_negatives(
                "gci2", self.model.training_datasets["gci2"])
        finally:
            self._reset_config()
        self.assertEqual(neg_int.shape, neg_list.shape)
        self.assertEqual(neg_list.shape[0], len(data))
        self.assertTrue(th.equal(neg_list[:, 0], data[:, 0]))
        self.assertTrue(th.equal(neg_list[:, 1], data[:, 1]))

    def test_per_column_pools(self):
        """index_pool may be a list, one pool per corrupted column (GDA has an ABox)."""
        model = ELEmbeddings(GDAHumanELDataset(), embed_dim=8, batch_size=16,
                             neg_sampling_gcis=["class_assertion"])
        model._DEFAULT_NEG_SAMPLING_CONFIG = {
            "class_assertion": {"index_pool": ["individuals", "classes"],
                                "corrupt_column": [0, 1]}
        }
        ind_ids = set(model.individual_index_dict.values())
        class_ids = set(model.class_index_dict.values())
        data = model.training_datasets["class_assertion"][:]
        neg = model.generate_negatives(
            "class_assertion", model.training_datasets["class_assertion"])
        n = len(data)
        block_a, block_b = neg[:n], neg[n:]
        # Block A: individual column corrupted from the individuals pool
        self.assertTrue(set(block_a[:, 0].tolist()) <= ind_ids)
        self.assertTrue(th.equal(block_a[:, 1], data[:, 1]))
        # Block B: class column corrupted from the classes pool
        self.assertTrue(set(block_b[:, 1].tolist()) <= class_ids)
        self.assertTrue(th.equal(block_b[:, 0], data[:, 0]))

    def test_pool_column_length_mismatch_raises(self):
        self._set_config({
            "gci2": {"index_pool": ["classes", "classes"], "corrupt_column": [2]}
        })
        try:
            with self.assertRaisesRegex(ValueError, "one pool per corrupted column"):
                self.model.generate_negatives(
                    "gci2", self.model.training_datasets["gci2"])
        finally:
            self._reset_config()

    def test_column_out_of_range_raises(self):
        self._set_config({"gci2": {"index_pool": "classes", "corrupt_column": 5}})
        try:
            with self.assertRaisesRegex(ValueError, "out of range"):
                self.model.generate_negatives(
                    "gci2", self.model.training_datasets["gci2"])
        finally:
            self._reset_config()

    def test_unknown_pool_raises(self):
        self._set_config({"gci2": {"index_pool": "properties", "corrupt_column": 2}})
        try:
            with self.assertRaisesRegex(ValueError, "Unknown index_pool"):
                self.model.generate_negatives(
                    "gci2", self.model.training_datasets["gci2"])
        finally:
            self._reset_config()

    def test_no_config_returns_none(self):
        self._set_config({})
        try:
            self.assertIsNone(self.model.generate_negatives(
                "gci2", self.model.training_datasets["gci2"]))
        finally:
            self._reset_config()

    def test_numpy_integer_columns_are_accepted(self):
        """Column indices computed from numpy (a tensor shape, an array) are integers too,
        and must not be mistaken for something out of range."""
        self._set_config({"gci2": {"index_pool": "classes",
                                   "corrupt_column": [np.int64(0), np.int64(2)]}})
        try:
            data = self.model.training_datasets["gci2"][:]
            neg = self.model.generate_negatives(
                "gci2", self.model.training_datasets["gci2"])
            self.assertEqual(neg.shape, th.Size([2 * len(data), 3]))
        finally:
            self._reset_config()

        self._set_config({"gci2": {"index_pool": "classes", "corrupt_column": np.int64(2)}})
        try:
            neg = self.model.generate_negatives(
                "gci2", self.model.training_datasets["gci2"])
            self.assertEqual(neg.shape, th.Size([len(data), 3]))
        finally:
            self._reset_config()

    def test_boolean_column_raises(self):
        """A bool is an int in Python, but it is never a column index."""
        self._set_config({"gci2": {"index_pool": "classes", "corrupt_column": True}})
        try:
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                self.model.generate_negatives(
                    "gci2", self.model.training_datasets["gci2"])
        finally:
            self._reset_config()

    def test_non_integer_column_raises(self):
        self._set_config({"gci2": {"index_pool": "classes", "corrupt_column": 2.0}})
        try:
            with self.assertRaisesRegex(ValueError, "must be an integer"):
                self.model.generate_negatives(
                    "gci2", self.model.training_datasets["gci2"])
        finally:
            self._reset_config()

    def test_empty_column_list_raises(self):
        """An empty list must be a configuration error, not a torch.cat crash."""
        self._set_config({"gci2": {"index_pool": "classes", "corrupt_column": []}})
        try:
            with self.assertRaisesRegex(ValueError, "'corrupt_column' is empty"):
                self.model.generate_negatives(
                    "gci2", self.model.training_datasets["gci2"])
        finally:
            self._reset_config()

    def test_missing_index_pool_raises_a_readable_error(self):
        """The base implementation needs a pool; several shipped examples omit the key and
        override generate_negatives instead, so the message has to point at both options."""
        self._set_config({"gci2": {"corrupt_column": 2}})
        try:
            with self.assertRaisesRegex(ValueError, "no 'index_pool' key"):
                self.model.generate_negatives(
                    "gci2", self.model.training_datasets["gci2"])
        finally:
            self._reset_config()

    def test_unknown_pool_in_a_list_raises(self):
        self._set_config({"gci2": {"index_pool": ["classes", "nope"],
                                   "corrupt_column": [0, 2]}})
        try:
            with self.assertRaisesRegex(ValueError, "Unknown index_pool"):
                self.model.generate_negatives(
                    "gci2", self.model.training_datasets["gci2"])
        finally:
            self._reset_config()

    def test_three_columns(self):
        """Nothing in the implementation is specific to K = 2."""
        self._set_config({"gci2": {"index_pool": "classes", "corrupt_column": [0, 1, 2]}})
        try:
            data = self.model.training_datasets["gci2"][:]
            neg = self.model.generate_negatives(
                "gci2", self.model.training_datasets["gci2"])
            self.assertEqual(neg.shape, th.Size([3 * len(data), 3]))
        finally:
            self._reset_config()


class TestMultiColumnTraining(TestCase):
    """End-to-end: one training epoch with multi-column negatives stays finite."""

    def test_train_with_multiple_corrupt_columns(self):
        model = ELEmbeddings(FamilyDataset(), embed_dim=8, batch_size=16)
        model._DEFAULT_NEG_SAMPLING_CONFIG = {
            "gci2": {"index_pool": "classes", "corrupt_column": [0, 2]},
        }
        model.train(epochs=1)
        param = next(model.module.parameters())
        self.assertTrue(th.isfinite(param).all().item())

    def test_bad_config_is_rejected_before_the_first_epoch(self):
        """A malformed entry must be reported before training starts, not once the loop
        reaches that normal form."""
        model = ELEmbeddings(FamilyDataset(), embed_dim=8, batch_size=16)
        model._DEFAULT_NEG_SAMPLING_CONFIG = {
            "gci2": {"index_pool": "classes", "corrupt_column": 7},
        }

        def fail_if_called(*args, **kwargs):
            raise AssertionError("training started despite a malformed config")

        model.generate_negatives = fail_if_called

        with self.assertRaisesRegex(ValueError, "out of range"):
            model.train(epochs=1)

    def test_config_without_index_pool_still_trains(self):
        """Several shipped examples (ELEmPPI and friends) configure only 'corrupt_column'
        and sample from their own pool, so the eager validation must tolerate the missing
        key even though the base implementation requires it."""

        class CustomPoolELEmbeddings(ELEmbeddings):
            def get_negative_sampling_config(self):
                return {"gci2": {"corrupt_column": 2}}

            def generate_negatives(self, gci_name, gci_dataset):
                data = gci_dataset[:]
                ids = np.random.choice(list(self.class_index_dict.values()),
                                       size=len(gci_dataset), replace=True)
                rand_index = th.tensor(ids, dtype=th.long, device=self.device)
                return th.cat([data[:, :2], rand_index.unsqueeze(1)], dim=1)

        model = CustomPoolELEmbeddings(FamilyDataset(), embed_dim=8, batch_size=16)
        model.train(epochs=1)
        param = next(model.module.parameters())
        self.assertTrue(th.isfinite(param).all().item())
