from unittest import TestCase
from mowl.utils.data import FastTensorDataLoader
import torch as th
import random


class TestDataLoaders(TestCase):

    def test_fast_tensor_data_loader_types(self):
        """Test the input types of FastTensorDataLoader"""

        with self.assertRaisesRegex(TypeError, "All non-optional parameters must be Tensors"):
            FastTensorDataLoader([1, 2, 3], [4, 5, 6])

        with self.assertRaisesRegex(TypeError,
                                    "Optional parameter batch_size must be of type int"):
            tensor = th.tensor([1, 2, 3])
            FastTensorDataLoader(tensor, tensor, batch_size=1.0)

        with self.assertRaisesRegex(TypeError,
                                    "Optional parameter shuffle must be of type bool"):
            tensor = th.tensor([1, 2, 3])
            FastTensorDataLoader(tensor, tensor, shuffle=1)

    def test_fast_tensor_data_loader(self):
        """Test the fast tensor data loader."""
        data = th.randn(100, 100)
        labels = th.randint(0, 10, (100,))
        extra_data = th.randn(100, 2)

        batch_size = random.randint(1, 100)
        old_batch_size = batch_size
        loader = FastTensorDataLoader(data, labels, extra_data, batch_size=batch_size)

        samples = 100
        for batch in loader:
            if samples < batch_size:
                batch_size = samples
            self.assertEqual(batch[0].shape, (batch_size, 100))
            self.assertEqual(batch[1].shape, (batch_size,))
            self.assertEqual(batch[2].shape, (batch_size, 2))
            samples -= batch_size

        # No shuffle
        batch_size = old_batch_size
        batch_data = data[:batch_size]
        batch_labels = labels[:batch_size]
        batch_extra_data = extra_data[:batch_size]

        first_batch = next(iter(loader))

        self.assertTrue(th.equal(first_batch[0], batch_data))
        self.assertTrue(th.equal(first_batch[1], batch_labels))
        self.assertTrue(th.equal(first_batch[2], batch_extra_data))

    def test_fast_tensor_data_loader_with_shuffle(self):
        """Test the fast tensor data loader with shuffle."""
        data = th.randn(100, 100)
        labels = th.randint(0, 10, (100,))
        extra_data = th.randn(100, 2)

        batch_data = data[:10]
        batch_labels = labels[:10]
        batch_extra_data = extra_data[:10]

        loader = FastTensorDataLoader(data, labels, extra_data, batch_size=10, shuffle=True)

        first_batch = next(iter(loader))
        self.assertFalse(th.equal(first_batch[0], batch_data))
        self.assertFalse(th.equal(first_batch[1], batch_labels))
        self.assertFalse(th.equal(first_batch[2], batch_extra_data))
