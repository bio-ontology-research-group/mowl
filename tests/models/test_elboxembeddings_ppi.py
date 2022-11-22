from unittest import TestCase

from mowl.datasets.builtin import PPIYeastSlimDataset
from mowl.models.elboxembeddings.examples.model_ppi import ELBoxEmbeddings


class TestELBoxEmbeddingsPPI(TestCase):

    def test_ppi(self):
        """Test ELBoxEmbeddings on PPI dataset. The test is not very strict, it just checks the \
correct syntax of the code."""

        dataset = PPIYeastSlimDataset()
        model = ELBoxEmbeddings(dataset, epochs=1, embed_dim=2)
        return_value = model.train()
        self.assertEqual(return_value, 1)
