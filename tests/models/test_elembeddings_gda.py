from unittest import TestCase

from mowl.datasets.builtin import GDAHumanELDataset
from mowl.models.elembeddings.examples.model_gda import ELEmbeddings


class TestELEmbeddingsGDA(TestCase):

    def test_ppi(self):
        """Test the ELEmbeddings model on a GDA dataset. The test is not very strict. It just \
checks the syntax of the code"""

        dataset = GDAHumanELDataset()
        model = ELEmbeddings(dataset, epochs=1, embed_dim=2)
        return_value = model.train()
        self.assertEqual(return_value, 1)
