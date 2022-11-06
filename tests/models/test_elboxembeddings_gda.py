from unittest import TestCase

from mowl.datasets.builtin import GDAHumanELDataset
from mowl.models.elboxembeddings.examples.model_gda import ELBoxEmbeddings


class TestELBoxEmbeddingsGDA(TestCase):

    def test_ppi(self):
        """Test the ELBoxEmbeddings model on a GDA dataset. The test is not very strict. It just \
checks the syntax of the code"""

        dataset = GDAHumanELDataset()
        model = ELBoxEmbeddings(dataset, epochs=1, embed_dim=1)
        return_value = model.train()
        self.assertEqual(return_value, 1)
