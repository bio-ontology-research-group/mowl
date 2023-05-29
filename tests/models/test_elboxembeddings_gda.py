from unittest import TestCase

from tests.datasetFactory import GDAHumanELDataset
from mowl.models.elboxembeddings.examples.model_gda import ELBoxGDA
import pytest

class TestELBoxEmbeddingsGDA(TestCase):

    @pytest.mark.slow
    def test_ppi(self):
        """Test the ELBoxEmbeddings model on a GDA dataset. The test is not very strict. It just \
checks the syntax of the code"""

        dataset = GDAHumanELDataset()
        model = ELBoxGDA(dataset, epochs=1, embed_dim=1)
        return_value = model.train()
        self.assertEqual(return_value, 1)
