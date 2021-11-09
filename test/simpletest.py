from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastLocalTestDataset
from mowl.onto2vec.model import Onto2Vec


def test_yeast():
    dataset = PPIYeastSlimDataset()
    # dataset = PPIYeastLocalTestDataset()
    o = Onto2Vec(dataset)
    o.train()
