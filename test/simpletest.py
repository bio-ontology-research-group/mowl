from mowl.datasets import PPIYeastDataset
from mowl.onto2vec.model import Onto2Vec


def test_yeast():
    dataset = PPIYeastDataset()
    o = Onto2Vec(dataset)
