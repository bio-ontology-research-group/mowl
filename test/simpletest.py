from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastLocalTestDataset
from mowl.onto2vec.model import Onto2Vec


def test_yeast():
    dataset = PPIYeastSlimDataset()
    o = Onto2Vec(dataset)
    mean_observed_ranks, rank_1, rank_10, rank_100 = o.evaluate_ppi()
    assert mean_observed_ranks > 0
    assert rank_1 <= rank_10 <= rank_100
