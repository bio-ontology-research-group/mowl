from mowl.datasets.ppi_yeast import PPIYeastSlimDataset, PPIYeastLocalTestDataset
from mowl.onto2vec.model import Onto2Vec


def test_onto2vec_yeast():
    dataset = PPIYeastSlimDataset()
    m = Onto2Vec(dataset)
    mean_observed_ranks, rank_1, rank_10, rank_100 = m.evaluate_ppi()
    assert mean_observed_ranks > 0
    assert rank_1 <= rank_10 <= rank_100
    # mean_observed_ranks = {float64: ()} 1092.205113257828
    # rank_1 = {int64: ()} 192
    # rank_10 = {int64: ()} 1250
    # rank_100 = {int64: ()} 4467
