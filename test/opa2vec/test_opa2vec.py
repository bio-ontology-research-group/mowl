from mowl.datasets.ppi_yeast import PPIYeastSlimDataset
from mowl.opa2vec.model import OPA2Vec


def test_opa2vec_yeast():
    dataset = PPIYeastSlimDataset()
    # need to reference the ontology file from which to retrieve class annotations
    m = OPA2Vec(dataset, '../data/ppi_yeast_localtest/goslim_yeast.owl')
    mean_observed_ranks, rank_1, rank_10, rank_100 = m.evaluate_ppi()
    assert mean_observed_ranks > 0
    assert rank_1 <= rank_10 <= rank_100
    # OPA2Vec, improving Onto2Vec by adding GO class annotations to the model:
    # mean_observed_ranks = {float64: ()} 1057.2340522984678
    # rank_1 = {int64: ()} 204
    # rank_10 = {int64: ()} 1326
    # rank_100 = {int64: ()} 4643
    # in addition, using a Pubmed pretrained model:
    # TODO ...