import sys
sys.path.append("../../")
import mowl
mowl.init_jvm("2g")
import pandas as pd

from mowl.datasets.ppi_yeast import PPIYeastSlimDataset
from mowl.embeddings.opa2vec.model import OPA2Vec


def test_opa2vec_yeast():
    dataset = PPIYeastSlimDataset()
    # need to reference the ontology file from which to retrieve class annotations
    # file sources:
    # http://current.geneontology.org/products/pages/downloads.html
    # https://bio2vec.cbrc.kaust.edu.sa/data/pubmed_model/
    m = OPA2Vec(dataset, '../data/ppi_yeast_localtest/goslim_yeast.owl', '../data/opa2vec_pubmed/RepresentationModel_pubmed.txt')
    mean_observed_ranks, rank_1, rank_10, rank_100 = m.evaluate_ppi()
    assert mean_observed_ranks > 0
    assert rank_1 <= rank_10 <= rank_100
    # OPA2Vec, improving Onto2Vec by adding GO class annotations to the model:
    # mean_observed_ranks = {float64: ()} 1057.2340522984678
    # rank_1 = {int64: ()} 204
    # rank_10 = {int64: ()} 1326
    # rank_100 = {int64: ()} 4643
    # current implementation, that uses the Pubmed pretrained model in addition:
    # mean_observed_ranks = {float64: ()} 292.20889792231253
    # rank_1 = {int64: ()} 520
    # rank_10 = {int64: ()} 2819
    # rank_100 = {int64: ()} 7273


def _most_similar_proteins(m, protein):
    def matching_proteins_in_pairs(pairs, protein):
        interacting_proteins = [p[1] for p in pairs if p[0] == protein]
        interacting_proteins.extend([p[0] for p in pairs if p[1] == protein])
        return set(interacting_proteins)

    similar = m.w2v_model.wv.most_similar(positive=[protein], topn=100)
    _, training_classes_pairs = m.get_classes_pairs_from_axioms(m.dataset.ontology,
                                                                ['<http://interacts_with>'])
    _, testing_classes_pairs = m.get_classes_pairs_from_axioms(m.dataset.testing,
                                                                ['<http://interacts_with>'])
    interacting_proteins_trained = matching_proteins_in_pairs(training_classes_pairs, protein)
    interacting_proteins_testing = matching_proteins_in_pairs(testing_classes_pairs, protein)
    df = pd.DataFrame(similar, columns=['protein', 'similarity'])
    df['training_data'] = df.apply( lambda r: r['protein'] in interacting_proteins_trained,  axis=1)
    df['testing_data'] = df.apply( lambda r: r['protein'] in interacting_proteins_testing,  axis=1)
    df.head()


def test_opa2vec_infer_example():
    dataset = PPIYeastSlimDataset()
    m = OPA2Vec(dataset, '../data/ppi_yeast_localtest/goslim_yeast.owl',
                '../data/opa2vec_pubmed/RepresentationModel_pubmed.txt')
    m.train_or_load_model()
    # Trying to match a real-world example of protein-protein interaction to the model predictions:
    # "Yeast α-tubulin suppressor Ats1/Kti13 relates to the Elongator complex and interacts with Elongator partner protein Kti11"
    # YAL020C - Ats1/Kti13
    # YBL071W-A - KTI11
    # present in the training data, and most similar in the result list.
    # 2nd hit: YIL064W - EMF4 - "May play a role in intracellular transport."
    _most_similar_proteins(m, '<http://4932.YAL020C>')


