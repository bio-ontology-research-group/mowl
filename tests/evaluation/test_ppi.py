from unittest import TestCase
from tests.datasetFactory import PPIYeastSlimDataset
from mowl.models import GraphPlusPyKEENModel
from mowl.projection import OWL2VecStarProjector
from mowl.evaluation import PPIEvaluator
from pykeen.models import TransE
import torch as th

allowed_diff = 1e-6

def auc_from_mr(mr, num_entities):
    auc = 1 - (mr-1) / (num_entities -1)
    return auc

class EvaluationModel(th.nn.Module):
    def __init__(self, kge_method, triples_factory, dataset, embedding_size, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.kge_method = kge_method
        
        classes = dataset.classes.as_str
        class_to_id = {class_: i for i, class_ in enumerate(classes)}

        ont_id_to_graph_id = dict()

        num_not_found = 0
        for class_ in classes:
            if class_ not in triples_factory.entity_to_id:
                ont_id_to_graph_id[class_to_id[class_]] = -1
                num_not_found += 1
            else:
                ont_id_to_graph_id[class_to_id[class_]] = triples_factory.entity_to_id[class_]

        assert list(ont_id_to_graph_id.keys()) == list(range(len(classes)))
        self.graph_ids = th.tensor(list(ont_id_to_graph_id.values())).to(self.device)
        
        relation_id = triples_factory.relation_to_id["http://interacts_with"]
        self.rel_embedding = th.tensor(relation_id).to(self.device)
        
    def forward(self, data, *args, **kwargs):
        if data.shape[1] == 2:
            x = data[:, 0]
            y = data[:, 1]
        elif data.shape[1] == 3:
            x = data[:, 0]
            y = data[:, 2]
        else:
            raise ValueError(f"Data shape {data.shape} not recognized")

        x = self.graph_ids[x].unsqueeze(1)
        y = self.graph_ids[y].unsqueeze(1)

        x_unique = self.graph_ids[data[:, 0].unique()]
        y_unique = self.graph_ids[data[:, 1].unique()]
        assert th.min(x_unique) >= 0, f"sum: {(x_unique==-1).sum()} min: {th.min(x_unique)} len: {len(x_unique)}"
        assert th.min(y_unique) >= 0, f"sum: {(y_unique==-1).sum()} min: {th.min(y_unique)} len: {len(y_unique)}"
                        
        r = self.rel_embedding.expand_as(x)
        
        triples = th.cat([x, r, y], dim=1)
        assert triples.shape[1] == 3
        scores = - self.kge_method.score_hrt(triples)
        return scores



    
class TestPPI(TestCase):

    def setUp(self):
        emb_dim = 20
        self.dataset = PPIYeastSlimDataset()
        self.model = GraphPlusPyKEENModel(self.dataset)
        self.model.set_projector(OWL2VecStarProjector())
        self.model.optimizer = th.optim.Adam
        self.model.lr = 0.01
        self.model.batch_size = 256
        self.model.set_kge_method(TransE, embedding_dim=emb_dim)
        self.model.set_evaluator(PPIEvaluator)
        self.model._evaluation_model = EvaluationModel(self.model.kge_method, self.model.triples_factory, self.dataset, emb_dim, self.model.device)
        
    def test_ppi_evaluator(self):
        self.model.train(epochs=5)
        self.model.evaluate(self.dataset.testing, filter_ontologies=[self.dataset.ontology, self.dataset.validation])
        mr = self.model.metrics["mr"]
        auc = self.model.metrics["auc"]

        num_classes = len(self.dataset.evaluation_classes[0])

        true_auc = auc_from_mr(mr, num_classes)

        diff_auc = abs(auc - true_auc)
        self.assertLess(diff_auc, allowed_diff)
        
        



