

from mowl.projection.factory import projector_factory
from mowl.projection.edge import Edge

#PyKEEN imports
from pykeen.triples import CoreTriplesFactory
from pykeen.models import TransE, TransH, TransR, TransD
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator

import torch as th
from torch.optim import Adam

import logging
logging.basicConfig(level=logging.DEBUG)   


class TranslationalOnt():

    '''
    :param edges: List of edges
    :type edges: mowl.projection.edge.Edge
    :param trans_method: Translational model. Choices are: "transE", "transH", "transR", "transD".
    :type trans_method: str
    :param embedding_dim: Dimension of embedding for each node
    :type embedding_dim: int
    :param epochs: Number of epochs
    :type epochs: int
    :param device: Device to run the model. Default is `cpu`
    :type device: str
    '''
    
    def __init__(self,
                 edges,
                 trans_method="transE",
                 embedding_dim = 50,
                 epochs = 5,
                 batch_size = 32,
                 device = "cpu"
    ):
        
        self.edges = edges
        self.trans_method = trans_method
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self._trained = False
        
    def train(self):

        entities, relations = Edge.getEntitiesAndRelations(self.edges)

        logging.debug("Number of ontology classes: %d, relations %d.", len(entities), len(relations))

        self.entities_idx = {ent: idx for idx, ent in enumerate(entities)}
        self.relations_idx = {rel: idx for idx, rel in enumerate(relations)}

        mapped_triples = [(self.entities_idx[e.src()], self.relations_idx[e.rel()], self.entities_idx[e.dst()]) for e in self.edges]

        mapped_triples = th.tensor(mapped_triples).long()

        triples_factory = CoreTriplesFactory(mapped_triples, len(entities), len(relations), self.entities_idx, self.relations_idx)


        self.model = self.trans_factory(self.trans_method, triples_factory, self.embedding_dim).to(self.device)

        optimizer = Adam(params=self.model.get_grad_params())

        training_loop = SLCWATrainingLoop(model=self.model, triples_factory=triples_factory, optimizer=optimizer)

        _ = training_loop.train(triples_factory=triples_factory, num_epochs=self.epochs, batch_size=self.batch_size)
        self._trained = True

    def get_embeddings(self):
        if not self._trained:
            raise ValueError("Model has not been trained yet")

        embeddings = self.model.entity_representations[0](indices = None).cpu().detach().numpy()
        embeddings = {item[0]: embeddings[item[1]] for item in self.entities_idx.items()}
        return embeddings

    def trans_factory(self, method_name, triples_factory, embedding_dim):
        methods = {
            "transE": TransE,
            "transH": TransH,
            "transR": TransR,
            "transD": TransD
        }

        if method_name in methods:
            return methods[method_name](triples_factory=triples_factory, embedding_dim=embedding_dim)
        else:
            raise Exception(f"Method name unrecognized. Recognized methods are: {methods}")
