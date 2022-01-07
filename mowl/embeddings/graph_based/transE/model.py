from mowl.model import Model
from mowl.graph.util import parser_factory
from mowl.graph.edge import Edge

#PyKEEN imports
from pykeen.triples import CoreTriplesFactory
from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator

import torch as th
from torch.optim import Adam

import logging
logging.basicConfig(level=logging.DEBUG)   


class OntTransE(Model):

    def __init__(self,
                 dataset,
                 parsing_method="taxonomy",
                 embedding_dim = 50,
                 epochs = 5,
                 batch_size = 32,
                 bidirectional_taxonomy = False
    ):
        super().__init__(dataset)

        self.parsing_method = parsing_method
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.parserTrain = parser_factory(self.parsing_method, self.dataset.ontology, bidirectional_taxonomy)
        self.parserTest = parser_factory(self.parsing_method, self.dataset.testing, bidirectional_taxonomy)

        self.transE_model = None
        
    def train(self):

        edges = self.parserTrain.parse()
        entities, relations = Edge.getEntitiesAndRelations(edges)

        edges_test = self.parserTest.parse()
        entities_test, relations_test = Edge.getEntitiesAndRelations(edges_test)

        total_entities = entities.union(entities_test)
        total_relations = relations.union(relations_test)

        logging.debug("Traininig entities: %d, relations %d. Testing entities: %d, relations %d.", len(entities), len(relations), len(entities_test), len(relations_test))

        self.entities_idx = {ent: idx for idx, ent in enumerate(total_entities)}
        self.relations_idx = {rel: idx for idx, rel in enumerate(total_relations)}

        mapped_triples = [(self.entities_idx[e.src()], self.relations_idx[e.rel()], self.entities_idx[e.dst()]) for e in edges]
        logging.debug("LEN OF TRAIN TRIPLES: %d", len(mapped_triples))

        mapped_triples = th.tensor(mapped_triples).long()

        triples_factory = CoreTriplesFactory(mapped_triples, len(total_entities), len(total_relations), self.entities_idx, self.relations_idx)


        self.transE_model = TransE(triples_factory = triples_factory, embedding_dim = self.embedding_dim)

        optimizer = Adam(params=self.transE_model.get_grad_params())

        training_loop = SLCWATrainingLoop(model=self.transE_model, triples_factory=triples_factory, optimizer=optimizer)

        _ = training_loop.train(triples_factory=triples_factory, num_epochs=self.epochs, batch_size=self.batch_size)


    def evaluate(self):
        if self.transE_model is None:
            raise ValueError("Train a model first.")

        edges = self.parserTest.parse()

        mapped_triples = [(self.entities_idx[e.src()], self.relations_idx[e.rel()], self.entities_idx[e.dst()]) for e in edges]

        logging.debug("LEN OF TEST TRIPLES: %d", len(mapped_triples))
        mapped_triples = th.tensor(mapped_triples).long()
        

        evaluator = RankBasedEvaluator()
        results = evaluator.evaluate(self.transE_model, mapped_triples, batch_size = self.batch_size)

        print(results)
