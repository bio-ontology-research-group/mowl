from mowl import Model
from mowl.graph.util import parser_factory
from model.graph.edge import Edge
from pykeen.models import TransE


class OntTransE(Model):

    def __init__(self, dataset, parsing_method="taxonomy", embedding_dim = 50):
        super().__init__(dataset)

        self.parsing_method = parsing_method
        self.embedding_dim = embedding_dim

        self.parserTrain = parser_factory(self.parsing_method, self.dataset.ontology)
        self.parserTest = parser_factory(self.parsing_method, self.dataset.test)

        
        def train(self):

            edges = self.parserTrain.parse()

            entities, relations = Edge.getEntitiesAndRelations(edges)

            entities_idx = {ent: idx for idx, ent in enumerate(entities)}
            relations_idx = {ent: idx for idx, ent in enumerate(relations)}


            
            model = TransE()
            
