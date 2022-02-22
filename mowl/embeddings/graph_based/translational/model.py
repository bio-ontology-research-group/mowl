from mowl.model import Model
from mowl.graph.factory import parser_factory
from mowl.graph.edge import Edge

#PyKEEN imports
from pykeen.triples import CoreTriplesFactory
from pykeen.models import TransE, TransH, TransR, TransD
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator

import torch as th
from torch.optim import Adam

import logging
logging.basicConfig(level=logging.DEBUG)   


class TranslationalOnt(Model):

    '''
    :param dataset: Dataset composed by training, validation and testing sets, each of which are in OWL format.
    :type dataset: :class:`mowl.datasets.base.Dataset`
    :param parsing_method: Method to generate the graph. The methods correspond to subclasses of :class:`mowl.graph.graph.GraphGenModel`. Choices are: "taxonomy", "taxonomy_rels", "dl2vec", "owl2vec_star"
    :type parsing_method: str
    :param edges_train: List of precomputed edges of the training set. This will replace the parsing method.
    :type edges_train: List of :class:`mowl.graph.edge.Edge`
    :param edges_test: List of precomputed edges of the testing set. This will replace the parsing method.
    :type edges_test: List of :class:`mowl.graph.edge.Edge`
    :param trans_method: Translational model. Choices are: "transE", "transH", "transR", "transD".
    :type trans_method: str
    :param embedding_dim: Dimension of embedding for each node
    :type embedding_dim: int
    :param epochs: Number of epochs
    :type epochs: int
    :param bidirectional_taxonomy: Parameter for the graph generation method. If true, then per each SubClass edge one SuperClass edge will be generated.
    :type bidirectional_taxonomy: bool
    '''
    
    def __init__(self,
                 dataset = None,
                 parsing_method="taxonomy",
                 edges_train = None,
                 edges_test = None,
                 trans_method="transE",
                 embedding_dim = 50,
                 epochs = 5,
                 batch_size = 32,
                 bidirectional_taxonomy = False
    ):
        super().__init__(dataset)

        self.parsing_method = parsing_method
        self.edges_train = edges_train
        self.edges_test = edges_test
        self.trans_method = trans_method
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size

        if not dataset is None: 
            self.parserTrain = parser_factory(self.parsing_method, self.dataset.ontology, bidirectional_taxonomy)
            self.parserTest = parser_factory(self.parsing_method, self.dataset.testing, bidirectional_taxonomy)

        self.model = None
        
    def train(self):

        if self.edges_train is None:
            self.edges_train = self.parserTrain.parse()

        entities, relations = Edge.getEntitiesAndRelations(self.edges_train)

        if self.edges_test is None:
            self.edges_test = self.parserTest.parse()
        entities_test, relations_test = Edge.getEntitiesAndRelations(self.edges_test)

        total_entities = entities.union(entities_test)
        total_relations = relations.union(relations_test)

        logging.debug("Traininig entities: %d, relations %d. Testing entities: %d, relations %d.", len(entities), len(relations), len(entities_test), len(relations_test))

        self.entities_idx = {ent: idx for idx, ent in enumerate(total_entities)}
        self.relations_idx = {rel: idx for idx, rel in enumerate(total_relations)}

        mapped_triples = [(self.entities_idx[e.src()], self.relations_idx[e.rel()], self.entities_idx[e.dst()]) for e in self.edges_train]


        mapped_triples = th.tensor(mapped_triples).long()

        triples_factory = CoreTriplesFactory(mapped_triples, len(total_entities), len(total_relations), self.entities_idx, self.relations_idx)


        self.model = self.trans_factory(self.trans_method, triples_factory, self.embedding_dim)

        optimizer = Adam(params=self.model.get_grad_params())

        training_loop = SLCWATrainingLoop(model=self.model, triples_factory=triples_factory, optimizer=optimizer)

        _ = training_loop.train(triples_factory=triples_factory, num_epochs=self.epochs, batch_size=self.batch_size)


    def evaluate(self):
        if self.model is None:
            raise ValueError("Train a model first.")

        if self.edges_test is None:
            self.edges_test = self.parserTest.parse()

        mapped_triples = [(self.entities_idx[e.src()], self.relations_idx[e.rel()], self.entities_idx[e.dst()]) for e in self.edges_test]

        logging.debug("LEN OF TEST TRIPLES: %d", len(mapped_triples))
        mapped_triples = th.tensor(mapped_triples).long()
        

        evaluator = RankBasedEvaluator()
        results = evaluator.evaluate(self.model, mapped_triples, batch_size = self.batch_size)

        return results
#        print(results)



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
