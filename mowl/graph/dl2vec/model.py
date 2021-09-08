

from mowl.graph.graph import GraphGenModel

import generate_graph as gen

from org.mowl.DL2Vec import OntologyParser

class DL2VecParser(GraphGenModel):
    
    def __init__(self, dataset, pref_reasoner='elk'):
        super().__init__(dataset)

        self.parser = OntologyParser(dataset.ontology, pref_reasoner)
        
    def parseOWL(self):


        edges = self.parser.parse()
        return edges
