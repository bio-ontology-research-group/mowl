from org.mowl.Porsers import DL2VecParser
from mowl.graph.graph import GraphGenModel


class DL2VecParser(GraphGenModel):
    
    def __init__(self, dataset, bidirectionalTaxonomy = False):
        super().__init__(dataset)
        
    def parseOWL(self):

        

        edges = gen.parseOWL(self.dataset.ontology)
        return edges
