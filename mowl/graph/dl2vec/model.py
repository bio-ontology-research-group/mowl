
from mowl.graph.graph import GraphGenModel
import mowl.graph.dl2vec.generate_graph as gen


class DL2VecParser(GraphGenModel):
    
    def __init__(self, dataset, pref_reasoner='elk'):
        super().__init__(dataset)
        
    def parseOWL(self):

        

        edges = gen.parseOWL(self.dataset.ontology)
        return edges
