
from mowl.graph.graph import GraphGenModel
from org.mowl.Parsers import DL2VecParser as Parser
from org.semanticweb.owlapi.model import OWLOntology

class DL2VecParser(GraphGenModel):
    
    def __init__(self, ontology: OWLOntology, bidirectional_taxonomy: bool = False):
        super().__init__(ontology)

        self.parser = Parser(ontology, bidirectional_taxonomy)
        
    def parseOWL(self):

        edges = self.parser.parse()
        return edges
