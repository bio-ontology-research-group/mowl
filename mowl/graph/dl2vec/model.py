from mowl.graph.graph import GraphGenModel

from org.mowl.Parsers import DL2VecParser as Parser
from org.semanticweb.owlapi.model import OWLOntology
from mowl.graph.edge import Edge
import logging

class DL2VecParser(GraphGenModel):

    '''
    :param ontology: The ontology to be processed.
    :param bidirectional_taxonomy: If true then per each SubClass edge one SuperClass edge will be generated.
    '''
    def __init__(self, ontology: OWLOntology, bidirectional_taxonomy: bool = False):
        super().__init__(ontology)
        
        logging.debug("BID IS: %s", str(bidirectional_taxonomy))
        self.parser = Parser(ontology, bidirectional_taxonomy)
        
    def parse(self):

        edges = self.parser.parse()
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges
