import os

from mowl.graph.graph import GraphGenModel
from mowl.graph.edge import Edge
from org.mowl.Parsers import OWL2VecStarParser as Parser
from org.semanticweb.owlapi.model import OWLOntology
class OWL2VecStarParser(GraphGenModel):

    '''
    :param ontology: The ontology to be processed.
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param bidirectional_taxonomy: If true then per each SubClass edge one SuperClass edge will be generated. Default is False.
    :type bidirectional_taxonomy: bool
    :param include_literals: If true the graph will also include triples involving data property assertions and annotations. Default is False.
    :type include_literals: bool
    :param only_taxonomy: If true, the projection will only include subClass edges
    :type only_taxonomy: bool
    '''

    
    def __init__(self, ontology: OWLOntology, bidirectional_taxonomy = False, only_taxonomy = False, include_literals = False):
        super().__init__(ontology)
        self.bidirectional_taxonomy = bidirectional_taxonomy
        self.include_literals = include_literals
        self.only_taxonomy = only_taxonomy
        self.parser = Parser(ontology, self.bidirectional_taxonomy, self.only_taxonomy, self.include_literals)
        
        
    def parse(self):
        edges = self.parser.parse()

        edges =[Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        
        return edges
