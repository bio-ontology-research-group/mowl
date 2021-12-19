from org.mowl.Porsers import DL2VecParser
from mowl.graph.graph import GraphGenModel

from org.mowl.Parsers import DL2VecParser as Parser
from org.semanticweb.owlapi.model import OWLOntology
from mowl.graph.edge import Edge

class DL2VecParser(GraphGenModel):

    '''
    :param ontology: The ontology to be processed.
    :param bidirectional_taxonomy: If true then per each SubClass edge one SuperClass edge will be generated.
    '''
    def __init__(self, ontology: OWLOntology, bidirectional_taxonomy: bool = False):
        super().__init__(ontology)

        self.parser = Parser(ontology, bidirectional_taxonomy)
        
    def parse(self):
        '''
        Performs the ontology parsing.

        :returns: A list of triples where each triple is of the form :math:`(head, relation, tail)`
        :rtype: List of :class:`mowl.graph.edge.Edge`
        '''

        edges = self.parser.parse()
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges
