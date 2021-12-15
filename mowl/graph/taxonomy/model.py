from org.mowl.Parsers import TaxonomyParser as Parser
from org.semanticweb.owlapi.model import OWLOntology


import sys

from mowl.graph.graph import GraphGenModel


class TaxonomyParser(GraphGenModel):

    '''
    This class will project the ontology considering only the axioms of the form :math:`A \sqsubseteq B` where A and B are ontology classes.
    
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
        '''

        
        edges = self.parser.parse()
        edges = [(e.src(), e.rel(), e.dst()) for e in edges]
        return edges
