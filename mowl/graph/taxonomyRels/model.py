from org.mowl.Parsers import TaxonomyParserWithRels as Parser
from org.semanticweb.owlapi.model import OWLOntology
from mowl.graph.edge import Edge

import sys

from mowl.graph.graph import GraphGenModel


class TaxonomyWithRelsParser(GraphGenModel):

    r'''
    This class will project the ontology considering the following axioms:
    
    * :math:`A \sqsubseteq B` will generate the triple :math:`\langle A, subClassOf, B \rangle`
    * :math:`A \sqsubseteq \exists R. B` will generate the triple :math:`\left\langle A, R, B \right\rangle`   


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