from mowl.graph.graph import GraphGenModel

from org.mowl.Parsers import TaxonomyParser as Parser
from org.semanticweb.owlapi.model import OWLOntology
from mowl.graph.edge import Edge

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
        edges = self.parser.parse()
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges

    def parseWithTransClosure(self):
        edges = self.parser.parseWithTransClosure()
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges
        
