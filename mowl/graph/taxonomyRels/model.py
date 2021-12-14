from org.mowl.Parsers import TaxonomyParserWithRels as Parser
from org.semanticweb.owlapi.model import OWLOntology


import sys

from mowl.graph.graph import GraphGenModel


class TaxonomyWithRelsParser(GraphGenModel):
    def __init__(self, ontology: OWLOntology, bidirectional_taxonomy: bool = False):
        super().__init__(ontology)

        self.parser = Parser(ontology, bidirectional_taxonomy)

    def parse(self):

        edges = self.parser.parse()
        return edges
