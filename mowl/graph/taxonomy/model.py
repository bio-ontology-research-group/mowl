
import sys

from mowl.graph.graph import GraphGenModel

from org.mowl.BasicParser import SimpleParser

class TaxonomyParser(GraphGenModel):
    def __init__(self, dataset, subclass = True, relations = False):
        super().__init__(dataset)

        self.parser = SimpleParser(dataset.ontology, subclass, relations)

    def parseOWL(self):

        edges = self.parser.parse()

        return edges
