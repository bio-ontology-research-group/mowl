from org.mowl.CatParser import CatParser

import sys

from mowl.graph.graph import GraphGenModel


class CatOnt(GraphGenModel):
    def __init__(self, dataset, subclass = True, relations = False):
        super().__init__(dataset)

        self.parser = CatParser(dataset.ontology)

    def parseOWL(self):

        edges = self.parser.parse()

        return edges
