from org.mowl.BasicParser import SimpleParser

import sys

from mowl.graph.graph import GraphGenModel


class TaxonomyParser(GraphGenModel):
    def __init__(self, dataset, subclass = True, relations = False):
        super().__init__(dataset)

        self.parserTrainSet = SimpleParser(dataset.ontology, subclass, relations)
        self.parserValSet = SimpleParser(dataset.validation, subclass, relations)

    def parseOWL(self, data = "train"):

        if data == "train":
            edges = self.parserTrainSet.parse()
        elif data == "val":
            edges = self.parserValSet.parse()
        elif data == "test":
            NotImplementedError()
        else:
            ValueError()
        return edges
