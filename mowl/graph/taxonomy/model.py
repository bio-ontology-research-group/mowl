from org.mowl.Parsers import TaxonomyParser

from mowl.graph.graph import GraphGenModel


class TaxonomyParser(GraphGenModel):
    def __init__(self, dataset, bidirectionalTaxonomy = False):
        super().__init__(dataset)

        self.parser = TaxonomyParser(dataset.ontology, subclass, relations)

    def parseOWL(self):

        edges = self.parser.parse()

        return edges
