import os
import tempfile

from java.io import File
from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.formats import OWLXMLDocumentFormat
from java.io import FileOutputStream
from mowl.graph.graph import GraphGenModel
from mowl.graph.edge import Edge


import mowl.graph.owl2vec_star.Onto_Projection as o2v


class OWL2VecParser(GraphGenModel):
    
    def __init__(self, dataset, include_literals = False, bidirectional_taxonomy = True, only_taxonomy = False):
        super().__init__(dataset)
        self.include_literals = include_literals
        self.bidirectional_taxonomy = bidirectional_taxonomy
        self.only_taxonomy = only_taxonomy
        
    def parseOWL(self):

        path = "temp.owl"
        man = OWLManager.createOWLOntologyManager()
        fileout = File(path)
        man.saveOntology(self.dataset.ontology, OWLXMLDocumentFormat(), FileOutputStream(fileout))
    

        parser = o2v.OntologyProjection(path, bidirectional_taxonomy = self.bidirectional_taxonomy, include_literals=self.include_literals, only_taxonomy = self.only_taxonomy)

        os.remove(path)

        parser.extractProjection()

        graph = parser.getProjectionGraph()

#        edges = [Edge(s, r, d) for s, r, d in graph]
        edges = []

        for s, r, d in graph:
            edges.append(Edge(str(s), str(r), str(d)))
    

        return edges
