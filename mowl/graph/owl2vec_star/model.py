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

    '''
    :param ontology: The ontology to be processed.
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param bidirectional_taxonomy: If true then per each SubClass edge one SuperClass edge will be generated. Default is False.
    :type bidirectional_taxonomy: bool
    :param include_literals: If true the graph will also include triples involving data property assertions and annotations. Default is False.
    :type include_literals: bool
    :param only_taxonomy: If true, the projection will only include subClass edges
    :type only_taxonomy: bool
    '''

    
    def __init__(self, ontology, bidirectional_taxonomy = False, include_literals = False, only_taxonomy = False):
        super().__init__(ontology)
        self.bidirectional_taxonomy = bidirectional_taxonomy
        self.include_literals = include_literals
        self.only_taxonomy = only_taxonomy
        
    def parse(self):        
        path = "temp.owl"
        man = OWLManager.createOWLOntologyManager()
        fileout = File(path)
        man.saveOntology(self.ontology, OWLXMLDocumentFormat(), FileOutputStream(fileout))
    
        parser = o2v.OntologyProjection(path, bidirectional_taxonomy = self.bidirectional_taxonomy, include_literals = self.include_literals, only_taxonomy = self.only_taxonomy )

        os.remove(path)

        parser.extractProjection()

        graph = parser.getProjectionGraph()

        edges = []

        for s, r, d in graph:
            edges.append(Edge(str(s), str(r), str(d)))
    

        return edges
