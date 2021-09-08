

from mowl.graph.graph import GraphGenModel

import generate_graph as gen

from org.mowl.DL2Vec import OntologyParser

class DL2VecParser(GraphGenModel):
    
    def __init__(self, dataset, annotations_file = None, pref_reasoner='elk'):
        super().__init__(dataset)

        self.parser = OntologyParser(dataset.ontology, pref_reasoner)
        self.annotatios_file = annotations_file
        
    def parseOWL(self):


        edges = self.parser.parse()
        return edges


    def addAnnotations(self):

        #Initialize ontology

        #Iterate over annotations and add them as axioms with relation "dl2vec_rel"
