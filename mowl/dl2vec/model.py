

from mowl.model import model
import generate_graph as gen

class DL2Vec(Model):
    def __init__(self, ontology_file, annotation_file, output_format):

        self.annotation_file = annotation_file
        self.ontology_file = ontology_file
        self.output_format = output_format

    def train():
        G = gen.generate_graph(self.annotation_file, se

        
