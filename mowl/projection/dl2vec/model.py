from mowl.projection.base import ProjectionModel

from org.mowl.Projectors import DL2VecProjector as Projector
from org.semanticweb.owlapi.model import OWLOntology
from mowl.projection.edge import Edge
import logging

class DL2VecProjector(ProjectionModel):

    '''
    :param ontology: The ontology to be processed.
    :param bidirectional_taxonomy: If true then per each SubClass edge one SuperClass edge will be generated.
    '''
    def __init__(self, bidirectional_taxonomy: bool = False):
        super().__init__()
        
        logging.debug("BID IS: %s", str(bidirectional_taxonomy))
        self.projector = Projector(bidirectional_taxonomy)
        
    def project(self, ontology):

        edges = self.projector.project(ontology)
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges
