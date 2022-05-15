from mowl.projection.base import ProjectionModel

from org.mowl.Projectors import TaxonomyProjector as Projector
from org.semanticweb.owlapi.model import OWLOntology
from mowl.projection.edge import Edge

class TaxonomyProjector(ProjectionModel):

    '''
    This class will project the ontology considering only the axioms of the form :math:`A \sqsubseteq B` where A and B are ontology classes.
    
    :param ontology: The ontology to be processed.
    :param bidirectional_taxonomy: If true then per each SubClass edge one SuperClass edge will be generated.
    '''
    
    def __init__(self, bidirectional_taxonomy: bool = False):
        super().__init__()

        self.projector = Projector(bidirectional_taxonomy)

    def project(self, ontology):        
        edges = self.projector.project(ontology)
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges

    def projectWithTransClosure(self, ontology):
        edges = self.projector.projectWithTransClosure(ontology)
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges
        
