from mowl.projection.base import ProjectionModel

from org.mowl.Projectors import DL2VecProjector as Projector
from org.semanticweb.owlapi.model import OWLOntology
from mowl.projection.edge import Edge
import logging


class DL2VecProjector(ProjectionModel):

    '''
    :param ontology: The ontology to be processed.
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param bidirectional_taxonomy: If true then per each SubClass edge one SuperClass edge will \
        be generated.
    :type bidirectional_taxonomy: bool
    '''

    def __init__(self, bidirectional_taxonomy: bool = False):
        super().__init__()

        if not isinstance(bidirectional_taxonomy, bool):
            raise TypeError("Optional parameter bidirectional_taxonomy must be of type boolean")
        self.projector = Projector(bidirectional_taxonomy)

    def project(self, ontology):
        if not isinstance(ontology, OWLOntology):
            raise TypeError(
                "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology")
        edges = self.projector.project(ontology)
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges
