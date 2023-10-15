from mowl.projection.base import ProjectionModel

from org.mowl.Projectors import TaxonomyProjector as Projector
from org.semanticweb.owlapi.model import OWLOntology
from mowl.projection.edge import Edge


class TaxonomyProjector(ProjectionModel):

    '''
    Projection of axioms :math:`A \sqsubseteq B`.

    :param ontology: The ontology to be processed.
    :param bidirectional_taxonomy: If true then per each SubClass edge one SuperClass edge wil be generated.
    '''

    def __init__(self, bidirectional_taxonomy: bool = False):
        super().__init__()

        if not isinstance(bidirectional_taxonomy, bool):
            raise TypeError("Optional parameter bidirectional_taxonomy must be of type boolean")
        self.projector = Projector(bidirectional_taxonomy)

    def project(self, ontology):
        r"""Generates the projection of the ontology.

        :param ontology: The ontology to be processed.
        :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
        """
        
        if not isinstance(ontology, OWLOntology):
            raise TypeError("Parameter ontology must be of \
type org.semanticweb.owlapi.model.OWLOntology")
        edges = self.projector.project(ontology)
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges

                            
