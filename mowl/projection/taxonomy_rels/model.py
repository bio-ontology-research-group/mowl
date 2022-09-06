from org.mowl.Projectors import TaxonomyWithRelsProjector as Projector
from org.semanticweb.owlapi.model import OWLOntology
from mowl.projection.edge import Edge

from mowl.projection.base import ProjectionModel

from java.util import ArrayList

class TaxonomyWithRelsProjector(ProjectionModel):

    r'''
    This class will project the ontology considering the following axioms:
    
    * :math:`A \sqsubseteq B` will generate the triple :math:`\langle A, subClassOf, B \rangle`
    * :math:`A \sqsubseteq \exists R. B` will generate the triple :math:`\left\langle A, R, B \right\rangle`   


    :param ontology: The ontology to be processed.
    :param bidirectional_taxonomy: If true then per each SubClass edge one SuperClass edge will be generated.
    '''
    
    def __init__(self, taxonomy = False, bidirectional_taxonomy: bool = False, relations = None):
        super().__init__()

        relations = [] if relations is None else relations
        relationsJ = ArrayList()
        for r in relations:
            relationsJ.add(r)

        self.projector = Projector(taxonomy, bidirectional_taxonomy, relationsJ)

    def project(self, ontology):        
        edges = self.projector.project(ontology)
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges
