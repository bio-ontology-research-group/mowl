from mowl.projection.base import ProjectionModel
from mowl.projection.edge import Edge
from org.mowl.Projectors import OWL2VecStarProjector as Projector
from org.semanticweb.owlapi.model import OWLOntology
class OWL2VecStarProjector(ProjectionModel):

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

    
    def __init__(self, bidirectional_taxonomy = False, only_taxonomy = False, include_literals = False):
        super().__init__()
        self.bidirectional_taxonomy = bidirectional_taxonomy
        self.include_literals = include_literals
        self.only_taxonomy = only_taxonomy
        self.projector = Projector(self.bidirectional_taxonomy, self.only_taxonomy, self.include_literals)
        
        
    def project(self, ontology):
        edges = self.projector.project(ontology)

        edges =[Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges if str(e.dst()) != ""]
        
        return edges
