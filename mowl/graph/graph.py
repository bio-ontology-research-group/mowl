
class GraphGenModel():
    """
    Abstract class for Ontology projection into a graph

    :param ontology: The ontology to be processed.
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    """
    
    def __init__(self, ontology):
        self.ontology = ontology

    def parse(self):
        '''
        Performs the ontology parsing.

        :returns: A list of triples where each triple is of the form :math:`(head, relation, tail)`
        :rtype: List of :class:`mowl.graph.edge.Edge`
        '''

        raise NotImplementedError()
    
