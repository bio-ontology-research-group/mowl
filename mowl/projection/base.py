from mowl.projection.edge import Edge

class ProjectionModel():
    """
    Abstract class for Ontology projection into a graph

    :param ontology: The ontology to be processed.
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    """
    
    def __init__(self):
        return
    
    def project(self, ontology):
        '''
        Performs the ontology parsing.

        :returns: A list of triples where each triple is of the form :math:`(head, relation, tail)`
        :rtype: List of :class:`mowl.graph.edge.Edge`
        '''

        raise NotImplementedError()




class Graph():

    '''
    Stores a graph in the form of an edgelist in which each edge is an instance of the class :class:`mowl.graph.edge.Edge`

    :param edges: A list of edges
    :type edges: :class:`mowl.graph.edge.Edge`
    '''
    
    def __init__(self, edges):
        self.edges = edges
        self.srcs_idx = None
        self.dsts_idx = None
        self.node_dict = None
        
    def index_edge_list(self):
        if self.node_dict is None:
            nodes, _ = Edge.getEntitiesAndRelations(self.edges)
            self.node_dict = {node:idx for idx, node in enumerate(nodes)}

        srcs = list(map(lambda x: self.node_dict[x.src()], self.edges))
        dsts = list(map(lambda x: self.node_dict[x.dst()], self.edges))

        return srcs, dsts
