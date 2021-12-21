
class WalkingModel():

    '''
    :param edges: List of :class:`mowl.graph.edge.Edge`
    '''
    def __init__(self, edges):
        
        self.edges = edges


    def walk(self):
        raise NotImplementedError()
