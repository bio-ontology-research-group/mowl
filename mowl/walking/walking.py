
class WalkingModel():

    '''
    :param edges: List of :class:`mowl.graph.edge.Edge`
    '''
    def __init__(self, edges, num_paths, path_length, alpha, num_workers, outfile):    
        self.edges = edges
        self.num_paths = num_paths
        self.path_length = path_length
        self.alpha = alpha
        self.num_workers = num_workers
        self.outfile = outfile

    def walk(self):
        raise NotImplementedError()
