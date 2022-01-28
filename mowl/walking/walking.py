
class WalkingModel():

    '''
    :param edges: List of edges
    :type edges: mowl.graph.edge.Edge
    :param num_walks: Number of walks per node
    :type num_walks: int
    :param walk_length: Length of each walk
    :type walk_length: int
    :param workers: Number of threads to be used for computing the walks. Default is 1'
    :type workers: int
    '''
    def __init__(self, edges, num_walks, walk_length, outfile, workers=1):    
        self.edges = edges
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.workers = workers
        self.outfile = outfile
    # Abstract methods
    def walk(self):

        '''
        This method will generate the walks which can be accessed from the attribute `walks`.
        '''
        
        raise NotImplementedError()

