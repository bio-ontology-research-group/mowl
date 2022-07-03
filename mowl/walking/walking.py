
class WalkingModel():

    '''
    :param num_walks: Number of walks per node
    :type num_walks: int
    :param walk_length: Length of each walk
    :type walk_length: int
    :param workers: Number of threads to be used for computing the walks, defaults to 1'
    :type workers: int, optional
    '''
    def __init__(self, num_walks, walk_length, outfile, workers=1):
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.workers = workers
        self.outfile = outfile
    # Abstract methods
    def walk(self, edges):

        '''
        This method will generate the walks.

        :param edges: List of edges
        :type edges: :class:`mowl.graph.edge.Edge`
        '''
        
        raise NotImplementedError()

