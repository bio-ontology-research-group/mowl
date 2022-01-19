
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
    def __init__(self, edges, num_walks, walk_length, workers=1):    
        self.edges = edges
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.workers = workers

    # Abstract methods
    def walk(self):

        '''
        This method will generate the walks which can be accessed from the attribute `walks`.
        '''
        
        raise NotImplementedError()


    # Non abstract method
    def num_paths_per_worker(self):
        
        if self.num_walks <= self.workers:
            self.workers = self.num_walks
            paths_per_worker = [1 for x in range(self.num_walks)]
        else:
            remainder = self.num_walks % self.workers
            aux = self.workers - remainder
            paths_per_worker = [int((self.num_walks+aux)/self.workers) for i in range(self.workers)]
            i = 0
            while aux > 0:
                paths_per_worker[i%self.workers] -= 1
                i += 1
                aux -= 1

        return paths_per_worker
