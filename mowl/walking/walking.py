
class WalkingModel():

    '''
    :param edges: List of edges
    :type edges: mowl.graph.edge.Edge
    :param num_walks: Number of walks per node
    :type num_walks: int
    :param walk_length: Length of each walk
    :type walk_length: int
    :param num_workers: Number of threads to be used for computing the walks
    :type num_workers: int
    :param outfile: Outfile path where the walks will be stored
    :type outfile: str
    '''
    def __init__(self, edges, num_walks, walk_length, num_workers, outfile):    
        self.edges = edges
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_workers = num_workers
        self.outfile = outfile

    # Abstract methods
    def walk(self):
        raise NotImplementedError()


    # Non abstract method
    def num_paths_per_worker(self):
        
        if self.num_walks <= self.num_workers:
            self.num_workers = self.num_walks
            paths_per_worker = [1 for x in range(self.num_walks)]
        else:
            remainder = self.num_walks % self.num_workers
            aux = self.num_workers - remainder
            paths_per_worker = [int((self.num_walks+aux)/self.num_workers) for i in range(self.num_workers)]
            i = 0
            while aux > 0:
                paths_per_worker[i%self.num_workers] -= 1
                i += 1
                aux -= 1

        return paths_per_worker
