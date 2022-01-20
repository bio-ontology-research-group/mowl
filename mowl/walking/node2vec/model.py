from mowl.walking.walking import WalkingModel
from node2vec.model import Node2Vec as N2V
from mowl.graph.graph import Graph
import numpy as np
class Node2Vec(WalkingModel):

    '''
    :param p: Return hyperparameter. Default is 1.
    :type p: float
    :param q: In-out hyperparameter. Default is 1.
    :type q: float
    '''
    
    def __init__(self,
                 edges,
                 num_walks,
                 walk_length,
                 p=1,
                 q=1,
                 workers=1
                 ):

        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        
        
        self.graph = Graph(edges)
        self.srcs, self.dsts = self.graph.index_edge_list()
        self.srcs = np.array(self.srcs, dtype = np.int32)
        self.dsts = np.array(self.dsts, dtype = np.int32)
        self.model = N2V(self.srcs, self.dsts, graph_is_directed = True)
        
    def walk(self):
        self.model.simulate_walks(
            walk_length = self.walk_length,
            n_walks = self.num_walks,
            p = self.p,
            q = self.q,
            workers = self.workers
        )

        nodes = list(self.graph.node_dict.keys())

        walks_list = []
        for row in self.model.walks:
            sentence = []
            for idx in row:
                sentence.append(nodes[idx])

            walks_list.append(sentence)

        self.walks = walks_list
