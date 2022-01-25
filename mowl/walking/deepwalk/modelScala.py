from mowl.walking.walking import WalkingModel
import random
import os
import logging

from org.mowl.Walking import DeepWalk as DW
from java.util import HashMap
from java.util import ArrayList
logging.basicConfig(level=logging.INFO)


class DeepWalk(WalkingModel):

    '''
    Implementation of DeepWalk based on <https://github.com/phanein/deepwalk/blob/master/deepwalk/graph.py>

    :param alpha: Probability of restart
    :type alpha: float
    '''
    
    def __init__(self,
                 edges,
                 num_walks,
                 walk_length,
                 alpha,
                 num_workers=1,
                 seed = 0,
                 outfile=None):

        super().__init__(edges, num_walks, walk_length, num_workers, outfile)
        self.edges = edges
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.alpha = alpha
        self.num_workers = num_workers
        self.seed = seed
        self.outfile = outfile


    def walk(self):

        graph = HashMap()

        for edge in self.edges:
            newEdge = edge.dst()
            if not edge.src() in graph:
                graph[edge.src()] = ArrayList()
            graph[edge.src()].add(newEdge)

        walker = DW(graph, self.num_walks, self.walk_length, self.alpha, self.num_workers, self.seed, self.outfile)

        walker.walk()
