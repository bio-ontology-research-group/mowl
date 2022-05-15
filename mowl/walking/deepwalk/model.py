from mowl.walking.walking import WalkingModel
import random
import os
import logging

from org.mowl.Walking import DeepWalk as DW
from java.util import HashMap
from java.util import ArrayList
from org.mowl import Edge
logging.basicConfig(level=logging.INFO)


class DeepWalk(WalkingModel):

    '''
    Implementation of DeepWalk based on <https://github.com/phanein/deepwalk/blob/master/deepwalk/graph.py>

    :param alpha: Probability of restart
    :type alpha: float
    '''
    
    def __init__(self,
                 num_walks,
                 walk_length,
                 alpha,
                 outfile,
                 workers=1,
):

        super().__init__(num_walks, walk_length, outfile, workers)

        self.alpha = alpha

    def walk(self, edges):

        edgesJ = ArrayList()
        for edge in edges:
            newEdge = Edge(edge.src(), edge.rel(), edge.dst())
            edgesJ.add(newEdge)

        walker = DW(edgesJ, self.num_walks, self.walk_length, self.alpha, self.workers, self.outfile)

        walker.walk()
