from mowl.walking.walking import WalkingModel
import random
import os
import logging

from org.mowl.Walking import DeepWalk as DW
from scala.collection.immutable import List as ScalaList
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
                 edges,
                 num_walks,
                 walk_length,
                 alpha,
                 num_workers=1,
                 outfile=None):

        super().__init__(edges, num_walks, walk_length, num_workers, outfile)

        self.alpha = alpha
        

    def walk(self):

        edgesJ = ArrayList()

        for edge in self.edges:
            newEdge = Edge(edge.src(), edge.dst())
            edgesJ.add(newEdge)

        walker = DW(edgesJ, self.num_walks, self.walk_length, self.alpha, self.num_workers, self.outfile)

        walker.walk()
