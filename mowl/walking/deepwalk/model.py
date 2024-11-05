from mowl.walking.walking import WalkingModel
import random
import os
import time
import logging
from org.mowl.Walking import DeepWalk as DW
from java.util import HashMap
from java.util import ArrayList
from org.mowl import Edge
from mowl.projection.edge import Edge as PyEdge
from deprecated.sphinx import versionchanged

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepwalk")


class DeepWalk(WalkingModel):

    '''
    Implementation of DeepWalk based on [perozzi2014]_.
    
    :param alpha: Probability of restart, defaults to 0
    :type alpha: float, optional
    '''

    def __init__(self,
                 num_walks,
                 walk_length,
                 alpha=0.,
                 outfile=None,
                 workers=1,
                 seed=0
                 ):
        super().__init__(num_walks, walk_length, outfile=outfile, workers=workers)

        # Type checking
        if not isinstance(alpha, float):
            if isinstance(alpha, int):
                alpha = float(alpha)
            else:
                raise TypeError("Optional parameter alpha must be a float")
        self.alpha = alpha
        self.seed = seed

    @versionchanged(version="0.1.0", reason="The method now can accept a list of entities to \
    focus on when generating the random walks.")
    def walk(self, edges, nodes_of_interest=None):
        if nodes_of_interest is None:
            nodes_of_interest = ArrayList()
        else:
            all_nodes, _ = PyEdge.get_entities_and_relations(edges)
            all_nodes = set(all_nodes)
            python_nodes = nodes_of_interest[:]
            nodes_of_interest = ArrayList()
            for node in python_nodes:
                if node in all_nodes:
                    nodes_of_interest.add(node)
                else:
                    logger.info(f"Node {node} does not exist in the graph. Ignoring it.")

        edgesJ = ArrayList()
        for edge in edges:
            newEdge = Edge(edge.src, edge.rel, edge.dst)
            edgesJ.add(newEdge)

        walker = DW(edgesJ, self.num_walks, self.walk_length, self.alpha, self.workers,
                    self.outfile, nodes_of_interest, self.seed)

        walker.walk()

        self.wait_for_all_walks()
        

