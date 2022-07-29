from mowl.walking.walking import WalkingModel
import random
import os
import logging
import tempfile
from org.mowl.Walking import DeepWalk as DW
from java.util import HashMap
from java.util import ArrayList
from org.mowl import Edge
from mowl.projection.edge import Edge as PyEdge
from deprecated.sphinx import versionchanged

logging.basicConfig(level=logging.INFO)


class DeepWalk(WalkingModel):

    '''
    Implementation of DeepWalk based on <https://github.com/phanein/deepwalk/blob/master/deepwalk/graph.py>

    :param alpha: Probability of restart
    :type alpha: float
    :param outfile: Path for saving the generated walks, defaults to :class:`tempfile.NamedTemporaryFile`
    :type outfile: str, optional
    '''
    
    def __init__(self,
                 num_walks,
                 walk_length,
                 alpha,
                 outfile = None,
                 workers=1,
):

        if outfile is None:
            tmp_file = tempfile.NamedTemporaryFile()
            outfile = tmp_file.name
        super().__init__(num_walks, walk_length, outfile, workers)

        self.alpha = alpha

    @versionchanged(version = "0.1.0", reason = "The method now can accept a list of entities to focus on when generating the random walks.")
    def walk(self, edges, nodes_of_interest = None):
        if nodes_of_interest is None:
            nodes_of_interest = ArrayList()
        else:
            all_nodes, _ = PyEdge.getEntitiesAndRelations(edges)
            all_nodes = set(all_nodes)
            python_nodes = nodes_of_interest[:]
            nodes_of_interest = ArrayList()
            for node in python_nodes:
                if node in all_nodes:
                    nodes_of_interest.add(node)
                else:
                    logging.info(f"Node {node} does not exist in graph. Ignoring it.")
                    
        edgesJ = ArrayList()
        for edge in edges:
            newEdge = Edge(edge.src(), edge.rel(), edge.dst())
            edgesJ.add(newEdge)

        walker = DW(edgesJ, self.num_walks, self.walk_length, self.alpha, self.workers, self.outfile, nodes_of_interest)

        walker.walk()
