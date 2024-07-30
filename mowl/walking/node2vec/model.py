from mowl.walking.walking import WalkingModel
import logging
import tempfile
from java.util import ArrayList
from org.mowl import Edge
from org.mowl.Walking import Node2Vec as N2V
from mowl.projection.edge import Edge as PyEdge
from deprecated.sphinx import versionchanged

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("node2vec")


class Node2Vec(WalkingModel):

    '''
    Implementation of Node2Vec based on [grover2016]_.

    :param p: Return hyperparameter. Default is 1.
    :type p: float
    :param q: In-out hyperparameter. Default is 1.
    :type q: float
    '''

    def __init__(self,
                 num_walks,
                 walk_length,
                 p=1,
                 q=1,
                 outfile=None,
                 workers=1
                 ):

        super().__init__(num_walks, walk_length, outfile=outfile, workers=workers)

        # Type checking
        if not isinstance(p, float):
            if isinstance(p, int):
                p = float(p)
            else:
                raise TypeError("Optional parameter p must be of type int or float")
        if not isinstance(q, float):
            if isinstance(q, int):
                q = float(q)
            else:
                raise TypeError("Optional parameter q must be of type int or float")
        self.p = p
        self.q = q

    def walk(self, edges, nodes_of_interest=None):
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
                    logger.info(f"Node {node} does not exist in the graph. Ignoring it.")

        edgesJ = ArrayList()
        for edge in edges:
            newEdge = Edge(edge.src, edge.rel, edge.dst, edge.weight)
            edgesJ.add(newEdge)

        walker = N2V(edgesJ, self.num_walks, self.walk_length, self.p, self.q, self.workers,
                     self.outfile, nodes_of_interest)

        walker.walk()

        self.wait_for_all_walks()
