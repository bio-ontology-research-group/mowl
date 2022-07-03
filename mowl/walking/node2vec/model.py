from mowl.walking.walking import WalkingModel
import logging
import tempfile
from java.util import ArrayList
from org.mowl import Edge
from org.mowl.Walking import Node2Vec as N2V

class Node2Vec(WalkingModel):

    '''
    :param p: Return hyperparameter. Default is 1.
    :type p: float
    :param q: In-out hyperparameter. Default is 1.
    :type q: float
    :param outfile: Path for saving the generated walks, defaults to :class:`tempfile.NamedTemporaryFile`"
    :type outfile: str, optional
    '''
    def __init__(self,
		 num_walks,
		 walk_length,
		 p,
		 q,
                 outfile = None,
                 workers=1
                 ):

        if outfile is None:
            tmp_file = tempfile.NamedTemporaryFile()
            outfile = tmp_file.name

        super().__init__(num_walks, walk_length, outfile, workers) 

        self.p = p
        self.q = q

    def walk(self, edges):

        edgesJ = ArrayList()

        for edge in edges:
            newEdge = Edge(edge.src(), edge.rel(),  edge.dst(), edge.weight())

            edgesJ.add(newEdge)

        walker = N2V(edgesJ, self.num_walks, self.walk_length, self.p, self.q, self.workers, self.outfile)

        walker.walk()
            
        

        
