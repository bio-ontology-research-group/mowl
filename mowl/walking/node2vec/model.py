from mowl.walking.walking import WalkingModel
import logging

from java.util import ArrayList
from org.mowl import Edge
from org.mowl.Walking import Node2Vec as N2V

class Node2Vec(WalkingModel):

    '''
    :param p: Return hyperparameter. Default is 1.
    :type p: float
    :param q: In-out hyperparameter. Default is 1.
    :type q: float
    '''
    def __init__(self,
		 num_walks,
		 walk_length,
		 p,
		 q,
                 outfile,
                 workers=1
                 ):
        
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
            
        

        
