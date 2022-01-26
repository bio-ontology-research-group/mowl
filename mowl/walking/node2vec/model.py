from mowl.walking.walking import WalkingModel
import logging

from java.util import ArrayList
from org.mowl import Edge
from org.mowl.Walking import Node2Vec as N2V

class Node2Vec(WalkingModel):
    
    '''
    Reference implementation of node2vec. 

    :param p: Return hyperparameter. Default is 1.
    :type p: float
    :param q: In-out hyperparameter. Default is 1.
    :type q: float

    Original author: Aditya Grover

    For more details, refer to the paper:
    node2vec: Scalable Feature Learning for Networks
    Aditya Grover and Jure Leskovec 
    Knowledge Discovery and Data Mining (KDD), 2016

    Adapted by Sonja Katz, 2021
    Readapted to subclass WalkingModel abstract class. By F. Zhapa 
    '''

    def __init__(self, 
                 edges, 
		 num_walks, 
		 walk_length, 
		 p, 
		 q,
                 num_workers=1,
        	 outfile = None): 
        
        super().__init__(edges, num_walks, walk_length, num_workers, outfile) 

        self.p = p
        self.q = q

    def walk(self):

        edgesJ = ArrayList()

        for edge in self.edges:
            newEdge = Edge(edge.src(), edge.dst(), edge.weight())

            edgesJ.add(newEdge)

        walker = N2V(edgesJ, self.num_walks, self.walk_length, self.p, self.q, self.num_workers, self.outfile)

        walker.walk()
            
        

        
