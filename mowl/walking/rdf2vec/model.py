from mowl.walking.walking import WalkingModel
import random
import os
import logging

from org.mowl.Walking import RDF2Vec as R2V
from java.util import HashMap
from java.util import ArrayList
from org.mowl import Edge
logging.basicConfig(level=logging.INFO)


class RDF2Vec(WalkingModel):

    '''
    Implementation of `Walking RDF and OWL <https://github.com/bio-ontology-research-group/walking-rdf-and-owl>`_
    '''
    
    def __init__(self,
                 edges,
                 num_walks,
                 walk_length,
                 outfile,
                 workers=1,
):

        super().__init__(edges, num_walks, walk_length, outfile, workers)


    def walk(self):

        edgesJ = ArrayList()
        for edge in self.edges:
            newEdge = Edge(edge.src(), edge.rel(), edge.dst())
            edgesJ.add(newEdge)

        walker = R2V(edgesJ, self.num_walks, self.walk_length, self.workers, self.outfile)

        walker.walk()
