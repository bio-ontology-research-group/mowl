import sys
sys.path.append('../../')
from unittest import TestCase
import os
import requests

import mowl
mowl.init_jvm("2g")

from mowl.datasets.base import PathDataset
from mowl.graph.util import parser_factory

from util import download




class TestingBT(TestCase):

    
    def setUp(self):

        if os.path.exists("data/pizza.owl"):
            self.ds = PathDataset("data/pizza.owl", None, None)
        else:
            url = "https://raw.githubusercontent.com/owlcs/pizza-ontology/master/pizza.owl"
            download(url, "data")
            self.ds = PathDataset("data/pizza.owl", None, None)


    def aux_test_bidirectional_taxonomy(self, method_name):

        parser = parser_factory(method_name, self.ds.ontology, bidirectional_taxonomy = True)

        edges = parser.parse()

        #The split part is needed for OWL2Vec
        subClassEdges = sum([1 if e.rel().split("#")[-1] == "subClassOf" else 0 for e in edges])
        superClassEdges = sum([1 if e.rel().split('#')[-1] == "superClassOf" else 0 for e in edges])


        self.assertNotEqual(0, subClassEdges, f"Method {method_name}: 0 subclass edges")
        self.assertNotEqual(0, superClassEdges, f"Method {method_name}: 0 superclass edges")
        self.assertEqual(subClassEdges, superClassEdges, f"Error in method: {method_name}")


    def test_all_BT(self):
        methods = ["taxonomy", "taxonomy_rels", "dl2vec", "owl2vec_star"]

        for method in methods:
            self.aux_test_bidirectional_taxonomy(method)
        
if __name__ == '__main__':
    unittest.main()
