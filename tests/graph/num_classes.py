import sys
sys.path.append("../../")
from unittest import TestCase
import os
import requests

import mowl
mowl.init_jvm("2g")
from mowl.datasets.base import PathDataset
from mowl.graph.taxonomy.model import TaxonomyParser
from mowl.graph.edge import Edge
from mowl.graph.util import prettyFormat

from util import download

onts = {
    "pizza.owl": "https://raw.githubusercontent.com/owlcs/pizza-ontology/master/pizza.owl",
    "goslim_yeast.owl": "http://current.geneontology.org/ontology/subsets/goslim_yeast.owl"
    
}


class TestingNumClasses(TestCase):

    def setUp(self):
        self.ont = "pizza.owl"
#        self.ont = "goslim_yeast.owl"
        
        if os.path.exists("data/" + self.ont):
            self.ds = PathDataset("data/" + self.ont, None, None)
        else:
            url = onts[self.ont]
            download(url, "data")
            self.ds = PathDataset("data/" + self.ont, None, None)

    def test_num_classes_num_nodes(self):

        classes = self.ds.ontology.getClassesInSignature(False)

        parser = TaxonomyParser(self.ds.ontology, bidirectional_taxonomy = False)
        edges = parser.parse()

        nodes, _ = Edge.getEntitiesAndRelations(edges)

        for c in classes:
            c = prettyFormat(str(c))
            if not c in nodes:
                print(c)
                            
        err_msg = f"Error: Test on {self.ont}. Number of nodes is {len(nodes)} and number of classes is {len(classes)}"

        self.assertEqual(len(classes), len(nodes)), err_msg

if __name__ == '__main__':
    unittest.main()
