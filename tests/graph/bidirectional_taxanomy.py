from unittest import TestCase
import os
import requests

from mowl.datasets.base import RemoteDataset
from mowl.graph.util import parser_factory

from util import download




class Testing(TestCase):

    def setUp(self):

        try:
            self.ds = PathDataset("data/pizza.owl", None, None)
        except:
            download("https://github.com/owlcs/pizza-ontology/blob/master/pizza.owl", "data/pizza.owl")
            self.ds = PathDataset("data/pizza.owl", None, None)
        

    def test_bidirectional_taxonomy_TaxonomyParser(self):

        '''
        NBT: Without bidirectional taxonomy
        BT: With bidirectional taxonomy
        '''
        parserNBT = parser_factory("taxonomy", bidirectional_taxonomy = False)
        parserBT = parser_factory("taxonomy", bidirectional_taxonomy = True)

        edgesNBT = parserNBT.parse()
        edgesBT = parserBT.parse()

        self.assertEqual(len(edgesNBT), 2*len(edgesBT))
