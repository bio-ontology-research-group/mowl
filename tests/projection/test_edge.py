from pykeen.triples.triples_factory import TriplesFactory
from mowl.projection import Edge
from unittest import TestCase


class TestEdge(TestCase):

    def test_edge_parameter_types(self):
        """This checks if type of parameters is checked"""

        self.assertRaisesRegex(TypeError, "Parameter src must be a string", Edge, 1, "rel", "dst")
        self.assertRaisesRegex(TypeError, "Parameter rel must be a string", Edge, "src", 1, "dst")
        self.assertRaisesRegex(TypeError, "Parameter dst must be a string", Edge, "src", "rel", 1)
        self.assertRaisesRegex(TypeError, "Optional parameter weight must be a float", Edge,
                               "src", "rel", "dst", weight=1)

    def test_edge_attributes(self):
        """This checks if Edge attributes are set correctly"""

        edge = Edge("src", "rel", "dst")
        self.assertEqual(edge.src, "src")
        self.assertEqual(edge.rel, "rel")
        self.assertEqual(edge.dst, "dst")
        self.assertEqual(edge.weight, 1)

    def test_as_tuple_method(self):
        """This checks if Edge.as_tuple method works correctly"""

        edge = Edge("src", "rel", "dst")
        self.assertEqual(edge.astuple(), ("src", "rel", "dst"))

    def test_get_ents_and_rels_method(self):
        """This checks if Edge.get_entities_and_relations method works correctly"""

        edge1 = Edge("src1", "rel1", "dst1")
        edge2 = Edge("src2", "rel2", "dst2")

        ents, rels = Edge.get_entities_and_relations([edge1, edge2])

        self.assertEqual(ents, ["dst1", "dst2", "src1", "src2"])
        self.assertEqual(rels, ["rel1", "rel2"])

    def test_zip_method(self):
        """This checks if Edge.zip method works correctly"""

        edge1 = Edge("src1", "rel1", "dst1")
        edge2 = Edge("src2", "rel2", "dst2")

        srcs, rels, dsts = Edge.zip([edge1, edge2])

        self.assertEqual(srcs, ("src1", "src2"))
        self.assertEqual(rels, ("rel1", "rel2"))
        self.assertEqual(dsts, ("dst1", "dst2"))

    def test_as_pykeen_method(self):
        """This checks if Edge.as_pykeen method works correctly"""

        edge1 = Edge("src1", "rel1", "dst1")
        edge2 = Edge("src2", "rel2", "dst2")

        triples = Edge.as_pykeen([edge1, edge2])

        self.assertIsInstance(triples, TriplesFactory)
