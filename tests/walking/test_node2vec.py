from mowl.walking import Node2Vec
from mowl.projection import Edge
from unittest import TestCase


class TestNode2Vec(TestCase):

    @classmethod
    def setUpClass(self):

        edge1 = Edge("A", "http://rel1", "B")
        edge2 = Edge("B", "http://rel1", "C")
        edge3 = Edge("C", "http://rel1", "D")
        edge4 = Edge("B", "http://rel2", "D")
        edge5 = Edge("A", "http://rel1", "C")
        edge6 = Edge("C", "http://rel2", "D")

        self.graph = [edge1, edge2, edge3, edge4, edge5, edge6]
        self.nodes, self.rels = Edge.get_entities_and_relations(self.graph)

    def test_node2vec_raise_error_with_incorrect_types(self):
        """This method tests if the exception is raised when the types are incorrect"""
        num_walks = "10"
        walk_length = 5
        self.assertRaisesRegex(TypeError, "Parameter num_walks must be an integer", Node2Vec,
                               num_walks, walk_length)

        num_walks = 10
        walk_length = "5"
        self.assertRaisesRegex(TypeError, "Parameter walk_length must be an integer", Node2Vec,
                               num_walks, walk_length)

        num_walks = 10
        walk_length = 5
        workers = "1"
        self.assertRaisesRegex(TypeError, "Optional parameter workers must be an integer",
                               Node2Vec, num_walks, walk_length, workers=workers)

        num_walks = 10
        walk_length = 5
        outfile = 16
        self.assertRaisesRegex(TypeError, "Optional parameter outfile must be a string",
                               Node2Vec, num_walks, walk_length, outfile=outfile)

        num_walks = 10
        walk_length = 5
        p = "0.1"
        self.assertRaisesRegex(TypeError,
                               "Optional parameter p must be of type int or float",
                               Node2Vec,
                               num_walks, walk_length, p=p)

        num_walks = 10
        walk_length = 5
        q = "0.1"
        self.assertRaisesRegex(TypeError,
                               "Optional parameter q must be of type int or float",
                               Node2Vec,
                               num_walks, walk_length, q=q)

    def test_node2vec_number_of_walks(self):
        """This method tests if the number of walks is correct"""
        num_walks = 10
        walk_length = 5
        node2vec = Node2Vec(num_walks, walk_length)
        node2vec.walk(self.graph)
        with open(node2vec.outfile, "r") as f:
            walks = f.readlines()

        self.assertEqual(len(walks), num_walks * len(self.nodes))

    def test_walking_with_list_of_nodes_ignore_unknown_nodes(self):
        """This method tests if the walking ignores unknown nodes"""
        num_walks = 10
        walk_length = 5
        node2vec = Node2Vec(num_walks, walk_length)

        with self.assertLogs("node2vec", level='INFO') as cm:
            node2vec.walk(self.graph, nodes_of_interest=["A", "X"])

        self.assertEqual(cm.output, ["INFO:node2vec:Node X does not exist in the graph. \
Ignoring it."])
