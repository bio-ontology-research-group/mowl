from mowl.walking import DeepWalk
from mowl.projection import Edge
from unittest import TestCase
import os


class TestDeepWalk(TestCase):

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

    def test_deepwalk_raise_error_with_incorrect_types(self):
        """This method tests if the exception is raised when the types are incorrect"""
        num_walks = "10"
        walk_length = 5
        self.assertRaisesRegex(TypeError, "Parameter num_walks must be an integer", DeepWalk,
                               num_walks, walk_length)

        num_walks = 10
        walk_length = "5"
        self.assertRaisesRegex(TypeError, "Parameter walk_length must be an integer", DeepWalk,
                               num_walks, walk_length)

        num_walks = 10
        walk_length = 5
        alpha = "0.1"
        self.assertRaisesRegex(TypeError, "Optional parameter alpha must be a float", DeepWalk,
                               num_walks, walk_length, alpha=alpha)

        num_walks = 10
        walk_length = 5
        workers = "1"
        self.assertRaisesRegex(TypeError, "Optional parameter workers must be an integer",
                               DeepWalk, num_walks, walk_length, workers=workers)

        num_walks = 10
        walk_length = 5
        outfile = 16
        self.assertRaisesRegex(TypeError, "Optional parameter outfile must be a string",
                               DeepWalk, num_walks, walk_length, outfile=outfile)

    def test_deepwalk_number_of_walks(self):
        """This method tests if the number of walks is correct"""
        num_walks = 10
        walk_length = 5
        deepwalk = DeepWalk(num_walks, walk_length)
        deepwalk.walk(self.graph)
        with open(deepwalk.outfile, "r") as f:
            walks = f.readlines()

        self.assertEqual(len(walks), num_walks * len(self.nodes))

    def test_walking_with_list_of_nodes_ignore_unknown_nodes(self):
        """This method tests if the walking ignores unknown nodes when list of nodes specified"""
        num_walks = 10
        walk_length = 5
        node2vec = DeepWalk(num_walks, walk_length)

        with self.assertLogs("deepwalk", level='INFO') as cm:
            node2vec.walk(self.graph, nodes_of_interest=["A", "X"])

        self.assertEqual(cm.output, ["INFO:deepwalk:Node X does not exist in the graph. \
Ignoring it."])

    def test_passing_outfile_name(self):
        """This checks that outfile name passed to walking method is created"""
        num_walks = 10
        walk_length = 5
        outfile = "test_outfile.txt"
        deepwalk = DeepWalk(num_walks, walk_length, outfile=outfile)

        deepwalk.walk(self.graph)
        self.assertTrue(os.path.exists(outfile))
        os.remove(outfile)

    def test_walking_on_updated_graph(self):
        """This should test that walks file get updated (not overwritten) when walking on updated graph"""
        num_walks = 10
        walk_length = 2
        
        deepwalk = DeepWalk(num_walks, walk_length)
        deepwalk.walk(self.graph)
        with open(deepwalk.outfile, "r") as f:
            walks = f.readlines()

        current_walks = len(walks)

        edge7 = Edge("E", "http://rel1", "A")
        self.graph.append(edge7)
        deepwalk.walk(self.graph, nodes_of_interest=["E"])
        with open(deepwalk.outfile, "r") as f:
            walks = f.readlines()
        new_walks = len(walks)

        self.assertEqual(new_walks, current_walks + 10)

        current_walks = new_walks + 10
        edge8 = Edge("B", "http://rel1", "E")
        self.graph.append(edge8)
        deepwalk.walk(self.graph, nodes_of_interest=["E"])
        with open(deepwalk.outfile, "r") as f:
            walks = f.readlines()
        new_walks = len(walks)

        self.assertGreater(new_walks, current_walks)
