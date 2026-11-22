"""The walkers empty their output file before a walk (DeepWalk's Java code appends to it);
``append=True`` keeps the old behaviour; node2vec says that it is not seeded."""
import os
import tempfile
from unittest import TestCase

import mowl
mowl.init_jvm("1g")

from mowl.projection import Edge
from mowl.walking import DeepWalk, Node2Vec, walker_factory

EDGES = [Edge("A", "http://rel1", "B"), Edge("B", "http://rel1", "C"),
         Edge("C", "http://rel1", "D"), Edge("B", "http://rel2", "D")]


def lines(path):
    with open(path) as f:
        return f.readlines()


class Outfile(TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.path = os.path.join(self.dir.name, "walks.txt")

    def tearDown(self):
        self.dir.cleanup()

    def test_a_second_walk_replaces_the_first(self):
        walker = DeepWalk(5, 4, outfile=self.path, seed=0)
        walker.walk(EDGES)
        first = lines(self.path)
        walker.walk(EDGES)
        self.assertEqual(lines(self.path), first)

    def test_a_file_left_from_another_run_is_emptied(self):
        with open(self.path, "w") as f:
            f.write("old walk\n")
        DeepWalk(5, 4, outfile=self.path, seed=0).walk(EDGES)
        self.assertNotIn("old walk\n", lines(self.path))

    def test_append_keeps_the_old_behaviour(self):
        walker = DeepWalk(5, 4, outfile=self.path, seed=0, append=True)
        walker.walk(EDGES)
        first = lines(self.path)
        walker.walk(EDGES)
        self.assertEqual(len(lines(self.path)), 2 * len(first))

    def test_node2vec_walks_replace_too_and_say_they_are_unseeded(self):
        with open(self.path, "w") as f:
            f.write("old walk\n")
        walker = Node2Vec(5, 4, outfile=self.path)
        walker.walk(EDGES)
        self.assertNotIn("old walk\n", lines(self.path))
        self.assertFalse(walker.seeded)
        self.assertTrue(DeepWalk(5, 4, outfile=self.path).seeded)
        self.assertFalse(walker_factory("node2vec", 5, 4).seeded)

    def test_node2vec_accepts_nodes_of_interest(self):
        walker = Node2Vec(5, 4, outfile=self.path)
        walker.walk(EDGES, nodes_of_interest=["A", "missing"])
        self.assertTrue(lines(self.path))
