"""Every catalogued method states its parts, and the listings use them."""
from unittest import TestCase

from mowl.catalog import list_methods, get_method, catalogue_to_rst
from mowl.representation import (REPRESENTATIONS, Representation, ClassModel, RelationModel,
                                 IndividualModel, Measure, View, Space, Construction,
                                 Composable, Parametrisation)
from mowl.benchmarks import methods_for, get_benchmark
from mowl.benchmarks.registry import missing_support


class TestParts(TestCase):

    def test_every_method_has_parts_and_no_entry_is_orphaned(self):
        keys = {c.key for c in list_methods()}
        self.assertEqual(keys, set(REPRESENTATIONS))
        for card in list_methods():
            self.assertIsInstance(card.representation, Representation, card.key)

    def test_values_are_coerced_and_checked(self):
        rep = Representation(view="graph", space="euclidean", classes=("box", "ball"),
                             relations="translation", individuals="point",
                             measure="bounds per coordinate", conjunction="exact")
        self.assertEqual(rep.classes, (ClassModel.BOX, ClassModel.BALL))
        self.assertIs(rep.conjunction, Construction.EXACT)
        self.assertTrue(rep.has(ClassModel.BALL) and rep.has("translation"))
        self.assertFalse(rep.has(ClassModel.DISK))
        with self.assertRaises(ValueError):
            Representation(view="graph", space="euclidean", classes="blob",
                           relations="none", individuals="none", measure="exact")

    def test_single_part_changes_of_the_literature(self):
        def parts(key):
            return get_method(key).representation
        # ELBE = ELEmbeddings with boxes; RegD = ELBE with another measure.
        self.assertEqual(parts("elem").relations, parts("elbe").relations)
        self.assertNotEqual(parts("elem").classes, parts("elbe").classes)
        self.assertEqual((parts("regd").classes, parts("regd").relations),
                         (parts("elbe").classes, parts("elbe").relations))
        self.assertNotEqual(parts("regd").measure, parts("elbe").measure)
        # TransBox and Query2Box share the relation model, not the view.
        self.assertEqual(parts("transbox").relations, parts("query2box").relations)
        self.assertNotEqual(parts("transbox").view, parts("query2box").view)
        # ball and ball is not a ball
        self.assertIs(parts("elem").conjunction, Construction.APPROXIMATE)
        self.assertIs(parts("elbe").conjunction, Construction.EXACT)

    def test_listing_by_parts(self):
        boxes = {c.key for c in list_methods(parts=(ClassModel.BOX, View.NORMAL_FORMS))}
        self.assertEqual(boxes, {"box2el", "boxel", "boxsel", "composed", "elbe", "elkem", "regd",
                                 "transbox"})
        self.assertEqual({c.key for c in list_methods(parts=Composable.MOSTLY)},
                         {"box2el", "multiboxel", "bcbox"})
        self.assertEqual({c.key for c in list_methods(inductive=True)},
                         {"eike", "hit", "ont", "reshuffle"})

    def test_inductive_benchmarks_ask_for_unseen_names(self):
        bench = get_benchmark("kg-inductive-fb15k237-v1")
        self.assertEqual(bench.requires, ("unseen names",))
        self.assertEqual({c.key for c in methods_for(bench)}, {"reshuffle", "ont"})
        self.assertNotIn("expressive", {c.key for c in methods_for(bench)})
        self.assertTrue(any("not seen in training" in m
                            for m in missing_support("expressive", bench)))
        self.assertEqual(get_benchmark("kg-fb15k237-link-prediction").requires, ())

    def test_rendered(self):
        text = catalogue_to_rst()
        self.assertIn("The parts of every method", text)
        self.assertIn("box of translations", text)
        rows = dict(get_method("dfalc").representation.rows())
        self.assertEqual(rows["An individual is"], "element of a finite domain")
        self.assertIn("exists: fuzzy operator", rows["Compound classes"])
