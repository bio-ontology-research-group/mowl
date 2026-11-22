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


#: Methods that cannot be built without a text encoder; their geometry is checked in the tests
#: of the language-model methods.
NEEDS_ENCODER = {"hit", "ont"}
#: Disagreements between the catalogue and the geometry that are known and not resolved,
#: as ``(key, field): reason``. A listed pair is skipped; the list must not grow silently.
KNOWN_DISAGREEMENTS = {}


def _build(key):
    from mowl.representation import METHODS, METHOD_KINDS
    if METHOD_KINDS[key] == "queries":
        return METHODS[key](nb_classes=6, nb_relations=6, num_classes=3, embed_dim=8)
    return METHODS[key](5, 2, 3, 8)


class TestGeometryAgreesWithCatalogue(TestCase):
    """A method that is rebuilt from parts names the same class model, relation model and
    measure as its catalogue entry says."""

    def values(self, rep, field):
        return {v.value for v in getattr(rep, field)}

    def test_every_rebuilt_method_has_an_entry(self):
        from mowl.representation import METHODS
        self.assertLessEqual(set(METHODS), set(REPRESENTATIONS))

    def test_parts_agree(self):
        from mowl.representation import METHODS
        disagreements = []
        for key in sorted(set(METHODS) - NEEDS_ENCODER):
            parts = _build(key).parts
            rep = REPRESENTATIONS[key]
            for field in ("classes", "relations", "measure"):
                if (key, field) in KNOWN_DISAGREEMENTS:
                    continue
                if parts[field] not in self.values(rep, field):
                    disagreements.append(
                        (key, field, parts[field], sorted(self.values(rep, field))))
        self.assertEqual(disagreements, [])

    def test_individuals_and_conjunction_agree(self):
        from mowl.representation import METHODS
        problems = []
        for key in sorted(set(METHODS) - NEEDS_ENCODER):
            parts = _build(key).parts
            rep = REPRESENTATIONS[key]
            if parts.get("individuals") == "none" and IndividualModel.NONE not in rep.individuals:
                problems.append((key, "individuals", parts["individuals"]))
            # a method without compound classes (NONE) does not use the constructor of
            # its region; a learned conjunction is realised as an approximation
            kind = parts.get("conjunction", "none")
            catalogue = rep.conjunction
            allowed = {Construction.NONE, Construction(kind) if kind in (
                c.value for c in Construction) else None}
            if kind == "approximate":
                allowed.add(Construction.LEARNED)
            if catalogue not in allowed:
                problems.append((key, "conjunction", kind, catalogue.value))
        self.assertEqual(problems, [])

    def test_fuzzy_methods_name_their_operator_family(self):
        from mowl.representation import METHODS
        for key in sorted(set(METHODS) - NEEDS_ENCODER):
            if "family" in _build(key).parts:
                self.assertTrue(REPRESENTATIONS[key].operators, key)
                self.assertIs(REPRESENTATIONS[key].conjunction, Construction.FUZZY, key)

    def test_vocabulary_has_the_parts_the_code_has(self):
        from mowl.representation.vocabulary import OperatorFamily
        from mowl.representation.library.logic_fuzzy import get_family
        self.assertIn(ClassModel.LINKED, REPRESENTATIONS["eike"].classes)
        self.assertIn(ClassModel.LINKED, REPRESENTATIONS["concept2box"].classes)
        self.assertEqual(RelationModel.HEAD_TAIL_BALLS.value, "head and tail balls")
        for family in (OperatorFamily.PRODUCT, OperatorFamily.GOEDEL,
                       OperatorFamily.LUKASIEWICZ):
            self.assertIsNotNone(get_family(family.value))
        rows = dict(REPRESENTATIONS["falcon"].rows())
        self.assertIn("product", rows["Fuzzy operators"])
