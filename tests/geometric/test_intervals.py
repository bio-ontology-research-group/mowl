"""Tests for 1-D interval-union exact geometric model in mowl.geometric.intervals."""
from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric.checker import ModelChecker
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.errors import InconsistentOntology, InternalConsistencyError
from mowl.geometric.intervals import IntervalUnionModel
from mowl.geometric.normal_forms import BOTTOM, NF1, NF3, TOP
from mowl.geometric.regions import EMPTY_1D, IntervalUnion, RectUnion
from mowl.owlapi import OWLAPIAdapter
from tests.datasetFactory import FamilyDataset
from tests.geometric.helpers import assert_canonical_model_satisfies_axioms

NS = "http://mowl.borg/geo#"
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


class TestIntervalUnionModel(TestCase):

    def test_fixtures_and_family_are_exact_nondegenerate_strongly_faithful_models(
        self,
    ) -> None:
        sources = [
            ("t3", PathDataset(_fixture_path("t3.owl"))),
            ("helly_ok", PathDataset(_fixture_path("helly_ok.owl"))),
            ("roles", PathDataset(_fixture_path("roles.owl"))),
            ("top_lhs", PathDataset(_fixture_path("top_lhs.owl"))),
            ("family", FamilyDataset()),
        ]
        for label, ds in sources:
            with self.subTest(fixture=label):
                cm = CanonicalModel.from_dataset(ds)
                assert_canonical_model_satisfies_axioms(cm)

                ivl = IntervalUnionModel.from_canonical_model(cm)
                self.assertEqual(ivl.family, "interval")
                self.assertIsInstance(ivl.check(), ModelChecker)
                self.assertIs(ivl.check(), ivl.check())
                self.assertTrue(ivl.is_model())
                self.assertTrue(ivl.is_nondegenerate())
                self.assertTrue(ivl.is_strongly_faithful_on_names())

                for ax in cm.axioms:
                    self.assertTrue(ivl.satisfies(ax), f"Failed on {label}: {ax}")

    def test_from_dataset_and_from_ontology_constructors(self) -> None:
        ds = PathDataset(_fixture_path("roles.owl"))
        ivl_ds = IntervalUnionModel.from_dataset(ds)
        ivl_ont = IntervalUnionModel.from_ontology(ds.ontology)

        self.assertTrue(ivl_ds.is_model())
        self.assertTrue(ivl_ont.is_model())
        self.assertEqual(ivl_ds.universe, ivl_ont.universe)
        self.assertEqual(
            ivl_ds.class_region(NS + "A"), ivl_ont.class_region(NS + "A")
        )
        self.assertEqual(
            ivl_ds.role_region(NS + "r"), ivl_ont.role_region(NS + "r")
        )

    def test_t3_regions(self) -> None:
        ds = PathDataset(_fixture_path("t3.owl"))
        cm = CanonicalModel.from_dataset(ds)
        ivl = IntervalUnionModel.from_canonical_model(cm)

        A, B, C = NS + "A", NS + "B", NS + "C"
        X1, X2, X3 = NS + "X1", NS + "X2", NS + "X3"

        # Element intervals: position i -> [i + 1 - 0.25, i + 1 + 0.25]
        for i in range(len(cm.domain)):
            self.assertEqual(
                ivl.element_interval(i), (i + 1 - 0.25, i + 1 + 0.25)
            )
        with self.assertRaises(IndexError):
            ivl.element_interval(-1)
        with self.assertRaises(IndexError):
            ivl.element_interval(len(cm.domain))

        # class_intervals(A) is a single interval and A ∩ B is empty
        ivls_a = ivl.class_intervals(A)
        self.assertEqual(len(ivls_a), 1)
        pos_a = cm.index["d_" + A]
        self.assertEqual(ivls_a, [(pos_a + 1 - 0.25, pos_a + 1 + 0.25)])

        self.assertTrue(
            ivl.class_region(A).intersection(ivl.class_region(B)).is_empty
        )
        self.assertTrue(
            ivl.class_region(B).intersection(ivl.class_region(C)).is_empty
        )
        self.assertTrue(
            ivl.class_region(A).intersection(ivl.class_region(C)).is_empty
        )

        # Pairwise intersections of X1, X2, X3 match A, B, C; triple intersection is empty
        reg_x1 = ivl.class_region(X1)
        reg_x2 = ivl.class_region(X2)
        reg_x3 = ivl.class_region(X3)
        self.assertEqual(reg_x1.intersection(reg_x2), ivl.class_region(A))
        self.assertEqual(reg_x2.intersection(reg_x3), ivl.class_region(B))
        self.assertEqual(reg_x1.intersection(reg_x3), ivl.class_region(C))
        self.assertTrue(
            reg_x1.intersection(reg_x2).intersection(reg_x3).is_empty
        )

        # BOTTOM and unknown names map to empty IntervalUnion; unknown role maps to empty RectUnion
        self.assertEqual(ivl.class_region(BOTTOM), EMPTY_1D)
        self.assertTrue(ivl.class_region(BOTTOM).is_empty)
        self.assertEqual(ivl.class_intervals(BOTTOM), [])
        self.assertEqual(ivl.class_region(NS + "DoesNotExist"), EMPTY_1D)
        self.assertEqual(ivl.class_intervals(NS + "DoesNotExist"), [])
        self.assertEqual(ivl.role_region(NS + "unknownRole"), RectUnion([]))
        self.assertEqual(ivl.class_region(TOP), ivl.universe)

    def test_all_unsat_names(self) -> None:
        adapter = OWLAPIAdapter()
        mgr = adapter.owl_manager
        ont = adapter.create_ontology("http://mowl.borg/geo/all_unsat")
        A = NS + "A"
        B = NS + "B"
        cls_a = adapter.create_class(A)
        cls_b = adapter.create_class(B)
        bot = adapter.create_class(BOTTOM)
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_a, bot))
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_b, bot))

        cm = CanonicalModel.from_ontology(ont)
        assert_canonical_model_satisfies_axioms(cm)

        ivl = IntervalUnionModel.from_canonical_model(cm)
        self.assertTrue(ivl.class_region(A).is_empty)
        self.assertTrue(ivl.class_region(B).is_empty)
        self.assertEqual(ivl.class_intervals(A), [])
        self.assertFalse(ivl.universe.is_empty)
        self.assertEqual(ivl.universe, IntervalUnion([(0.75, 1.25)]))
        self.assertTrue(ivl.is_model())
        self.assertTrue(ivl.is_nondegenerate())
        self.assertTrue(ivl.is_strongly_faithful_on_names())

    def test_top_on_lhs(self) -> None:
        ds = PathDataset(_fixture_path("top_lhs.owl"))
        cm = CanonicalModel.from_dataset(ds)
        assert_canonical_model_satisfies_axioms(cm)

        ivl = IntervalUnionModel.from_canonical_model(cm)
        r = NS + "r"
        B = NS + "B"
        self.assertTrue(ivl.satisfies(NF3(TOP, r, B)))
        self.assertTrue(ivl.check().satisfies(NF3(TOP, r, B)))

    def test_inconsistent_propagates_and_internal_consistency_error_raised(
        self,
    ) -> None:
        adapter = OWLAPIAdapter()
        mgr = adapter.owl_manager
        ont = adapter.create_ontology("http://mowl.borg/geo/inconsistent_ivl")
        top = adapter.create_class(TOP)
        bot = adapter.create_class(BOTTOM)
        cls_a = adapter.create_class(NS + "A")
        mgr.addAxiom(ont, adapter.create_subclass_of(top, cls_a))
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_a, bot))

        with self.assertRaises(InconsistentOntology):
            IntervalUnionModel.from_ontology(ont)

        # Verify constructor raises InternalConsistencyError listing violated axioms if checker fails
        ds = PathDataset(_fixture_path("t3.owl"))
        cm = CanonicalModel.from_dataset(ds)
        bad_axiom = NF1(TOP, BOTTOM)
        with patch.object(
            ModelChecker, "violations", return_value=[bad_axiom]
        ):
            with self.assertRaises(InternalConsistencyError) as ctx:
                IntervalUnionModel.from_canonical_model(cm)
            self.assertIn(repr(bad_axiom), str(ctx.exception))
