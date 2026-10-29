"""Tests for convex polyhedral exact geometric model in mowl.geometric.polyhedral."""
from __future__ import annotations

import os
import random
import subprocess
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric.checker import ModelChecker
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.errors import (
    InconsistentOntology,
    InternalConsistencyError,
    UnsupportedAxiom,
)
from mowl.geometric.normal_forms import (
    BOTTOM,
    NF1,
    NF2,
    NF3,
    NF4,
    NF6,
    NF7,
    TOP,
    NormalisedOntology,
)
from mowl.geometric.polyhedral import ConvexPolyhedralModel
from mowl.geometric.regions import Face, PairHull, ThickFace, simplex_points
from mowl.owlapi import OWLAPIAdapter
from tests.datasetFactory import FamilyDataset
from tests.geometric.helpers import assert_canonical_model_satisfies_axioms

NS = "http://mowl.borg/geo#"
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


class TestConvexPolyhedralModel(TestCase):

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

                poly = ConvexPolyhedralModel.from_canonical_model(cm)
                self.assertEqual(poly.family, "polyhedral")
                self.assertEqual(poly.dimension, len(cm.names) - 1)
                self.assertIsInstance(poly.check(), ModelChecker)
                self.assertIs(poly.check(), poly.check())
                self.assertTrue(poly.is_model())
                self.assertTrue(poly.is_nondegenerate())
                self.assertTrue(poly.is_strongly_faithful_on_names())

                for ax in cm.axioms:
                    self.assertTrue(poly.satisfies(ax), f"Failed on {label}: {ax}")

    def test_from_dataset_and_from_ontology_constructors(self) -> None:
        ds = PathDataset(_fixture_path("roles.owl"))
        poly_ds = ConvexPolyhedralModel.from_dataset(ds)
        poly_ont = ConvexPolyhedralModel.from_ontology(ds.ontology)

        self.assertTrue(poly_ds.is_model())
        self.assertTrue(poly_ont.is_model())
        self.assertEqual(poly_ds.dimension, poly_ont.dimension)
        self.assertEqual(poly_ds.universe, poly_ont.universe)
        self.assertEqual(
            poly_ds.class_region(NS + "A"), poly_ont.class_region(NS + "A")
        )
        np.testing.assert_array_equal(
            poly_ds.role_region(NS + "r").points,
            poly_ont.role_region(NS + "r").points,
        )

    def test_t3_dimension(self) -> None:
        ds = PathDataset(_fixture_path("t3.owl"))
        cm = CanonicalModel.from_dataset(ds)
        poly = ConvexPolyhedralModel.from_canonical_model(cm)

        n = len(cm.names)
        k = n - 1
        self.assertEqual(poly.dimension, k)
        pts = poly.element_points()
        self.assertEqual(pts.shape, (n, k))
        np.testing.assert_array_equal(pts, simplex_points(n))

        A, B, C = NS + "A", NS + "B", NS + "C"
        X1, X2, X3 = NS + "X1", NS + "X2", NS + "X3"

        # class_points(A) returns rows of A's vertices
        pos_a = sorted(cm.extension(A))
        np.testing.assert_array_equal(poly.class_points(A), pts[pos_a])

        # Pairwise intersections of X1, X2, X3 match A, B, C; triple intersection is empty
        reg_x1 = poly.class_region(X1)
        reg_x2 = poly.class_region(X2)
        reg_x3 = poly.class_region(X3)
        self.assertEqual(reg_x1.intersection(reg_x2), poly.class_region(A))
        self.assertEqual(reg_x2.intersection(reg_x3), poly.class_region(B))
        self.assertEqual(reg_x1.intersection(reg_x3), poly.class_region(C))
        self.assertTrue(
            reg_x1.intersection(reg_x2).intersection(reg_x3).is_empty
        )

        # Pairwise intersections of A, B, C are empty
        self.assertTrue(
            poly.class_region(A).intersection(poly.class_region(B)).is_empty
        )
        self.assertTrue(
            poly.class_region(B).intersection(poly.class_region(C)).is_empty
        )
        self.assertTrue(
            poly.class_region(A).intersection(poly.class_region(C)).is_empty
        )

        # BOTTOM and unknown names map to empty Face; unknown role maps to empty PairHull
        self.assertTrue(poly.class_region(BOTTOM).is_empty)
        self.assertEqual(poly.class_points(BOTTOM).shape, (0, k))
        self.assertTrue(poly.class_region(NS + "DoesNotExist").is_empty)
        self.assertEqual(poly.class_points(NS + "DoesNotExist").shape, (0, k))
        self.assertTrue(poly.role_region(NS + "unknownRole").is_empty)
        self.assertEqual(
            poly.role_region(NS + "unknownRole").points.shape, (0, 2 * k)
        )
        self.assertEqual(poly.class_region(TOP), poly.universe)
        self.assertEqual(poly.universe.vertex_ids, frozenset(range(n)))

    def test_all_unsat_names(self) -> None:
        adapter = OWLAPIAdapter()
        mgr = adapter.owl_manager
        ont = adapter.create_ontology("http://mowl.borg/geo/all_unsat_poly")
        A = NS + "A"
        B = NS + "B"
        cls_a = adapter.create_class(A)
        cls_b = adapter.create_class(B)
        bot = adapter.create_class(BOTTOM)
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_a, bot))
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_b, bot))

        cm = CanonicalModel.from_ontology(ont)
        assert_canonical_model_satisfies_axioms(cm)

        poly = ConvexPolyhedralModel.from_canonical_model(cm)
        self.assertEqual(poly.dimension, 0)
        self.assertEqual(poly.element_points().shape, (1, 0))
        self.assertTrue(poly.class_region(A).is_empty)
        self.assertTrue(poly.class_region(B).is_empty)
        self.assertEqual(poly.class_points(A).shape, (0, 0))
        self.assertFalse(poly.universe.is_empty)
        self.assertTrue(poly.is_model())
        self.assertTrue(poly.is_nondegenerate())
        self.assertTrue(poly.is_strongly_faithful_on_names())

    def test_top_on_lhs_gci2(self) -> None:
        ds = PathDataset(_fixture_path("top_lhs.owl"))
        cm = CanonicalModel.from_dataset(ds)
        assert_canonical_model_satisfies_axioms(cm)

        poly = ConvexPolyhedralModel.from_canonical_model(cm)
        r = NS + "r"
        B = NS + "B"
        self.assertTrue(poly.satisfies(NF3(TOP, r, B)))
        self.assertTrue(poly.check().satisfies(NF3(TOP, r, B)))

    def test_random_role_tboxes(self) -> None:
        names = [f"{NS}N{i}" for i in range(6)]
        roles = [f"{NS}r0", f"{NS}r1"]
        axiom_kinds = ("NF1", "NF2", "NF3", "NF4", "NF6")

        consistent_count = 0
        for seed in range(30):
            rng = random.Random(seed)
            n_axioms = rng.randint(5, 10)
            axioms = []
            for _ in range(n_axioms):
                kind = rng.choice(axiom_kinds)
                if kind == "NF1":
                    sub = rng.choice(names)
                    sup_pool = [c for c in names if c != sub] + [BOTTOM]
                    sup = rng.choice(sup_pool)
                    axioms.append(NF1(sub, sup))
                elif kind == "NF2":
                    sub1, sub2 = rng.sample(names, 2)
                    sup_pool = [c for c in names if c not in (sub1, sub2)] + [BOTTOM]
                    sup = rng.choice(sup_pool)
                    axioms.append(NF2(sub1, sub2, sup))
                elif kind == "NF3":
                    sub, filler = rng.sample(names, 2)
                    role = rng.choice(roles)
                    axioms.append(NF3(sub, role, filler))
                elif kind == "NF4":
                    filler = rng.choice(names)
                    sup_pool = [c for c in names if c != filler] + [BOTTOM]
                    sup = rng.choice(sup_pool)
                    role = rng.choice(roles)
                    axioms.append(NF4(role, filler, sup))
                else:  # NF6
                    r_sub, r_sup = rng.sample(roles, 2)
                    axioms.append(NF6(r_sub, r_sup))

            no = NormalisedOntology(
                axioms=axioms,
                concept_names=set(names) | {TOP, BOTTOM},
                role_names=set(roles),
                dropped_axioms=0,
            )
            try:
                cm = CanonicalModel.from_normalised(no)
            except InconsistentOntology:
                continue

            consistent_count += 1
            with self.subTest(seed=seed):
                assert_canonical_model_satisfies_axioms(cm)
                poly = ConvexPolyhedralModel.from_canonical_model(cm)
                self.assertTrue(poly.is_model())
                self.assertTrue(poly.is_nondegenerate())
                self.assertTrue(poly.is_strongly_faithful_on_names())

        self.assertGreater(consistent_count, 0)

    def test_thicken_propositional(self) -> None:
        ds_helly = PathDataset(_fixture_path("helly_ok.owl"))
        cm = CanonicalModel.from_dataset(ds_helly)
        poly_thick = ConvexPolyhedralModel.from_canonical_model(cm, thicken=True)
        poly_thin = ConvexPolyhedralModel.from_canonical_model(cm, thicken=False)

        n = len(cm.names)
        expected_eps = 1.0 / (2 * n)

        self.assertTrue(poly_thick.is_model())
        self.assertTrue(poly_thick.is_nondegenerate())
        self.assertTrue(poly_thick.is_strongly_faithful_on_names())
        self.assertIsInstance(poly_thick.universe, ThickFace)
        self.assertAlmostEqual(poly_thick.universe.eps, expected_eps)

        # Universe centroid has all barycentric coordinates equal to 1 / n
        univ_centroid = poly_thick.element_points().mean(axis=0)
        self.assertTrue(poly_thick.universe.contains(univ_centroid))

        # Every satisfiable class region contains an interior point shifted toward universe centroid
        for cname in cm.names:
            reg_thick = poly_thick.class_region(cname)
            reg_thin = poly_thin.class_region(cname)
            self.assertIsInstance(reg_thick, ThickFace)
            self.assertAlmostEqual(reg_thick.eps, expected_eps)

            face_centroid = reg_thick.vertices.mean(axis=0)
            # Shift face centroid 25% of the way to the universe centroid:
            # for j not in S, lambda_j = 0.25 / n = eps / 2 < eps, and for j in S, lambda_j > 0.
            shifted = 0.75 * face_centroid + 0.25 * univ_centroid
            self.assertTrue(
                reg_thick.contains(shifted),
                f"Thickened region for {cname} should contain interior point",
            )
            if len(reg_thin.vertex_ids) < n:
                self.assertFalse(
                    reg_thin.contains(shifted),
                    f"Unthickened proper face for {cname} must not contain interior point",
                )

        # Raises UnsupportedAxiom on roles fixture and on any NF3/NF4/NF6/NF7 axiom
        ds_roles = PathDataset(_fixture_path("roles.owl"))
        with self.assertRaises(UnsupportedAxiom):
            ConvexPolyhedralModel.from_dataset(ds_roles, thicken=True)

        for role_ax in (
            NF3(NS + "A", NS + "r", NS + "B"),
            NF4(NS + "r", NS + "A", NS + "B"),
            NF6(NS + "r", NS + "s"),
            NF7(NS + "r", NS + "s", NS + "t"),
        ):
            no_role = NormalisedOntology(
                axioms=[role_ax],
                concept_names={NS + "A", NS + "B", TOP, BOTTOM},
                role_names={NS + "r", NS + "s", NS + "t"},
                dropped_axioms=0,
            )
            cm_role = CanonicalModel.from_normalised(no_role)
            with self.assertRaises(UnsupportedAxiom):
                ConvexPolyhedralModel.from_canonical_model(cm_role, thicken=True)

    def test_inconsistent_propagates_and_internal_consistency_error_raised(
        self,
    ) -> None:
        adapter = OWLAPIAdapter()
        mgr = adapter.owl_manager
        ont = adapter.create_ontology("http://mowl.borg/geo/inconsistent_poly")
        top = adapter.create_class(TOP)
        bot = adapter.create_class(BOTTOM)
        cls_a = adapter.create_class(NS + "A")
        mgr.addAxiom(ont, adapter.create_subclass_of(top, cls_a))
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_a, bot))

        with self.assertRaises(InconsistentOntology):
            ConvexPolyhedralModel.from_ontology(ont)

        ds = PathDataset(_fixture_path("t3.owl"))
        cm = CanonicalModel.from_dataset(ds)
        bad_axiom = NF1(TOP, BOTTOM)
        with patch.object(
            ModelChecker, "violations", return_value=[bad_axiom]
        ):
            with self.assertRaises(InternalConsistencyError) as ctx:
                ConvexPolyhedralModel.from_canonical_model(cm)
            self.assertIn(repr(bad_axiom), str(ctx.exception))

    def test_import_without_jvm(self) -> None:
        cmd = [
            "/home/leechuck/.venvs/mowl-dev/bin/python",
            "-c",
            "import mowl.geometric; from mowl.geometric import ConvexPolyhedralModel, ThickFace; print('ok')",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        self.assertEqual(res.returncode, 0, f"stderr: {res.stderr}")
        self.assertIn("ok", res.stdout)
