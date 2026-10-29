"""Tests for EL++ normal forms and ontology normalisation in mowl.geometric."""
from __future__ import annotations

import os
from unittest import TestCase

import mowl
mowl.init_jvm("2g")

from java.util import ArrayList
from org.semanticweb.owlapi.model import IRI

from mowl.datasets import PathDataset
from mowl.geometric.errors import (
    BoxModelNotFound,
    GeometricError,
    InconsistentOntology,
    InternalConsistencyError,
    MissingDependency,
    NotBoxRepresentable,
    UnsupportedAxiom,
    UnsupportedRegion,
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
    normalise,
)
from mowl.owlapi import OWLAPIAdapter

NS = "http://mowl.borg/geo#"
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


class TestNormalForms(TestCase):

    def test_t3_normalises_to_expected(self) -> None:
        ds = PathDataset(_fixture_path("t3.owl"))
        no = normalise(ds)
        self.assertIsInstance(no, NormalisedOntology)

        axioms = set(no.axioms)
        A, B, C = NS + "A", NS + "B", NS + "C"
        X1, X2, X3 = NS + "X1", NS + "X2", NS + "X3"
        Z = NS + "Z"

        # A ≡ X1 ⊓ X2 -> NF2(X1, X2, A) (or symmetric), NF1(A, X1), NF1(A, X2)
        self.assertTrue(
            NF2(X1, X2, A) in axioms or NF2(X2, X1, A) in axioms,
            f"Expected NF2(X1, X2, A) in {axioms}",
        )
        self.assertIn(NF1(A, X1), axioms)
        self.assertIn(NF1(A, X2), axioms)

        # B ≡ X2 ⊓ X3
        self.assertTrue(
            NF2(X2, X3, B) in axioms or NF2(X3, X2, B) in axioms,
            f"Expected NF2(X2, X3, B) in {axioms}",
        )
        self.assertIn(NF1(B, X2), axioms)
        self.assertIn(NF1(B, X3), axioms)

        # C ≡ X1 ⊓ X3
        self.assertTrue(
            NF2(X1, X3, C) in axioms or NF2(X3, X1, C) in axioms,
            f"Expected NF2(X1, X3, C) in {axioms}",
        )
        self.assertIn(NF1(C, X1), axioms)
        self.assertIn(NF1(C, X3), axioms)

        # Disjointness: A ⊓ B ⊑ ⊥, B ⊓ C ⊑ ⊥, A ⊓ C ⊑ ⊥
        self.assertTrue(NF2(A, B, BOTTOM) in axioms or NF2(B, A, BOTTOM) in axioms)
        self.assertTrue(NF2(B, C, BOTTOM) in axioms or NF2(C, B, BOTTOM) in axioms)
        self.assertTrue(NF2(A, C, BOTTOM) in axioms or NF2(C, A, BOTTOM) in axioms)

        expected_names = {A, B, C, X1, X2, X3, Z, TOP, BOTTOM}
        self.assertTrue(expected_names.issubset(no.concept_names))
        self.assertEqual(no.dropped_axioms, 0)

    def test_roles_normalises_role_axioms(self) -> None:
        ds = PathDataset(_fixture_path("roles.owl"))
        no = normalise(ds)
        axioms = set(no.axioms)

        r, s, t = NS + "r", NS + "s", NS + "t"
        A, B, C, D, E = NS + "A", NS + "B", NS + "C", NS + "D", NS + "E"

        self.assertIn(NF6(r, s), axioms)
        self.assertIn(NF7(r, s, t), axioms)
        self.assertIn(NF3(A, r, B), axioms)
        self.assertIn(NF4(s, B, C), axioms)
        self.assertIn(NF4(t, B, D), axioms)
        self.assertIn(NF3(B, r, B), axioms)
        self.assertIn(NF3(E, r, B), axioms)
        self.assertTrue(NF2(E, C, BOTTOM) in axioms or NF2(C, E, BOTTOM) in axioms)
        self.assertEqual(no.role_names, {r, s, t})

    def test_dataset_and_ontology_sources_agree(self) -> None:
        for fixture in ["t3.owl", "helly_ok.owl", "roles.owl", "top_lhs.owl"]:
            ds = PathDataset(_fixture_path(fixture))
            no_ds = normalise(ds)
            no_ont = normalise(ds.ontology)
            self.assertEqual(set(no_ds.axioms), set(no_ont.axioms))
            self.assertEqual(no_ds.concept_names, no_ont.concept_names)
            self.assertEqual(no_ds.role_names, no_ont.role_names)
            self.assertEqual(no_ds.dropped_axioms, no_ont.dropped_axioms)

    def test_top_lhs_normalises_to_expected(self) -> None:
        ds = PathDataset(_fixture_path("top_lhs.owl"))
        no = normalise(ds)
        axioms = set(no.axioms)
        A, B, C, r = NS + "A", NS + "B", NS + "C", NS + "r"
        self.assertIn(NF3(TOP, r, B), axioms)
        self.assertIn(NF3(A, r, C), axioms)
        self.assertIn(NF1(C, B), axioms)

    def test_long_role_chain_splitting(self) -> None:
        adapter = OWLAPIAdapter()
        mgr = adapter.owl_manager
        df = adapter.data_factory
        ont = adapter.create_ontology("http://mowl.borg/geo/long_chain")
        r1 = adapter.create_object_property(NS + "r1")
        r2 = adapter.create_object_property(NS + "r2")
        r3 = adapter.create_object_property(NS + "r3")
        sup = adapter.create_object_property(NS + "sup")
        chain = ArrayList()
        chain.add(r1)
        chain.add(r2)
        chain.add(r3)
        mgr.addAxiom(ont, df.getOWLSubPropertyChainOfAxiom(chain, sup))

        no = normalise(ont)
        axioms = set(no.axioms)
        aux = "http://mowl.borg/geo/aux#sup_chain_1"
        self.assertIn(NF7(NS + "r1", NS + "r2", aux), axioms)
        self.assertIn(NF7(aux, NS + "r3", NS + "sup"), axioms)
        self.assertIn(aux, no.role_names)

    def test_errors_hierarchy(self) -> None:
        for cls in [
            InconsistentOntology,
            UnsupportedAxiom,
            UnsupportedRegion,
            NotBoxRepresentable,
            BoxModelNotFound,
            InternalConsistencyError,
            MissingDependency,
        ]:
            self.assertTrue(issubclass(cls, GeometricError))

        err_msg_report = NotBoxRepresentable("not box rep", report={"dummy": True})
        self.assertEqual(err_msg_report.report, {"dummy": True})
        self.assertIn("not box rep", str(err_msg_report))

        err_report_only = NotBoxRepresentable({"dummy": True})
        self.assertEqual(err_report_only.report, {"dummy": True})
