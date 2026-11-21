"""Tests for EL++ normal forms and ontology normalisation in mowl.geometric."""
from __future__ import annotations

import itertools
import os
from typing import Dict, FrozenSet, Set
from unittest import TestCase

import mowl
mowl.init_jvm("2g")

from java.util import ArrayList, HashSet
from org.semanticweb.elk.owlapi import ElkReasonerFactory
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

HELLY_OK_MINIMAL_UNSAT_SETS: Set[FrozenSet[str]] = {
    frozenset({NS + "A", NS + "C"}),
    frozenset({NS + "A", NS + "E"}),
    frozenset({NS + "B", NS + "E"}),
    frozenset({NS + "C", NS + "D"}),
    frozenset({NS + "C", NS + "E"}),
    frozenset({NS + "D", NS + "E"}),
    frozenset({NS + "D", NS + "F"}),
}


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

    def test_helly_ok_minimal_unsat_sets(self) -> None:
        ds = PathDataset(_fixture_path("helly_ok.owl"))
        ont = ds.ontology
        adapter = OWLAPIAdapter()
        mgr = adapter.owl_manager

        names = ["A", "B", "C", "D", "E", "F"]
        cls = {NS + n: adapter.create_class(NS + n) for n in names}

        probe_axioms = []
        z_map: Dict[FrozenSet[str], object] = {}
        for idx, r in enumerate([1, 2, 3]):
            for comb_idx, comb in enumerate(itertools.combinations(sorted(cls.keys()), r)):
                fset = frozenset(comb)
                z = adapter.create_class(f"{NS}_probe_{r}_{comb_idx}")
                z_map[fset] = z
                if len(comb) == 1:
                    ax = adapter.create_subclass_of(z, cls[comb[0]])
                else:
                    ax = adapter.create_subclass_of(
                        z, adapter.create_object_intersection_of(*[cls[x] for x in comb])
                    )
                mgr.addAxiom(ont, ax)
                probe_axioms.append(ax)

        reasoner = ElkReasonerFactory().createReasoner(ont)
        reasoner.precomputeInferences()
        try:
            # 1. Every single class is satisfiable
            for cname in cls:
                self.assertTrue(
                    reasoner.isSatisfiable(z_map[frozenset({cname})]),
                    f"Single class {cname} should be satisfiable",
                )

            # 2. Each of the seven recorded pairs is jointly unsatisfiable
            self.assertEqual(len(HELLY_OK_MINIMAL_UNSAT_SETS), 7)
            for pair in HELLY_OK_MINIMAL_UNSAT_SETS:
                self.assertFalse(
                    reasoner.isSatisfiable(z_map[pair]),
                    f"Recorded pair {pair} should be jointly unsatisfiable",
                )

            # Also check that every other pair is jointly satisfiable
            for comb in itertools.combinations(sorted(cls.keys()), 2):
                pair = frozenset(comb)
                if pair not in HELLY_OK_MINIMAL_UNSAT_SETS:
                    self.assertTrue(
                        reasoner.isSatisfiable(z_map[pair]),
                        f"Non-recorded pair {pair} should be jointly satisfiable",
                    )

            # 3. Every triple of classes that contains none of those pairs is jointly satisfiable
            for comb in itertools.combinations(sorted(cls.keys()), 3):
                triple = frozenset(comb)
                if not any(pair.issubset(triple) for pair in HELLY_OK_MINIMAL_UNSAT_SETS):
                    self.assertTrue(
                        reasoner.isSatisfiable(z_map[triple]),
                        f"Triple {triple} containing no unsat pair should be jointly satisfiable",
                    )
        finally:
            reasoner.dispose()
            for ax in probe_axioms:
                mgr.removeAxiom(ont, ax)

    def test_equivalent_transitive_and_domain_role_axioms(self) -> None:
        adapter = OWLAPIAdapter()
        mgr = adapter.owl_manager
        df = adapter.data_factory
        ont = adapter.create_ontology("http://mowl.borg/geo/rbox_supported")

        r = adapter.create_object_property(NS + "r")
        s = adapter.create_object_property(NS + "s")
        t = adapter.create_object_property(NS + "t")
        A = adapter.create_class(NS + "A")

        eq_props = HashSet()
        eq_props.add(r)
        eq_props.add(s)
        mgr.addAxiom(ont, df.getOWLEquivalentObjectPropertiesAxiom(eq_props))
        mgr.addAxiom(ont, df.getOWLTransitiveObjectPropertyAxiom(t))
        mgr.addAxiom(ont, df.getOWLObjectPropertyDomainAxiom(r, A))

        no = normalise(ont)
        axioms = set(no.axioms)

        self.assertIn(NF6(NS + "r", NS + "s"), axioms)
        self.assertIn(NF6(NS + "s", NS + "r"), axioms)
        self.assertIn(NF7(NS + "t", NS + "t", NS + "t"), axioms)
        self.assertIn(NF4(NS + "r", TOP, NS + "A"), axioms)
        self.assertEqual(
            len([ax for ax in no.axioms if ax == NF4(NS + "r", TOP, NS + "A")]),
            1,
        )
        self.assertEqual(no.dropped_axioms, 0)
        self.assertEqual(no.role_names, {NS + "r", NS + "s", NS + "t"})
        self.assertEqual(no.concept_names, {NS + "A", TOP, BOTTOM})

    def test_unsupported_inverse_subproperty_and_range_axioms_dropped(self) -> None:
        adapter = OWLAPIAdapter()
        mgr = adapter.owl_manager
        df = adapter.data_factory
        ont = adapter.create_ontology("http://mowl.borg/geo/rbox_dropped")

        r = adapter.create_object_property(NS + "r")
        s = adapter.create_object_property(NS + "s")
        inv_r = df.getOWLObjectInverseOf(r)
        B = adapter.create_class(NS + "B")

        mgr.addAxiom(ont, df.getOWLSubObjectPropertyOfAxiom(inv_r, s))
        mgr.addAxiom(ont, df.getOWLObjectPropertyRangeAxiom(r, B))

        no = normalise(ont)
        self.assertEqual(no.axioms, [])
        self.assertEqual(no.dropped_axioms, 2)

