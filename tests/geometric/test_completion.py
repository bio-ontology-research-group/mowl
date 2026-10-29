"""Tests for ELK-style completion and CanonicalModel in mowl.geometric."""
from __future__ import annotations

import logging
import os
import random
from typing import Dict, Set, Tuple
from unittest import TestCase

import pytest

import mowl
mowl.init_jvm("2g")

from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.owlapi.model import OWLOntology

from mowl.datasets import PathDataset
from mowl.geometric.completion import CanonicalModel, Completion, CompletionResult
from mowl.geometric.errors import InconsistentOntology
from mowl.geometric.normal_forms import BOTTOM, TOP, normalise, to_owl
from mowl.owlapi import OWLAPIAdapter
from mowl.reasoning import MOWLReasoner
from tests.datasetFactory import FamilyDataset
from tests.geometric.helpers import random_el_tbox

logger = logging.getLogger(__name__)

NS = "http://mowl.borg/geo#"
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


def _elk_subsumptions_and_unsat(
    ont: OWLOntology, concept_names: Set[str]
) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    """Compute all entailed (sub, sup) pairs and unsatisfiable names using ELK and MOWLReasoner."""
    adapter = OWLAPIAdapter()
    names = sorted(concept_names - {BOTTOM})
    cls_map: Dict[str, object] = {n: adapter.create_class(n) for n in names}
    owl_classes = [cls_map[n] for n in names]

    reasoner = ElkReasonerFactory().createReasoner(ont)
    reasoner.precomputeInferences()
    try:
        mowl_reasoner = MOWLReasoner(reasoner)
        sub_axioms = mowl_reasoner.infer_subclass_axioms(owl_classes, direct=False)
        eq_axioms = mowl_reasoner.infer_equivalent_class_axioms(owl_classes)

        entailed: Set[Tuple[str, str]] = set()
        name_set = set(names)
        for ax in sub_axioms:
            sub_c = str(ax.getSubClass().toStringID())
            sup_c = str(ax.getSuperClass().toStringID())
            if sub_c in name_set and sup_c in name_set:
                entailed.add((sub_c, sup_c))

        for ax in eq_axioms:
            eqs = [str(c.toStringID()) for c in ax.getClassesInSignature()]
            for c1 in eqs:
                for c2 in eqs:
                    if c1 in name_set and c2 in name_set:
                        entailed.add((c1, c2))

        unsat = {n for n in names if not reasoner.isSatisfiable(cls_map[n])}
    finally:
        reasoner.dispose()

    return entailed, unsat


class TestCompletion(TestCase):

    def test_t3_entailments(self) -> None:
        ds = PathDataset(_fixture_path("t3.owl"))
        cm = CanonicalModel.from_dataset(ds)

        A, B, C = NS + "A", NS + "B", NS + "C"
        X1, X2, X3 = NS + "X1", NS + "X2", NS + "X3"

        self.assertTrue(cm.entails(A, X1))
        self.assertTrue(cm.entails(A, X2))
        self.assertFalse(cm.entails(X1, A))
        self.assertTrue(cm.is_satisfiable(A))
        self.assertFalse(cm.is_satisfiable(BOTTOM))

        self.assertTrue(cm.joint_satisfiable([X1, X2]))
        self.assertTrue(cm.joint_satisfiable([X2, X3]))
        self.assertTrue(cm.joint_satisfiable([X1, X3]))
        self.assertFalse(cm.joint_satisfiable([X1, X2, X3]))

        self.assertTrue(cm.entails_conjunction([X1, X2], A))
        self.assertFalse(cm.entails_conjunction([X1, X2], B))
        self.assertFalse(cm.entails_conjunction([X1, X2], C))
        self.assertTrue(cm.entails_conjunction([X1, X2, X3], BOTTOM))

        d_A = "d_" + A
        self.assertIn(d_A, cm.index)
        self.assertEqual(cm.extension(A), frozenset({cm.index[d_A]}))
        self.assertEqual(cm.extension(TOP), frozenset(range(len(cm.domain))))
        self.assertEqual(cm.extension(BOTTOM), frozenset())

    def test_role_chain_and_hierarchy(self) -> None:
        # Hand derivation on roles.owl with ELK completion rules:
        # Axioms in roles.owl:
        #   (1) A ⊑ ∃r.B          [NF3(A, r, B)]
        #   (2) ∃s.B ⊑ C          [NF4(s, B, C)]
        #   (3) r ⊑ s             [NF6(r, s)]
        #   (4) r ∘ s ⊑ t         [NF7(r, s, t)]
        #   (5) ∃t.B ⊑ D          [NF4(t, B, D)]
        #   (6) B ⊑ ∃r.B          [NF3(B, r, B)]
        #   (7) E ⊓ C ⊑ ⊥         [NF2(E, C, ⊥)]
        #   (8) E ⊑ ∃r.B          [NF3(E, r, B)]
        #
        # Step-by-step saturation:
        # - Role hierarchy (NF6): role_sub[r] = {r, s}, role_sub[s] = {s}, role_sub[t] = {t}.
        # - Init (R0): X ∈ S(X), ⊤ ∈ S(X) for X ∈ {A, B, C, D, E, ⊤}.
        # - Rule R∃ (NF3):
        #     from A ∈ S(A) and (1) A ⊑ ∃r.B  => (A, B) ∈ R(r)
        #     from B ∈ S(B) and (6) B ⊑ ∃r.B  => (B, B) ∈ R(r)
        #     from E ∈ S(E) and (8) E ⊑ ∃r.B  => (E, B) ∈ R(r)
        # - Role hierarchy lifting (r ⊑ s):
        #     (A, B) ∈ R(s), (B, B) ∈ R(s), (E, B) ∈ R(s).
        # - Rule R∃⁻ (NF4) with (2) ∃s.B ⊑ C and B ∈ S(B):
        #     (A, B) ∈ R(s) => C ∈ S(A)  (so A ⊑ C)
        #     (B, B) ∈ R(s) => C ∈ S(B)  (so B ⊑ C)
        #     (E, B) ∈ R(s) => C ∈ S(E)  (so E ⊑ C)
        # - Rule R∘ (NF7) with (4) r ∘ s ⊑ t:
        #     (A, B) ∈ R(r) and (B, B) ∈ R(s) => (A, B) ∈ R(t)
        #     (B, B) ∈ R(r) and (B, B) ∈ R(s) => (B, B) ∈ R(t)
        #     (E, B) ∈ R(r) and (B, B) ∈ R(s) => (E, B) ∈ R(t)
        # - Rule R∃⁻ (NF4) with (5) ∃t.B ⊑ D and B ∈ S(B):
        #     (A, B) ∈ R(t) => D ∈ S(A)  (so A ⊑ D)
        #     (B, B) ∈ R(t) => D ∈ S(B)  (so B ⊑ D)
        #     (E, B) ∈ R(t) => D ∈ S(E)  (so E ⊑ D)
        # - Rule R⊓ (NF2) with (7) E ⊓ C ⊑ ⊥:
        #     E ∈ S(E) and C ∈ S(E) => ⊥ ∈ S(E) (E is unsatisfiable, excluded from domain).
        ds = PathDataset(_fixture_path("roles.owl"))
        cm = CanonicalModel.from_dataset(ds)

        A, B, C, D, E = NS + "A", NS + "B", NS + "C", NS + "D", NS + "E"
        r, s, t = NS + "r", NS + "s", NS + "t"

        self.assertTrue(cm.entails(A, C))
        self.assertTrue(cm.entails(A, D))
        self.assertTrue(cm.entails(B, C))
        self.assertTrue(cm.entails(B, D))
        self.assertFalse(cm.entails(A, B))
        self.assertFalse(cm.entails(C, D))

        self.assertFalse(cm.is_satisfiable(E))
        self.assertNotIn(E, cm.names)
        self.assertEqual(cm.extension(E), frozenset())

        idx_a = cm.index["d_" + A]
        idx_b = cm.index["d_" + B]
        self.assertIn((idx_a, idx_b), cm.role(r))
        self.assertIn((idx_a, idx_b), cm.role(s))
        self.assertIn((idx_a, idx_b), cm.role(t))
        self.assertIn((idx_b, idx_b), cm.role(r))
        self.assertIn((idx_b, idx_b), cm.role(s))
        self.assertIn((idx_b, idx_b), cm.role(t))
        self.assertEqual(cm.role(NS + "unknown"), frozenset())

    def test_isolated_name_is_element(self) -> None:
        ds = PathDataset(_fixture_path("t3.owl"))
        cm = CanonicalModel.from_dataset(ds)
        Z = NS + "Z"
        d_Z = "d_" + Z

        self.assertIn(Z, cm.names)
        self.assertIn(d_Z, cm.index)
        self.assertEqual(cm.extension(Z), frozenset({cm.index[d_Z]}))

    def test_inconsistent_raises(self) -> None:
        adapter = OWLAPIAdapter()
        mgr = adapter.owl_manager
        ont = adapter.create_ontology("http://mowl.borg/geo/inconsistent")
        top = adapter.create_class(TOP)
        bot = adapter.create_class(BOTTOM)
        cls_a = adapter.create_class(NS + "A")

        mgr.addAxiom(ont, adapter.create_subclass_of(top, cls_a))
        mgr.addAxiom(ont, adapter.create_subclass_of(cls_a, bot))

        with self.assertRaises(InconsistentOntology):
            CanonicalModel.from_ontology(ont)

    def test_bot_edge_propagation_and_transitive_role_hierarchy(self) -> None:
        from mowl.geometric.normal_forms import NF1, NF3, NF4, NF6

        A, B, C, D = NS + "A", NS + "B", NS + "C", NS + "D"
        r1, r2, r3 = NS + "r1", NS + "r2", NS + "r3"
        axioms = [
            NF6(r2, r3),
            NF6(r1, r2),
            NF3(A, r1, B),
            NF4(r3, B, C),
            NF3(D, r1, A),
            NF1(C, BOTTOM),
        ]
        comp = Completion(axioms, {A, B, C, D}, {r1, r2, r3}).saturate()
        # r1 <= r2 <= r3 gives r3 in role_sub[r1], so A <= exists r3.B <= C <= BOTTOM
        self.assertIn(r3, comp.role_sub[r1])
        self.assertTrue(comp.is_unsatisfiable(C))
        self.assertTrue(comp.is_unsatisfiable(A))
        # Backwards propagation along (D, r1, A): since A <= BOTTOM, D <= BOTTOM
        self.assertTrue(comp.is_unsatisfiable(D))
        self.assertFalse(comp.is_unsatisfiable(B))
        self.assertFalse(comp.is_inconsistent())

    def test_matches_elk_on_normalised(self) -> None:
        sources = [
            ("t3", PathDataset(_fixture_path("t3.owl"))),
            ("helly_ok", PathDataset(_fixture_path("helly_ok.owl"))),
            ("roles", PathDataset(_fixture_path("roles.owl"))),
            ("top_lhs", PathDataset(_fixture_path("top_lhs.owl"))),
            ("family", FamilyDataset()),
        ]
        for label, ds in sources:
            with self.subTest(fixture=label):
                no = normalise(ds)
                comp = Completion(no.axioms, no.concept_names, no.role_names).saturate()
                self.assertIsInstance(comp, CompletionResult)
                cm = CanonicalModel.from_normalised(no)

                ont_norm = to_owl(no)
                elk_entailed, elk_unsat = _elk_subsumptions_and_unsat(
                    ont_norm, no.concept_names
                )

                names = sorted(no.concept_names - {BOTTOM})
                comp_unsat = {n for n in names if comp.is_unsatisfiable(n)}
                self.assertEqual(
                    comp_unsat,
                    elk_unsat,
                    f"Unsatisfiable set mismatch on {label}: {comp_unsat} vs {elk_unsat}",
                )

                for a in names:
                    for b in names:
                        expected = (a, b) in elk_entailed
                        actual = comp.entails(a, b)
                        self.assertEqual(
                            actual,
                            expected,
                            f"Subsumption mismatch on {label} for ({a}, {b}): "
                            f"completion={actual}, ELK={expected}",
                        )
                        self.assertEqual(cm.entails(a, b), expected)
                        self.assertEqual(
                            cm.extension(a).issubset(cm.extension(b)),
                            expected,
                            f"Canonical extension inclusion mismatch on {label} for ({a}, {b})",
                        )

    def test_sound_wrt_original_family(self) -> None:
        ds = FamilyDataset()
        no = normalise(ds)
        logger.info("FamilyDataset dropped_axioms: %d", no.dropped_axioms)

        comp = Completion(no.axioms, no.concept_names, no.role_names).saturate()
        orig_names = {
            str(cls.toStringID()) for cls in ds.ontology.getClassesInSignature()
        } | {TOP}
        orig_names.discard(BOTTOM)

        elk_entailed, _ = _elk_subsumptions_and_unsat(ds.ontology, orig_names)
        comp_entailed = {
            (a, b)
            for a in orig_names
            for b in orig_names
            if comp.entails(a, b)
        }
        self.assertTrue(
            comp_entailed.issubset(elk_entailed),
            f"Unsound derivations wrt original FamilyDataset: {comp_entailed - elk_entailed}",
        )

    @pytest.mark.slow
    def test_random_tboxes_match_elk(self) -> None:
        """Differential test: completion vs ELK on 100 seeded random EL++ TBoxes with roles."""
        from mowl.geometric.intervals import IntervalUnionModel
        from mowl.geometric.polyhedral import ConvexPolyhedralModel

        n_consistent = 0
        n_inconsistent = 0
        for seed in range(100):
            rng = random.Random(seed)
            no = random_el_tbox(rng, allow_roles=True)
            comp = Completion(no.axioms, no.concept_names, no.role_names).saturate()
            ont = to_owl(no)

            reasoner = ElkReasonerFactory().createReasoner(ont)
            try:
                reasoner.precomputeInferences()
                elk_consistent = bool(reasoner.isConsistent())
            finally:
                reasoner.dispose()

            with self.subTest(seed=seed):
                self.assertEqual(
                    comp.is_inconsistent(),
                    not elk_consistent,
                    f"Consistency mismatch on seed {seed}",
                )
            if not elk_consistent:
                n_inconsistent += 1
                continue
            n_consistent += 1

            elk_entailed, elk_unsat = _elk_subsumptions_and_unsat(
                ont, no.concept_names
            )
            names = sorted(no.concept_names - {BOTTOM})
            with self.subTest(seed=seed):
                comp_unsat = {n for n in names if comp.is_unsatisfiable(n)}
                self.assertEqual(comp_unsat, elk_unsat)
                for a in names:
                    if a in elk_unsat:
                        continue
                    for b in names:
                        self.assertEqual(
                            comp.entails(a, b),
                            (a, b) in elk_entailed,
                            f"Subsumption mismatch on seed {seed} for ({a}, {b})",
                        )

                # Both exact constructions post-check themselves and raise
                # InternalConsistencyError on failure.
                cm = CanonicalModel.from_normalised(no)
                IntervalUnionModel(cm)
                ConvexPolyhedralModel(cm)

        logger.info(
            "test_random_tboxes_match_elk: %d consistent, %d inconsistent",
            n_consistent,
            n_inconsistent,
        )
        self.assertGreater(n_consistent, 0)

    def test_edge_indexed_saturation_agrees_with_elk(self) -> None:
        """Verify inverse-edge indexed saturation against ELK on ``roles.owl`` and 20 seeded random role-bearing TBoxes.

        ELK's OWLAPI ``OWLReasoner`` interface exposes class satisfiability and
        named-class subsumptions (``sub_super``) rather than internal completion
        role-edge queues, so this test asserts exact ``sub_super`` and
        unsatisfiability agreement with ELK across all concepts (and specifically
        for every derived edge target ``d`` in ``comp.edges``).
        """
        tboxes = [("roles", normalise(PathDataset(_fixture_path("roles.owl"))))]
        for seed in range(20):
            rng = random.Random(seed)
            tboxes.append((f"seed_{seed}", random_el_tbox(rng, allow_roles=True)))

        n_random_with_edges = 0
        for label, no in tboxes:
            comp = Completion(no.axioms, no.concept_names, no.role_names).saturate()
            if label.startswith("seed_") and any(comp.edges.values()):
                n_random_with_edges += 1
            ont = to_owl(no)

            reasoner = ElkReasonerFactory().createReasoner(ont)
            try:
                reasoner.precomputeInferences()
                elk_consistent = bool(reasoner.isConsistent())
            finally:
                reasoner.dispose()

            with self.subTest(tbox=label):
                self.assertEqual(
                    comp.is_inconsistent(),
                    not elk_consistent,
                    f"Consistency mismatch on {label}",
                )
                if not elk_consistent:
                    continue

                elk_entailed, elk_unsat = _elk_subsumptions_and_unsat(
                    ont, no.concept_names
                )
                names = sorted(no.concept_names - {BOTTOM})
                comp_unsat = {n for n in names if comp.is_unsatisfiable(n)}
                self.assertEqual(comp_unsat, elk_unsat)

                for a in names:
                    if a in elk_unsat:
                        continue
                    for b in names:
                        self.assertEqual(
                            comp.entails(a, b),
                            (a, b) in elk_entailed,
                            f"Subsumption mismatch on {label} for ({a}, {b})",
                        )

                edge_targets = {
                    target
                    for role_set in comp.edges.values()
                    for (_role, target) in role_set
                }
                for target in edge_targets:
                    if target == BOTTOM or target in elk_unsat:
                        self.assertTrue(comp.is_unsatisfiable(target))
                    else:
                        target_sups = {
                            b for b in names if comp.entails(target, b)
                        }
                        elk_target_sups = {
                            b for b in names if (target, b) in elk_entailed
                        }
                        self.assertEqual(
                            target_sups,
                            elk_target_sups,
                            f"Edge target subsumers mismatch on {label} for {target}",
                        )

        # The edge-driven rules (and the inverse-edge index) are only exercised if
        # the random sample actually derives role edges.
        self.assertGreater(n_random_with_edges, 0)
