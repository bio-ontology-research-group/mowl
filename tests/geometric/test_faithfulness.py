"""Strong faithfulness (Lacerda et al. 2024) and the finite-model convex construction
(Gutiérrez-Basulto & Schockaert 2018, Proposition 3)."""
from __future__ import annotations

import random
from unittest import TestCase

import mowl
mowl.init_jvm("2g")

from mowl.datasets import PathDataset
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.convex_regions import ConvexRegionModel, FiniteInterpretation
from mowl.geometric.faithfulness import (
    StronglyFaithfulModel,
    candidate_axioms,
    check_strong_faithfulness,
    entailed,
)
from mowl.geometric.normal_forms import BOTTOM, NF1, NF2, NF3, NF4, NF6, TOP, normalise
from mowl.geometric.polyhedral import ConvexPolyhedralModel
from tests.geometric.helpers import fixture_path, random_el_tbox

NS = "http://mowl.borg/geo#"


def _nos():
    out = [(n, normalise(PathDataset(fixture_path(n + ".owl"))))
           for n in ("t3", "roles", "top_lhs")]
    return out


def _random_nos(count=2):
    out = []
    for seed in range(40):
        if len(out) >= count:
            break
        no = random_el_tbox(random.Random(seed), allow_roles=True)
        try:
            CanonicalModel.from_normalised(no)
        except Exception:
            continue
        out.append((f"rand{seed}", no))
    return out


def _cms():
    return [(n, CanonicalModel.from_normalised(no)) for n, no in _nos()]


class TestEntailmentOracle(TestCase):

    def test_oracle_agrees_with_hand_checked_cases(self):
        a, b, c, r, s = (NS + x for x in ("A", "B", "C", "r", "s"))
        from mowl.geometric.normal_forms import NormalisedOntology
        no = NormalisedOntology(
            axioms=[NF1(a, b), NF3(b, r, c), NF6(r, s), NF4(s, c, NS + "D")],
            concept_names={a, b, c, NS + "D"}, role_names={r, s}, dropped_axioms=0)
        cm = CanonicalModel.from_normalised(no)
        self.assertTrue(entailed(cm, NF3(a, r, c)))      # via A <= B
        self.assertTrue(entailed(cm, NF3(a, s, c)))      # via r <= s
        self.assertFalse(entailed(cm, NF3(c, r, c)))
        self.assertTrue(entailed(cm, NF4(r, c, NS + "D")))  # via r <= s
        self.assertTrue(entailed(cm, NF6(r, s)))
        self.assertFalse(entailed(cm, NF6(s, r)))
        self.assertTrue(entailed(cm, NF3(a, s, TOP)))
        self.assertFalse(entailed(cm, NF1(b, a)))
        self.assertTrue(entailed(cm, NF2(a, c, b)))  # A <= B
        self.assertFalse(entailed(cm, NF2(b, c, a)))


class TestStrongFaithfulness(TestCase):

    def test_both_constructions_are_strongly_tbox_faithful(self):
        checked_entailed = 0
        for label, no in _nos() + _random_nos():
            for regions in ("intervals", "convex"):
                with self.subTest(ontology=label, regions=regions):
                    m = StronglyFaithfulModel(no, regions=regions)
                    rep = m.check_strong_faithfulness()
                    self.assertGreater(rep.checked, 20)
                    self.assertEqual(rep.unsound, [])
                    self.assertEqual(rep.incomplete, [])
                    checked_entailed += rep.entailed_count
        self.assertGreater(checked_entailed, 50)

    def test_report_detects_a_model_that_is_not_faithful(self):
        """A model of the axioms that merges two classes is not strongly faithful."""
        a, b, r = NS + "A", NS + "B", NS + "r"
        from mowl.geometric.normal_forms import NormalisedOntology
        no = NormalisedOntology(axioms=[NF3(a, r, b)], concept_names={a, b}, role_names={r},
                                dropped_axioms=0)
        cm = CanonicalModel.from_normalised(no)
        fin = FiniteInterpretation(1, {a: {0}, b: {0}}, {r: {(0, 0)}})
        eta = ConvexRegionModel(fin, cm.axioms)
        rep = check_strong_faithfulness(eta, cm)
        self.assertFalse(rep.is_strongly_faithful)
        self.assertIn(NF1(a, b), rep.unsound)

    def test_one_dimensional_and_dimension(self):
        no = _nos()[0][1]
        self.assertEqual(StronglyFaithfulModel(no, "intervals").dimension, 1)
        m = StronglyFaithfulModel(no, "convex")
        self.assertEqual(m.dimension, len(m.canonical_model.names) - 1)
        with self.assertRaises(ValueError):
            StronglyFaithfulModel(no, "boxes")

    def test_mowl_name_only_canonical_model_is_not_strongly_faithful(self):
        """Why Definition 26 needs the extra elements: with one element per name, the
        model satisfies A and Z disjoint-by-position, hence non-entailed A and Z <= B."""
        for label, no in _nos():
            cm = CanonicalModel.from_normalised(no)
            rep = check_strong_faithfulness(ConvexPolyhedralModel.from_canonical_model(cm), cm)
            self.assertEqual(rep.incomplete, [], label)
        t3 = CanonicalModel.from_normalised(_nos()[0][1])
        rep = check_strong_faithfulness(ConvexPolyhedralModel.from_canonical_model(t3), t3)
        self.assertTrue(any(isinstance(x, NF2) for x in rep.unsound))

    def test_candidates_cover_all_forms(self):
        cm = _cms()[1][1]
        kinds = {type(x).__name__ for x in candidate_axioms(cm)}
        self.assertEqual(kinds, {"NF1", "NF2", "NF3", "NF4", "NF6"})


class TestConvexRegionModel(TestCase):

    def test_phi_of_eta_is_m_for_arbitrary_finite_models(self):
        rng = random.Random(0)
        for trial in range(12):
            m = rng.randint(1, 5)
            concepts = {f"C{i}": {x for x in range(m) if rng.random() < 0.5} for i in range(3)}
            roles = {f"r{i}": {(x, y) for x in range(m) for y in range(m) if rng.random() < 0.3}
                     for i in range(2)}
            fin = FiniteInterpretation(m, concepts, roles)
            eta = ConvexRegionModel(fin)
            self.assertEqual(eta.dimension, m - 1)
            self.assertTrue(eta.phi().same_as(fin), f"trial {trial}")

    def test_eta_satisfies_an_axiom_iff_m_does(self):
        rng = random.Random(1)
        for trial in range(8):
            m = rng.randint(2, 4)
            concepts = {f"C{i}": {x for x in range(m) if rng.random() < 0.6} for i in range(3)}
            roles = {f"r{i}": {(x, y) for x in range(m) for y in range(m) if rng.random() < 0.4}
                     for i in range(2)}
            fin = FiniteInterpretation(m, concepts, roles)
            eta = ConvexRegionModel(fin)
            names, rs = sorted(concepts), sorted(roles)
            axioms = []
            for x in names + [TOP]:
                for y in names + [TOP, BOTTOM]:
                    axioms.append(NF1(x, y))
                for r in rs:
                    for y in names + [TOP]:
                        axioms.append(NF3(x, r, y))
                        axioms.append(NF4(r, y, x if x != TOP else BOTTOM))
            for x in names:
                for y in names:
                    for z in names + [BOTTOM]:
                        axioms.append(NF2(x, y, z))
            axioms.append(NF6(rs[0], rs[1]))
            axioms.append(NF6(rs[1], rs[0]))
            holds = 0
            for ax in axioms:
                self.assertEqual(eta.satisfies(ax), fin.satisfies(ax), (trial, ax))
                holds += fin.satisfies(ax)
            self.assertGreater(holds, 0)

    def test_model_of_ontology_gives_convex_model_of_ontology(self):
        """Proposition 3 with a non-canonical model: the canonical model plus an extra,
        arbitrary element is still a finite model, and so is its convex image."""
        for label, cm in _cms():
            with self.subTest(ontology=label):
                base = FiniteInterpretation.from_canonical_model(cm)
                # duplicate the whole domain: two disjoint copies is again a model
                n = base.size
                dup = FiniteInterpretation(
                    2 * n,
                    {k: set(v) | {x + n for x in v} for k, v in base.concepts.items()},
                    {k: set(v) | {(x + n, y + n) for x, y in v} for k, v in base.roles.items()})
                self.assertTrue(dup.is_model(cm.axioms))
                eta = ConvexRegionModel(dup, cm.axioms)
                self.assertTrue(eta.phi().same_as(dup))
                self.assertTrue(eta.check().is_model(cm.axioms))

    def test_not_a_model_is_rejected(self):
        a, b = NS + "A", NS + "B"
        fin = FiniteInterpretation(2, {a: {0}, b: {1}})
        with self.assertRaises(ValueError):
            ConvexRegionModel(fin, [NF1(a, b)])

    def test_agrees_with_polyhedral_construction_on_the_canonical_model(self):
        cm = _cms()[1][1]
        eta = ConvexRegionModel.from_canonical_model(cm)
        poly = ConvexPolyhedralModel.from_canonical_model(cm)
        for ax in candidate_axioms(cm, forms=("NF1", "NF3", "NF4")):
            self.assertEqual(eta.satisfies(ax), poly.satisfies(ax), ax)

    def test_role_disjointness_example_has_no_convex_model_but_construction_fails_there(self):
        """Example 3: R1(x,y) and R2(x,y) -> bottom is outside the quasi-chained fragment.
        M = D is a model, yet the convex image satisfies R1(b,b) and R2(b,b) at the
        midpoint b, so the disjointness 'R1 and R2 empty intersection' fails in eta."""
        fin = FiniteInterpretation(2, {}, {"R1": {(0, 0), (1, 1)}, "R2": {(0, 1), (1, 0)}})
        eta = ConvexRegionModel(fin)
        import numpy as np
        mid = 0.5 * eta.element_point(0) + 0.5 * eta.element_point(1)
        pt = np.concatenate([mid, mid])
        chk = eta.check()
        self.assertTrue(chk._point_in_pair_hull(pt, eta.role_region("R1")))
        self.assertTrue(chk._point_in_pair_hull(pt, eta.role_region("R2")))
