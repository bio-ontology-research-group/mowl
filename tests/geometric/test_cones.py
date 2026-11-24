"""Tests for mowl.geometric.cones (axis-aligned cone semantics of ALC).

Source: Özçep, Leemhuis & Wolter, JAIR 78 (2023) 287-337 ("JAIR") and IJCAI 2020.
"""
import itertools
import os
import random

import numpy as np
import pytest

from mowl.datasets import PathDataset
from mowl.geometric.cones import (
    ALCone, ALConeChecker, ALConeInterpretation, And, Atomic, BOTTOM_CONCEPT,
    ConceptAssertion, Exists, FiniteInterpretation, Forall, GCI, Not, Or, RoleAssertion,
    TOP_CONCEPT, axioms_from_owl, halfaxis_vector, point_support)
from mowl.geometric.errors import InconsistentOntology, UnsupportedAxiom

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
A, B, C, D = Atomic("A"), Atomic("B"), Atomic("C"), Atomic("D")


def _sample_points(dim, rng, k=60):
    """Grid points of {-1, 0, 1}^dim plus random Gaussian points."""
    pts = [np.array(p, dtype=float) for p in itertools.product((-1, 0, 1), repeat=dim)]
    pts += [rng.normal(size=dim) for _ in range(k)]
    return pts


def _sample_cone_points(cone, rng, k=20):
    """Random points of the cone: non-negative combinations of its generators."""
    g = cone.generators()
    if len(g) == 0:
        return [np.zeros(cone.dim)]
    return [rng.random(len(g)) @ g for _ in range(k)]


def _in_conic_hull(x, gens):
    """Brute-force membership in the conic hull of generators (non-negative least
    squares by projected gradient; exact enough for unit half-axis generators)."""
    if len(gens) == 0:
        return bool(np.allclose(x, 0))
    G = np.asarray(gens).T
    lam = np.zeros(G.shape[1])
    for _ in range(2000):
        grad = G.T @ (G @ lam - x)
        lam = np.maximum(lam - 0.5 * grad, 0.0)
    return bool(np.linalg.norm(G @ lam - x) < 1e-6)


# ---------------------------------------------------------------------------
# Cone operations (JAIR Prop. 1, Prop. 2, Corollary 1)
# ---------------------------------------------------------------------------

class TestConeOperations:

    def test_all_cones_count(self):
        """There are 4^n al-cones in R^n (JAIR Sec. 4)."""
        assert len(set(ALCone.all_cones(2))) == 16
        assert len(set(ALCone.all_cones(3))) == 64

    def test_factors_roundtrip(self):
        X = ALCone.from_factors(["+", "0", "R", "-"])
        assert X.factors() == ("+", "0", "R", "-")
        assert X.halfaxes() == {0, 4, 5, 7}

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_membership_matches_product_definition(self, dim):
        """x ∈ X iff x_i ∈ X_i for every coordinate."""
        rng = np.random.default_rng(0)
        sets = {"0": lambda v: v == 0, "+": lambda v: v >= 0, "-": lambda v: v <= 0,
                "R": lambda v: True}
        for X in ALCone.all_cones(dim):
            for x in _sample_points(dim, rng, 20):
                expected = all(sets[f](v) for f, v in zip(X.factors(), x))
                assert X.contains_point(x) == expected

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_polar_against_definition(self, dim):
        """X° = {v | <v, w> <= 0 for all w in X}: checked on the generators (exact) and on
        random points of X (sampling)."""
        rng = np.random.default_rng(1)
        for X in ALCone.all_cones(dim):
            P = X.polar()
            gens = X.generators()
            for v in _sample_points(dim, rng, 30):
                by_def = all(float(v @ g) <= 1e-12 for g in gens)
                assert P.contains_point(v) == by_def
                if P.contains_point(v):
                    for w in _sample_cone_points(X, rng, 5):
                        assert float(v @ w) <= 1e-9

    @pytest.mark.parametrize("dim", [1, 2])
    def test_intersection_and_join_by_sampling(self, dim):
        """X ∩ Y is set intersection; the join is conHull(X ∪ Y) (JAIR Prop. 2)."""
        rng = np.random.default_rng(2)
        for X, Y in itertools.product(list(ALCone.all_cones(dim)), repeat=2):
            gens = list(X.generators()) + list(Y.generators())
            for x in _sample_points(dim, rng, 10):
                assert (X & Y).contains_point(x) == \
                    (X.contains_point(x) and Y.contains_point(x))
                assert (X | Y).contains_point(x) == _in_conic_hull(x, gens)
            # the convex union is (X° ∩ Y°)° and is larger than the set union in general
            assert (X | Y) == (X.polar() & Y.polar()).polar()

    def test_join_is_not_set_union(self):
        """The convex union of the two positive half-axes is the whole quadrant."""
        X, Y = ALCone.from_factors("+0"), ALCone.from_factors("0+")
        p = np.array([1.0, 1.0])
        assert (X | Y).contains_point(p)
        assert not X.contains_point(p) and not Y.contains_point(p)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_polar_is_involution(self, dim):
        for X in ALCone.all_cones(dim):
            assert X.polar().polar() == X

    def test_top_bottom(self):
        assert ALCone.top(3).polar() == ALCone.bottom(3)
        assert ALCone.bottom(3).is_bottom() and ALCone.top(3).is_top()
        assert ALCone.bottom(2).contains_point(np.zeros(2))
        assert not ALCone.bottom(2).contains_point(np.array([1.0, 0.0]))

    def test_de_morgan(self):
        cones = list(ALCone.all_cones(2))
        for X, Y in itertools.product(cones, repeat=2):
            assert (X & Y).polar() == X.polar() | Y.polar()
            assert (X | Y).polar() == X.polar() & Y.polar()
            # orthocomplement laws (JAIR Sec. 4.2.1)
            assert (X & X.polar()).is_bottom()
            assert (X | X.polar()).is_top()
            if X <= Y:
                assert Y.polar() <= X.polar()

    def test_distributivity_holds_for_al_cones(self):
        """Al-cones form the Boolean algebra O_4^n, so ⊓ distributes over ⊔ (JAIR
        Prop. 7 proof); exhaustive in R^2."""
        cones = list(ALCone.all_cones(2))
        for X, Y, Z in itertools.product(cones, repeat=3):
            assert X & (Y | Z) == (X & Y) | (X & Z)
            assert X | (Y & Z) == (X | Y) & (X | Z)

    def test_prop5_condition_holds_for_al_cones(self):
        """JAIR Prop. 5: a class of cones is distributive iff for any two cones X, Y,
        X ∩ Y ≠ {0} or <x, y> <= 0 for all x ∈ X, y ∈ Y. Exhaustive in R^2."""
        cones = list(ALCone.all_cones(2))
        for X, Y in itertools.product(cones, repeat=2):
            if (X & Y).is_bottom():
                for gx in X.generators():
                    for gy in Y.generators():
                        assert float(gx @ gy) <= 0
                assert X <= Y.polar()

    def test_distributivity_fails_for_paper_non_al_cone(self):
        """JAIR Fig. 3(c) / Sec. 4.2: with X the upper-left quadrant, Y the positive
        x-axis and U a non-axis-aligned cone between them,
        conHull(X ∪ Y) ∩ U = U but X ∩ U = Y ∩ U = {0}. We take U = cone{(1,1), (1,2)}."""
        X = ALCone.from_factors("-+")
        Y = ALCone.from_factors("+0")
        U_gens = np.array([[1.0, 1.0], [1.0, 2.0]])
        rng = np.random.default_rng(3)
        U_pts = [rng.random(2) @ U_gens + 1e-3 for _ in range(200)]
        # X ∩ U = {0} and Y ∩ U = {0}: nonzero points of U have x > 0 and y > 0
        assert not any(X.contains_point(u) for u in U_pts)
        assert not any(Y.contains_point(u) for u in U_pts)
        # but U ⊆ conHull(X ∪ Y), the upper half-plane
        assert (X | Y) == ALCone.from_factors("R+")
        assert all((X | Y).contains_point(u) for u in U_pts)
        # Prop. 5 condition violated: X ∩ U = {0} yet <x, u> > 0 for some x ∈ X, u ∈ U
        assert float(np.array([-1.0, 2.0]) @ U_gens[1]) > 0


# ---------------------------------------------------------------------------
# Worked examples of the paper
# ---------------------------------------------------------------------------

class TestPaperExamples:

    def test_example7_narcissus_incidence_matrix(self):
        """JAIR Example 7: V = R+ × R-, R^T as printed; (∃R.V)^I = R+ × R+."""
        RT = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        interp = ALConeInterpretation(2, {"V": ALCone.from_factors("+-")}, {"R": RT.T})
        assert interp.cone(Exists("R", Atomic("V"))) == ALCone.from_factors("++")
        # incidence vector of y = [1, -1] is [1, 0, 0, 1]; R^T y' = [1, 0, 1, 0]
        y = np.array([1.0, -1.0])
        assert point_support(y) == {0, 3}

    def test_example1_figure10(self):
        """JAIR Example 1 / Fig. 10: A is the upper-left, B the upper-right quadrant."""
        interp = ALConeInterpretation(2, {"A": ALCone.from_factors("-+"),
                                          "B": ALCone.from_factors("++")})
        chk = interp.checker()
        assert chk.cone(And((B, Not(A)))) == ALCone.from_factors("+0")
        assert chk.cone(Not(And((B, Not(A))))) == ALCone.from_factors("-R")  # left area
        assert chk.cone(And((A, B))) == ALCone.from_factors("0+")
        assert chk.cone(Not(A)) == ALCone.from_factors("+-")
        # the 16 Boolean concepts over A, B are 16 distinct al-cones
        atoms = [And((A, B)), And((A, Not(B))), And((Not(A), B)), And((Not(A), Not(B)))]
        cones = set()
        for k in range(5):
            for S in itertools.combinations(atoms, k):
                cones.add(chk.cone(Or(tuple(S))) if S else chk.cone(BOTTOM_CONCEPT))
        assert len(cones) == 16
        # a2 ∈ B placed inside the quadrant is neither in A nor in ¬A (partiality, 5.1)
        a2 = np.array([1.0, 1.0])
        assert chk.cone(B).contains_point(a2)
        assert not chk.cone(A).contains_point(a2)
        assert not chk.cone(Not(A)).contains_point(a2)
        # a3 ∈ B ⊓ ¬A on the positive x-axis is completely specified
        a3 = np.array([1.0, 0.0])
        assert chk.cone(B).contains_point(a3) and chk.cone(Not(A)).contains_point(a3)

    def test_example1_abox_with_15_constants(self):
        """JAIR Example 1: one constant per non-bottom Boolean concept over {A, B}; the
        Prop. 7 construction gives a model in R^2."""
        atoms = [And((A, B)), And((A, Not(B))), And((Not(A), B)), And((Not(A), Not(B)))]
        axioms = []
        for k in range(1, 5):
            for i, S in enumerate(itertools.combinations(atoms, k)):
                axioms.append(ConceptAssertion(Or(S), f"a_{k}_{i}"))
        assert len(axioms) == 15
        interp = ALConeInterpretation.from_boolean_ontology(axioms)
        assert interp.dim == 2
        assert interp.checker().is_model(axioms)

    def test_example2_figure11(self):
        """JAIR Example 2 / Fig. 11: TBox {A ⊑ B}; 3 atoms in R^2 and A ⊓ ¬B = ⊥."""
        interp = ALConeInterpretation.from_boolean_ontology([GCI(A, B)])
        assert interp.dim == 2
        chk = interp.checker()
        assert chk.cone(And((A, Not(B)))).is_bottom()
        assert chk.cone(A) <= chk.cone(B) and chk.cone(A) != chk.cone(B)
        assert not chk.cone(And((Not(A), B))).is_bottom()

    def test_prop11_exists_is_normal_and_additive(self):
        """JAIR Prop. 11: (∃R.⊥) = ⊥ and ∃R.(C ⊔ D) = ∃R.C ⊔ ∃R.D, for random
        incidence matrices."""
        rng = np.random.default_rng(4)
        for _ in range(30):
            R = rng.random((6, 6)) < 0.3
            X = ALCone(rng.random(6) < 0.5)
            Y = ALCone(rng.random(6) < 0.5)
            interp = ALConeInterpretation(3, {"X": X, "Y": Y}, {"r": R})
            chk = interp.checker()
            assert chk.cone(Exists("r", BOTTOM_CONCEPT)).is_bottom()
            assert chk.cone(Exists("r", Or((Atomic("X"), Atomic("Y"))))) == \
                chk.cone(Exists("r", Atomic("X"))) | chk.cone(Exists("r", Atomic("Y")))
            assert chk.cone(Forall("r", TOP_CONCEPT)).is_top()


# ---------------------------------------------------------------------------
# Construction from finite classical models
# ---------------------------------------------------------------------------

def _random_concept(rng, names, roles, depth):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice([Atomic(n) for n in names] + [TOP_CONCEPT, BOTTOM_CONCEPT])
    k = rng.randrange(5)
    if k == 0:
        return Not(_random_concept(rng, names, roles, depth - 1))
    if k == 1:
        return And((_random_concept(rng, names, roles, depth - 1),
                    _random_concept(rng, names, roles, depth - 1)))
    if k == 2:
        return Or((_random_concept(rng, names, roles, depth - 1),
                   _random_concept(rng, names, roles, depth - 1)))
    r = rng.choice(roles)
    f = _random_concept(rng, names, roles, depth - 1)
    return Exists(r, f) if k == 3 else Forall(r, f)


def _random_interpretation(rng, size, names, roles, inds):
    concepts = {n: {e for e in range(size) if rng.random() < 0.5} for n in names}
    rels = {r: {(a, b) for a in range(size) for b in range(size) if rng.random() < 0.3}
            for r in roles}
    return FiniteInterpretation(size, concepts, rels,
                                {i: rng.randrange(size) for i in inds})


class TestFiniteModelConstruction:

    @pytest.mark.parametrize("seed", range(6))
    def test_cone_semantics_equals_classical_semantics(self, seed):
        """For every ALC concept, halfAxes(C^cone) is the pre-image of C^I (our lemma);
        hence the cone model satisfies exactly the axioms the finite model satisfies.
        Odd and even domain sizes are covered (odd sizes duplicate one element)."""
        rng = random.Random(seed)
        names, roles, inds = ["A", "B", "C"], ["r", "s"], ["a", "b", "c"]
        fi = _random_interpretation(rng, 3 + seed, names, roles, inds)
        interp = ALConeInterpretation.from_finite_model(fi)
        chk = interp.checker()
        f = interp.halfaxis_elements
        for _ in range(150):
            c = _random_concept(rng, names, roles, 3)
            ext = fi.extension(c)
            assert chk.cone(c).halfaxes() == {h for h in range(2 * interp.dim) if f[h] in ext}
            d = _random_concept(rng, names, roles, 2)
            for ax in (GCI(c, d), ConceptAssertion(c, rng.choice(inds)),
                       RoleAssertion(rng.choice(roles), rng.choice(inds), rng.choice(inds))):
                assert chk.satisfies(ax) == fi.satisfies(ax)

    def _ontologies(self):
        """Small ALC ontologies with negation, disjunction and roles, with a finite model
        of each written down by hand."""
        o1 = [GCI(A, Or((B, C))), GCI(And((B, C)), BOTTOM_CONCEPT),
              GCI(Not(A), Forall("r", A)), ConceptAssertion(A, "a"),
              ConceptAssertion(Not(B), "a"), RoleAssertion("r", "a", "b"),
              ConceptAssertion(Not(A), "b")]
        m1 = FiniteInterpretation(3, {"A": {0, 2}, "B": set(), "C": {0, 2}},
                                  {"r": {(0, 1), (1, 2)}}, {"a": 0, "b": 1})
        o2 = [GCI(TOP_CONCEPT, Or((A, Not(A)))), GCI(A, Exists("r", Not(A))),
              GCI(Not(A), Exists("r", A)), GCI(Exists("r", A), Not(A)),
              ConceptAssertion(A, "x")]
        m2 = FiniteInterpretation(2, {"A": {0}}, {"r": {(0, 1), (1, 0)}}, {"x": 0})
        o3 = [GCI(A, Or((B, C, D))), GCI(B, Not(C)), GCI(C, Not(D)), GCI(B, Not(D)),
              GCI(D, Forall("s", BOTTOM_CONCEPT)), GCI(Or((B, C)), Exists("s", D)),
              ConceptAssertion(And((A, Not(D))), "p")]
        m3 = FiniteInterpretation(3, {"A": {0}, "B": {0}, "C": set(), "D": {1}},
                                  {"s": {(0, 1)}}, {"p": 0})
        return [(o1, m1), (o2, m2), (o3, m3)]

    def test_construction_yields_models(self):
        for axioms, fi in self._ontologies():
            assert fi.violations(axioms) == []
            interp = ALConeInterpretation.from_finite_model(fi, axioms)
            assert ALConeChecker(interp).is_model(axioms)
            for p in interp.individuals.values():
                assert np.any(p != 0)

    def test_checker_rejects_broken_interpretation(self):
        axioms, fi = self._ontologies()[0]
        interp = ALConeInterpretation.from_finite_model(fi, axioms)
        # flip one half-axis of B so that B ⊓ C ≠ {0}
        c_bits = interp.concepts["C"].bits.copy()
        b_bits = interp.concepts["B"].bits.copy()
        b_bits[np.flatnonzero(c_bits)[0]] = True
        interp.concepts["B"] = ALCone(b_bits)
        bad = ALConeChecker(interp).violations(axioms)
        assert GCI(And((B, C)), BOTTOM_CONCEPT) in bad
        # move an individual to the origin: assertions fail (JAIR Def. 2)
        I2 = ALConeInterpretation.from_finite_model(fi, axioms)
        I2.individuals["a"] = np.zeros(I2.dim)
        assert ConceptAssertion(A, "a") in ALConeChecker(I2).violations(axioms)
        # drop the role edge: the role assertion fails
        I3 = ALConeInterpretation.from_finite_model(fi, axioms)
        I3.roles["r"] = np.zeros_like(I3.roles["r"])
        assert RoleAssertion("r", "a", "b") in ALConeChecker(I3).violations(axioms)

    def test_rejects_non_model(self):
        axioms, fi = self._ontologies()[0]
        with pytest.raises(ValueError):
            ALConeInterpretation.from_finite_model(fi, axioms + [ConceptAssertion(B, "a")])

    def test_partial_individual_role_semantics(self):
        """Role assertions are b ∈ conHull(R a'): a partial subject spreads its
        successors."""
        R = np.zeros((4, 4), dtype=bool)
        R[2, 0] = True  # half-axis 0 -> half-axis 2
        points = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]),
                  "c": np.array([0.0, -1.0])}
        interp = ALConeInterpretation(2, {}, {"r": R}, points)
        chk = interp.checker()
        assert chk.satisfies(RoleAssertion("r", "a", "b"))
        assert not chk.satisfies(RoleAssertion("r", "a", "c"))
        assert not chk.satisfies(RoleAssertion("r", "b", "a"))


class TestCanonicalModelRoute:

    @pytest.mark.parametrize("fixture", ["t3.owl", "roles.owl", "helly_ok.owl"])
    def test_el_fixtures(self, fixture):
        """EL++ ontologies: mOWL's CanonicalModel is a finite model; the al-cone model
        built from it is verified against the ALC reading of the ontology."""
        ds = PathDataset(os.path.join(FIXTURES, fixture))
        axioms, skipped = axioms_from_owl(ds.ontology)
        interp = ALConeInterpretation.from_ontology(ds.ontology)
        assert ALConeChecker(interp).is_model(axioms)
        if fixture == "roles.owl":
            # role inclusions and chains are outside ALC and are reported, not used
            assert len(skipped) == 2
        else:
            assert skipped == []


# ---------------------------------------------------------------------------
# Boolean construction (JAIR Props. 7 and 8)
# ---------------------------------------------------------------------------

def _boolean_concepts(rng, names, n):
    out = []
    for _ in range(n):
        out.append(_random_concept(rng, names, ["unused"], 3))
    return [c for c in out if "∃" not in str(c) and "∀" not in str(c)]


def _types(names, tbox):
    from mowl.geometric.cones import _eval_boolean
    out = []
    for bits in itertools.product((False, True), repeat=len(names)):
        v = dict(zip(names, bits))
        if all((not _eval_boolean(g.sub, v)) or _eval_boolean(g.sup, v) for g in tbox):
            out.append(v)
    return out


class TestBooleanConstruction:

    def _ontology(self):
        tbox = [GCI(A, Or((B, C))), GCI(And((B, C)), BOTTOM_CONCEPT), GCI(D, Not(A))]
        abox = [ConceptAssertion(A, "a"), ConceptAssertion(Or((C, D)), "b"),
                ConceptAssertion(Not(B), "c")]
        return tbox, abox

    @pytest.mark.parametrize("faithful", [False, True])
    def test_is_model(self, faithful):
        tbox, abox = self._ontology()
        interp = ALConeInterpretation.from_boolean_ontology(tbox + abox, faithful=faithful)
        assert interp.checker().is_model(tbox + abox)

    def test_tbox_faithful(self):
        """C ⊑ D holds in the Prop. 7 model iff the TBox entails it (all atoms are
        realised on half-axes)."""
        tbox, abox = self._ontology()
        names = ["A", "B", "C", "D"]
        types = _types(names, tbox)
        from mowl.geometric.cones import _eval_boolean
        interp = ALConeInterpretation.from_boolean_ontology(tbox + abox)
        chk = interp.checker()
        rng = random.Random(5)
        concepts = _boolean_concepts(rng, names, 200)
        for c, d in zip(concepts[::2], concepts[1::2]):
            entailed = all((not _eval_boolean(c, t)) or _eval_boolean(d, t) for t in types)
            assert chk.satisfies(GCI(c, d)) == entailed

    def test_strong_concept_faithfulness_prop8(self):
        """JAIR Prop. 8: in the doubled model, a ∈ C iff O ⊨ C(a) for Boolean C. The
        Prop. 7 model (one atom per individual) is not strongly faithful."""
        tbox, abox = self._ontology()
        names = ["A", "B", "C", "D"]
        types = _types(names, tbox)
        from mowl.geometric.cones import _eval_boolean
        interp = ALConeInterpretation.from_boolean_ontology(tbox + abox, faithful=True)
        I7 = ALConeInterpretation.from_boolean_ontology(tbox + abox, faithful=False)
        assert interp.dim == 2 * I7.dim
        for model in (interp, I7):
            elems = model.halfaxis_elements
            for n in names:
                ha = model.checker().cone(Atomic(n)).halfaxes()
                for h in range(2 * model.dim):
                    assert (h in ha) == types[elems[h]][n]
        chk, chk7 = interp.checker(), I7.checker()
        rng = random.Random(6)
        unfaithful_7 = 0
        for c in _boolean_concepts(rng, names, 300):
            for ind in ("a", "b", "c"):
                own = [ax.concept for ax in abox if ax.individual == ind]
                compat = [t for t in types if all(_eval_boolean(x, t) for x in own)]
                entailed = all(_eval_boolean(c, t) for t in compat)
                assert chk.instance_of(ind, c) == entailed
                if chk7.instance_of(ind, c) and not entailed:
                    unfaithful_7 += 1
        assert unfaithful_7 > 0

    def test_inconsistent(self):
        with pytest.raises(InconsistentOntology):
            ALConeInterpretation.from_boolean_ontology([GCI(TOP_CONCEPT, A),
                                                        GCI(A, BOTTOM_CONCEPT)])
        with pytest.raises(InconsistentOntology):
            ALConeInterpretation.from_boolean_ontology([GCI(A, B), ConceptAssertion(A, "x"),
                                                        ConceptAssertion(Not(B), "x")])

    def test_roles_rejected(self):
        with pytest.raises(UnsupportedAxiom):
            ALConeInterpretation.from_boolean_ontology([GCI(A, Exists("r", B))])


# ---------------------------------------------------------------------------
# Helly-type behaviour: T3 pattern and nontrivial intersections
# ---------------------------------------------------------------------------

class TestHelly:

    def test_t3_pattern(self):
        """T3 (Leemhuis & Kutz, NeSy 2025; tests/geometric/fixtures/make_fixtures.py):
        A ≡ X1 ⊓ X2, B ≡ X2 ⊓ X3, C ≡ X1 ⊓ X3, A, B, C pairwise disjoint. Boxes cannot
        make A, B, C all non-empty (Helly number 2 per axis). The faithful al-cone model
        has A, B, C ≠ {0}, pairwise X_i ∩ X_j ≠ {0}, and X1 ∩ X2 ∩ X3 = {0}: disjointness
        in cone semantics is "intersection = {0}", and pairwise non-trivial intersection
        does not force a common non-zero point."""
        X1, X2, X3 = Atomic("X1"), Atomic("X2"), Atomic("X3")
        tbox = []
        for N, (P, Q) in ((A, (X1, X2)), (B, (X2, X3)), (C, (X1, X3))):
            tbox += [GCI(N, And((P, Q))), GCI(And((P, Q)), N)]
        tbox += [GCI(And((A, B)), BOTTOM_CONCEPT), GCI(And((B, C)), BOTTOM_CONCEPT),
                 GCI(And((A, C)), BOTTOM_CONCEPT)]
        interp = ALConeInterpretation.from_boolean_ontology(tbox)
        chk = interp.checker()
        assert chk.is_model(tbox)
        for N in (A, B, C, X1, X2, X3):
            assert not chk.cone(N).is_bottom()
        for P, Q in ((X1, X2), (X2, X3), (X1, X3)):
            assert not chk.cone(And((P, Q))).is_bottom()
        assert chk.cone(And((X1, X2, X3))).is_bottom()
        # every cone contains the origin, so the "empty intersection" notion of Helly
        # theorems never applies: the triple intersection is {0}, not empty
        assert chk.cone(And((X1, X2, X3))).contains_point(np.zeros(interp.dim))
        # A, B, C are determined by the X's and X1 ⊓ X2 ⊓ X3 is excluded: 7 atoms on
        # 7 of the 8 half-axes of R^4
        assert interp.dim == 4

    def test_t3_fixture_via_canonical_model(self):
        ds = PathDataset(os.path.join(FIXTURES, "t3.owl"))
        interp = ALConeInterpretation.from_ontology(ds.ontology)
        ns = "http://mowl.borg/geo#"
        chk = interp.checker()
        for n in ("A", "B", "C"):
            assert not chk.cone(Atomic(ns + n)).is_bottom()
        assert chk.cone(And((Atomic(ns + "X1"), Atomic(ns + "X2"),
                             Atomic(ns + "X3")))).is_bottom()

    @pytest.mark.parametrize("dim", [1, 2])
    def test_nontrivial_helly_number_is_2n(self, dim):
        """Our observation: for 'non-trivial intersection' (≠ {0}) the Helly number of
        the al-cones of R^n is 2n. Tightness: the 2n cones that each miss one half-axis
        intersect non-trivially in every 2n-1 of them but not all together. Bound: an
        exhaustive check that every family of 2n+1 cones whose 2n-subfamilies all
        intersect non-trivially intersects non-trivially."""
        m = 2 * dim
        family = [ALCone(np.array([h != j for h in range(m)])) for j in range(m)]
        total = ALCone.top(dim)
        for X in family:
            total = total & X
        assert total.is_bottom()
        for sub in itertools.combinations(family, m - 1):
            t = ALCone.top(dim)
            for X in sub:
                t = t & X
            assert not t.is_bottom()
        cones = [X for X in ALCone.all_cones(dim) if not X.is_bottom()]
        for fam in itertools.combinations(cones, m + 1):
            ok = all(not _meet(sub, dim).is_bottom()
                     for sub in itertools.combinations(fam, m))
            if ok:
                assert not _meet(fam, dim).is_bottom()


def _meet(cones, dim):
    t = ALCone.top(dim)
    for X in cones:
        t = t & X
    return t


def test_halfaxis_vector():
    assert list(halfaxis_vector(2, 3)) == [0.0, -1.0]
    assert point_support(halfaxis_vector(3, 4)) == {4}
