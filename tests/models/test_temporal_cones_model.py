"""Tests of the temporal al-cone construction (mowl.geometric.temporal_cones).

Source: Leemhuis, FCR 2024 (CEUR Vol-3763, paper 3): Definition 2, Equation (1),
Theorem 1 and Theorem 2. The classical side is evaluated by an independent evaluator
(least fixpoint for until), the geometric side by cone operations.
"""
import random

import pytest

from mowl.geometric.cones import And, Atomic, BOTTOM_CONCEPT, GCI, Not, Or, TOP_CONCEPT
from mowl.geometric.temporal_cones import (
    Always, Eventually, LassoModel, Next, TemporalALConeInterpretation, TemporalOntology,
    Until, find_lasso_model, parse_axiom, parse_concept)

A, B, C = Atomic("A"), Atomic("B"), Atomic("C")


def _random_concept(rng, depth, names=("A", "B", "C")):
    if depth == 0 or rng.random() < 0.2:
        return Atomic(rng.choice(names))
    kind = rng.choice(["not", "and", "or", "next", "until", "ev", "alw"])
    sub = lambda: _random_concept(rng, depth - 1, names)  # noqa: E731
    if kind == "not":
        return Not(sub())
    if kind == "and":
        return And((sub(), sub()))
    if kind == "or":
        return Or((sub(), sub()))
    if kind == "next":
        return Next(sub())
    if kind == "until":
        return Until(sub(), sub())
    if kind == "ev":
        return Eventually(sub())
    return Always(sub())


def _random_model(rng, n_traces, length, loop_start):
    traces = [[frozenset(n for n in "ABC" if rng.random() < 0.5) for _ in range(length)]
              for _ in range(n_traces)]
    inds = {f"i{k}": k for k in range(n_traces)}
    return LassoModel(traces, loop_start, inds)


def test_parser_and_derived_operators():
    assert parse_concept("A & B | C") == Or((And((A, B)), C))
    assert parse_concept("X A U B") == Until(Next(A), B)
    assert parse_concept("F A") == Eventually(A)
    assert parse_concept("G ~A") == Always(Not(A))
    assert parse_axiom("A <= F B") == GCI(A, Eventually(B))
    with pytest.raises(ValueError):
        parse_concept("A &")


def test_next_and_until_semantics_on_a_hand_built_cycle():
    # one element: A at 0, B at 1, C at 2, then 1, 2, 1, 2, ...  (L = 3, loop at 1)
    model = LassoModel([[{"A"}, {"B"}, {"C"}]], loop_start=1, individuals={"a": 0})
    g = TemporalALConeInterpretation.from_lasso_model(model)
    assert g.successor(2) == 1 and g.canonical_time(3) == 1 and g.canonical_time(6) == 2
    assert g.instance_of("a", Next(B), 0) and not g.instance_of("a", Next(C), 0)
    assert g.instance_of("a", Until(Or((A, B)), C), 0)
    assert not g.instance_of("a", Until(A, C), 0)      # B interrupts
    assert g.instance_of("a", Eventually(C), 1)
    assert g.instance_of("a", Always(Or((B, C))), 1)   # periodic part only
    assert not g.instance_of("a", Always(Or((B, C))), 0)
    assert g.instance_of("a", B, 3) and g.instance_of("a", C, 100)  # time 100 -> position 2


def test_geometry_agrees_with_classical_semantics_on_random_models():
    """Theorem 2 (<=, =>) for the construction: membership and inclusion coincide with the
    classical temporal semantics, for random concepts and ultimately periodic models."""
    rng = random.Random(7)
    checked = 0
    for _ in range(25):
        length = rng.randint(1, 4)
        model = _random_model(rng, rng.randint(1, 5), length, rng.randrange(length))
        g = TemporalALConeInterpretation.from_lasso_model(model)
        for _ in range(12):
            c = _random_concept(rng, 3)
            d = _random_concept(rng, 3)
            for t in range(length + 3):
                for a, el in model.individuals.items():
                    assert g.instance_of(a, c, t) == model.holds(c, el, t), (str(c), t)
                    checked += 1
            assert g.satisfies(GCI(c, d)) == model.satisfies(GCI(c, d)), (str(c), str(d))
    assert checked > 500


def test_negation_is_polar_and_concepts_are_cones():
    model = LassoModel([[{"A"}, set()], [set(), {"A"}], [{"A"}, {"A"}]], 0, {"x": 0})
    g = TemporalALConeInterpretation.from_lasso_model(model)
    for t in (0, 1):
        assert g.cone(Not(A), t) == g.cone(A, t).polar()
        assert g.cone(And((A, Not(A))), t).is_bottom()
        assert g.cone(Or((A, Not(A))), t).is_top()
    assert g.cone(Next(A), 0) == g.cone(A, 1)


def test_individuals_are_rigid_rays():
    model = LassoModel([[{"A"}, set()]], 0, {"a": 0})
    g = TemporalALConeInterpretation.from_lasso_model(model)
    x = g.point("a")
    for scale in (0.5, 1.0, 7.0):  # a ray: the length does not matter
        g.individuals["a"] = scale * x
        assert g.instance_of("a", A, 0) and not g.instance_of("a", A, 1)


def test_convex_union_inside_until_is_needed_for_correctness():
    """Two traces, one with A now and one with A next. B <= F A holds on both elements.
    The cone of B contains both half-axes and is not covered by the *set* union of the cone
    of A now and of A next; with the convex union used here the inclusion holds."""
    model = LassoModel([[{"A", "B"}, set()], [{"B"}, {"A"}]], loop_start=1,
                       individuals={"a": 0, "b": 1})
    assert model.satisfies(GCI(B, Eventually(A)))
    g = TemporalALConeInterpretation.from_lasso_model(model)
    assert g.satisfies(GCI(B, Eventually(A)))
    cone_b = g.cone(B, 0)
    assert len(cone_b.halfaxes()) == 2
    union_pieces = [g.cone(A, 0), g.cone(A, 1)]
    assert not any(cone_b.subset_of(p) for p in union_pieces)


def test_example_3_of_the_paper_has_a_geometric_model():
    """Non-EU members that eventually become members must first be candidates (Example 3)."""
    onto = TemporalOntology.parse(
        tbox=["~M & F M <= F (C U M)"],
        abox=["~M(france)", "(~M & F M)(turkey)", "M(germany)"])
    g = TemporalALConeInterpretation.from_ontology(onto, max_prefix=2, max_period=2)
    assert g.is_model(onto)
    assert g.lasso_model.is_model(onto)
    assert g.dim >= 1
    # turkey is not a member now and becomes one later, passing through candidacy:
    assert g.instance_of("turkey", parse_concept("~M & F M"), 0)
    assert g.instance_of("turkey", parse_concept("F (C U M)"), 0)
    assert g.instance_of("germany", parse_concept("M"), 0)


def test_next_chain_needs_a_period():
    onto = TemporalOntology.parse(tbox=["A <= X B", "B <= X A", "A <= ~B"], abox=["A(a)"])
    g = TemporalALConeInterpretation.from_ontology(onto, max_prefix=0, max_period=2)
    assert g.is_model(onto)
    assert [g.instance_of("a", A, t) for t in range(6)] == [True, False] * 3
    assert [g.instance_of("a", B, t) for t in range(6)] == [False, True] * 3


def test_unsatisfiable_within_bounds_returns_none():
    onto = TemporalOntology.parse(tbox=["top <= F ~A"], abox=["(G A)(a)"])
    assert find_lasso_model(onto, max_prefix=2, max_period=2) is None
    with pytest.raises(ValueError):
        TemporalALConeInterpretation.from_ontology(onto, max_prefix=2, max_period=2)
    # ... and a satisfiable variant is found:
    onto2 = TemporalOntology.parse(tbox=["A <= F ~A"], abox=["A(a)"])
    assert find_lasso_model(onto2, max_prefix=1, max_period=2) is not None


def test_violation_is_reported_by_the_geometric_checker():
    model = LassoModel([[{"A"}, set()]], 0, {"a": 0})
    g = TemporalALConeInterpretation.from_lasso_model(model)
    bad = GCI(A, Next(A))
    assert not g.satisfies(bad)
    assert g.violations([bad, GCI(A, TOP_CONCEPT)]) == [bad]
    assert g.satisfies(GCI(BOTTOM_CONCEPT, A))


def test_odd_number_of_traces_and_no_atoms():
    model = LassoModel([[{"A"}], [{"B"}], [{"A", "B"}]], 0, {"a": 2})
    g = TemporalALConeInterpretation.from_lasso_model(model)
    assert g.dim == 2 and g.instance_of("a", And((A, B)), 0)
    empty = LassoModel([[set(), set()]], 1, {"a": 0})
    g2 = TemporalALConeInterpretation.from_lasso_model(empty)
    assert g2.length == 2 and not g2.instance_of("a", A, 0)
