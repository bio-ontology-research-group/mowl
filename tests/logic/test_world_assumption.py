"""Open and closed world: the assumption of regimes, methods and benchmarks."""

import pytest

from mowl.benchmarks import get_benchmark, list_benchmarks
from mowl.catalog import get_method, get_regime, list_regimes
from mowl.logic import LogicalStatus, TASKS, WorldAssumption

OPEN, ENTAILED, STATED = (WorldAssumption.OPEN, WorldAssumption.CLOSED_ENTAILED,
                          WorldAssumption.CLOSED_STATED)


def test_what_each_assumption_takes_to_be_false():
    e, c, u = LogicalStatus.ENTAILED, LogicalStatus.CONTRADICTORY, LogicalStatus.UNDETERMINED
    assert [OPEN.is_false(s) for s in (e, c, u)] == [False, True, False]
    assert [ENTAILED.is_false(s) for s in (e, c, u)] == [False, True, True]
    assert [STATED.is_false(s) for s in (e, c, u)] == [True, True, True]
    # a stated axiom is never a negative, under any assumption
    assert not any(a.is_false(u, stated=True) for a in WorldAssumption)
    assert OPEN.is_false("contradictory")


@pytest.mark.parametrize("key, expected", [
    ("either-side-corruption", STATED),
    ("known-positive-filtering", STATED),
    ("dele-closure-filtered-negatives", ENTAILED),
    ("contradiction-negatives", OPEN),
    ("common-filtered-ranking", STATED),
    ("query-answering", STATED),
    ("dele-evaluation", ENTAILED),
    ("faithfulness-hierarchy", OPEN),
    ("exact-model-check", OPEN),
])
def test_regimes_state_their_assumption(key, expected):
    assert WorldAssumption(get_regime(key).assumption) is expected


def test_regimes_that_do_not_decide_truth_state_none():
    for key in ("pairwise-margin-ranking", "linear-lr-decay", "loss-scoring",
                "transitive-closure-positives"):
        assert get_regime(key).assumption == ""


def test_every_negative_sampling_regime_states_an_assumption():
    undeclared = [r.key for r in list_regimes()
                  if r.components == ("negative sampling",) and not r.assumption]
    assert undeclared == []


def test_an_unknown_assumption_is_rejected():
    import dataclasses
    with pytest.raises(ValueError, match="assumption"):
        dataclasses.replace(get_regime("bpr-loss"), assumption="half open")


def test_benchmarks_inherit_the_assumption_of_their_evaluation_regime():
    b = get_benchmark("common-go-subsumption-prediction")
    assert b.assumption is STATED
    closed = list_benchmarks(assumption="closed world over stated axioms")
    assert b in closed
    open_world = list_benchmarks(assumption=OPEN)
    assert open_world and all(x.assumption is OPEN for x in open_world)
    assert not set(x.key for x in closed) & set(x.key for x in open_world)


def test_methods_state_the_assumption_of_their_own_objective():
    assert WorldAssumption(get_method("query2box").assumption) is STATED
    assert WorldAssumption(get_method("dfalc").assumption) is OPEN
    assert WorldAssumption(get_method("convex_regions").assumption) is OPEN


def test_query_answering_is_defined_open_world_and_says_so():
    summary = TASKS["query-answering"].summary
    assert "certain answers" in summary and "closed world" in summary
