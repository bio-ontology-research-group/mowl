"""A comparison of two trainings says nothing about a parameter that neither training moves.
For the methods in which some table did not move in a Pair, the Pairs of the method together
must move every trainable parameter of the combination, except those listed here with the
reason (none are defects)."""
import pytest
import torch as th

from tests.baselines.harness import all_pairs
from tests.baselines.test_equivalence import _both, _seed

#: ``(method, parameter): reason`` for a trainable parameter no Pair moves.
EXPLAINED = {}

METHODS = ("eike", "concept2box", "emelvar", "falcon", "dfalc")


def _trainable(model):
    return {n: p.detach().clone() for n, p in model.module.named_parameters()
            if p.requires_grad}


@pytest.mark.parametrize("method", METHODS)
def test_every_trainable_parameter_moves_in_some_pair_of_the_method(method):
    pairs = [p for p in all_pairs() if p.method == method]
    assert pairs
    moved, seen = set(), set()
    for pair in pairs:
        _, new = _both(pair)
        before = _trainable(new)
        _seed(pair.seed)
        pair.train(new, pair.epochs)
        after = _trainable(new)
        seen |= set(before)
        moved |= {n for n in before if not th.equal(before[n], after[n])}
    stuck = {n for n in seen - moved if (method, n) not in EXPLAINED}
    # tables that are views of another table are the same parameter under two names
    assert not stuck, f"{method}: parameters that no Pair moves: {sorted(stuck)}"


def test_frozen_tables_are_not_parameters():
    """FALCON and DF-ALC: an individual is the element ``i`` of the domain, a one-hot row of a
    frozen table (``DomainElements``), so it has no gradient by construction."""
    pair = next(p for p in all_pairs() if p.key == "falcon")
    _, new = _both(pair)
    assert not new.module.individual_model.centres.weight.requires_grad
