"""Every method, built from parts, gives the same numbers as its direct implementation."""
import random

import numpy as np
import pytest
import torch as th

from tests.baselines.harness import all_pairs, assert_same

PAIRS = all_pairs()
IDS = [p.key for p in PAIRS]

from tests.baselines.own_code import (TEMPORARY_EXPECTED_FAILURES, foreign_members,
                                      named_units, own_computation)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)


def _both(pair):
    dataset = pair.dataset()
    _seed(pair.seed)
    old = pair.old(dataset)
    _seed(pair.seed)
    new = pair.new(dataset)
    pair.copy(old, new)
    return old, new


def _skip(pair, check):
    if check in pair.skip:
        pytest.skip(pair.skip[check])


@pytest.mark.parametrize("pair", PAIRS, ids=IDS)
def test_the_method_users_get_is_built_from_parts(pair, request):
    _skip(pair, "parts")
    if pair.key in TEMPORARY_EXPECTED_FAILURES:
        request.applymarker(pytest.mark.xfail(strict=True,
                                              reason=TEMPORARY_EXPECTED_FAILURES[pair.key]))
    from mowl.representation.geometry import Geometry
    _, new = _both(pair)
    assert isinstance(new.module, Geometry), type(new.module)
    assert "mowl.baselines" not in type(new).__module__
    assert "mowl.baselines" not in type(new.module).__module__
    for part in vars(new.module).values():
        assert "mowl.baselines" not in type(part).__module__, type(part)
    # The parts are named and there is more than one.
    parts = new.module.parts
    parts = parts() if callable(parts) else parts
    assert isinstance(parts, dict) and len(parts) >= 2, parts
    # Every object the module is made of is a general library class.
    assert not foreign_members(new.module)
    # Neither the module class nor the model class computes with tensors itself, nor do the
    # module-level functions of the method's own package, nor the general models they are
    # built on (they wire, hold options and call parts): the computation belongs to the parts.
    units = named_units(pair.key)
    assert not own_computation(type(new.module), units)
    assert not own_computation(type(new), units)


@pytest.mark.parametrize("pair", PAIRS, ids=IDS)
def test_the_direct_implementation_is_the_baseline(pair):
    old, _ = _both(pair)
    assert type(old).__module__.startswith("mowl.baselines"), type(old).__module__


@pytest.mark.parametrize("pair", PAIRS, ids=IDS)
def test_same_losses_from_the_same_parameters(pair):
    _skip(pair, "losses")
    old, new = _both(pair)
    _seed(pair.seed)
    a = pair.losses(old)
    _seed(pair.seed)
    b = pair.losses(new)
    assert_same(a, b, pair, "loss")


@pytest.mark.parametrize("pair", PAIRS, ids=IDS)
def test_same_scores_from_the_same_parameters(pair):
    _skip(pair, "scores")
    old, new = _both(pair)
    assert_same(pair.scores(old), pair.scores(new), pair, "scores")


@pytest.mark.parametrize("pair", PAIRS, ids=IDS)
def test_same_training_run(pair):
    """Same start, same seed, same number of epochs: same losses, scores and parameters after."""
    _skip(pair, "training")
    old, new = _both(pair)
    _seed(pair.seed)
    pair.train(old, pair.epochs)
    _seed(pair.seed)
    pair.train(new, pair.epochs)
    assert_same(pair.scores(old), pair.scores(new), pair, "scores after training")
    if "losses" not in pair.skip:
        _seed(pair.seed)
        a = pair.losses(old)
        _seed(pair.seed)
        b = pair.losses(new)
        assert_same(a, b, pair, "loss after training")
    if pair.parameters is not None:
        assert_same(pair.parameters(old), pair.parameters(new), pair, "parameters after training")
