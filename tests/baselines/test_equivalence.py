"""Every method, built from parts, gives the same numbers as its direct implementation."""
import random

import numpy as np
import pytest
import torch as th

from tests.baselines.harness import all_pairs, assert_same

PAIRS = all_pairs()
IDS = [p.key for p in PAIRS]


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
def test_the_method_users_get_is_built_from_parts(pair):
    _skip(pair, "parts")
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
    # Every object the module is made of is a general library class, not code
    # from a per-method model file.
    general = ("mowl.representation", "mowl.training", "mowl.evaluation",
               "mowl.projection", "mowl.walking", "mowl.corpus", "mowl.reasoning",
               "mowl.datasets", "mowl.owlapi", "mowl.ontology", "mowl.utils")
    members = list(vars(new.module).values())
    if hasattr(new.module, "modules"):
        members += [m for m in new.module.modules() if m is not new.module]
    for member in members:
        where = type(member).__module__
        if where.startswith("mowl."):
            assert where.startswith(general), (type(member).__name__, where)
    # The module is a general class, or a named combination that adds no
    # computation of its own.
    cls = type(new.module)
    if not cls.__module__.startswith("mowl.representation"):
        own = sorted(k for k, v in vars(cls).items()
                     if callable(v) and not k.startswith("__"))
        assert not own, (cls.__module__, cls.__name__, own)


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
