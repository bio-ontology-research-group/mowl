"""Shared description of one "combination of parts versus direct implementation" comparison.

Each group of methods has a file ``tests/baselines/pairs_<group>.py`` that defines

- ``PAIRS``: a list of :class:`Pair`, one per method (several if a method has variants);
- ``EXEMPT``: ``{catalogue key: reason}`` for methods that have no direct implementation to
  compare with (the reason is shown in the documentation, so write it for a reader).

``test_equivalence.py`` and ``test_layout.py`` collect these files; nobody edits a shared list.
"""
import glob
import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch as th


def default_dataset():
    from mowl.datasets.builtin import FamilyDataset
    return FamilyDataset()


def default_losses(model):
    """Loss of every training axiom form, positives and (where the module has one) negatives."""
    out = {}
    module = model.module
    for form, data in model.training_datasets.items():
        if len(data) == 0:
            continue
        out[form] = module(data[:], form)
        if form in getattr(module, "neg_capable_gcis", ()):
            out[form + "/neg"] = module(data[:], form, neg=True)
    return out


def default_train(model, epochs):
    model.train(epochs=epochs)


@dataclass
class Pair:
    """One comparison.

    :param key: catalogue key of the method (``mowl.catalog``); add ``"/variant"`` for variants.
    :param new: ``dataset -> model`` building the class users get (``mowl.models``).
    :param old: ``dataset -> model`` building the direct implementation (``mowl.baselines``).
    :param copy: ``(old_model, new_model) -> None`` copying every parameter and buffer of the
        direct implementation into the combination, so both start from the same numbers.
    :param scores: ``model -> Tensor`` (or dict of tensors): what the evaluation regime ranks,
        on a fixed input.
    :param losses: ``model -> {name: Tensor}``; default: every axiom form of the training data.
    :param train: ``(model, epochs) -> None``; default ``model.train(epochs=epochs)``.
    :param parameters: ``model -> {name: Tensor}`` in a naming common to both sides, compared
        after training; default: compare ``scores`` and ``losses`` only.
    :param skip: ``{check: reason}`` for a check that cannot hold (checks: ``"parts"``,
        ``"losses"``, ``"training"``, ``"scores"``). Every entry must be justified in the report.
    """
    key: str
    new: Callable
    old: Callable
    copy: Callable
    scores: Callable
    losses: Callable = default_losses
    train: Callable = default_train
    parameters: Optional[Callable] = None
    dataset: Callable = default_dataset
    epochs: int = 3
    seed: int = 0
    atol: float = 1e-5
    rtol: float = 1e-4
    skip: dict = field(default_factory=dict)

    @property
    def method(self):
        return self.key.split("/")[0]


def _modules():
    here = os.path.dirname(__file__)
    for path in sorted(glob.glob(os.path.join(here, "pairs_*.py"))):
        yield importlib.import_module("tests.baselines." + os.path.basename(path)[:-3])


def all_pairs():
    return [p for m in _modules() for p in getattr(m, "PAIRS", [])]


def all_exempt():
    out = {}
    for m in _modules():
        out.update(getattr(m, "EXEMPT", {}))
    return out


def as_dict(value):
    if isinstance(value, dict):
        return value
    return {"value": value}


def assert_same(a, b, pair, what):
    a, b = as_dict(a), as_dict(b)
    assert sorted(a) == sorted(b), f"{pair.key} {what}: {sorted(a)} != {sorted(b)}"
    for name in a:
        x, y = th.as_tensor(a[name]).detach().double(), th.as_tensor(b[name]).detach().double()
        assert x.shape == y.shape, f"{pair.key} {what} {name}: shape {x.shape} != {y.shape}"
        assert th.allclose(x, y, atol=pair.atol, rtol=pair.rtol, equal_nan=True), (
            f"{pair.key} {what} {name}: largest difference {(x - y).abs().max().item():.3g}")
