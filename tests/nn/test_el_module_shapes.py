"""Every per-axiom loss of every ELModule subclass is a column of shape (B, 1); the option
``per_axiom_column = False`` gives back the shape of the formula, and the values are the same.

``golden/el_module_losses_flat.json`` holds the values (flattened, 6 significant digits) that
every module returned before the change (with the shape it had), for seeded modules and batches of
``tests/nn/el_module_zoo.py``.
"""
import json
from pathlib import Path

import pytest
import torch as th

from tests.nn.el_module_zoo import B, FORMS, all_el_module_classes, batch, make

GOLDEN = json.loads((Path(__file__).parent / "golden" / "el_module_losses_flat.json").read_text())

# These two were made columns by the previous change, so their golden shapes (taken from the
# commit before this one) are columns; their option still gives the flat (B,) of the formula.
FLAT_WITH_OPTION = {"BoxSquaredELModule", "MultiBoxELModule"}


# These two refused a negative of a form without one before the change as well.
ALWAYS_STRICT = {"TransBoxModule", "VolumeMultiBoxELModule"}


def _old_shape(name, form, golden_shape):
    if name in FLAT_WITH_OPTION and not form.startswith("role_"):
        return (B,)
    return tuple(golden_shape)


def _call(module, form, neg):
    # The golden values are those of the earlier fall-through (neg=True on a form without a
    # negative loss returned the positive loss); that is what strict_negatives=False selects.
    if type(module).__name__ not in ALWAYS_STRICT:
        module.strict_negatives = False
    try:
        out = module(batch(form), form, neg=neg)
    except NotImplementedError:
        return None
    return out if th.is_tensor(out) else None


CLASSES = all_el_module_classes()


def _constructible(cls):
    try:
        make(cls)
        return True
    except NotImplementedError:  # abstract bases
        return False


def _implements_something(cls):
    module = make(cls)
    return any(_call(module, f, False) is not None for f in FORMS)


CLASSES = [c for c in CLASSES if _constructible(c) and _implements_something(c)]


def test_the_survey_covers_every_el_module():
    names = {c.__name__ for c in CLASSES}
    assert {"ELBEModule", "MultiELBEModule", "PrismModule", "BoxELModule", "TransBoxModule",
            "CanonicalMixtureModule", "HyperbolicConesModule"} <= names


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("neg", [False, True], ids=["pos", "neg"])
def test_every_implemented_form_is_a_column(cls, neg):
    module = make(cls)
    assert module.per_axiom_column is True
    seen = 0
    for form in FORMS:
        out = _call(module, form, neg)
        if out is None:
            continue
        seen += 1
        assert tuple(out.shape) == (B, 1), (cls.__name__, form, neg, tuple(out.shape))
    if not neg:
        assert seen > 0, f"{cls.__name__} implements no form"


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("neg", [False, True], ids=["pos", "neg"])
def test_option_gives_the_old_shape_and_values_are_unchanged(cls, neg):
    module = make(cls)
    columns = {f: _call(module, f, neg) for f in FORMS}
    module.per_axiom_column = False
    for form in FORMS:
        col, flat = columns[form], _call(module, form, neg)
        assert (col is None) == (flat is None)
        if col is None:
            continue
        key = f"{cls.__name__}|{form}|{int(neg)}"
        assert key in GOLDEN, key
        assert tuple(flat.shape) == _old_shape(cls.__name__, form, GOLDEN[key]["shape"]), key
        assert th.equal(col.reshape(-1), flat.reshape(-1))
        assert col.detach().reshape(-1).tolist() == pytest.approx(GOLDEN[key]["values"],
                                                                  rel=1e-5, abs=1e-6), key


def test_golden_values_cover_every_checked_loss():
    expected = {f"{c.__name__}|{f}|{int(n)}" for c in CLASSES for f in FORMS for n in (0, 1)
                if _call(make(c), f, bool(n)) is not None}
    assert expected == set(GOLDEN)


def test_a_loss_added_to_a_flat_vector_is_the_hazard_the_column_removes():
    """The reason for one shape: (B,) + (B, 1) broadcasts to (B, B), (B, 1) + (B, 1) does not."""
    module = make(next(c for c in CLASSES if c.__name__ == "ELBEModule"))
    for form in ("gci0_bot", "gci3_bot", "gci0"):
        out = module(batch(form), form)
        assert (out + th.zeros(B, 1)).shape == (B, 1), form
