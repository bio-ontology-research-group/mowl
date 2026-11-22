"""``neg=True`` on a form without a negative loss is an error for every ELModule, and every
module's ``neg_capable_gcis`` is truthful."""
import pytest
import torch as th

from mowl.nn import ELModule
from tests.nn.el_module_zoo import FORMS, all_el_module_classes, batch, make


def _constructible(cls):
    try:
        make(cls)
        return True
    except NotImplementedError:
        return False


CLASSES = [c for c in all_el_module_classes() if _constructible(c)]


def _positive(module, form, seed):
    try:
        out = module(batch(form, seed=seed), form)
    except NotImplementedError:
        return None
    return out if th.is_tensor(out) else None


def test_the_default_is_strict():
    assert ELModule.strict_negatives is True


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_declared_negative_forms_differ_from_the_positive_loss(cls):
    module = make(cls)
    for form in sorted(module.neg_capable_gcis):
        pos = _positive(module, form, 7)
        if pos is None:  # a form not implemented as a (B, ...) loss by this module
            continue
        different = False
        for seed in (7, 8, 9):
            data = batch(form, seed=seed)
            p, n = module(data, form), module(data, form, neg=True)
            different |= p.shape != n.shape or not th.allclose(p, n)
        assert different, (cls.__name__, form, "neg_capable_gcis lists a form without a "
                           "negative loss of its own")


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_every_other_form_refuses_a_negative(cls):
    module = make(cls)
    for form in FORMS:
        if form in module.neg_capable_gcis:
            continue
        with pytest.raises(NotImplementedError):
            module(batch(form), form, neg=True)
        method = getattr(module, form + "_loss")
        with pytest.raises(NotImplementedError):  # called directly, positionally too
            method(batch(form), True)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_the_option_gives_the_positive_loss_back(cls):
    module = make(cls)
    module.strict_negatives = False
    for form in FORMS:
        if form in module.neg_capable_gcis:
            continue
        pos = _positive(module, form, 7)
        if pos is None:
            continue
        assert th.equal(module(batch(form), form, neg=True), pos), (cls.__name__, form)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_every_form_the_module_has_a_negative_for_is_one_the_module_implements(cls):
    module = make(cls)
    for form in module.neg_capable_gcis:
        assert form in FORMS or form.endswith("_bot"), (cls.__name__, form)
