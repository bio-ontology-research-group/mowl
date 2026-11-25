import pytest
import torch as th

import mowl.baselines.nn.el.transbox.code_losses as CL
import mowl.baselines.nn.el.transbox.losses as L
from mowl.nn.el.transbox.module import TransBoxModule

D2 = th.tensor([[0, 1], [2, 3]])
D3 = th.tensor([[0, 1, 2], [1, 3, 4]])


def _module(**kw):
    th.manual_seed(0)
    return TransBoxModule(5, 2, nb_inds=3, embed_dim=4, margin=0.0, **kw)


def test_reference_bottom_forms_follow_the_code_by_default():
    m = _module(formulation="reference")
    assert m.reference_bottom_losses == "code"
    expected0 = CL.gci0_loss(D2, m.class_center, m.class_offset, m.margin)
    expected3 = CL.gci3_loss(D3, m.class_center, m.class_offset, m.relation_center,
                             m.relation_offset, m.margin)
    assert th.allclose(m(D2, "gci0_bot").reshape(-1), expected0)
    assert th.allclose(m(D3, "gci3_bot").reshape(-1), expected3)
    # the code's loss carries the centre-norm terms, the mowl formula does not
    assert not th.allclose(m(D2, "gci0_bot").reshape(-1),
                           L.gci0_loss(D2, m.class_center, m.class_offset, m.margin))


def test_reference_bottom_forms_old_option_gives_the_mowl_formulas():
    m = _module(formulation="reference", reference_bottom_losses="mowl")
    assert th.allclose(m(D2, "gci0_bot").reshape(-1),
                       L.gci0_loss(D2, m.class_center, m.class_offset, m.margin))
    assert th.allclose(m(D3, "gci3_bot").reshape(-1),
                       L.gci3_loss(D3, m.class_center, m.class_offset, m.relation_center,
                                   m.relation_offset, m.margin))


def test_mowl_formulation_bottom_forms_unchanged():
    m = _module()
    assert th.allclose(m(D2, "gci0_bot").reshape(-1),
                       L.gci0_loss(D2, m.class_center, m.class_offset, m.margin))


def test_invalid_option_is_rejected():
    with pytest.raises(ValueError):
        _module(reference_bottom_losses="x")


@pytest.mark.parametrize("form,data", [("object_property_assertion", th.tensor([[0, 1, 1]])),
                                       ("class_assertion", th.tensor([[0, 1]])),
                                       ("gci0", D2), ("gci3", D3),
                                       ("role_inclusion", th.tensor([[0, 1]]))])
def test_negative_of_a_form_without_negative_loss_raises(form, data):
    m = _module()
    with pytest.raises(NotImplementedError):
        m(data, form, neg=True)


def test_strict_negatives_off_returns_the_positive_loss():
    m = _module(strict_negatives=False)
    assert th.allclose(m(D2, "gci0", neg=True), m(D2, "gci0"))


def test_gci2_negative_still_works():
    m = _module()
    data = th.tensor([[0, 1, 2]])
    assert m(data, "gci2", neg=True).shape == (1, 1)
