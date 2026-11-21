"""Gumbel box geometry, vector-to-box distance and loss semantics of Concept2Box."""
import pytest
import torch as th

import mowl
mowl.init_jvm("1g")
import mowl.datasets  # noqa: F401,E402
import mowl.nn.el.concept2box.losses as L  # noqa: E402
from mowl.nn.el.concept2box.module import Concept2BoxModule  # noqa: E402

SHARP = 1e-3     # beta -> 0: Gumbel boxes become hard boxes


def _box(lo, hi):
    lo, hi = th.tensor([lo], dtype=th.float64), th.tensor([hi], dtype=th.float64)
    return lo, hi


def _phi(h, t, beta=SHARP):
    return L.conditional_probability(*h, *t, beta).item()


def test_hard_limit_of_the_conditional_probability():
    big, small, far = _box(0., 4.), _box(1., 2.), _box(10., 12.)
    # box t inside box h: phi = 1
    assert _phi(big, small) == pytest.approx(1.0, abs=1e-2)
    # box t contains box h: phi = Vol(h) / Vol(t)
    assert _phi(small, big) == pytest.approx(1.0 / 4.0, abs=1e-2)
    # partial overlap [0,4] and [3,5]: Vol(inter)=1, Vol(t)=2
    assert _phi(_box(0., 4.), _box(3., 5.)) == pytest.approx(0.5, abs=1e-2)
    # disjoint boxes: phi = 0
    assert _phi(big, far) == pytest.approx(0.0, abs=1e-3)


def test_gumbel_boxes_give_a_gradient_for_disjoint_boxes():
    lo = th.tensor([10.0], requires_grad=True)
    phi = L.conditional_probability(*_box(0., 4.), lo, lo + 2, 1.0)
    phi.backward()
    assert phi.item() > 0 and lo.grad is not None and lo.grad.abs().item() > 0


def test_volume_grows_with_the_box():
    a = L.gumbel_log_volume(*_box(0., 1.), 1.0)
    b = L.gumbel_log_volume(*_box(0., 2.), 1.0)
    assert b > a


def test_vector_to_box_distance():
    center, offset = th.zeros(1, 2), th.ones(1, 2)
    # alpha = 0 isolates the outside part: L1 distance to the box
    out = L.vector_to_box_distance(th.tensor([[3.0, 0.0]]), center, offset, 0.0, 1.0)
    assert out.item() == pytest.approx(2.0)
    inside = L.vector_to_box_distance(th.tensor([[0.5, -0.5]]), center, offset, 0.0, 1.0)
    assert inside.item() == 0.0
    # with alpha > 0 the inside part is the distance to the centre, scaled by alpha(c)
    with_alpha = L.vector_to_box_distance(th.tensor([[0.5, -0.5]]), center, offset, 0.5, 1.0)
    assert 0.0 < with_alpha.item() <= 0.5 * 1.0 + 1e-6
    assert (L.vector_to_box_distance(th.tensor([[3.0, 0.0]]), center, offset, 0.5, 1.0)
            > out).all()


def test_alpha_is_smaller_for_larger_boxes():
    """Equation 9: the inside weight decreases as the box volume increases, so entities of
    a general (large) concept are not pulled to its centre."""
    volumes = [L.gumbel_log_volume(*_box(0., side), 0.1) for side in (0.5, 1.0, 3.0, 10.0)]
    weights = [L.inside_weight(v, 0.9).item() for v in volumes]
    assert weights == sorted(weights, reverse=True)
    assert 0.9 * 0.5 <= weights[-1] < weights[0] <= 0.9


def test_losses_are_zero_when_the_condition_holds():
    assert L.ontology_view_loss(th.tensor([1.0]), neg=False).item() == 0.0
    assert L.ontology_view_loss(th.tensor([0.0]), neg=True).item() == 0.0
    assert L.ontology_view_loss(th.tensor([0.0]), neg=False).item() == 1.0
    assert L.ontology_view_loss(th.tensor([1.0]), neg=True).item() == 1.0
    assert L.instance_view_loss(th.tensor([0.0]), False, 1.0).item() == 0.0
    assert L.instance_view_loss(th.tensor([2.0]), True, 1.0).item() == 0.0
    assert L.instance_view_loss(th.tensor([0.0]), True, 1.0).item() == 1.0
    assert L.cross_view_loss(th.tensor([0.0]), False, 0.1).item() == 0.0
    assert L.cross_view_loss(th.tensor([20.0]), True, 0.1).item() == 0.0
    assert L.cross_view_loss(th.tensor([0.0]), True, 0.1).item() == pytest.approx(0.1)


def _module(**kw):
    th.manual_seed(0)
    return Concept2BoxModule(4, 2, nb_inds=3, embed_dim=6, **kw)


def test_module_declares_what_it_implements():
    m = _module()
    assert m.neg_capable_gcis == {"gci0", "gci2", "class_assertion",
                                  "object_property_assertion"}
    assert not m.role_axiom_capable
    for name in ("gci1", "gci3", "gci0_bot", "gci1_bot", "gci3_bot"):
        with pytest.raises(NotImplementedError):
            m(th.zeros(1, 3, dtype=th.long), name)


def test_module_losses_have_one_value_per_row_and_are_nonnegative():
    m = _module()
    gci0 = th.tensor([[0, 1], [1, 2]])
    gci2 = th.tensor([[0, 0, 1], [2, 1, 3]])
    ca = th.tensor([[0, 1], [2, 3]])
    opa = th.tensor([[0, 1, 2], [1, 0, 2]])
    for data, name in ((gci0, "gci0"), (gci2, "gci2"), (ca, "class_assertion"),
                       (opa, "object_property_assertion")):
        for neg in (False, True):
            out = m(data, name, neg=neg)
            assert out.shape == (2,) and th.isfinite(out).all() and (out >= 0).all()


def test_training_reduces_the_loss_of_a_subsumption():
    """Gradient steps on C ⊑ D move the box of C into the box of D."""
    m = _module(beta=0.2)
    data = th.tensor([[0, 1], [1, 2]])
    opt = th.optim.Adam(m.parameters(), lr=0.05)
    before = m(data, "gci0").mean().item()
    for _ in range(150):
        opt.zero_grad()
        loss = m(data, "gci0").mean() + m(data[:, [1, 0]], "gci0", neg=True).mean()
        loss.backward()
        opt.step()
    assert m(data, "gci0").mean().item() < before / 2
    # the converse is not entailed: it keeps a high loss
    assert m(data[:, [1, 0]], "gci0").mean().item() > m(data, "gci0").mean().item()
