import pytest
import torch as th
import torch.nn.functional as F

import mowl.nn.el.boxel.losses as L
from mowl.nn.el.boxel.module import BoxELModule
from mowl.nn.el.boxsel.module import BoxSELModule
from mowl.nn.el.boxsel import losses as S


def _module(cls=BoxELModule, **kw):
    th.manual_seed(1)
    m = cls(5, 3, embed_dim=4, **kw)
    with th.no_grad():
        m.scaling_embedding.weight.copy_(th.tensor([[1.0] * 4, [-0.8, 0.9, 1.1, -1.2],
                                                    [0.5] * 4]))
    return m


GCI2 = th.tensor([[0, 1, 2], [3, 1, 4]])
GCI3 = th.tensor([[1, 0, 2], [1, 3, 4]])
CHAIN = th.tensor([[0, 1, 2], [1, 2, 0]])


def test_negative_scale_does_not_invert_boxes_by_default():
    m = _module()
    abs_scale = _module()
    with th.no_grad():
        abs_scale.scaling_embedding.weight.abs_()
    for name, data in (("gci2", GCI2), ("gci3", GCI3)):
        assert th.allclose(m(data, name), abs_scale(data, name)), name


def test_signed_scale_option_gives_the_old_numbers():
    m = _module(positive_scale=False)
    assert m.positive_scale is False
    data = GCI2
    c_min, c_delta = m.min_embedding(data[:, 0]), m.delta_embedding(data[:, 0])
    d_min, d_delta = m.min_embedding(data[:, 2]), m.delta_embedding(data[:, 2])
    rel, sc = m.relation_embedding(data[:, 1]), m.scaling_embedding(data[:, 1])
    c = L.Box(c_min, delta_embed=c_delta)
    t = L.Box(c.min_embed * (sc + L.eps) + rel, max_embed=c.max_embed * (sc + L.eps) + rel)
    expected = L.inclusion_loss(t, L.Box(d_min, delta_embed=d_delta), m.temperature)
    assert th.allclose(m(data, "gci2").reshape(-1), expected)
    assert not th.allclose(m(data, "gci2"), _module()(data, "gci2"))


def test_positive_scales_are_unchanged():
    new = _module()
    old = _module(positive_scale=False)
    with th.no_grad():
        for mod in (new, old):
            mod.scaling_embedding.weight.fill_(1.05)
    for name, data in (("gci2", GCI2), ("gci3", GCI3)):
        assert th.allclose(new(data, name), old(data, name))


def test_role_losses_are_exposed():
    m = _module()
    assert m.role_axiom_capable
    inc = m(th.tensor([[0, 2], [1, 2]]), "role_inclusion")
    chain = m(CHAIN, "role_chain")
    assert inc.shape == (2, 1) and chain.shape == (2, 1)
    assert th.isfinite(inc).all() and th.isfinite(chain).all()


def test_role_chain_composes_the_maps_in_order():
    m = _module()
    t = m.relation_embedding(CHAIN[:, :3].T[0]), m.relation_embedding(CHAIN[:, 1]), \
        m.relation_embedding(CHAIN[:, 2])
    f = [m.scaling_embedding(CHAIN[:, i]).abs() + L.eps for i in range(3)]
    expected_1 = th.norm(f[1] * t[0] + t[1] - t[2], dim=1, keepdim=True)
    expected_2 = th.norm(F.relu(f[0] * f[1] / f[2] - 1), dim=1, keepdim=True)
    assert th.allclose(m(CHAIN, "role_chain"), expected_1 + expected_2, atol=1e-6)


def test_role_chain_as_released_gives_the_old_expression():
    m = _module(positive_scale=False, role_chain_composition="as_released")
    t = [m.relation_embedding(CHAIN[:, i]) for i in range(3)]
    f = [m.scaling_embedding(CHAIN[:, i]) for i in range(3)]
    first = th.norm(f[0] * t[0] + t[1] - t[2], dim=1, keepdim=True)
    second = th.norm(F.relu(f[0] * f[1] / (f[2] + L.eps) - 1), dim=1, keepdim=True)
    old = first + second
    assert th.allclose(m(CHAIN, "role_chain"), old, atol=1e-6)
    assert not th.allclose(m(CHAIN, "role_chain"),
                           _module(positive_scale=False)(CHAIN, "role_chain"))
    with pytest.raises(ValueError):
        L.role_chain_loss(CHAIN, m.relation_embedding, m.scaling_embedding, composition="x")


def test_gci1_bot_is_the_volume_product_ratio_not_scale_invariant():
    # Published code reading (kept): Vol(C ∩ D) / (Vol(C) Vol(D)).
    m = _module(temperature=1.0)
    data = th.tensor([[0, 1], [2, 3]])
    c = L.Box(m.min_embedding(data[:, 0]), delta_embed=m.delta_embedding(data[:, 0]))
    d = L.Box(m.min_embedding(data[:, 1]), delta_embed=m.delta_embedding(data[:, 1]))
    vol = lambda b: th.clamp(L.volumes(b, 1.0), 1e-10, 1e4)  # noqa: E731
    expected = vol(L.intersection(c, d)) / (vol(c) * vol(d))
    assert th.allclose(m(data, "gci1_bot").squeeze(), expected, rtol=1e-4)


def test_boxsel_follows_the_scale_option():
    leaves = th.tensor([[1, 0]])  # role 1, class 0
    struct = ("some", ("atom",))
    for positive in (True, False):
        m = _module(BoxSELModule, positive_scale=positive)
        assert m.positive_scale is positive
        lo, hi, _ = S.expression_box(struct, leaves, 0, m.min_embedding,
                                     m.delta_embedding, m.relation_embedding,
                                     m.scaling_embedding, positive)
        assert th.allclose(m.box(struct, leaves)[0], lo)
        if positive:
            assert (hi >= lo).all()
        else:
            assert not (hi >= lo).all()
