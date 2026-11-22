"""Behaviours that are the published method's formulas and are kept (documented on the cards):
EIKE's printed SubClassOf score. (Concept2Box's direction of gci2 and inside weight, read on the
direct module, is in ``tests/baselines/test_concept2box_published_reading.py``.)"""
import torch as th

from mowl.nn.el.eike.module import EIKEModule


def test_eike_printed_subclass_score_is_not_containment():
    th.manual_seed(0)
    m = EIKEModule(2, 1, 1, embed_dim=2, alpha=0.0)
    with th.no_grad():
        # child centre 1.4 / radius 1, parent centre 0 / radius 2, all axes 1
        m.class_center.weight.copy_(th.tensor([[1.4, 0.0], [0.0, 0.0]]))
        m.class_axes_raw.weight.fill_(0.5413)
        raw = th.log(th.expm1(th.tensor([1.0, 2.0])))
        m.class_radius_raw.weight.copy_(raw.reshape(2, 1))
    # 1.4^2 + 1 - 4 < 0: zero cost, although 1.4 + 1 > 2, so the child ball sticks out
    assert float(m.gci0_loss(th.tensor([[0, 1]]))) < 1e-4


def test_eike_intensional_length_term_is_signed():
    th.manual_seed(0)
    m = EIKEModule(2, 1, 1, embed_dim=3, alpha=1.0)
    with th.no_grad():
        m.class_intensional.weight.copy_(th.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
        m.class_center.weight.zero_()
    # child shorter than parent: 1 - cos + (|c_i| - |c_j|) = 0 + (1 - 3) < 0
    assert float(m.gci0_loss(th.tensor([[0, 1]]))) < 0
