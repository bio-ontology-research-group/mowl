"""Behaviours that are the published method's formulas and are kept (documented on the cards):
EIKE's printed SubClassOf score and Concept2Box's direction of gci2 and inside weight."""
import torch as th

from mowl.nn.el.concept2box import losses as L
from mowl.nn.el.concept2box.module import Concept2BoxModule
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


def test_concept2box_gci2_is_the_triple_c_r_d_and_the_inside_weight_is_kept():
    th.manual_seed(0)
    m = Concept2BoxModule(4, 2, nb_inds=2, embed_dim=3, gci2_reading="triple")
    data = th.tensor([[0, 1, 2]])
    phi = m.triple_probability(data[:, 0], data[:, 1], data[:, 2])
    assert th.allclose(m.gci2_loss(data), (phi - 1) ** 2)
    # the default reads the axiom: the box of C inside the box of D (tail C, head D)
    th.manual_seed(0)
    d = Concept2BoxModule(4, 2, nb_inds=2, embed_dim=3)
    phi = d.triple_probability(data[:, 2], data[:, 1], data[:, 0])
    assert th.allclose(d.gci2_loss(data), (phi - 1) ** 2)
    # a point inside the box (off the centre) has a positive distance: alpha * inside part
    center, offset = m.box(th.tensor([0]), th.tensor([m.null_id]))
    inside = center + 0.5 * offset
    assert float(L.vector_to_box_distance(inside, center, offset, 0.5, 1.0)) > 0
