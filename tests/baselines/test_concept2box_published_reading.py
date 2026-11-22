"""Concept2Box's direction of gci2 and inside weight, as the direct module has them (documented
on the card): the formulas that the combination of parts reproduces."""
import torch as th

from mowl.baselines.nn.el.concept2box import losses as L
from mowl.baselines.nn.el.concept2box.module import Concept2BoxModule


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
