import torch as th

from mowl.nn.el.prism import losses as L
from mowl.nn.el.prism.module import PrismModule

NEG_ROW = th.tensor([[0, 0, 1]])


def _regions(c0, r0, c1, r1, dim=3):
    """Two prisms with triangle centres/inradii and identical unit boxes."""
    def one(c, r):
        h = L.supports(th.tensor([c], dtype=th.float32), th.tensor([[r]], dtype=th.float32))
        return L.PrismRegions(h=h, cen=th.zeros(1, dim - 2), w=th.ones(1, dim - 2))
    return one(c0, r0), one(c1, r1)


def _neg(c, d, mode, rel=None):
    rel = th.zeros(1, 3) if rel is None else rel
    return L.gci2_score_neg_prisms(c, d, rel, 0.0, triangle_negative=mode)


def test_small_triangle_in_the_corner_of_a_large_one_is_not_penalised():
    # the apex-up triangle of inradius 1 around the origin has its top corner at (0, 2);
    # a tiny triangle at (0, 1.8) lies inside it, so the two overlap.
    big, small = _regions([0.0, 0.0], 1.0, [0.0, 1.8], 0.05)
    assert float(_neg(small, big, "overlap")) == 0.0
    assert float(_neg(big, small, "overlap")) == 0.0
    # the earlier per-direction test calls this pair separated
    assert float(_neg(big, small, "inradius")) > 0.5


def test_disjoint_triangles_are_penalised_and_the_old_option_gives_the_old_value():
    a, b = _regions([0.0, 0.0], 0.5, [5.0, 5.0], 0.5)
    assert float(_neg(a, b, "overlap")) > 1.0
    hc, hd = a.h, b.h
    rc = th.tensor([[0.5]])
    old = L.gci2_score_neg(hc, rc, a.cen, a.w, hd, b.cen, b.w, th.zeros(1, 3), 0.0,
                           triangle_negative="inradius")
    tri = th.relu(hc - 2.0 * rc - hd)
    box = th.relu(th.abs(a.cen - b.cen) - a.w.abs() - b.w.abs())
    assert th.allclose(old, th.linalg.norm(th.cat([tri, box], dim=-1), dim=-1))


def test_overlapping_triangles_zero_in_both_halves():
    a, b = _regions([0.0, 0.0], 0.5, [0.1, 0.0], 0.5)
    assert float(_neg(a, b, "overlap")) == 0.0


def test_module_option_reaches_the_losses():
    th.manual_seed(0)
    new = PrismModule(4, 2, 2, embed_dim=4)
    old = PrismModule(4, 2, 2, embed_dim=4, gci2_negative_triangle="inradius")
    old.load_state_dict(new.state_dict())
    data = th.tensor([[0, 1, 2], [3, 0, 1]])
    assert new.gci2_negative_triangle == "overlap"
    assert not th.allclose(new(data, "gci2", neg=True), old(data, "gci2", neg=True))
    # positive losses do not depend on the option
    assert th.allclose(new(data, "gci2"), old(data, "gci2"))
