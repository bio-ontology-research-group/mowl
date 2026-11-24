"""Poincaré-ball operations and the HiT / OnT losses on plain tensors (no language model
needed)."""

import math

import pytest
import torch as th

import mowl.datasets  # noqa: F401  (import order: datasets before nn)
from mowl.nn.lm import hyperbolic as hb
from mowl.nn.lm.hit import (HiTModule, centripetal_loss, clustering_loss, hierarchy_loss,
                            subsumption_score)
from mowl.nn.lm.ont import (OnTModule, RoleTransformation, conjunction_loss,
                            existential_loss)
from mowl.nn.lm.encoders import BagOfWordsEncoder, as_encoder

C = 0.25  # ball of radius 2


def _points(n=20, d=6, radius=1.9, seed=0):
    g = th.Generator().manual_seed(seed)
    x = th.randn(n, d, generator=g, dtype=th.float64)
    r = th.rand(n, 1, generator=g, dtype=th.float64) * radius
    return x / x.norm(dim=1, keepdim=True) * r


# --------------------------------------------------------------------------- geometry

def test_curvature_for_dim():
    assert hb.curvature_for_dim(384) == pytest.approx(1 / 384)
    with pytest.raises(ValueError):
        hb.curvature_for_dim(0)


def test_mobius_addition_identities():
    x, y = _points(), _points(seed=1)
    zero = th.zeros_like(x)
    assert th.allclose(hb.mobius_add(x, zero, C), x)
    assert th.allclose(hb.mobius_add(zero, y, C), y)
    assert th.allclose(hb.mobius_add(-x, x, C), zero, atol=1e-12)
    # left cancellation: -x ⊕ (x ⊕ y) = y
    assert th.allclose(hb.mobius_add(-x, hb.mobius_add(x, y, C), C), y, atol=1e-10)


def test_mobius_addition_is_euclidean_for_small_curvature():
    x, y = _points() * 1e-3, _points(seed=1) * 1e-3
    assert th.allclose(hb.mobius_add(x, y, 1e-9), x + y, atol=1e-12)


def test_distance_closed_forms():
    # c = 1, one dimension: d(0, t) = 2 artanh(t)
    t = th.tensor([[0.5]], dtype=th.float64)
    assert float(hb.distance(th.zeros_like(t), t, 1.0)) == pytest.approx(2 * math.atanh(0.5))
    assert float(hb.hyperbolic_norm(t, 1.0)) == pytest.approx(2 * math.atanh(0.5))
    # c = 1/4: norm 2/sqrt(c) artanh(sqrt(c)|y|)
    y = th.tensor([[0.6, 0.8]], dtype=th.float64)  # |y| = 1
    assert float(hb.hyperbolic_norm(y, C)) == pytest.approx(4 * math.atanh(0.5))
    # symmetric points on a line through the origin: d(-a, a) = 2 d(0, a)
    a = th.tensor([[0.3, 0.0]], dtype=th.float64)
    assert float(hb.distance(-a, a, 1.0)) == pytest.approx(4 * math.atanh(0.3))


def test_distance_properties():
    x, y, z = _points(), _points(seed=1), _points(seed=2)
    dxy = hb.distance(x, y, C)
    assert th.allclose(dxy, hb.distance(y, x, C))                      # symmetry
    assert th.allclose(hb.distance(x, x, C), th.zeros(len(x), dtype=x.dtype), atol=1e-6)
    assert (dxy > 0).all()
    assert (hb.distance(x, z, C) <= dxy + hb.distance(y, z, C) + 1e-9).all()  # triangle
    # Eq. 1 of HiT (Möbius form) equals Eq. 2 of OnT (arcosh form)
    assert th.allclose(dxy, hb.distance_arcosh(x, y, C), atol=1e-8)


def test_pairwise_distance_matches_distance():
    x, y = _points(7), _points(5, seed=3)
    pw = hb.pairwise_distance(x, y, C)
    full = hb.distance(x.unsqueeze(1), y.unsqueeze(0), C)
    assert pw.shape == (7, 5)
    assert th.allclose(pw, full, atol=1e-9)


def test_exp_log_maps_are_inverse_and_stay_in_ball():
    x, y = _points(), _points(seed=1)
    assert th.allclose(hb.expmap0(hb.logmap0(y, C), C), y, atol=1e-10)
    u = th.randn(20, 6, dtype=th.float64)
    assert th.allclose(hb.logmap0(hb.expmap0(u, C), C), u, atol=1e-8)
    assert th.allclose(hb.expmap(x, hb.logmap(x, y, C), C), y, atol=1e-8)
    big = 100 * th.randn(20, 6, dtype=th.float64)
    for out in (hb.expmap0(big, C), hb.expmap(x, big, C)):
        assert (out.norm(dim=1) <= 1 / math.sqrt(C) + 1e-9).all()
    # distance from x to exp_x(u) is the Riemannian length lambda_x |u|
    u = 0.1 * th.randn(20, 6, dtype=th.float64)
    lam = hb.conformal_factor(x, C).squeeze(1)
    assert th.allclose(hb.distance(x, hb.expmap(x, u, C), C), lam * u.norm(dim=1), atol=1e-8)


def test_project_and_artanh_clipping():
    x = th.tensor([[3.0, 4.0], [0.1, 0.2]])  # first point outside the ball of radius 2
    p = hb.project(x, C)
    assert float(p[0].norm()) == pytest.approx(2 * (1 - 4e-3), rel=1e-6)
    assert th.equal(p[1], x[1])
    # points outside the ball give large but finite distances (saturated artanh)
    d = hb.distance(x[:1], th.zeros(1, 2), C)
    assert th.isfinite(d).all()


def test_mobius_scalar_multiplication():
    x = _points()
    assert th.allclose(hb.mobius_scalar_mul(1.0, x, C), x, atol=1e-10)
    assert th.allclose(hb.mobius_scalar_mul(2.0, x, C), hb.mobius_add(x, x, C), atol=1e-9)
    # the hyperbolic norm scales linearly
    assert th.allclose(hb.hyperbolic_norm(hb.mobius_scalar_mul(0.5, x, C), C),
                       0.5 * hb.hyperbolic_norm(x, C), atol=1e-8)


@pytest.mark.parametrize("pairing", ["half", "adjacent"])
def test_rotation_is_an_isometry(pairing):
    x, y = _points(), _points(seed=1)
    theta = th.rand(20, 3, dtype=th.float64) * 6
    rx, ry = hb.rotate(x, theta, pairing), hb.rotate(y, theta, pairing)
    assert th.allclose(rx.norm(dim=1), x.norm(dim=1))
    assert th.allclose(hb.distance(rx, ry, C), hb.distance(x, y, C), atol=1e-9)
    assert th.allclose(hb.rotate(rx, -theta, pairing), x, atol=1e-12)


def test_rotation_pairings_differ_by_a_permutation():
    x = _points(4)
    theta = th.rand(4, 3, dtype=th.float64)
    perm = [0, 2, 4, 1, 3, 5]   # puts the adjacent pairs (2i, 2i+1) at (i, i+3)
    inv = [perm.index(i) for i in range(6)]
    half = hb.rotate(x[:, perm], theta, "half")
    assert th.allclose(half[:, inv], hb.rotate(x, theta, "adjacent"))


def test_rotation_by_quarter_turn():
    x = th.tensor([[1.0, 0.0]])
    assert th.allclose(hb.rotate(x, th.tensor([[math.pi / 2]])), th.tensor([[0.0, 1.0]]),
                       atol=1e-7)


def test_proposition_1_rotation_invariance():
    """OnT, Proposition 1: rotations preserve norms and distances, so with k_r = 1 the
    score of role images equals the score of the fillers."""
    x, y = _points(), _points(seed=1)
    theta = th.rand(1, 3, dtype=th.float64) * 6
    for lam in (0.0, 0.3, 1.0):
        s = subsumption_score(x, y, C, lam)
        s_rot = subsumption_score(hb.rotate(x, theta), hb.rotate(y, theta), C, lam)
        assert th.allclose(s, s_rot, atol=1e-9)


# --------------------------------------------------------------------------- losses

def _line(*norms):
    """Points on the first axis with the given Euclidean norms (ball of radius 2)."""
    return th.tensor([[n, 0.0] for n in norms], dtype=th.float64)


def test_clustering_loss_semantics():
    child = _line(1.0)
    parent = _line(0.9)
    far = th.tensor([[-1.5, 0.0]], dtype=th.float64)
    d_pos = hb.distance(child, parent, C)
    d_neg = hb.distance(child, far, C)
    margin = float(d_neg - d_pos) - 0.01
    assert float(clustering_loss(child, parent, far, C, margin)) == 0.0
    loss = clustering_loss(child, far, parent, C, 0.5)   # swapped: positive is far
    assert float(loss) == pytest.approx(float(d_neg - d_pos) + 0.5)


def test_centripetal_loss_semantics():
    child, parent = _line(1.5), _line(0.5)
    # parent closer to the origin than the child by more than the margin: zero loss
    gap = float(hb.hyperbolic_norm(child, C) - hb.hyperbolic_norm(parent, C))
    assert float(centripetal_loss(child, parent, C, gap - 0.01)) == 0.0
    assert float(centripetal_loss(child, parent, C, gap + 0.2)) == pytest.approx(0.2)
    # parent further out than the child: positive loss
    assert float(centripetal_loss(parent, child, C, 0.1)) == pytest.approx(gap + 0.1)


def test_hierarchy_loss_and_score():
    child, parent, neg = _line(1.5), _line(1.2), th.tensor([[-1.5, 0.0]], dtype=th.float64)
    total = hierarchy_loss(child, parent, neg, C, 1.0, 0.1, 2.0, 3.0)
    expected = 2 * clustering_loss(child, parent, neg, C, 1.0) + \
        3 * centripetal_loss(child, parent, C, 0.1)
    assert th.allclose(total, expected)
    # Eq. 6: score of the true direction is higher once lambda > 0
    s_true = subsumption_score(child, parent, C, 1.0)
    s_rev = subsumption_score(parent, child, C, 1.0)
    assert float(s_true) > float(s_rev)
    assert float(subsumption_score(child, parent, C, 0.0)) == pytest.approx(
        -float(hb.distance(child, parent, C)))


def test_existential_loss_has_floor_beta():
    """L_r averages two opposite centripetal terms, so it is at least beta and equals beta
    exactly when x_exists and f_r(x_D) coincide and negatives are far enough."""
    x = _line(1.0)
    neg = th.tensor([[-1.8, 0.0]], dtype=th.float64)
    beta = 0.5
    val = existential_loss(x, x.clone(), neg, neg, C, 1.0, beta)
    assert float(val) == pytest.approx(beta)
    other = _line(1.4)
    assert float(existential_loss(x, other, neg, neg, C, 1.0, beta)) > beta


def test_conjunction_loss_is_eq7_for_two_conjuncts():
    x, a, b = _line(1.5), _line(1.0), th.tensor([[0.0, 1.0]], dtype=th.float64)
    n1 = th.tensor([[-1.5, 0.0]], dtype=th.float64)
    n2 = th.tensor([[0.0, -1.5]], dtype=th.float64)
    val = conjunction_loss(x, [a, b], [n1, n2], C, 2.0, 0.3)
    l1, l2 = hierarchy_loss(x, a, n1, C, 2.0, 0.3), hierarchy_loss(x, b, n2, C, 2.0, 0.3)
    expected = 0.5 * (l1 + l2)
    assert th.allclose(val, expected)


def test_role_transformation():
    th.manual_seed(0)
    role = RoleTransformation(6)
    x = _points(5, 6).float()
    r = th.randn(5, 6)
    y = role(x, r, C)
    assert th.allclose(y.norm(dim=1), x.norm(dim=1), atol=1e-6)   # rotation only
    scaling, angles = role.parameters_of(r)
    assert (angles.abs() <= math.pi).all() and (scaling > 0).all()
    scaled = RoleTransformation(6, use_scaling=True)
    scaled.load_state_dict(role.state_dict())
    z = scaled(x, r, C)
    assert th.allclose(hb.hyperbolic_norm(z, C), scaling * hb.hyperbolic_norm(x, C),
                       atol=1e-4)
    with pytest.raises(ValueError):
        RoleTransformation(5)


# --------------------------------------------------------------------------- modules

def test_bag_of_words_encoder():
    enc = BagOfWordsEncoder(dim=8, num_buckets=64)
    out = enc(["Heart", "heart", "left ventricle", ""])
    assert out.shape == (4, 8)
    assert th.allclose(out[0], out[1])
    assert (out.abs() <= 1).all()
    assert as_encoder(enc) is enc
    with pytest.raises(TypeError):
        as_encoder(42)


def test_modules_backpropagate_into_encoder_and_roles():
    th.manual_seed(0)
    enc = BagOfWordsEncoder(dim=8, num_buckets=64)
    hit = HiTModule(enc)
    assert hit.c == pytest.approx(1 / 8)
    loss = hit(["dog", "cat"], ["animal", "animal"], ["car", "rock"]).mean()
    loss.backward()
    assert enc.embedding.weight.grad is not None
    ont = OnTModule(BagOfWordsEncoder(dim=8, num_buckets=64))
    neg = ont.embed(["rock", "car"])
    l_r = ont.existential_loss(["has part some wheel", "has part some leg"],
                               ["has part", "has part"], ["wheel", "leg"], neg, neg)
    l_c = ont.conjunction_loss(["dog and pet", "cat and pet"], [["dog", "cat"], ["pet", "pet"]],
                               [neg, neg])
    (l_r.mean() + l_c.mean()).backward()
    assert ont.role_model.linear.weight.grad is not None
    assert l_r.shape == (2,) and l_c.shape == (2,)
