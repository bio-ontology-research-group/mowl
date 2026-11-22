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
            assert out.shape == (2, 1) and th.isfinite(out).all() and (out >= 0).all()


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


def test_gumbel_volume_matches_the_exact_bessel_expression():
    """Dasgupta et al. (2020), Proposition 1: the expected side length is
    2 beta K0(2 exp(-x / (2 beta))) with x = mu_M - mu_m; Equation 2 of Concept2Box (and
    Appendix of Dasgupta et al.) approximates it by beta log(1 + exp(x / beta - 2 gamma)),
    with an error below 0.0617 beta for x / beta in [-100, 100]."""
    scipy_special = pytest.importorskip("scipy.special")
    for beta in (0.1, 1.0, 2.5):
        x = th.linspace(-20 * beta, 20 * beta, 81, dtype=th.float64)
        exact = 2 * beta * th.from_numpy(
            scipy_special.kv(0, (2 * th.exp(-x / (2 * beta))).numpy()))
        approx = th.exp(L.gumbel_log_volume(th.zeros_like(x)[:, None], x[:, None], beta))
        assert (approx - exact).abs().max().item() < 0.0617 * beta


def test_gumbel_intersection_volume_against_monte_carlo():
    """Lemma 1 of Dasgupta et al. (2020): the min corner of an intersection is
    MaxGumbel(beta ln(e^(mu1/beta) + e^(mu2/beta))) and the max corner is
    MinGumbel(-beta ln(e^(-mu1/beta) + e^(-mu2/beta))). The expected side of the
    intersection of two sampled boxes is compared with the closed form."""
    scipy_special = pytest.importorskip("scipy.special")
    beta = 0.7
    gen = th.Generator().manual_seed(0)
    n = 400_000
    gumbel = -th.log(-th.log(th.rand(n, generator=gen, dtype=th.float64)))   # standard Gumbel
    gumbel2 = -th.log(-th.log(th.rand(n, generator=gen, dtype=th.float64)))
    gumbel3 = -th.log(-th.log(th.rand(n, generator=gen, dtype=th.float64)))
    gumbel4 = -th.log(-th.log(th.rand(n, generator=gen, dtype=th.float64)))
    (lo1, hi1), (lo2, hi2) = (0.0, 2.0), (1.0, 4.0)
    # corners: min corner ~ MaxGumbel(mu_m), max corner ~ MinGumbel(mu_M)
    x1, x2 = lo1 + beta * gumbel, lo2 + beta * gumbel2
    y1, y2 = hi1 - beta * gumbel3, hi2 - beta * gumbel4
    sampled = th.relu(th.minimum(y1, y2) - th.maximum(x1, x2)).mean().item()
    mu_lo = beta * th.logaddexp(th.tensor(lo1 / beta), th.tensor(lo2 / beta)).item()
    mu_hi = -beta * th.logaddexp(th.tensor(-hi1 / beta), th.tensor(-hi2 / beta)).item()
    arg = 2 * float(th.exp(th.tensor(-(mu_hi - mu_lo) / (2 * beta))))
    exact = 2 * beta * scipy_special.kv(0, arg)
    # the module's intersection and softplus volume
    lo_t, hi_t = L.gumbel_intersection(th.tensor([lo1]), th.tensor([hi1]),
                                       th.tensor([lo2]), th.tensor([hi2]), beta)
    assert lo_t.item() == pytest.approx(mu_lo, abs=1e-5)
    assert hi_t.item() == pytest.approx(mu_hi, abs=1e-5)
    softplus = th.exp(L.gumbel_log_volume(lo_t, hi_t, beta)).item()
    assert sampled == pytest.approx(exact, abs=0.02)
    assert softplus == pytest.approx(sampled, abs=0.0617 * beta + 0.02)


def _gci2_inside_phi(m, c, r, d):
    """phi of the box of C inside the box of D in the context of r (hand-built)."""
    cc = th.tensor([c]), th.tensor([r])
    dd = th.tensor([d]), th.tensor([r])
    lo_c, hi_c = m._corners(*cc)
    lo_d, hi_d = m._corners(*dd)
    return L.conditional_probability(lo_d, hi_d, lo_c, hi_c, m.beta)   # tail C inside head D


def test_gci2_reads_the_axiom_by_default_and_the_paper_triple_on_request():
    """C ⊑ ∃r.D: the box of C inside the box of D (as for gci0); the paper's triple
    (C, r, D) is the reverse and stays an option."""
    row = th.tensor([[0, 1, 2]])
    default = _module()
    assert default.gci2_reading == "axiom"
    expected = (_gci2_inside_phi(default, 0, 1, 2) - 1) ** 2
    assert default(row, "gci2").item() == pytest.approx(expected.item(), rel=1e-5)
    old = _module(gci2_reading="triple")
    reverse = (_gci2_inside_phi(old, 2, 1, 0) - 1) ** 2
    assert old(row, "gci2").item() == pytest.approx(reverse.item(), rel=1e-5)
    assert default(row, "gci2").item() != pytest.approx(old(row, "gci2").item())
    # same parameters: the two readings differ exactly by swapping C and D
    assert old(row[:, [2, 1, 0]], "gci2").item() == pytest.approx(
        default(row, "gci2").item(), rel=1e-5)
    with pytest.raises(ValueError):
        _module(gci2_reading="other")


def test_training_gci2_puts_c_inside_d_in_the_context_of_r():
    m = _module(beta=0.2)
    data = th.tensor([[0, 1, 2]])
    opt = th.optim.Adam(m.parameters(), lr=0.05)
    before = _gci2_inside_phi(m, 0, 1, 2).item()
    for _ in range(150):
        opt.zero_grad()
        m(data, "gci2").mean().backward()
        opt.step()
    assert _gci2_inside_phi(m, 0, 1, 2).item() > max(before, 0.9)


def test_cross_view_cost_of_a_point_inside_a_box_is_not_zero():
    """Published (Equation 8): the inside distance to the centre is weighted by alpha(c)."""
    m = _module(alpha=0.5)
    cls = th.tensor([1])
    center, offset = m.box(cls, th.full_like(cls, m.null_id))
    m.ind_embed.weight.data[0] = center[0].detach()      # a point at the centre: zero
    m.ind_embed.weight.data[1] = (center[0] + 0.5 * offset[0]).detach()   # inside, off centre
    at_centre = m.cross_distance(th.tensor([0]), cls).item()
    inside = m.cross_distance(th.tensor([1]), cls).item()
    assert at_centre == pytest.approx(0.0, abs=1e-6) and inside > 0.0
