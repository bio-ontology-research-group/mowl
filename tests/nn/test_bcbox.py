"""Tests for the binary code box module and losses (Zhang et al., NeurIPS 2022)."""
import math

import mowl
mowl.init_jvm("10g")
import mowl.datasets  # noqa: F401,E402  (import order: avoids a circular import in mowl.nn)
import pytest  # noqa: E402
import torch as th  # noqa: E402

from mowl.nn.graph.bcbox.losses import (  # noqa: E402
    bce_negative_sampling_loss,
    gumbel_box_log_score,
    lse,
    neg_log1mexp,
)
from mowl.nn.graph.bcbox.module import BinaryCodeBoxModule  # noqa: E402
from mowl.models.bcbox.model import best_f1  # noqa: E402

INF = float("inf")
CYCLE3 = [(0, 1), (1, 2), (2, 0)]


def _all_pairs(n):
    return th.tensor([(a, b) for a in range(n) for b in range(n) if a != b])


def _adjacency(pairs, edges):
    edges = set(edges)
    return th.tensor([tuple(p) in edges for p in pairs.tolist()])


def _hard(variant, n, d, **kw):
    """Module with tau = nu = 0, i.e. the hard box score F_Box."""
    return BinaryCodeBoxModule(n, d, variant=variant, intersection_temp=0.0,
                               volume_temp=0.0, **kw)


def _fit(module, pos, neg, epochs, lr=0.05, code_reg=0.0, seed=0):
    th.manual_seed(seed)
    opt = th.optim.Adam(module.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        neg_e = module(neg).unsqueeze(0).expand(len(pos), -1)
        loss = bce_negative_sampling_loss(module(pos), neg_e, self_adversarial=False).mean()
        if code_reg:
            loss = loss + code_reg * module.code_regularization()
        loss.backward()
        opt.step()
    return loss.item()


# ---------------------------------------------------------------- energies
def test_lse_zero_temperature_is_max_and_soft_bounds():
    a, b = th.tensor([0.3, -1.0]), th.tensor([0.1, 2.0])
    assert th.equal(lse(a, b, 0.0), th.maximum(a, b))
    soft = lse(a, b, 0.5)
    assert (soft >= th.maximum(a, b)).all()
    assert (soft <= th.maximum(a, b) + 0.5 * math.log(2) + 1e-6).all()


def test_hard_box_energy_on_hand_built_boxes():
    m = _hard("gbox", 2, 2)
    # u = [0,2]x[0,2], v = [1,3]x[0,1]: |u ∩ v| = 1, |v| = 2, |u| = 4
    m.set_boxes([[0., 0.], [1., 0.]], [[2., 2.], [3., 1.]])
    u, v = th.tensor([0]), th.tensor([1])
    assert m.energy(u, v).item() == pytest.approx(math.log(2))
    assert m.energy(v, u).item() == pytest.approx(math.log(4))
    # containment: v inside u gives energy 0
    m.set_boxes([[0., 0.], [0.5, 0.5]], [[2., 2.], [1., 1.]])
    assert m.energy(u, v).item() == pytest.approx(0.0)
    assert m.energy(v, u).item() == pytest.approx(math.log(16))


def test_gumbel_box_score_matches_closed_form():
    tau, nu = 0.1, 0.7
    xm, xp, ym, yp = 0.0, 2.0, 1.0, 3.0

    def lse_(t, *z):
        return t * math.log(sum(math.exp(zi / t) for zi in z))

    num = lse_(nu, -lse_(tau, -xp, -yp) - lse_(tau, xm, ym), 0.0)
    den = lse_(nu, yp - ym, 0.0)
    got = gumbel_box_log_score(*(th.tensor([z]) for z in (xm, xp, ym, yp)), tau, nu)
    assert got.item() == pytest.approx(math.log(num / den), rel=1e-5)
    assert got.item() <= 0.0


def test_gumbel_score_is_at_most_one_on_random_boxes():
    th.manual_seed(0)
    m = BinaryCodeBoxModule(20, 5, variant="gbox", intersection_temp=0.05, volume_temp=0.5)
    pairs = _all_pairs(20)
    assert (m.neg_log_scores(pairs[:, 0], pairs[:, 1]) >= -1e-6).all()


def test_gbc_energy_is_min_over_codes_of_weighted_sum():
    th.manual_seed(1)
    m = BinaryCodeBoxModule(5, 4, variant="gbc", num_codes=3)
    pairs = _all_pairs(5)
    nls = m.neg_log_scores(pairs[:, 0], pairs[:, 1])
    B = m.global_codes()
    expected = th.stack([(nls * B[:, j]).sum(-1) for j in range(3)], -1).min(-1).values
    assert th.allclose(m(pairs), expected)


def test_vbc_energy_is_code_product_weighted_sum():
    th.manual_seed(2)
    m = BinaryCodeBoxModule(5, 4, variant="vbc")
    pairs = _all_pairs(5)
    nls = m.neg_log_scores(pairs[:, 0], pairs[:, 1])
    cu, cv = m.vertex_codes(pairs[:, 0]), m.vertex_codes(pairs[:, 1])
    assert th.allclose(m(pairs), (cu * cv * nls).sum(-1))


def test_all_codes_one_reduces_to_gbox():
    """Section 4.2 ('Generalization'): with all codes 1 both models equal G-Box."""
    th.manual_seed(3)
    pairs = _all_pairs(6)
    for variant in ("gbc", "vbc"):
        m = BinaryCodeBoxModule(6, 3, variant=variant)
        g = BinaryCodeBoxModule(6, 3, variant="gbox")
        g.set_boxes(m.box_min, m.box_max)
        m.set_code_logits(th.full_like(m.code_logits, INF))
        assert th.allclose(m(pairs), g(pairs))


def test_code_dims_fixes_leading_codes_to_one():
    m = BinaryCodeBoxModule(4, 5, variant="gbc", num_codes=2, code_dims=2)
    B = m.global_codes()
    assert B.shape == (5, 2)
    assert th.equal(B[:3], th.ones(3, 2))
    v = BinaryCodeBoxModule(4, 5, variant="vbc", code_dims=2)
    codes = v.vertex_codes(th.arange(4))
    assert th.equal(codes[:, :3], th.ones(4, 3))
    assert v.code_logits.shape == (4, 2)


# ---------------------------------------------------------- regulariser
def test_code_regularizer_is_l1_of_one_minus_codes():
    th.manual_seed(4)
    for variant in ("gbc", "vbc"):
        m = BinaryCodeBoxModule(5, 4, variant=variant, code_dims=3)
        expected = (1 - th.sigmoid(m.code_logits)).abs().sum()
        assert m.code_regularization().item() == pytest.approx(expected.item())
        m.set_code_logits(th.full_like(m.code_logits, INF))
        assert m.code_regularization().item() == 0.0
    assert BinaryCodeBoxModule(3, 2, variant="gbox").code_regularization().item() == 0.0


def test_code_regularizer_pushes_codes_to_one():
    th.manual_seed(5)
    m = BinaryCodeBoxModule(5, 4, variant="vbc", code_init=-1.0)
    before = m.trainable_codes().mean().item()
    opt = th.optim.SGD([m.code_logits], lr=1.0)
    for _ in range(50):
        opt.zero_grad()
        m.code_regularization().backward()
        opt.step()
    assert m.trainable_codes().mean().item() > max(before, 0.9)


def test_bce_loss_values_and_self_adversarial_weights():
    pos = th.tensor([0.5])
    neg = th.tensor([[1.0, 2.0]])
    plain = bce_negative_sampling_loss(pos, neg, self_adversarial=False)
    expected = 0.5 - math.log(1 - math.exp(-1.0)) - math.log(1 - math.exp(-2.0))
    assert plain.item() == pytest.approx(expected, rel=1e-5)
    # masked negatives do not count
    masked = bce_negative_sampling_loss(pos, neg, neg_mask=th.tensor([[True, False]]),
                                        self_adversarial=False)
    assert masked.item() == pytest.approx(0.5 - math.log(1 - math.exp(-1.0)), rel=1e-5)
    # self-adversarial: a weighted average, the harder negative (lower energy) weighs more
    adv = bce_negative_sampling_loss(pos, neg, self_adversarial=True)
    nl = neg_log1mexp(neg[0])
    w = th.softmax(-neg[0] + nl, dim=0)  # logit of exp(-E)
    assert w[0] > w[1]
    assert adv.item() == pytest.approx(0.5 + (w * nl).sum().item(), rel=1e-5)


def test_neg_log1mexp_is_stable():
    e = th.tensor([1e-8, 1e-3, 0.5, 5.0, 50.0])
    ref = -th.log1p(-th.exp(-e.double())).float()
    assert th.allclose(neg_log1mexp(e)[1:], ref[1:], rtol=1e-4)
    assert th.isfinite(neg_log1mexp(e)).all()


# -------------------------------------------------- Theorem 3 / Corollary 4
def test_theorem3_premise_holds_for_gbox():
    """E(u,v) - E(v,u) = g(u) - g(v) with g(u) = -log prod_i LSE_nu(u+ - u-, 0)."""
    th.manual_seed(6)
    nu = 0.7
    m = BinaryCodeBoxModule(6, 3, variant="gbox", intersection_temp=0.05, volume_temp=nu)
    pairs = _all_pairs(6)
    u, v = pairs[:, 0], pairs[:, 1]
    side = m.box_max - m.box_min
    g = -th.log(nu * th.nn.functional.softplus(side / nu)).sum(-1)
    assert th.allclose(m.energy(u, v) - m.energy(v, u), g[u] - g[v], atol=1e-5)


def test_gbox_cannot_fit_a_directed_3_cycle_but_vbc_can():
    """Corollary 4: no G-Box parameters reconstruct the chordless 3-cycle; VBC-Box can."""
    pairs = _all_pairs(3)
    adj = _adjacency(pairs, CYCLE3)
    pos, neg = th.tensor(CYCLE3), pairs[~adj]

    th.manual_seed(0)
    gbox = BinaryCodeBoxModule(3, 4, variant="gbox")
    loss_gbox = _fit(gbox, pos, neg, epochs=600)
    f1_gbox, _ = best_f1(gbox(pairs).detach(), adj)
    # By Theorem 3 at least one cycle edge is ranked at or below its reverse edge, so the
    # best threshold misclassifies at least one pair: F1 <= 0.8.
    assert f1_gbox <= 0.8 + 1e-9
    e = gbox(pairs).detach()
    energy = {tuple(p): e[i].item() for i, p in enumerate(pairs.tolist())}
    assert any(energy[(a, b)] >= energy[(b, a)] - 1e-6 for a, b in CYCLE3)

    th.manual_seed(0)
    vbc = BinaryCodeBoxModule(3, 4, variant="vbc")
    loss_vbc = _fit(vbc, pos, neg, epochs=600)
    f1_vbc, _ = best_f1(vbc(pairs).detach(), adj)
    assert f1_vbc == 1.0
    assert loss_vbc < loss_gbox


@pytest.mark.parametrize("seed", [0, 1])
def test_gbc_can_be_trained_to_fit_the_3_cycle(seed):
    """GBC-Box is trainable on the 3-cycle with k = 4 codes in d = 4. (With k = 2 the hard
    minimum of Eq. 8 often stays at the G-Box optimum; see the documentation.)"""
    pairs = _all_pairs(3)
    adj = _adjacency(pairs, CYCLE3)
    th.manual_seed(seed)
    gbc = BinaryCodeBoxModule(3, 4, variant="gbc", num_codes=4)
    _fit(gbc, th.tensor(CYCLE3), pairs[~adj], epochs=600)
    assert best_f1(gbc(pairs).detach(), adj)[0] == 1.0


# ------------------------------------------------- Theorems 5 and 6 (constructions)
def test_theorem5_gbc_construction_models_the_3_cycle():
    """The proof of Theorem 5: split the cycle into two DAGs (order 0 < 1 < 2), embed each
    with hard boxes in its own dimension, and select each dimension with one code."""
    m = _hard("gbc", 3, 2, num_codes=2)
    # dim 0 models D1 = {(0,1), (1,2)} with T1 = 0.5; dim 1 models D2 = {(2,0)} with T2 = 0.5
    m.set_boxes([[0., 0.], [5., 0.9], [9., 0.]], [[10., 1.], [12., 3.], [12., 2.]])
    m.set_code_logits([[INF, -INF], [-INF, INF]])  # B = [[1, 0], [0, 1]] = T / T_j
    pairs = _all_pairs(3)
    predicted = {tuple(p) for p, e in zip(pairs.tolist(), m(pairs).tolist()) if e < 0.5}
    assert predicted == set(CYCLE3)
    # the same boxes with all codes 1 (G-Box) do not model the cycle
    g = _hard("gbox", 3, 2)
    g.set_boxes(m.box_min, m.box_max)
    predicted_g = {tuple(p) for p, e in zip(pairs.tolist(), g(pairs).tolist()) if e < 0.5}
    assert predicted_g != set(CYCLE3)


def test_theorem6_vbc_construction_models_any_graph():
    """The proof of Theorems 6/7 with Lemma 2: one dimension per vertex pair h = {u, v},
    codes 1 exactly on u and v; in dimension h, x = (0, 1), y = (0, e^{-2T})."""
    T = 1.0
    edges = CYCLE3 + [(0, 3), (3, 0)]  # a 3-cycle, one 2-cycle, vertex 3 not linked to 1, 2
    n = 4
    hs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    m = _hard("vbc", n, len(hs))
    box_min = th.zeros(n, len(hs))
    box_max = th.ones(n, len(hs))
    logits = th.full((n, len(hs)), -INF)
    small = math.exp(-2 * T)
    for i, (a, b) in enumerate(hs):
        logits[a, i] = logits[b, i] = INF
        if (a, b) in edges and (b, a) not in edges:
            box_max[b, i] = small       # b inside a: only a -> b
        elif (b, a) in edges and (a, b) not in edges:
            box_max[a, i] = small       # only b -> a
        elif (a, b) not in edges:       # no edge: disjoint intervals
            box_min[b, i], box_max[b, i] = 2.0, 3.0
    m.set_boxes(box_min, box_max)
    m.set_code_logits(logits)
    pairs = _all_pairs(n)
    with th.no_grad():
        energy = m(pairs)  # dimensions with code 0 are ignored, even where F = 0
    assert not th.isnan(energy).any()
    predicted = {tuple(p) for p, e in zip(pairs.tolist(), energy.tolist()) if e < T}
    assert predicted == set(edges)


# --------------------------------------------------------- transitivity bias
def test_box_models_generalise_to_transitive_edges():
    """Trained on the transitive reduction of a chain only, with the closure pairs left
    unlabelled, G-Box and GBC-Box predict the closure edges (box containment is
    transitive), which is the transitivity bias the paper keeps (Section 5.1, Table 2)."""
    n = 5
    chain = [(i, i + 1) for i in range(n - 1)]
    closure = [(i, j) for i in range(n) for j in range(i + 2, n)]
    reverse = [(j, i) for i in range(n) for j in range(i + 1, n)]
    pos, neg = th.tensor(chain), th.tensor(reverse)
    for variant in ("gbox", "gbc"):
        th.manual_seed(0)
        m = BinaryCodeBoxModule(n, 4, variant=variant, code_init=3.0)
        _fit(m, pos, neg, epochs=400)
        with th.no_grad():
            e_pos = m(pos)
            e_closure = m(th.tensor(closure))
            e_rev = m(neg)
        assert e_pos.max() < e_rev.min(), variant
        assert e_closure.max() < e_rev.min(), variant


def test_best_f1_threshold_helper():
    e = th.tensor([0.1, 0.2, 0.9, INF])
    y = th.tensor([True, True, False, False])
    f1, t = best_f1(e, y)
    assert f1 == 1.0 and 0.2 < t < 0.9
