"""Gumbel box embeddings (Dasgupta et al., NeurIPS 2020) over the class taxonomy."""
import math

import numpy as np
import pytest
import torch as th

import mowl
mowl.init_jvm("2g")

from mowl.evaluation.model_scoring import axiom_scoring_of  # noqa: E402
from mowl.models.gumbel_box import GumbelBox  # noqa: E402
from mowl.nn.el.gumbel_box import GumbelBoxModule, softplus_volume  # noqa: E402
from tests.models.taxonomy_helpers import (check_trained_taxonomy, total_loss,  # noqa: E402
                                           train)


def test_lemma_1_max_of_max_gumbels():
    """max of MaxGumbel(mu_x, b), MaxGumbel(mu_y, b) is MaxGumbel(b ln(e^{mu_x/b}+e^{mu_y/b}), b);
    the min of two MinGumbels is MinGumbel(-b ln(e^{-mu_x/b}+e^{-mu_y/b}), b)."""
    rng = np.random.default_rng(0)
    b, mx_, my_ = 0.3, 0.2, 0.9
    n = 400000
    x = rng.gumbel(mx_, b, n)
    y = rng.gumbel(my_, b, n)
    loc = b * math.log(math.exp(mx_ / b) + math.exp(my_ / b))
    assert np.max([x, y], axis=0).mean() == pytest.approx(loc + 0.5772156649 * b, abs=5e-3)
    # the module's intersection locations are exactly these log-sum-exps
    lo = b * th.logaddexp(th.tensor(mx_ / b), th.tensor(my_ / b)).item()
    assert lo == pytest.approx(loc)
    # min of MinGumbels (= -MaxGumbel of negated locations)
    xm, ym = -rng.gumbel(-mx_, b, n), -rng.gumbel(-my_, b, n)
    loc_min = -b * math.log(math.exp(-mx_ / b) + math.exp(-my_ / b))
    assert np.min([xm, ym], axis=0).mean() == pytest.approx(loc_min - 0.5772156649 * b,
                                                            abs=5e-3)


def test_softplus_volume_approximates_the_bessel_expected_length():
    """Proposition 1: E[max(Y - X, 0)] = 2 b K0(2 exp(-(mu_y - mu_x) / (2 b)))."""
    scipy_special = pytest.importorskip("scipy.special")
    b = 0.1
    rng = np.random.default_rng(1)
    for diff in (-0.3, 0.0, 0.4):
        x = rng.gumbel(0.0, b, 1_000_000)        # MaxGumbel lower corner
        y = -rng.gumbel(-diff, b, 1_000_000)     # MinGumbel upper corner
        empirical = np.maximum(y - x, 0).mean()
        exact = 2 * b * scipy_special.k0(2 * math.exp(-diff / (2 * b)))
        approx = softplus_volume(th.tensor(diff), b).item()
        assert empirical == pytest.approx(exact, abs=2e-3)
        assert abs(approx - exact) < 0.0617013 * b + 1e-4      # the paper's error bound
    # beta -> 0: the hinge
    assert softplus_volume(th.tensor(0.5), 1e-4).item() == pytest.approx(0.5, abs=1e-3)
    assert softplus_volume(th.tensor(-0.5), 1e-4).item() == pytest.approx(0.0, abs=1e-6)


def _module(rows, beta=0.01):
    module = GumbelBoxModule(len(rows), embed_dim=2, beta=beta)
    with th.no_grad():
        module.class_embed.weight.copy_(th.tensor(rows))
    return module


def test_conditional_probability_and_gradient_for_disjoint_boxes():
    # [min | delta]: a = [0,.5]^2, b = [0,.5]x[0,1] (contains a), c = [.6,1]^2 (disjoint)
    module = _module([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 1.0], [0.6, 0.6, 0.4, 0.4]])
    p = lambda s, t: math.exp(module.log_conditional(th.tensor([s]), th.tensor([t])).item())  # noqa: E731,E501
    assert p(0, 1) > 0.5 > p(1, 0)         # a inside b is more probable than b inside a
    assert p(0, 1) > p(0, 2)
    assert 0 < p(0, 2) < 1e-6              # tiny but positive for disjoint boxes
    # in the limit beta -> 0 the model is the hard box lattice: P(b|a) -> 1, P(a|b) -> 1/2
    sharp = _module([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 1.0]], beta=1e-4)
    q = lambda s, t: math.exp(sharp.log_conditional(th.tensor([s]), th.tensor([t])).item())  # noqa: E731,E501
    assert q(0, 1) == pytest.approx(1.0, abs=1e-2) and q(1, 0) == pytest.approx(0.5, abs=1e-2)
    # local identifiability: a disjoint pair still gets a gradient (hard boxes do not)
    module.class_embed.weight.grad = None
    module(th.tensor([[0, 2]]), "gci0").sum().backward()
    assert module.class_embed.weight.grad.abs().sum() > 0


def test_negative_loss_is_low_for_disjoint_pairs():
    module = _module([[0.0, 0.0, 0.5, 0.5], [0.6, 0.6, 0.4, 0.4], [0.0, 0.0, 0.5, 0.5]])
    neg = lambda s, t: module(th.tensor([[s, t]]), "gci0", neg=True).item()  # noqa: E731
    assert neg(0, 1) < 1e-6 < neg(0, 2)


def test_trains_on_the_taxonomy_and_declares_scoring():
    model = train(GumbelBox, epochs=400, embed_dim=4, beta=0.02, learning_rate=0.02)
    assert axiom_scoring_of(model) == "module" and model.scored_forms == ("gci0",)
    check_trained_taxonomy(model)
    first = train(GumbelBox, epochs=1, embed_dim=4, beta=0.02, learning_rate=0.02)
    assert total_loss(model) < total_loss(first)


def test_log_guard_stable_matches_epsilon_where_both_are_exact_and_keeps_gradient():
    from mowl.nn.el.gumbel_box import log_softplus_volume
    x = th.tensor([0.5, 0.05, 0.0, -0.1], dtype=th.double)
    old = log_softplus_volume(x, 0.01, guard="epsilon")
    new = log_softplus_volume(x, 0.01)
    # z = x / 0.01 - 2 gamma is above -15 for the first three: same values
    assert th.allclose(old[:3], new[:3], atol=1e-12)
    # very negative: the old guard returns a constant with no gradient, the new one does not
    far = th.tensor([-2.0], dtype=th.double, requires_grad=True)
    log_softplus_volume(far, 0.01, guard="epsilon").sum().backward()
    assert abs(far.grad.item()) < 1e-20
    far2 = th.tensor([-2.0], dtype=th.double, requires_grad=True)
    val = log_softplus_volume(far2, 0.01)
    val.sum().backward()
    assert far2.grad.item() == pytest.approx(100.0, rel=1e-6)
    assert val.item() < log_softplus_volume(far, 0.01, guard="epsilon").item()


def test_negative_loss_guards_agree_away_from_certain_inclusion():
    new = GumbelBoxModule(4, embed_dim=3, beta=0.01)
    old = GumbelBoxModule(4, embed_dim=3, beta=0.01, log_guard="epsilon")
    old.load_state_dict(new.state_dict())
    sub, sup = th.tensor([0, 1, 2]), th.tensor([1, 2, 3])
    assert th.allclose(new.negative_loss(sub, sup), old.negative_loss(sub, sup), rtol=1e-4)
    assert th.allclose(new.positive_loss(sub, sup), old.positive_loss(sub, sup), atol=1e-6)
    with pytest.raises(ValueError):
        GumbelBoxModule(4, log_guard="x")
