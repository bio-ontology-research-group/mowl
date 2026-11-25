"""Why Concept2Box's centre bias is the one parameter that the Pairs do not compare strictly.

The last layer of the network of Concept2Box gives the centre and the offset of a box; its bias
for the centre shifts the box of both sides of every axiom by the same vector, which changes no
loss. Its gradient is therefore zero in theory and what the two implementations compute is the
rounding noise of a float32 sum of 1e-11 (the gradients of the other parameters are 1e-4).
Adam divides by ``sqrt(v) + 1e-8``, so a noise gradient g moves the parameter by
``lr * g / (|g| + 1e-8)``, about ``lr * 1e-3 = 1e-5`` per step, in a direction that depends on
the noise, which the two orders of summation do not share. After three epochs the biases of the
two sides differ by 2e-5; every other parameter agrees to 2e-7 and so do the scores.
"""
import pytest
import torch as th

from tests.baselines.harness import all_pairs
from tests.baselines.pairs_shapes import _network
from tests.baselines.test_equivalence import _both, _seed

KEYS = ["concept2box", "concept2box/triple"]


def _pair(key):
    return [p for p in all_pairs() if p.key == key][0]


@pytest.mark.parametrize("key", KEYS)
def test_the_gradient_of_the_centre_bias_is_rounding_noise_on_both_sides(key):
    pair = _pair(key)
    old, new = _both(pair)
    for model in (old, new):
        model.module.zero_grad()
        sum(v.sum() for v in pair.losses(model).values()).backward()
        grad = _network(model.module)[2].bias.grad
        half = grad.numel() // 2
        assert grad[:half].abs().max() < 1e-9
        assert grad[half:].abs().min() > 1e-6


@pytest.mark.parametrize("key", KEYS)
def test_the_drift_of_the_centre_bias_is_bounded_by_adams_epsilon(key):
    pair = _pair(key)
    old, new = _both(pair)
    _seed(pair.seed)
    pair.train(old, pair.epochs)
    _seed(pair.seed)
    pair.train(new, pair.epochs)
    a, b = (_network(m.module)[2].bias.detach() for m in (old, new))
    half = a.numel() // 2
    # offset half: same as any parameter
    assert th.allclose(a[half:], b[half:], atol=pair.atol, rtol=pair.rtol)
    # centre half: noise of 1e-10 through Adam (lr 0.01, eps 1e-8) is at most about
    # lr * 1e-10 / 1e-8 = 1e-4 per step; observed 2e-5 after three epochs
    assert (a[:half] - b[:half]).abs().max() < 1e-4
