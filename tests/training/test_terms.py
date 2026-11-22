"""The general loss terms of ``mowl.training.terms``: known values, properties, and agreement
with the frozen ELK-Em functions of the baseline."""
import math

import pytest
import torch as th

from mowl.training import (correlation_contrastive_loss, lexical_similarity_loss,
                           pearson_correlation)


def test_pearson_correlation_known_values():
    x = th.tensor([1.0, 2.0, 3.0, 4.0])
    assert float(pearson_correlation(x, 2 * x + 5)) == pytest.approx(1.0, abs=1e-6)
    assert float(pearson_correlation(x, -x)) == pytest.approx(-1.0, abs=1e-6)
    assert float(pearson_correlation(x, th.ones(4))) == 0.0      # constant: defined, zero


def test_contrastive_term_is_zero_for_perfect_separation_and_two_for_reversed():
    assert float(correlation_contrastive_loss(th.tensor([1.0, 1.0]), th.tensor([0.0, 0.0, 0.0]))
                 ) == pytest.approx(0.0, abs=1e-6)
    assert float(correlation_contrastive_loss(th.tensor([0.0, 0.0]), th.tensor([1.0, 1.0, 1.0]))
                 ) == pytest.approx(2.0, abs=1e-6)


def test_contrastive_term_accepts_columns_and_different_lengths_and_is_differentiable():
    pos = th.tensor([[0.9], [0.8]], requires_grad=True)
    neg = th.tensor([[0.2], [0.3], [0.1]])
    loss = correlation_contrastive_loss(pos, neg)
    loss.backward()
    assert 0 <= float(loss) < 0.2 and bool(th.isfinite(pos.grad).all())


def test_lexical_term_is_zero_when_the_geometry_follows_the_features():
    # two features at 90 degrees have external similarity 0.5, and two points at squared
    # distance (n / lambda) * ln 2 have geometric similarity 0.5
    d = math.sqrt(math.log(2.0) * 2 / 1.0)
    points = th.tensor([[0.0, 0.0], [d, 0.0]])
    features = th.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert float(lexical_similarity_loss(points, features, 1.0)) == pytest.approx(0.0, abs=1e-6)
    # identical points, opposite features: external 0, geometric 1
    assert float(lexical_similarity_loss(th.zeros(2, 2), th.tensor([[1.0, 0.0], [-1.0, 0.0]]),
                                         1.0)) == pytest.approx(1.0)
    assert float(lexical_similarity_loss(th.zeros(1, 2), th.ones(1, 3), 1.0)) == 0.0


def test_the_terms_agree_with_the_direct_functions_of_the_baseline():
    import mowl.baselines.nn.el.elkem.losses as B
    g = th.Generator().manual_seed(0)
    pos, neg = th.rand(7, generator=g), th.rand(11, generator=g)
    assert float(correlation_contrastive_loss(pos, neg)) == pytest.approx(
        float(B.pearson_contrastive_loss(pos, neg)), abs=1e-7)
    points, features = th.randn(9, 4, generator=g), th.randn(9, 6, generator=g)
    assert float(lexical_similarity_loss(points, features, 0.7)) == pytest.approx(
        float(B.lexical_loss(points, features, 0.7)), abs=1e-7)
