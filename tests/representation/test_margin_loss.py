"""``margin_loss`` (the named objective of CatE, Eq. 7): known inputs, known outputs."""
import pytest
import torch as th

from mowl.representation.library.hierarchy_objectives import margin_loss


def test_known_value_per_negative():
    pos = th.tensor([0.5, 1.5])
    neg = th.tensor([[0.2, 2.0], [0.0, 0.9]])       # hinges (margin 1): 0.8, 0, 1.0, 0.1
    assert float(margin_loss(pos, neg, 1.0)) == pytest.approx(1.0 + (0.8 + 0.0 + 1.0 + 0.1) / 4)


def test_mean_aggregation_averages_the_costs_before_the_hinge():
    pos = th.tensor([0.0, 0.0])
    neg = th.tensor([[0.2, 2.0], [0.0, 0.8]])       # edge costs 1.1 and 0.4: hinges 0 and 0.6
    assert float(margin_loss(pos, neg, 1.0, "mean")) == pytest.approx(0.3)


def test_a_corruption_that_is_a_stated_edge_is_left_out():
    pos = th.tensor([0.0])
    neg = th.tensor([[0.0, 5.0]])
    mask = th.tensor([[False, True]])
    assert float(margin_loss(pos, neg, 1.0, mask=mask)) == 0.0
    assert float(margin_loss(pos, neg, 1.0, "mean", mask=mask)) == 0.0
    # an edge whose corruptions are all masked contributes nothing at all
    two = margin_loss(th.tensor([0.0, 0.0]), th.tensor([[0.0], [0.0]]), 1.0, "mean",
                      mask=th.tensor([[False], [True]]))
    assert float(two) == pytest.approx(1.0)


def test_zero_when_stated_edges_cost_nothing_and_corruptions_cost_the_margin():
    pos, neg = th.zeros(3), th.full((3, 4), 0.7)
    assert float(margin_loss(pos, neg, 0.7)) == 0.0
    assert float(margin_loss(pos, neg, 0.7, "mean")) == 0.0


def test_a_flat_negative_cost_is_one_corruption_each_and_the_loss_is_differentiable():
    neg = th.tensor([0.1, 0.4], requires_grad=True)
    loss = margin_loss(th.zeros(2), neg, 1.0)
    loss.backward()
    assert float(loss) == pytest.approx((0.9 + 0.6) / 2)
    assert th.allclose(neg.grad, th.tensor([-0.5, -0.5]))
    with pytest.raises(ValueError):
        margin_loss(th.zeros(1), th.zeros(1, 1), 1.0, "median")


def test_agrees_with_the_direct_loss_of_the_baseline():
    from mowl.baselines.nn.graph.cate.losses import order_margin_loss
    g = th.Generator().manual_seed(3)
    pos, neg = th.rand(6, generator=g), th.rand(6, 5, generator=g) * 2
    mask = th.rand(6, 5, generator=g) > 0.3
    for aggregation in ("per_negative", "mean"):
        assert float(margin_loss(pos, neg, 1.0, aggregation, mask)) == pytest.approx(
            float(order_margin_loss(pos, neg, 1.0, aggregation, mask)), abs=1e-7)
