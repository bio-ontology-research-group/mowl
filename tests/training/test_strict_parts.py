"""Parts that the tightened strict check made necessary: the anchor penalty, the sum of loss
terms, the interval error of a conditional, the margin logit of a query geometry."""
import math
from types import SimpleNamespace

import torch as th

from mowl.evaluation.sel_probability import conditional_errors, interval_error
from mowl.representation.library.query_parts import margin_logit
from mowl.training import AnchorPenalty
from mowl.training.regularization import add_penalties, sum_terms


class _Init:
    def anchor_loss(self, module):
        return module.weight.pow(2).mean()


def _model(weight, init=None):
    return SimpleNamespace(init_boxes=init, anchor_weight=weight,
                           module=SimpleNamespace(weight=th.tensor([1.0, 3.0])))


def test_anchor_penalty_is_the_weighted_anchor_loss():
    model = _model(0.5, _Init())
    (penalty,) = AnchorPenalty.of(model)
    assert float(penalty.penalty(model)) == 0.5 * 5.0
    assert float(add_penalties(th.tensor(1.0), [penalty], model)) == 3.5


def test_anchor_penalty_is_absent_without_initialiser_or_weight():
    assert AnchorPenalty.of(_model(0.5)) == ()
    assert AnchorPenalty.of(_model(0.0, _Init())) == ()
    assert AnchorPenalty.of(SimpleNamespace()) == ()
    assert add_penalties(0, (), _model(0.0)) == 0


def test_anchor_penalty_works_for_any_model_with_an_initialiser():
    # a swap: another geometry's module, a different weight, no method involved
    model = _model(2.0, _Init())
    assert float(AnchorPenalty(model.init_boxes, 2.0).penalty(model)) == 10.0


def test_sum_terms_skips_none():
    assert sum_terms(None, None) is None
    assert sum_terms(None, 2.0) == 2.0
    assert sum_terms(1.0, None, 2.0) == 3.0


def test_interval_error_is_the_distance_to_the_interval():
    assert interval_error(0.5, 0.2, 0.6) == 0.0
    assert math.isclose(interval_error(0.1, 0.2, 0.6), 0.1)
    assert math.isclose(interval_error(0.9, 0.2, 0.6), 0.3)
    assert math.isnan(interval_error(float("nan"), 0.2, 0.6))


def test_conditional_errors_follow_the_conditionals_of_the_model():
    conditionals = [SimpleNamespace(conclusion="d", premise="c", lower=0.2, upper=0.6),
                    SimpleNamespace(conclusion="e", premise="c", lower=0.5, upper=0.5)]
    model = SimpleNamespace(conditionals=conditionals,
                            probability=lambda d, c: {"d": 0.7, "e": 0.5}[d])
    errors = conditional_errors(model)
    assert math.isclose(errors[0], 0.1) and errors[1] == 0.0


def test_margin_logit_is_gamma_minus_distance():
    out = margin_logit(th.tensor(4.0), th.tensor([[1.0, 5.0]]))
    assert out.tolist() == [[3.0, -1.0]]
