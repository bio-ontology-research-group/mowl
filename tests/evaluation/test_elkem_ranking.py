"""The ELK-Em subsumption-ranking evaluation regime: margin-free scoring, owl:Thing
removed from the candidates, average-rank ties."""

import os

import pytest
import torch as th

from mowl.evaluation.el_scoring import margin_free_loss_scores, resolve_scorer
from mowl.evaluation.normal_form_ranking import ELKEmSubsumptionRankingRegime
from mowl.models import ELKEm
from tests.benchmarks.tiny import scratch_dir, tiny_dataset


def _model():
    model = ELKEm(tiny_dataset(), embed_dim=4, batch_size=4,
                  model_filepath=os.path.join(scratch_dir(), "elkem_rank.pt"))
    with th.no_grad():
        model.module.class_center.weight.zero_()
        model.module.class_offset.weight.zero_()
    return model


def test_margin_free_scores_ignore_the_margin():
    model = _model()
    rows = model._training_tensors()["gci0"]
    scores = margin_free_loss_scores(model, "gci0", rows)
    assert th.allclose(scores, model.module.zero_margin_loss(rows, "gci0"))
    # equal boxes have zero loss at any margin, so compare through a non-satisfied pair
    with th.no_grad():
        model.module.class_center.weight[1] = 10.0
    rows = th.tensor([[0, 1]])
    loss = model.module(rows, "gci0")
    free = margin_free_loss_scores(model, "gci0", rows)
    assert float(free) < float(loss)
    assert resolve_scorer(model, "margin_free")[1] == "margin_free"


def test_all_tied_candidates_share_their_average_rank():
    """With every trained box identical, every candidate of every query has the same
    score, so each rank is the average rank (n + 1) / 2 of the n effective candidates
    (all classes except owl:Thing and the query)."""
    model = _model()
    dataset = model.dataset
    regime = ELKEmSubsumptionRankingRegime(split="test")
    results = regime.evaluate(model, dataset)
    m = results["all"]
    n_eff = regime.info["num_candidates"] - 1
    # owl:Nothing is a candidate whose empty box never ties with the others
    untied = 1 if "http://www.w3.org/2002/07/owl#Nothing" in regime.candidate_iris(dataset) \
        else 0
    rank = (n_eff - untied + 1) / 2
    assert m["med"] == pytest.approx(rank)
    assert m["mr"] == pytest.approx(rank)
    assert m["mrr"] == pytest.approx(1 / rank)
    assert m["hits@1"] == 0.0            # average ties: no query ranks first
    assert m["hits@100"] == 1.0
    assert m["auc"] == pytest.approx((n_eff - rank) / (n_eff - 1))
    assert regime.info["scoring"] == "margin_free"
    assert regime.info["ties"] == "average"
    assert "http://www.w3.org/2002/07/owl#Thing" not in regime.candidate_iris(dataset)
    assert regime.info["queries"]["gci0"]["queries"] == 2
