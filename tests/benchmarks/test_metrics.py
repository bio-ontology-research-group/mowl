"""Rank computations and metrics on hand-computed examples."""

import math

import numpy as np
import pytest
import torch as th

from mowl.evaluation.ranking_metrics import (
    ordinal_ranks, average_ranks, auc_box2el, auc_rank_curve, auc_mowl, summarise_ranks,
    micro_metrics, best_threshold, binary_f1)


def test_ordinal_and_average_ranks():
    scores = th.tensor([[0.1, 0.5, 0.5, 0.2]])
    assert ordinal_ranks(scores, th.tensor([2])).tolist() == [4]
    assert ordinal_ranks(scores, th.tensor([1])).tolist() == [3]
    assert ordinal_ranks(scores, th.tensor([0])).tolist() == [1]
    assert average_ranks(scores, th.tensor([2])).tolist() == [3.5]
    assert average_ranks(scores, th.tensor([3])).tolist() == [2.0]


def test_auc_box2el():
    assert auc_box2el([1], 2) == pytest.approx(1.0)
    assert auc_box2el([2], 2) == pytest.approx(0.5)
    # N = 3, ranks 1 and 3: thresholds 1 and 3 give (FPR, TPR) = (0.25, 0.5), (1, 1).
    assert auc_box2el([1, 3], 3) == pytest.approx(0.5 * 0.25 * 0.5 + 0.75 * 0.75)


def test_auc_rank_curve():
    assert auc_rank_curve([1, 3], 4) == pytest.approx(2.5 / 4)


def test_auc_mowl_closed_form():
    assert auc_mowl([1, 3], 4) == pytest.approx((3 + 1) / 2 / 3)
    assert auc_mowl([1], 5) == pytest.approx(1.0)


def test_auc_mowl_matches_compute_rank_roc():
    from collections import Counter
    from mowl.evaluation.base import compute_rank_roc
    rng = np.random.default_rng(0)
    for n in (5, 50, 300):
        ranks = rng.integers(1, n + 1, size=40).tolist()
        assert auc_mowl(ranks, n) == pytest.approx(compute_rank_roc(dict(Counter(ranks)), n))


def test_summarise_ranks():
    m = summarise_ranks([1, 2, 10], 20, hits=(1, 10), auc="mowl")
    assert m["hits@1"] == pytest.approx(1 / 3)
    assert m["hits@10"] == pytest.approx(1.0)
    assert m["med"] == 2
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0.1) / 3)
    assert m["mr"] == pytest.approx(13 / 3)
    empty = summarise_ranks([], 5, prefix="f_")
    assert math.isnan(empty["f_mr"])


def test_micro_metrics():
    m = micro_metrics([1, 3, 2], ["a", "a", "b"], 4)
    assert m["micro_mr"] == pytest.approx(2.0)
    assert m["micro_auc"] == pytest.approx(np.mean([auc_mowl([1, 3], 4), auc_mowl([2], 4)]))


def test_threshold_and_f1():
    t, f1 = best_threshold([0.1, 0.2, 0.9, 0.3], [1, 1, 0, 0])
    assert t == pytest.approx(0.2) and f1 == pytest.approx(1.0)
    assert binary_f1([1, 1, 0], [1, 0, 0]) == pytest.approx(2 / 3)


def test_rank_queries_exclusion_and_filter():
    from mowl.evaluation.axiom_ranking import rank_queries, build_filter

    def scorer(model, gci, rows):
        return (rows[:, 1] - 2).abs().float()

    class Dummy:
        device = "cpu"

    cands = th.arange(5)
    rows = th.tensor([[0, 3]])
    raw, filt, kept = rank_queries(Dummy(), "gci0", rows, 1, cands, scorer)
    assert raw.tolist() == [3.0] and filt is None and kept.tolist() == [True]
    # Excluding column 0 removes candidate 0, which is ranked below the target anyway.
    raw, _, _ = rank_queries(Dummy(), "gci0", rows, 1, cands, scorer, exclude_cols=(0,))
    assert raw.tolist() == [3.0]
    filters = build_filter(th.tensor([[0, 1]]), 1)
    assert filters == {(0,): {1}}
    raw, filt, _ = rank_queries(Dummy(), "gci0", rows, 1, cands, scorer, filters=filters)
    assert raw.tolist() == [3.0] and filt.tolist() == [2.0]
    # A target outside the candidate set is not ranked.
    _, _, kept = rank_queries(Dummy(), "gci0", th.tensor([[0, 7]]), 1, cands, scorer)
    assert kept.tolist() == [False]


def test_rank_queries_reports_progress_of_a_long_ranking(capsys):
    """A long ranking prints a progress line now and then (a ranking of the anatomy
    inferences ran for hours without one); a short one prints nothing."""
    from mowl.evaluation.axiom_ranking import rank_queries

    def scorer(model, gci, rows):
        return (rows[:, 1] - 2).abs().float()

    class Dummy:
        device = "cpu"

    rows = th.tensor([[0, 3]] * 4)
    quiet, _, _ = rank_queries(Dummy(), "gci0", rows, 1, th.arange(5), scorer, max_rows=5)
    assert capsys.readouterr().err == ""
    loud, _, _ = rank_queries(Dummy(), "gci0", rows, 1, th.arange(5), scorer, max_rows=5,
                              progress=1e-9)
    err = capsys.readouterr().err
    assert "ranking gci0: 1 of 4 queries (5 candidates each)" in err
    assert "4 of 4 queries" in err and "s left" in err
    assert loud.tolist() == quiet.tolist()
