"""Tests of the filtered role-assertion ranking regime (no solver needed)."""

import numpy as np
import pytest

import mowl.datasets  # noqa: F401  (import order: datasets before nn)
from mowl.datasets.dllite import DLLiteKB
from mowl.evaluation.assertion_ranking import RoleAssertionRankingRegime
from mowl.owlapi import OWLAPIAdapter

NS = "http://mowl.borg/rank#"
IND = [NS + n for n in ("a", "b", "c", "d")]
R = NS + "r"


class _Dataset:
    def __init__(self, valid, test):
        ad = OWLAPIAdapter()
        self.validation = self._ont(ad, "valid", valid)
        self.testing = self._ont(ad, "test", test)

    @staticmethod
    def _ont(ad, name, triples):
        ont = ad.create_ontology(f"http://mowl.borg/rank/{name}")
        r = ad.create_object_property(R)
        for s, o in triples:
            ad.owl_manager.addAxiom(ont, ad.create_object_property_assertion(
                r, ad.create_individual(NS + s), ad.create_individual(NS + o)))
        return ont


class _Model:
    """Scores s(r, x, y) = table[x, y]."""

    def __init__(self, table, train):
        self.table = np.asarray(table, dtype=float)
        self.kb = DLLiteKB.from_lists(
            roles=[R], individuals=IND,
            role_assertions=[(R, NS + s, NS + o) for s, o in train])

    def role_scores(self, role, subject=None, obj=None):
        assert role == 0
        return self.table[subject] if obj is None else self.table[:, obj]


def test_filtered_ranks_by_hand():
    # train r(a, b); valid r(b, d); test r(a, c), r(d, c)
    table = [[0.0, 0.9, 0.5, 0.7],   # a -> b (train, filtered), d (0.7) beats c (0.5)
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.8, 0.0]]
    model = _Model(table, train=[("a", "b")])
    ds = _Dataset(valid=[("b", "d")], test=[("a", "c"), ("d", "c")])
    regime = RoleAssertionRankingRegime(split="test", ties="ordinal", hits=(1, 3))
    # r(a, c): tail query r(a, ?) -> d scores higher, not filtered: rank 2.
    #          head query r(?, c): column c = [0.5, 0, 0, 0.8]; d is a test answer
    #          (filtered), b and c score 0 < 0.5: rank 1.
    # r(d, c): tail query r(d, ?): c is best: rank 1.
    #          head query r(?, c): a is filtered (test), d is best: rank 1.
    res = regime.evaluate(model, ds)["all"]
    assert res["mrr"] == pytest.approx((1 / 2 + 1 + 1 + 1) / 4)
    assert res["hits@1"] == pytest.approx(3 / 4)
    assert res["hits@3"] == pytest.approx(1.0)
    assert regime.info["ranked"] == 4


def test_without_filtering_known_answers_count():
    table = [[0.0, 0.9, 0.5, 0.7]] + [[0.0] * 4] * 3
    model = _Model(table, train=[("a", "b")])
    ds = _Dataset(valid=[], test=[("a", "c")])
    regime = RoleAssertionRankingRegime(split="test", filter_by=(), hits=(1,))
    regime.evaluate(model, ds)
    ranks = regime.ranks(model, [(0, 0, 2)], set())
    assert ranks[0] == 3  # b and d score higher


@pytest.mark.parametrize("ties,expected", [("optimistic", 1), ("pessimistic", 3),
                                           ("average", 2), ("ordinal", 2)])
def test_tie_policies(ties, expected):
    regime = RoleAssertionRankingRegime(ties=ties)
    scores = np.array([0.5, 0.5, 0.5, 0.1])
    assert regime.rank(scores, 1, np.zeros(4, dtype=bool)) == expected


def test_validation_split_filters_train_and_valid():
    table = [[0.0, 0.9, 0.5, 0.7]] + [[0.0] * 4] * 3
    model = _Model(table, train=[("a", "b")])
    ds = _Dataset(valid=[("a", "c"), ("a", "d")], test=[])
    regime = RoleAssertionRankingRegime(split="valid", hits=(1,))
    res = regime.evaluate(model, ds)["all"]
    # tail queries: c (0.5) with b, d filtered -> 1; d (0.7) -> 1.
    # head queries r(?, c) and r(?, d): a scores above all others -> 1.
    assert res["mrr"] == pytest.approx(1.0)
    assert regime.filter_by == ("train", "valid")


def test_bad_arguments():
    with pytest.raises(ValueError):
        RoleAssertionRankingRegime(split="train")
    with pytest.raises(ValueError):
        RoleAssertionRankingRegime(ties="random")
    with pytest.raises(ValueError):
        RoleAssertionRankingRegime(filter_by=("closure",))
