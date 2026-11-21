"""Logical query answering as an evaluation regime."""

import numpy as np
import pytest
import torch as th


from mowl.evaluation import (LogicalQueryAnswering, QUERY_STRUCTURES, answer_query,
                             sample_queries)

KNOWN = np.array([[0, 0, 1], [1, 1, 2], [0, 0, 3], [4, 0, 1], [5, 1, 2]])
HELD_OUT = np.array([[3, 1, 6], [1, 1, 7], [4, 1, 2]])
COMPLETE = np.concatenate([KNOWN, HELD_OUT])


def test_exact_answers():
    S = QUERY_STRUCTURES
    assert answer_query(S["1p"], (0, (0,)), KNOWN) == {1, 3}
    assert answer_query(S["2p"], (0, (0, 1)), KNOWN) == {2}
    assert answer_query(S["2p"], (0, (0, 1)), COMPLETE) == {2, 6, 7}
    assert answer_query(S["2i"], ((1, (1,)), (5, (1,))), KNOWN) == {2}
    assert answer_query(S["2u"], ((0, (0,)), (1, (1,)), (-1,)), KNOWN) == {1, 2, 3}
    assert answer_query(S["ip"], (((0, (0,)), (4, (0,))), (1,)), COMPLETE) == {2, 7}
    assert answer_query(S["up"], (((0, (0,)), (4, (0,)), (-1,)), (1,)), KNOWN) == {2}


def test_sampled_queries_have_hard_answers():
    queries = sample_queries(KNOWN, HELD_OUT, structures=("1p", "2p", "2i", "2u"),
                             num_queries=5, seed=1)
    assert queries["1p"] and queries["2p"]
    for name, items in queries.items():
        for query, easy, hard in items:
            assert hard and not (easy & hard)
            assert answer_query(QUERY_STRUCTURES[name], query, KNOWN) == easy
            assert answer_query(QUERY_STRUCTURES[name], query, COMPLETE) == easy | hard


class _Oracle:
    """Scores 1 for the answers on a graph, 0 for the others."""

    def __init__(self, triples, n=8):
        self.triples, self.n = triples, n

    def score_query(self, structure, queries):
        scores = np.zeros((len(queries), self.n))
        for i, q in enumerate(queries):
            scores[i, sorted(answer_query(structure, q, self.triples))] = 1.0
        return scores


def test_ranking_metrics():
    queries = sample_queries(KNOWN, HELD_OUT, structures=("1p", "2p"), num_queries=5, seed=0)
    regime = LogicalQueryAnswering(queries)
    perfect = regime.evaluate(_Oracle(COMPLETE))
    assert perfect["average"]["mrr"] == 1.0 and perfect["2p"]["hits@1"] == 1.0
    # A model that only knows the training graph ties every hard answer with the
    # non-answers: the tie policy decides.
    blind = _Oracle(KNOWN)
    assert LogicalQueryAnswering(queries, ties="optimistic").evaluate(blind)["1p"]["mrr"] == 1.0
    assert LogicalQueryAnswering(queries, ties="pessimistic").evaluate(blind)["1p"]["mrr"] < 0.5
    assert regime.evaluate(blind)["1p"]["mrr"] < 0.5
    with pytest.raises(TypeError):
        regime.evaluate(object())


def test_grounded_el_answers_queries():
    from mowl.models import GroundedEL
    model = GroundedEL.from_tables(
        {"C0": 0, "C1": 1}, {"r": 0, "s": 1}, gci0=th.tensor([[0, 1]]),
        object_property_assertion=th.as_tensor(KNOWN),
        individual_index_dict={f"a{i}": i for i in range(8)},
        role_mode="translation", embed_dim=6, n_role_components=3, epochs=300, seed=0,
        learning_rate=0.05)
    model.train()
    one_hop = model.score_query(QUERY_STRUCTURES["1p"], [(0, (0,)), (1, (1,))])
    assert tuple(one_hop.shape) == (2, 8)
    boxes = model.score_query(QUERY_STRUCTURES["1p"], [(0, (0,)), (1, (1,))], method="box")
    assert th.allclose(one_hop, boxes)
    # The stated facts are answers of the trained model.
    assert one_hop[0, 1] > 0 and one_hop[0, 3] > 0 and one_hop[1, 2] > 0
    queries = sample_queries(KNOWN, HELD_OUT, structures=("1p", "2p"), num_queries=5, seed=0)
    result = LogicalQueryAnswering(queries, score_kwargs={"k": 8}).evaluate(model)
    assert set(result) == {"1p", "2p", "average"}
    assert 0.0 < result["average"]["mrr"] <= 1.0
