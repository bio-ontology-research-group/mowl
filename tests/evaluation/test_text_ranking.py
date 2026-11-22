"""The general helpers of text-encoder methods, on their own and with another encoder."""
import numpy as np
import torch as th

from mowl.evaluation.signature_costs import class_assertion_costs
from mowl.evaluation.text_ranking import (best_on_grid, candidate_scores, embed_unique_pairs,
                                          encode_batched, encode_cached)
from mowl.representation.library.vector_tables import (load_vector_table, pca_reduce,
                                                       save_vector_table)


def test_encode_batched_restores_the_mode_and_handles_no_texts():
    module = th.nn.Linear(2, 2)
    module.train()
    out = encode_batched(lambda texts: th.ones(len(texts), 3), ["a"] * 5, 2, module, 3, "cpu")
    assert out.shape == (5, 3) and module.training
    assert encode_batched(lambda t: None, [], 2, module, lambda: 7, "cpu").shape == (0, 7)


def test_encode_cached_encodes_each_text_once():
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return th.tensor([[float(len(t))] for t in texts])

    cache = {}
    first = encode_cached(encode, ["aa", "b", "aa"], cache)
    again = encode_cached(encode, ["b", "ccc"], cache)
    assert first.flatten().tolist() == [2.0, 1.0, 2.0] and again.flatten().tolist() == [1.0, 3.0]
    assert calls == [["aa", "b"], ["ccc"]]


def test_embed_unique_pairs_expands_back_to_rows():
    seen = []

    def embed(a, b):
        seen.append(len(a))
        return (10 * a + b).float().unsqueeze(1)

    out = embed_unique_pairs(embed, th.tensor([1, 2, 1]), th.tensor([3, 4, 3]))
    assert out.flatten().tolist() == [13.0, 24.0, 13.0] and seen == [2]


def test_candidate_scores_chunks_give_the_same_as_one_block():
    q, c = th.randn(4, 3), th.randn(7, 3)
    assert th.allclose(candidate_scores(lambda a, b: a @ b.T, q, c, 3), q @ c.T)


def test_best_on_grid_prefers_the_first_on_ties():
    scores = th.tensor([[1.0, 0.0], [0.0, 1.0]])
    mrr, value = best_on_grid(lambda v, a, b: scores[a:b], th.tensor([0, 1]), (0.2, 0.7))
    assert mrr == 1.0 and value == 0.2


def test_class_assertion_costs_are_infinite_outside_the_signature():
    rows = th.tensor([[0, 0], [1, 0], [0, 1]])
    out = class_assertion_costs(rows, {"A": 0, "B": 1}, {"a": 0, "b": 1}, ["A"], ["a"],
                                lambda concept, individuals: [0.5] * len(individuals))
    assert out.tolist() == [0.5, float("inf"), float("inf")]


def test_vector_tables_round_trip_and_reduce(tmp_path):
    vectors = {"a": np.arange(6.0), "b": np.arange(6.0)[::-1], "c": np.ones(6)}
    path = tmp_path / "v.npz"
    save_vector_table(path, vectors)
    loaded = load_vector_table(path)
    assert set(loaded) == set(vectors) and np.allclose(loaded["a"], vectors["a"])
    small = pca_reduce(vectors, 2)
    assert small["a"].shape == (2,)
    assert pca_reduce(vectors, 5)["a"].shape == (5,)        # padded: only 3 components


def test_old_import_paths_are_the_general_modules():
    import mowl.nn.lm.encoders as old_encoders
    import mowl.nn.lm.hyperbolic as old_ball
    import mowl.nn.el.structural as old_structural
    import mowl.models.query.model as old_query
    from mowl.models.composed.query import QueryEmbeddingModel
    from mowl.representation.library import poincare_ball, structural_tables, text_encoders
    assert old_encoders is text_encoders and old_ball is poincare_ball
    assert old_structural.DefinitionTable is structural_tables.DefinitionTable
    assert old_query.QueryEmbeddingModel is QueryEmbeddingModel
