"""Message passing and scoring of RESHUFFLE: closed path rules as matrix products."""
import pytest
import torch as th
from pykeen.triples import TriplesFactory

import mowl
mowl.init_jvm("1g")
import mowl.datasets  # noqa: F401,E402
from mowl.nn.graph.reshuffle import Reshuffle, propagate, relation_matrices, random_features  # noqa: E402
from mowl.nn.graph.reshuffle.module import reshuffle_interaction  # noqa: E402

L = 6


def _permutation_raw(perm, sharp=60.0):
    """Raw ``(l + 1) x l`` parameters whose softmax is (almost) the permutation matrix with
    ``R[perm[j], j] = 1``."""
    raw = th.zeros(L + 1, L)
    for j, i in enumerate(perm):
        raw[i, j] = sharp
    raw[L] = -sharp          # the dropped row gets no weight
    return raw


def test_relation_matrices_are_column_substochastic():
    raw = th.randn(3, L + 1, L)
    m = relation_matrices(raw)
    assert m.shape == (3, L, L)
    assert (m >= 0).all() and (m.sum(dim=-2) <= 1 + 1e-6).all()


def test_permutation_matrix_is_recovered():
    perm = [2, 0, 1, 5, 4, 3]
    m = relation_matrices(_permutation_raw(perm))
    expected = th.zeros(L, L)
    for j, i in enumerate(perm):
        expected[i, j] = 1
    assert th.allclose(m, expected, atol=1e-4)


def test_score_is_zero_exactly_when_the_ordering_holds():
    r = th.eye(L)
    x = th.tensor([[1., 0, 1, 0, 0, 1]])
    assert reshuffle_interaction(x[None], r, x[None]).item() == 0.0
    bigger = x.clone()
    bigger[0, 1] = 1
    assert reshuffle_interaction(x[None], r, bigger[None]).item() == 0.0
    assert reshuffle_interaction(bigger[None], r, x[None]).item() < 0.0


def test_closed_path_rule_is_captured_by_the_product_of_matrices():
    """Graph a -r1-> b -r2-> c. With R_r3 = R_r1 R_r2, two layers of message passing make
    x_c at least x_a R_r3, so the triple (a, r3, c) is captured; the triple (c, r3, a)
    (the path read backwards) is not."""
    th.manual_seed(0)
    p1, p2 = [1, 2, 3, 4, 5, 0], [3, 4, 5, 0, 1, 2]
    r1, r2 = relation_matrices(_permutation_raw(p1)), relation_matrices(_permutation_raw(p2))
    r3 = r1 @ r2
    # relations 0, 1 with their inverses (transposes) at 2i + 1; relation 2 is the head
    rel = th.stack([r1, r1.T, r2, r2.T, r3, r3.T])
    edge_index = th.tensor([[0, 1], [1, 2]])
    edge_type = th.tensor([0, 1])
    x0 = random_features(3, 8, L, density=0.4, generator=th.Generator().manual_seed(1))
    x = propagate(x0, rel, th.eye(L), edge_index, edge_type, num_layers=2)
    score = lambda h, r, t: reshuffle_interaction(x[h], rel[2 * r], x[t]).item()  # noqa: E731
    assert score(0, 0, 1) == pytest.approx(0.0, abs=1e-3)       # a r1 b
    assert score(1, 1, 2) == pytest.approx(0.0, abs=1e-3)       # b r2 c
    assert score(0, 2, 2) == pytest.approx(0.0, abs=1e-3)       # a r3 c  (the rule)
    assert score(2, 2, 0) < -0.1                                # not entailed


def test_no_loop_aggregation_differs_from_loop_aggregation():
    th.manual_seed(1)
    rel = th.rand(2, L, L)
    loop = th.rand(L, L)
    x = random_features(3, 4, L, generator=th.Generator().manual_seed(2))
    args = (x, rel, loop, th.tensor([[0], [1]]), th.tensor([0]), 1)
    assert not th.allclose(propagate(*args, aggregation="max_self_and_selfLoop"),
                           propagate(*args, aggregation="max_self_noLoop"))
    with pytest.raises(ValueError):
        propagate(*args, aggregation="mean")


def _factory():
    triples = th.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 0]])
    return TriplesFactory(mapped_triples=triples, entity_to_id={str(i): i for i in range(4)},
                          relation_to_id={"r": 0, "s": 1}, create_inverse_triples=False)


def test_model_scores_and_only_relation_matrices_are_learned():
    model = Reshuffle(triples_factory=_factory(), l=4, k=5, num_layers=2, random_seed=0)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any("entity_representations" in n for n in trainable)
    assert any("relation_representations" in n for n in trainable)
    batch = th.tensor([[0, 0, 1], [1, 1, 2]])
    assert model.score_hrt(batch).shape == (2, 1)
    assert model.score_t(batch[:, :2]).shape == (2, 4)
    assert model.score_h(batch[:, 1:]).shape == (2, 4)
    assert (model.score_hrt(batch) <= 0).all()


def test_inductive_scoring_on_a_new_graph():
    """The model encodes a graph it has not seen (fresh entities, same relations)."""
    model = Reshuffle(triples_factory=_factory(), l=4, k=5, num_layers=2, random_seed=0)
    graph = th.tensor([[0, 0, 1], [1, 1, 2]])
    out = model.score_on_graph(th.tensor([[0, 0, 1], [2, 1, 0]]), graph, num_entities=3)
    assert out.shape == (2,) and (out <= 0).all()
