"""Message passing and scoring of RESHUFFLE: closed path rules as matrix products."""
import pytest
import torch as th

import mowl
mowl.init_jvm("1g")
import mowl.datasets  # noqa: F401,E402
from mowl.representation.library.graph_interactions import (propagate, random_features,  # noqa: E402
                                            relation_matrices, reshuffle_interaction)

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


# ---- the stated properties of the paper (Pavlovic, Sallinger, Schockaert, KR 2025) ------

#: Rule graph for P = {r1(X,Y) & r2(Y,Z) -> r3(X,Z), r4(X,Y) & r5(Y,Z) -> r2(X,Z)}, built
#: for these tests after Definition 4 (R1-R4): nodes n0..n3, the edge r3 closes the path
#: r1;r2 and also r1;r4;r5, the edge r2 closes the path r4;r5.
RULE_GRAPH = {1: [(0, 1)], 2: [(1, 2)], 3: [(0, 2)], 4: [(1, 3)], 5: [(3, 2)]}
NODES = 4


def _rule_graph_model():
    """Relation matrices of Equation 7, in mOWL's row-vector convention: ``R[i, j] = 1`` iff
    the rule graph has an edge from node i to node j (``R = B^T``). Inverse relations and
    the loop get zero/identity matrices so that the message passing is exactly
    Equation 6 (self and forward edges, and the eq relation as the identity)."""
    rel = th.zeros(2 * 6, NODES, NODES)       # relations 0 (unused), 1..5; inverse at 2i+1
    for r, edges in RULE_GRAPH.items():
        for i, j in edges:
            rel[2 * r, i, j] = 1.0
    return rel


def _captured(x, rel, h, r, t):
    return bool(reshuffle_interaction(x[h], rel[2 * r], x[t]).item() == 0.0)


def _encode(graph, rel, layers, k, seed=0):
    n = 1 + max(max(h, t) for h, _, t in graph)
    x0 = random_features(n, k, NODES, density=0.5, generator=th.Generator().manual_seed(seed))
    triples = th.tensor(graph)
    return propagate(x0, rel, th.eye(NODES), th.stack([triples[:, 0], triples[:, 2]]),
                     triples[:, 1], layers)


def test_proposition_1_every_entailed_triple_is_captured():
    """P u G |= (a, r, b) implies the triple is captured, for any entity embedding that
    captures G (Proposition 1; the GNN of Equation 6 gives such an embedding)."""
    rel = _rule_graph_model()
    a, b, c, d = 0, 1, 2, 3
    graph = [(a, 1, b), (b, 4, c), (c, 5, d)]        # r1, r4, r5
    x = _encode(graph, rel, layers=4, k=64)
    for h, r, t in graph:
        assert _captured(x, rel, h, r, t)             # G itself
    assert _captured(x, rel, b, 2, d)                 # r4;r5 -> r2
    assert _captured(x, rel, a, 3, d)                 # r1;r2 -> r3, with r2 derived


def test_proposition_2_non_entailed_triples_are_not_captured_for_large_k():
    """With enough columns k, a triple that P u G does not entail is captured only with
    small probability (Proposition 2): for k = 256 none of the checked ones is."""
    rel = _rule_graph_model()
    a, b, c, d = 0, 1, 2, 3
    graph = [(a, 1, b), (b, 4, c), (c, 5, d)]
    x = _encode(graph, rel, layers=4, k=256)
    entailed = {(a, 1, b), (b, 4, c), (c, 5, d), (b, 2, d), (a, 3, d)}
    for h in range(4):
        for r in range(1, 6):
            for t in range(4):
                if h != t and (h, r, t) not in entailed:
                    assert not _captured(x, rel, h, r, t), (h, r, t)


def test_proposition_7_bounded_reasoning_needs_one_layer_per_inference_step():
    """With m layers the model captures what m rule applications derive (Propositions 7
    and 16): a(r1)b, b(r2)c entails (a, r3, c) after one rule step, which needs two
    layers because the information of a must first reach b and then c."""
    rel = _rule_graph_model()
    graph = [(0, 1, 1), (1, 2, 2)]
    one = _encode(graph, rel, layers=1, k=256)
    two = _encode(graph, rel, layers=2, k=256)
    assert not _captured(one, rel, 0, 3, 2)
    assert _captured(two, rel, 0, 3, 2)


def test_square_row_function_is_the_reshuffle2_ablation():
    raw = th.randn(L + 1, L)
    assert th.allclose(relation_matrices(raw, "square"), raw.square()[:-1])
    with pytest.raises(ValueError):
        relation_matrices(raw, "relu")
