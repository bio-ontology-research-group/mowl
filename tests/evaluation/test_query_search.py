"""The query evaluators of mowl.evaluation.query_search on region families other than
GroundedEL: a crisp graph (sets of elements) and Query2Box-style boxes (points with one box
of translations per relation and one box per class)."""
from types import SimpleNamespace

import pytest
import torch as th

from mowl.evaluation.query_search import (EMPTY, TOP, Anchored, Chain, HopPlan, Join,
                                          MaxMinBeam, RegionCompiler, WitnessChain, max_min,
                                          read_batch, sparsify, top_witnesses)
from mowl.nn.el.structural import Regions
from mowl.representation.library.box_union_algebra import BoundedBoxUnions
from mowl.representation.library.element_box_family import ElementBoxFamily


# -- a crisp graph as a family ------------------------------------------------------------
class GraphFamily:
    """Elements of a graph; a relation holds of a pair iff it is an edge (degree TOP),
    otherwise EMPTY. Not a geometry: the beam only reads degrees."""

    def __init__(self, n, edges):
        self.n_elements, self.n_classes, self.embed_dim, self.parts_per_degree = n, 0, 1, 1
        self.dtype, self.device = th.float32, th.device("cpu")
        self.adj = th.zeros((max(r for r, _, _ in edges) + 1, n, n), dtype=th.bool)
        for r, a, b in edges:
            self.adj[r, a, b] = True

    def role_degree(self, ids, role_ids, targets, tau=None):
        held = self.adj[role_ids[:, None, None], ids[:, None, :], targets[:, :, None]]
        return th.where(held, th.full(held.shape, TOP), th.full(held.shape, EMPTY))


EDGES = [(0, 0, 1), (0, 0, 2), (0, 3, 2), (1, 1, 4), (1, 2, 5), (1, 3, 6), (0, 4, 6)]


def answers(scores):
    return [set(th.nonzero(row > EMPTY).view(-1).tolist()) for row in scores]


def test_beam_answers_graph_queries_exactly():
    family = GraphFamily(7, EDGES)
    beam = MaxMinBeam(k=7)
    one = beam.scores(family, read_batch(("e", ("r",)), [(0, (0,)), (3, (1,))]))
    assert answers(one) == [{1, 2}, {6}]
    two = beam.scores(family, read_batch(("e", ("r", "r")), [(0, (0, 1))]))
    assert answers(two) == [{4, 5}]
    both = beam.scores(family, read_batch((("e", ("r",)), ("e", ("r",))),
                                          [((0, (0,)), (3, (0,)))]))
    assert answers(both) == [{2}]
    either = beam.scores(family, read_batch((("e", ("r",)), ("e", ("r",)), ("u",)),
                                            [((0, (0,)), (3, (1,)), (-1,))]))
    assert answers(either) == [{1, 2, 6}]
    then = beam.scores(family, read_batch(((("e", ("r",)), ("e", ("r",))), ("r",)),
                                          [(((0, (0,)), (3, (0,))), (1,))]))
    assert answers(then) == [{5}]
    # a narrow beam keeps only the first witness: a subset of the answers
    narrow = MaxMinBeam(k=1).scores(family, read_batch(("e", ("r", "r")), [(0, (0, 1))]))
    assert answers(narrow)[0] <= {4, 5}


def test_witness_chain_on_the_graph_scores_targets_like_the_beam():
    family = GraphFamily(7, EDGES)
    node = read_batch(("e", ("r", "r")), [(0, (0, 1))])
    targets = th.tensor([[4, 5, 6]])
    out = WitnessChain(k=7).scores(family, family, node, targets, tau=0.1)
    dense = MaxMinBeam(k=7).scores(family, node)
    assert th.equal(out, dense[:, [4, 5, 6]])


def test_reading_a_batch_and_planning_a_hop():
    node = read_batch(((("e", ("r",)), ("e", ("r", "r"))), ("r",)),
                      [(((0, (1,)), (2, (3, 4))), (5,))])
    assert isinstance(node, Chain) and node.hops == ((5,),)
    assert isinstance(node.child, Join) and node.child.kind == "and"
    assert node.child.children[1] == Anchored((2,), ((3,), (4,)))
    with pytest.raises(ValueError):
        read_batch(("x",), [(0,)])
    plan = HopPlan((0, 7, 3), "cpu", isa=7, isa_inv=3, first=False)
    assert plan.kinds == ["role", "isa", "isa_inv"] and plan.only is None
    assert plan.role_ids.tolist() == [0, 0, 0] and plan.rows("isa").tolist() == [0, 1, 0]
    assert HopPlan((7, 7), "cpu", isa=7).only == "isa"


def test_max_min_and_witnesses():
    scores = th.tensor([[3.0, EMPTY, 1.0], [EMPTY, EMPTY, EMPTY]])
    ids, best, valid = top_witnesses(scores, 3)
    assert ids.shape == (2, 2) and valid.tolist() == [[True, True], [False, False]]
    degree = th.tensor([[[2.0, 5.0]], [[1.0, 1.0]]])
    assert max_min(best, valid, degree).tolist() == [[2.0], [EMPTY]]
    assert sparsify(scores, 1).tolist()[0] == [3.0, EMPTY, EMPTY]


# -- Query2Box-style boxes as a family ----------------------------------------------------
class PlainBoxes:
    """Entities are points; a relation is one box of translations (the projection of
    Query2Box: centres add, widths add); a class is one box. No top, no bottom."""

    def __init__(self, n_classes=2, n_entities=6, n_rels=2, dim=2, seed=0):
        g = th.Generator().manual_seed(seed)
        self.nb_classes, self.nb_elements, self.embed_dim = n_classes, n_entities, dim
        self.points = th.rand((n_entities, dim), generator=g) * 2 - 1
        self.class_cen = th.rand((n_classes, 1, dim), generator=g) - 0.5
        self.class_w = th.full((n_classes, 1, dim), 0.6)
        self.rel_cen = th.rand((n_rels, 1, dim), generator=g) - 0.5
        self.rel_w = th.full((n_rels, 1, dim), 0.4)

    def positions(self, ids=None):
        return self.points if ids is None else self.points[ids]

    def class_regions(self, ids):
        return Regions(cen=self.class_cen[ids], w=self.class_w[ids],
                       active=th.ones((len(ids), 1), dtype=th.bool))

    def translation_regions(self, ids):
        return Regions(cen=self.rel_cen[ids], w=self.rel_w[ids],
                       active=th.ones((len(ids), 1), dtype=th.bool))


def test_compiler_beam_and_witness_on_plain_boxes():
    geometry = PlainBoxes()
    family = ElementBoxFamily(geometry)
    queries = [(0, (0,)), (3, (1,))]
    node = read_batch(("e", ("r",)), queries)
    region = RegionCompiler().compile(family, node)
    # the Query2Box projection of a point: the box of the relation moved to the point
    assert th.allclose(region.cen[:, 0], geometry.points[[0, 3]] + geometry.rel_cen[[0, 1], 0])
    assert th.allclose(region.w[:, 0], geometry.rel_w[[0, 1], 0])
    boxes = family.entity_scores(region)
    beam = MaxMinBeam(k=6).scores(ElementBoxFamily(geometry, geometry.points), node)
    assert th.equal(beam, boxes)
    targets = th.arange(6).repeat(2, 1)
    soft = WitnessChain(k=6).scores(family, family, node, targets, tau=1e-4)
    assert th.allclose(soft, boxes, atol=1e-3)
    # an intersection with a class box (isa as relation 0 would need classes: use the
    # algebra directly) is inside both
    alg = BoundedBoxUnions()
    both = alg.intersect(region, family.class_region(th.tensor([0, 0])))
    inside = family.entity_scores(both) > 0
    assert bool((inside <= (boxes > 0)).all())
