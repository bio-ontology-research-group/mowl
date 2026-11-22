"""The general parts that the graph, lattice, box-code and walk methods are made of.

Each part is exercised on its own (stub modules, known inputs and outputs) and in a
combination other than the method it came from. The comparison with the direct
implementations is in ``tests/baselines``."""
import pytest
import torch as th

import mowl
mowl.init_jvm("1g")
pytest.importorskip("pykeen")

from mowl.evaluation.inductive_triples import InductiveTripleCosts  # noqa: E402
from mowl.evaluation.lattice_ranking import (order_mean_reciprocal_rank,  # noqa: E402
                                             score_columns)
from mowl.evaluation.triple_axioms import AxiomTriples  # noqa: E402
from mowl.representation.library import graph_interactions  # noqa: E402
from mowl.representation.library.graph_projection import (  # noqa: E402
    PROJECTIONS, ProjectionView, projection, projection_view)
from mowl.representation.library.hierarchy_classes import CodedBoxes, OrderPoints  # noqa: E402
from mowl.representation.library.hierarchy_geometry import (  # noqa: E402
    ExistentialElements, LatticeGeometry)
from mowl.representation.library.hierarchy_measures import (  # noqa: E402
    LogVolumeRatio, OrderViolation)
from mowl.representation.library.other_graph import (SelfAdversarial,  # noqa: E402
                                                     UniformCorruption)
from mowl.training.triples import (carry_rows, extend_embeddings,  # noqa: E402
                                   positive_triples, train_on_triples)
from mowl.utils.device import default_device  # noqa: E402
from mowl.utils.random import seed_generators  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


# -- the training of triples ------------------------------------------------------------
def test_positive_triples_add_the_inverse_triples():
    triples = th.tensor([[0, 1, 2], [3, 0, 4]])
    assert th.equal(positive_triples(triples, False), triples)
    both = positive_triples(triples, True)
    assert both.tolist() == [[0, 2, 2], [3, 0, 4], [2, 3, 0], [4, 1, 3]]


def test_train_on_triples_trains_any_triple_geometry_without_a_model_class():
    from mowl.representation import methods as M
    th.manual_seed(0)
    module = M.METHODS["graph_kge"](6, 2, 0, embed_dim=4, subclass_id=0,
                                    sampler=UniformCorruption(2))
    triples = th.tensor([[0, 1, 2], [2, 1, 3], [4, 0, 5], [1, 1, 5]])
    before = [p.clone() for p in module.parameters()]
    train_on_triples(module, triples, 6, 2, th.optim.Adam, 0.1, batch_size=2, epochs=2)
    assert any(not th.equal(a, b) for a, b in zip(before, module.parameters()))
    assert all(th.isfinite(p).all() for p in module.parameters())


def test_train_on_triples_with_another_objective_and_sampler():
    from mowl.representation import methods as M
    th.manual_seed(0)
    module = M.METHODS["boxe"](6, 2, 0, embed_dim=4, subclass_id=0,
                               sampler=UniformCorruption(3, sides=("t",)),
                               loss=SelfAdversarial(margin=1.0))
    triples = th.tensor([[0, 1, 2], [2, 1, 3], [4, 0, 5]])
    train_on_triples(module, triples, 6, 2, th.optim.SGD, 0.01, batch_size=3, epochs=1)
    assert th.isfinite(module.triple_cost(triples[:, 0], triples[:, 1], triples[:, 2])).all()


def test_an_empty_graph_has_no_batch():
    from mowl.representation import methods as M
    module = M.METHODS["graph_kge"](3, 1, 0, embed_dim=2, subclass_id=0)
    with pytest.raises(ValueError, match="no training batch"):
        train_on_triples(module, th.zeros(0, 3, dtype=th.long), 3, 1, th.optim.Adam, 0.1, 2, 1)


def test_extend_embeddings_keeps_known_rows_and_draws_the_others():
    previous = [{"a": [1.0, 2.0]}, {"b": [3.0, 4.0]}]
    out = extend_embeddings(previous, ["b", "x", "a"], 2)
    assert out.shape == (3, 2)
    assert out[0].tolist() == [3.0, 4.0] and out[2].tolist() == [1.0, 2.0]
    assert out[1].tolist() != [0.0, 0.0]


def test_carry_rows_copies_the_rows_of_shared_keys():
    old, new = th.nn.Embedding(2, 3), th.nn.Embedding(3, 3)
    carry_rows(old, new, {"x": 0, "y": 1}, {"y": 0, "z": 2}, False)
    assert th.equal(new.weight[0], old.weight[1])
    carry_rows(old, new, {"x": 0}, {"x": 1}, False)
    assert th.equal(new.weight[1], old.weight[0])


# -- scoring axioms as triples --------------------------------------------------------------
class _Sum(th.nn.Module):
    def triple_cost(self, heads, relations, tails):
        return (heads * 100 + relations * 10 + tails).float().reshape(-1, 1)


def _axioms(inverse=False, type_label="http://type"):
    return AxiomTriples(
        {"C0": 0, "C1": 1}, {"a": 0}, {"r": 0}, (2, 1, 1),
        {"C0": 3, "C1": 4, "a": 5}, {"http://subclassof": 0, "r": 1, type_label: 2},
        "http://subclassof", "http://type", inverse)


def test_axiom_triples_read_each_form_as_its_edge():
    axioms, module = _axioms(), _Sum()
    assert axioms.costs(module, "gci0", th.tensor([[0, 1]])).tolist() == [304.0]
    assert axioms.costs(module, "gci2", th.tensor([[1, 0, 0]])).tolist() == [413.0]
    assert axioms.costs(module, "class_assertion", th.tensor([[0, 1]])).tolist() == [524.0]
    assert axioms.costs(module, "object_property_assertion",
                        th.tensor([[0, 0, 0]])).tolist() == [515.0]


def test_axiom_triples_without_the_edge_label_cost_infinity_and_inverse_relations_double():
    axioms = AxiomTriples({"C0": 0}, {"a": 0}, {"r": 0}, (1, 1, 1), {"C0": 0, "a": 1}, {},
                          "http://subclassof", "http://type")
    assert axioms.costs(_Sum(), "gci2", th.tensor([[0, 0, 0]])).isinf().all()
    inverse = _axioms(inverse=True)
    assert inverse.costs(_Sum(), "gci2", th.tensor([[1, 0, 0]])).tolist() == [423.0]


def test_axiom_triples_refuse_other_forms():
    from mowl.evaluation.model_scoring import UnsupportedAxiomForm
    with pytest.raises(UnsupportedAxiomForm):
        _axioms().costs(_Sum(), "gci1", th.tensor([[0, 0, 0]]))


# -- inductive scoring ----------------------------------------------------------------------
class _Relations:
    def __init__(self):
        self.seen = []

    def cost(self, ids, head, tail):
        self.seen.append(ids.tolist())
        return (head.centre.sum(1) - tail.centre.sum(1)).reshape(-1, 1)


class _Encoder:
    def encode_graph(self, facts, n, seed=0, all_relations=False):
        return th.arange(n * 2, dtype=th.float).reshape(n, 1, 2), None


class _Stub(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.class_model, self.relation_model = _Encoder(), _Relations()


class _Graph:
    relations = ["r", "unknown"]
    entities = ["x", "y", "z"]
    triples = th.tensor([[0, 0, 1], [1, 1, 2]])


def test_inductive_costs_use_the_inverse_relation_for_heads_and_skip_unknown_relations():
    module = _Stub()
    costs = InductiveTripleCosts(module, {"r": 0}, inverse=True)
    rows = th.tensor([[0, 0, 1], [0, 1, 2]])
    tail = costs(_Graph(), rows, target="tail")
    assert th.isfinite(tail[0]) and th.isinf(tail[1])
    assert module.relation_model.seen[-1] == [0]
    costs(_Graph(), rows, target="head")
    assert module.relation_model.seen[-1] == [1]
    with pytest.raises(ValueError):
        costs(_Graph(), rows, target="middle")


# -- the lattice ------------------------------------------------------------------------------
class _Order(th.nn.Module):
    def order_violation(self, a, b):
        return (b - a).abs().float()

    def forward(self, rows, form):
        return rows.sum(1).float() + (form == "gci2")


def test_score_columns_stacks_the_columns_into_rows():
    assert score_columns(_Order(), "gci2", ([1, 2], [3, 4], [5, 6])).tolist() == [10.0, 13.0]


def test_the_reciprocal_rank_is_one_over_the_rank_with_ties_averaged():
    module = _Order()
    queries = [(0, 1, th.tensor([0, 1, 2, 3])), (0, 2, th.tensor([0, 1, 2, 3]))]
    # head 0: candidates 1, 2, 3 cost 1, 2, 3: ranks 1 and 2
    assert order_mean_reciprocal_rank(module, queries) == pytest.approx((1 + 0.5) / 2)
    tie = [(0, 1, th.tensor([1, 1, 2]))]
    assert order_mean_reciprocal_rank(module, tie) == pytest.approx(1 / 1.5)
    assert order_mean_reciprocal_rank(module, []) == 0.0


class _Elements(ExistentialElements, LatticeGeometry):
    """A lattice of another method: order embeddings in the reversed order, with elements
    for existential restrictions."""

    def __init__(self, n):
        LatticeGeometry.__init__(self, OrderPoints(n, 3, True, "uniform"),
                                 OrderViolation("reversed", 2))
        self._init_existentials(n)


def test_existential_elements_are_looked_up_in_any_lattice_geometry():
    module = _Elements(6)
    module.set_existentials([0, 1], [2, 2], [4, 5])
    found = module.existential_ids(th.tensor([0, 1, 0]), th.tensor([2, 2, 3]))
    assert found.tolist() == [4, 5, -1]
    assert {"existential_keys", "existential_nodes"} <= set(module.state_dict())
    assert module.parts["classes"]
    empty = _Elements(3)
    assert empty.existential_ids(th.tensor([0]), th.tensor([1])).tolist() == [-1]


# -- boxes with codes -----------------------------------------------------------------------
def test_coded_boxes_give_codes_vectors_and_probabilities():
    th.manual_seed(0)
    for variant, width in (("gbox", 8), ("gbc", 8), ("vbc", 12)):
        boxes = CodedBoxes(5, 4, variant=variant, num_codes=2, code_dims=3)
        ids = th.tensor([0, 3])
        assert boxes.vectors(ids).shape == (2, width)
        codes = boxes.trainable_codes()
        assert (codes is None) == (variant == "gbox")
        if codes is not None:
            assert ((codes > 0) & (codes < 1)).all()
    assert LogVolumeRatio.probability(th.tensor([0.0, 1.0])).tolist() == pytest.approx(
        [1.0, 0.36787944])


# -- the projection as one part ---------------------------------------------------------------
def test_one_projection_part_serves_the_walks_and_the_graph_models():
    from mowl.projection import (DL2VecProjector, GDAProjector, TaxonomyProjector,
                                 TaxonomyWithRelationsProjector)
    from mowl.representation.library import walks_views
    assert walks_views.ProjectionView is ProjectionView
    assert walks_views.projection_view is projection_view
    assert walks_views.PROJECTIONS is PROJECTIONS
    # the short names of the graph models and the vocabulary names of the walks are the same
    for short, vocabulary in (("dl2vec", "DL2Vec projection"), ("gda", "GDA projection"),
                              ("owl2vec_star", "OWL2Vec* projection")):
        assert PROJECTIONS[short] is PROJECTIONS[vocabulary]
        assert type(projection(short)) is type(projection(vocabulary))
    assert isinstance(ProjectionView("gda").projector, GDAProjector)
    assert isinstance(ProjectionView(DL2VecProjector).projector, DL2VecProjector)
    assert isinstance(ProjectionView(default="taxonomy").projector, TaxonomyProjector)
    instance = TaxonomyProjector()
    assert ProjectionView(instance).projector is instance
    view = projection_view("taxonomy with relations", bidirectional_taxonomy=True)
    assert isinstance(view.projector, TaxonomyWithRelationsProjector)
    assert view.part == "taxonomy with relations"
    with pytest.raises(ValueError):
        projection("no such projection")


def test_the_walk_model_and_the_graph_models_read_the_graph_through_the_same_view():
    from mowl.datasets.builtin import FamilyDataset
    from mowl.models import GraphKGEModel, RandomWalkPlusW2VModel
    from mowl.projection import DL2VecProjector, GDAProjector
    dataset = FamilyDataset()
    kge = GraphKGEModel(dataset, embed_dim=4, projector="dl2vec")
    assert isinstance(kge.projector, DL2VecProjector)
    walks = RandomWalkPlusW2VModel(dataset)
    walks.set_projector(GDAProjector())
    assert isinstance(walks.module.source.view, ProjectionView)
    assert isinstance(walks.module.source.view.projector, GDAProjector)


# -- small general helpers ---------------------------------------------------------------------
def test_the_interactions_live_with_the_parts_and_the_old_path_still_works():
    from mowl.nn.graph import kge_interactions
    assert kge_interactions.ExpressivEInteraction is graph_interactions.ExpressivEInteraction
    assert kge_interactions.reshuffle_interaction is graph_interactions.reshuffle_interaction
    h, t = th.rand(2, 2, 3), th.rand(2, 2, 3)
    r = th.rand(2, 3, 3)
    assert graph_interactions.reshuffle_interaction(h, r, t).shape == (2,)


def test_seed_generators_make_the_global_draws_repeat_and_the_device_is_a_device():
    seed_generators(3)
    a = (th.rand(2), th.randint(0, 100, (1,)))
    seed_generators(3)
    b = (th.rand(2), th.randint(0, 100, (1,)))
    assert th.equal(a[0], b[0]) and th.equal(a[1], b[1])
    assert isinstance(default_device(), th.device)


# -- the knowledge graph model as seen by PyKEEN-style callers ----------------------------------
def test_the_scorer_of_a_graph_model_reads_scores_tables_and_modes_from_the_geometry():
    from mowl.datasets.builtin import FamilyDataset
    from mowl.models import GraphKGEModel
    th.manual_seed(0)
    model = GraphKGEModel(FamilyDataset(), method="boxe", embed_dim=4)
    kge, module = model.kge_method, model.module
    triples = model.training_triples()[:5]
    expected = -module.triple_cost(triples[:, 0], triples[:, 1], triples[:, 2])
    assert th.equal(kge.score_hrt(triples), expected)
    assert list(kge.parameters()) == list(module.parameters())
    assert kge.interaction is module.measure
    # BoxE: entities have a base and a bump table, relations six
    assert len(kge.entity_representations) == 2 and len(kge.relation_representations) == 6
    ids = th.tensor([0, 1])
    assert th.equal(kge.entity_representations[0](indices=ids), module.class_model.centres(ids))
    assert kge.entity_representations[0]().shape == module.class_model.centres.weight.shape
    assert kge.eval() is kge and not module.training
    assert kge.train() is kge and module.training
    kge.zero_grad()
    transe = GraphKGEModel(FamilyDataset(), method="graph_kge", embed_dim=4).kge_method
    assert len(transe.entity_representations) == 1 and len(transe.relation_representations) == 1


def test_best_f1_finds_the_threshold_and_keeps_its_old_import_path():
    from mowl.evaluation.reconstruction import best_f1
    from mowl.models.bcbox.common import best_f1 as old
    assert old is best_f1
    energies = th.tensor([0.1, 0.2, 0.9, 1.0, float("inf")])
    truth = th.tensor([True, True, False, False, True])
    f1, threshold = best_f1(energies, truth)
    assert f1 == pytest.approx(0.8) and 0.2 < threshold < 0.9
    assert best_f1(energies, th.zeros(5, dtype=th.bool))[0] == 0.0
