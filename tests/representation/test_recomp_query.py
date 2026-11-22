"""Query2Box, BetaE, ConE and AConE are combinations of parts: read the combination back, swap
its parts, and check the catalogue against the parts.

The numbers against the direct implementations are in ``tests/baselines`` (``pairs_query.py``).
"""
import numpy as np
import pytest
import torch as th

from mowl.evaluation import LogicalQueryAnswering, NormalFormRankingRegime, sample_queries
from mowl.models import AConE, BetaE, ConE, Query2Box
from mowl.models.query.model import QueryEmbeddingModel
from mowl.models.query.sampler import ReleasedQueries, SampledQueries
from mowl.nn.query.base import UnsupportedOperator
from mowl.representation import (METHOD_KINDS, METHODS, REPRESENTATIONS, ClassModel, Geometry,
                                 Measure)
from mowl.representation.library import query_parts as P
from mowl.representation.library.query_methods import (
    AConEGeometry, BetaEGeometry, ComposedQueryGeometry, ConEGeometry, Query2BoxGeometry,
    betae, query2box)
from mowl.representation.library.query_objective import QueryObjective
from mowl.representation.library.boxrel_relations import TranslationBoxes
from mowl.representation.regions import ConstructorUnavailable
from mowl.training.schedules import StepDecay
from tests.datasetFactory import FamilyDataset

KEYS = {Query2Box: "query2box", BetaE: "betae", ConE: "cone_qe", AConE: "acone"}
COMMON = dict(embed_dim=8, learning_rate=3e-3, batch_size=24, negatives=4, seed=0,
              steps_per_epoch=3)


@pytest.fixture(scope="module")
def dataset():
    return FamilyDataset()


def build(cls, dataset, **kwargs):
    th.manual_seed(0)
    return cls(dataset, **{**COMMON, **kwargs})


def finite_after_one_epoch(model):
    model.train(epochs=1)
    assert np.isfinite(model.loss_history[-1])
    assert all(bool(th.isfinite(p).all()) for p in model.module.parameters())
    scores = model.score_query(("e", ("r",)), [(0, (1,)), (2, (3,))])
    assert scores.shape == (2, model.graph.num_entities)
    assert bool(th.isfinite(scores).all())
    costs = model.axiom_costs("gci0", th.tensor([[0, 1], [1, 2]]))
    assert bool(th.isfinite(costs).all())
    return model


# -- (a) the named combination, read back --------------------------------------------------

EXPECTED = {
    Query2Box: dict(module=Query2BoxGeometry, entities=P.PointBoxes,
                    relations=TranslationBoxes, intersection=P.BoxAttentionIntersection,
                    distance=P.BoxDistance, negation=None),
    BetaE: dict(module=BetaEGeometry, entities=P.BetaPoints, relations=P.BetaNetwork,
                intersection=P.BetaAttentionIntersection, distance=P.BetaDistance,
                negation=P.BetaInversion),
    ConE: dict(module=ConEGeometry, entities=P.ConePoints, relations=P.ConeNetwork,
               intersection=P.ConeAttentionIntersection, distance=P.ConeDistance,
               negation=P.ConeComplement),
    AConE: dict(module=AConEGeometry, entities=P.ConePoints, relations=P.ConeRotation,
                intersection=P.ConeAttentionIntersection, distance=P.ConeDistance,
                negation=P.ConeComplement),
}


@pytest.mark.parametrize("cls", list(KEYS))
def test_the_method_is_the_named_combination_of_parts(cls, dataset):
    model = build(cls, dataset)
    want = EXPECTED[cls]
    module = model.module
    assert isinstance(module, Geometry) and isinstance(module, ComposedQueryGeometry)
    assert type(module) is want["module"]
    assert type(module.entities) is want["entities"]
    assert type(module.relations) is want["relations"]
    assert type(module.intersection) is want["intersection"]
    assert type(module.distance_part) is want["distance"]
    assert (module.negation_part is None) if want["negation"] is None \
        else type(module.negation_part) is want["negation"]
    assert module.supports_negation is (want["negation"] is not None)
    # the method is the registered builder; the model reads its name
    key = KEYS[cls]
    assert model.method == key and METHOD_KINDS[key] == "queries"
    assert type(METHODS[key](5, 2, 1, embed_dim=4)) is want["module"]
    # the other parts of the regime
    assert isinstance(model.objective, QueryObjective) and model.objective.kind == "logsigmoid"
    assert isinstance(model.sampler, SampledQueries)
    assert model.parts["method"] == key and model.parts["sampler"] == "SampledQueries"
    assert not any("baselines" in type(p).__module__ for p in module.modules())


def test_cone_and_acone_differ_in_the_relation_model_only(dataset):
    cone, acone = build(ConE, dataset), build(AConE, dataset)
    assert type(cone.module.relations) is not type(acone.module.relations)
    for part in ("entities", "intersection", "distance_part", "negation_part", "measure"):
        assert type(getattr(cone.module, part)) is type(getattr(acone.module, part))
    assert cone.module.parts["relations"] == "neural network"
    assert acone.module.parts["relations"] == "rotation"


def test_the_module_class_of_old_is_the_part_built_module():
    """``mowl.nn.query`` still exports the four geometries, with their constructors."""
    from mowl.nn import query as nnq
    for name in ("Query2BoxGeometry", "BetaEGeometry", "ConEGeometry", "AConEGeometry"):
        assert issubclass(getattr(nnq, name), ComposedQueryGeometry)
    g = nnq.Query2BoxGeometry(6, 4, 3, gamma=4.0, center_weight=0.5, activation="none",
                              clamp_extent=True)
    assert g.cen == 0.5 and g.clamp_extent is True and g.num_entities == 6
    assert nnq.AConEGeometry(6, 4, 3).aperture_mode == "wrap"
    assert nnq.ConEGeometry(6, 4, 3, signed_negation=True).signed_negation is True


def test_a_seed_gives_the_initial_numbers_of_the_direct_model(dataset):
    """The parts create and initialise their tables and layers in the order of the reference
    codes, so the same seed starts the combination and the direct model from the same point."""
    import importlib
    from mowl.representation.library import query_methods as qm
    for cls, key in KEYS.items():
        direct = getattr(importlib.import_module(f"mowl.baselines.models.{key}"),
                         cls.__name__)
        th.manual_seed(0)
        old = direct(dataset, embed_dim=8, seed=0)
        th.manual_seed(0)
        new = cls(dataset, embed_dim=8, seed=0)
        shared = getattr(qm, key)(tables=old.module)
        a, b = shared.state_dict(), new.module.state_dict()
        assert sorted(a) == sorted(b)
        for name in a:
            assert th.equal(a[name], b[name]), (key, name)


# -- (b) swaps -----------------------------------------------------------------------------

def test_query2box_with_another_intersection(dataset):
    model = finite_after_one_epoch(build(Query2Box, dataset, intersection="exact"))
    assert isinstance(model.module.intersection, P.BoxExactIntersection)
    assert model.module.parts["conjunction"] == "exact"
    assert not list(model.module.intersection.parameters())


def test_query2box_with_another_distance(dataset):
    model = finite_after_one_epoch(build(Query2Box, dataset, distance="l2"))
    assert model.module.distance_part.norm == 2
    other = build(Query2Box, dataset)
    state = model.module.anchor(th.tensor([0, 1]))
    model.module.load_state_dict(other.module.state_dict())
    a = model.module.distance(model.module.project(state, th.tensor([3, 3])),
                              th.tensor([[2, 3], [4, 5]]))
    b = other.module.distance(other.module.project(state, th.tensor([3, 3])),
                              th.tensor([[2, 3], [4, 5]]))
    assert not th.allclose(a, b)                 # the distance is a part: it changes the score


def test_query2box_with_the_negation_of_cones_is_refused_clearly(dataset):
    with pytest.raises(ConstructorUnavailable, match="ConeComplement.*SectorCone.*Box"):
        build(Query2Box, dataset, negation="complement")
    with pytest.raises(ConstructorUnavailable, match="BetaInversion.*BetaRegion.*Box"):
        build(Query2Box, dataset, negation="inversion")
    with pytest.raises(ConstructorUnavailable, match="BetaDistance"):
        ComposedQueryGeometry(P.PointBoxes(5, 4), TranslationBoxes(4, 4, init_range=0.1),
                              P.BoxExactIntersection(), P.BetaDistance(),
                              Query2BoxGeometry(5, 4, 4).measure, dim=4)
    # without a negation the queries with one raise the error of the operator
    model = build(Query2Box, dataset)
    assert "2in" not in model.training_structures
    with pytest.raises(UnsupportedOperator):
        model.score_query((("e", ("r",)), ("e", ("r", "n"))), [((0, (1,)), (1, (1, -2)))])


def test_query2box_with_another_sampler_setting_and_objective(dataset):
    model = finite_after_one_epoch(build(
        Query2Box, dataset, batch_per_structure=True, structures=("1p", "2i"),
        objective=QueryObjective(kind="hinge", margin=2.0)))
    assert set(model.training_structures) == {"1p", "2i"}
    assert model.parts["objective"] == "hinge negative sampling"
    plain = finite_after_one_epoch(build(Query2Box, dataset, extension=False))
    assert plain.module.extension is False


def test_betae_with_another_intersection_and_distance(dataset):
    model = finite_after_one_epoch(build(BetaE, dataset, intersection="mean"))
    assert isinstance(model.module.intersection, P.BetaMeanIntersection)
    model = finite_after_one_epoch(build(BetaE, dataset, distance="reverse kl"))
    assert model.module.distance_part.reverse is True


def test_betae_with_self_adversarial_negatives_and_without_negation(dataset):
    model = finite_after_one_epoch(build(
        BetaE, dataset, objective=QueryObjective(adversarial_temperature=1.0)))
    assert model.objective.adversarial_temperature == 1.0
    model = finite_after_one_epoch(build(BetaE, dataset, negation=None))
    assert not model.module.supports_negation and "2in" not in model.training_structures
    assert model.parts["negation"] == "none"


def test_cone_with_the_rotation_and_with_another_negation_reading(dataset):
    model = finite_after_one_epoch(build(ConE, dataset, relations="rotation"))
    assert isinstance(model.module.relations, P.ConeRotation)
    model = finite_after_one_epoch(build(ConE, dataset, signed_negation=True))
    assert model.module.signed_negation is True
    model = finite_after_one_epoch(build(ConE, dataset, negation=None, center_weight=0.1))
    assert not model.module.supports_negation and model.module.cen == 0.1


def test_cone_with_a_learning_rate_schedule(dataset):
    model = build(ConE, dataset)
    model.training_regime = StepDecay(milestones=[1], gamma=0.5)
    model.train(epochs=2)
    assert model.training_regime.history == [3e-3, 1.5e-3]


def test_acone_with_the_clipped_aperture_and_another_distance_weight(dataset):
    model = finite_after_one_epoch(build(AConE, dataset, aperture="clip"))
    assert model.module.aperture_mode == "clip"
    model = finite_after_one_epoch(build(AConE, dataset, aperture="wrap", signed_negation=True,
                                         center_weight=0.05))
    assert model.module.signed_negation is True and model.module.cen == 0.05


def test_acone_with_the_network_relation_model_is_cone(dataset):
    model = finite_after_one_epoch(build(AConE, dataset, objective=QueryObjective(
        kind="hinge")))
    cone = build(QueryEmbeddingModel, dataset, method="cone_qe", gamma=20.0)
    assert type(cone.module) is ConEGeometry and cone.method is None
    rot = build(QueryEmbeddingModel, dataset, method="cone_qe", relations="rotation",
                gamma=20.0)
    assert type(rot.module) is AConEGeometry
    assert type(model.module.relations) is P.ConeRotation


def test_the_general_model_takes_any_builder(dataset):
    def my_method(nb_classes, nb_relations, nb_individuals=0, embed_dim=8, gamma=24.0, **kw):
        """Query2Box boxes with BetaE's attention-free mean? no: boxes with exact meet."""
        return query2box(nb_classes, nb_relations, nb_individuals, embed_dim, gamma,
                         intersection="exact", distance="l2", **kw)
    model = finite_after_one_epoch(build(QueryEmbeddingModel, dataset, method=my_method,
                                         gamma=24.0))
    assert model.module.parts["conjunction"] == "exact"
    with pytest.raises(ValueError, match="trained on axioms"):
        build(QueryEmbeddingModel, dataset, method="elbe")
    with pytest.raises(ValueError, match="Name the combination"):
        build(QueryEmbeddingModel, dataset)
    # an option the method has no part for
    with pytest.raises(TypeError):
        build(Query2Box, dataset, hidden=16)


def test_swapping_a_part_of_betae_in_the_builder_changes_the_geometry():
    plain, mean = betae(5, 2, 1, embed_dim=4), betae(5, 2, 1, embed_dim=4, intersection="mean")
    assert any("int_layer" in n for n, _ in plain.named_parameters())
    assert not any("int_layer" in n for n, _ in mean.named_parameters())


def test_the_objective_is_a_part_with_options():
    pos, neg = th.tensor([[2.0], [0.5]]), th.tensor([[-1.0, 0.0], [1.0, -3.0]])
    base = QueryObjective()
    p, n = base._terms(pos, neg)
    assert th.allclose(p, -th.nn.functional.logsigmoid(pos[:, 0]))
    assert th.allclose(n, -th.nn.functional.logsigmoid(-neg).mean(1))
    hinge = QueryObjective(kind="hinge", margin=1.0)
    p, n = hinge._terms(pos, neg)
    assert th.allclose(p, th.tensor([0.0, 0.5])) and th.allclose(n, th.tensor([0.5, 1.0]))
    adv = QueryObjective(adversarial_temperature=0.0)        # uniform weights: the mean
    assert th.allclose(adv._terms(pos, neg)[1], base._terms(pos, neg)[1])
    with pytest.raises(ValueError):
        QueryObjective(kind="mse")


def test_released_queries_are_a_sampler_that_the_model_takes(dataset):
    """The released regime on a stand-in dataset: weights per query reach the objective."""
    th.manual_seed(0)
    model = Query2Box(_with_queries(dataset), embed_dim=8, batch_size=6, negatives=3,
                      queries="released", steps_per_epoch=2, seed=0)
    assert isinstance(model.sampler, ReleasedQueries)
    assert set(model.released) == {"1p"} and model.training_structures == {}
    batch = model.sampler.updates()[0]
    sig, rows, pos, neg, weights = batch["1p"]
    assert weights.shape == (6,) and float(weights.min()) < float(weights.max())
    weighted = float(model.loss(batch))
    model.objective = QueryObjective(use_weights=False)
    assert weighted != float(model.loss(batch))
    model.train(epochs=1)
    assert np.isfinite(model.loss_history[-1])


def _with_queries(dataset):
    """A copy of the dataset with ``training_queries`` (a few released queries of 1p)."""
    import copy
    from mowl.datasets.builtin.query_sets import ReleasedTrainingQueries
    stand_in = copy.copy(dataset)

    def training_queries(entity_id, relation_id):
        rows = [[0, 1], [1, 1], [2, 1]]
        return {"1p": ReleasedTrainingQueries(("e", ("r",)), rows, [0, 1, 3, 7],
                                              [1, 2, 3, 4, 5, 6, 0])}
    stand_in.training_queries = training_queries
    return stand_in


# -- evaluation regimes ----------------------------------------------------------------------

def test_two_evaluation_regimes_rank_the_same_model(dataset):
    model = finite_after_one_epoch(build(ConE, dataset))
    known = model.triples
    complete = np.concatenate([known, [[0, 1, 2]]])
    queries = sample_queries(known, complete, structures=("1p",), num_queries=3, seed=0)
    result = LogicalQueryAnswering(queries).evaluate(model)
    assert 0 < result["average"]["mrr"] <= 1
    ranking = NormalFormRankingRegime(normal_forms=("gci0",), scoring="loss", combined=False)
    outcome = ranking.evaluate(model, dataset)
    assert outcome is not None


# -- (c) the catalogue and the parts -------------------------------------------------------

@pytest.mark.parametrize("cls", list(KEYS))
def test_the_catalogue_cards_agree_with_the_parts(cls, dataset):
    from mowl.catalog.registry import get_method
    key = KEYS[cls]
    card = get_method(key)
    model = build(cls, dataset)
    parts = model.module.parts
    # the representation card (vocabulary of parts)
    rep = REPRESENTATIONS[key]
    ClassModel(parts["classes"])
    Measure(parts["measure"])
    assert rep.has(parts["classes"]) and rep.has(parts["relations"])
    assert rep.negation.value == parts["negation"]
    # the method card
    assert card.implementation.endswith(cls.__name__)
    assert card.module.endswith("Geometry")
    assert {"subsumption", "instance-checking", "query-answering"} <= set(card.tasks)
    assert set(model.scored_forms) == {"gci0", "gci2", "class_assertion"}
    assert hasattr(model, "score_query")
    assert {"subclass_of", "class_assertion", "object_property_assertion"} <= set(card.axioms)
    # negation: the card says what the parts have
    assert ("negation" in card.logic.lower() or "¬" in card.notes) is True
    assert model.module.supports_negation is (key != "query2box")
    # the region of the card
    region = {"query2box": "box", "betae": "Beta", "cone_qe": "cone", "acone": "cone"}[key]
    assert region.lower() in card.region.lower()
