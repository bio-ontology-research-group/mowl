"""Inductive link prediction: the GraIL-split dataset, the sampled-negative regime, RESHUFFLE's
inductive scoring, and the registered benchmarks, published numbers and recipes. Offline: the
dataset is built from small fake release files."""

import os

import pytest
import torch as th

from mowl.benchmarks import get_benchmark, list_recipes, published_results
from mowl.datasets.builtin.inductive_kg import (InductiveKnowledgeGraphDataset, entity_iri,
                                                relation_iri)
from mowl.evaluation.inductive_ranking import (InductiveSampledRankingRegime,
                                               UnsupportedInductiveModel)

# training graph: people and what they work for; test graph: other entities, same relations
TRAIN = [("a", "works", "x"), ("b", "works", "x"), ("c", "works", "y"), ("a", "knows", "b"),
         ("b", "knows", "c"), ("d", "works", "y"), ("d", "knows", "a")]
VALID = [("c", "knows", "a")]
INFERENCE = [("p", "works", "u"), ("q", "works", "u"), ("p", "knows", "q"), ("r", "works", "v"),
             ("s", "knows", "p"), ("t", "works", "v"), ("t", "knows", "r"), ("u", "partof", "w"),
             ("v", "partof", "w")]
TEST = [("q", "knows", "r"), ("s", "works", "u")]


def _release(root, name="FB15k237", version="v1"):
    raw = os.path.join(root, "inductive_kg", f"{name}_{version}", "raw")
    os.makedirs(raw)
    for file, rows in {"train.txt": TRAIN, "valid.txt": VALID, "inference.txt": INFERENCE,
                       "test.txt": TEST}.items():
        with open(os.path.join(raw, file), "w") as f:
            f.writelines("\t".join(row) + "\n" for row in rows)


@pytest.fixture
def dataset(tmp_path):
    _release(str(tmp_path))
    return InductiveKnowledgeGraphDataset("FB15k237", "v1", data_root=str(tmp_path))


def _e(label):
    return entity_iri("FB15k237", "v1", label)


def _r(label):
    return relation_iri("FB15k237", "v1", label)


def _existentials(ontology):
    from org.semanticweb.owlapi.model import AxiomType
    out = set()
    for axiom in ontology.getAxioms(AxiomType.SUBCLASS_OF):
        sup = axiom.getSuperClass()
        out.add((str(axiom.getSubClass().toStringID()), str(sup.getProperty().toString())[1:-1],
                 str(sup.getFiller().toStringID())))
    return out


# -- dataset ------------------------------------------------------------------------------

def test_a_triple_is_the_axiom_head_subclass_of_some_r_tail(dataset):
    assert _existentials(dataset.ontology) == {(_e(h), _r(r), _e(t)) for h, r, t in TRAIN}
    assert _existentials(dataset.validation) == {(_e("c"), _r("knows"), _e("a"))}
    assert dataset.ontology.getClassesInSignature().size() == 4 + 2     # a b c d, x y


def test_the_test_graph_is_given_apart_and_its_entities_are_not_classes(dataset):
    assert dataset.testing is None
    classes = set(dataset.classes.as_str)
    assert _e("p") not in classes and _e("a") in classes
    graph = dataset.inference_graph
    assert len(graph.entities) == 8 and len(graph.triples) == len(INFERENCE)
    assert set(graph.relations) == {_r("works"), _r("knows"), _r("partof")}
    assert set(graph.iri_triples()) == {(_e(h), _r(r), _e(t)) for h, r, t in INFERENCE}
    assert dataset.test_triples[0] == (_e("q"), _r("knows"), _e("r"))
    index = graph.index_triples(dataset.test_triples)
    assert graph.entities[index[1, 0]] == _e("s") and graph.relations[index[1, 1]] == _r("works")


def test_the_dataset_is_cached_and_validates_its_arguments(dataset, tmp_path):
    for name in os.listdir(os.path.join(dataset.directory, "raw")):
        os.remove(os.path.join(dataset.directory, "raw", name))      # no network needed
    again = InductiveKnowledgeGraphDataset("FB15k237", "v1", data_root=str(tmp_path))
    assert len(again.inference_graph.triples) == len(INFERENCE)
    assert not [f for f in os.listdir(dataset.directory) if f.endswith(".part")]
    with pytest.raises(ValueError):
        InductiveKnowledgeGraphDataset("NotAGraph", "v1", data_root=str(tmp_path))
    with pytest.raises(ValueError):
        InductiveKnowledgeGraphDataset("WN18RR", "v5", data_root=str(tmp_path))


# -- regime -------------------------------------------------------------------------------

class _Oracle:
    """Scores over a graph: 0 for a triple of ``truth``, ``other`` otherwise."""

    def __init__(self, truth, other=1.0, true_cost=0.0):
        self.truth, self.other, self.true_cost = truth, other, true_cost
        self.graphs = []

    def inductive_costs(self, graph, rows):
        self.graphs.append(graph)
        return th.tensor([self.true_cost if (graph.entities[h], graph.relations[r],
                                             graph.entities[t]) in self.truth else self.other
                          for h, r, t in rows.tolist()])


def _truth(dataset):
    return set(dataset.inference_graph.iri_triples()) | set(dataset.test_triples)


def test_an_oracle_ranks_every_test_triple_first(dataset):
    regime = InductiveSampledRankingRegime(num_negatives=5)
    oracle = _Oracle(_truth(dataset))
    out = regime.evaluate(oracle, dataset)
    assert set(out) == {"all", "tail", "head"}
    assert set(out["all"]) == {"hits@1", "hits@3", "hits@10", "mrr", "mr"}
    assert all(v == 1.0 for v in (out["all"]["hits@1"], out["all"]["mrr"], out["all"]["mr"]))
    assert regime.info["queries"] == 2 and regime.info["directions"] == 2
    assert oracle.graphs[0] is dataset.inference_graph        # the model receives the graph


def test_a_model_that_ranks_the_truth_last_and_one_without_information(dataset):
    truth = _truth(dataset)
    regime = InductiveSampledRankingRegime(num_negatives=5)
    worst = regime.evaluate(_Oracle(truth, other=0.0, true_cost=1.0), dataset)["all"]
    assert worst["mr"] == 6.0 and worst["hits@1"] == 0.0 and worst["hits@10"] == 1.0
    flat = regime.evaluate(_Oracle(set(), other=1.0), dataset)["all"]
    assert flat["mr"] == 1 + 0.5 * 5                            # average rank of ties
    optimistic = InductiveSampledRankingRegime(num_negatives=5, ties="optimistic")
    assert optimistic.evaluate(_Oracle(set()), dataset)["all"]["mr"] == 1.0
    pessimistic = InductiveSampledRankingRegime(num_negatives=5, ties="pessimistic")
    assert pessimistic.evaluate(_Oracle(set()), dataset)["all"]["mr"] == 6.0


def test_negatives_are_corruptions_of_one_side_that_are_not_known_true(dataset):
    graph = dataset.inference_graph
    test_rows = graph.index_triples(dataset.test_triples)
    regime = InductiveSampledRankingRegime(num_negatives=4, seed=3)
    negatives = regime.sample_negatives(graph, test_rows)
    truth = {tuple(x) for x in th.cat([graph.triples, test_rows]).tolist()}
    for side, column in (("tail", 2), ("head", 0)):
        assert negatives[side].shape == (2, 4)
        for i, row in enumerate(test_rows.tolist()):
            assert len(set(negatives[side][i].tolist())) == 4          # without replacement
            for entity in negatives[side][i].tolist():
                candidate = list(row)
                candidate[column] = entity
                assert tuple(candidate) not in truth
    again = regime.sample_negatives(graph, test_rows)
    assert th.equal(again["tail"], negatives["tail"])                   # same seed
    other = InductiveSampledRankingRegime(num_negatives=4, seed=4)
    assert not th.equal(other.sample_negatives(graph, test_rows)["tail"], negatives["tail"])
    # fewer admissible entities than negatives: the pool is repeated, as PyKEEN does
    many = InductiveSampledRankingRegime(num_negatives=50).sample_negatives(graph, test_rows)
    assert many["tail"].shape == (2, 50)


def test_a_model_that_cannot_score_a_new_graph_is_refused_with_the_reason(dataset):
    class Transductive:
        class_embeddings = {}

    with pytest.raises(UnsupportedInductiveModel, match="transductive"):
        InductiveSampledRankingRegime().evaluate(Transductive(), dataset)
    with pytest.raises(ValueError, match="inference_graph"):
        InductiveSampledRankingRegime().evaluate(_Oracle(set()), object())


def test_the_seed_of_the_regime_reaches_a_model_that_takes_one(dataset):
    seen = []

    class Seeded(_Oracle):
        def inductive_costs(self, graph, rows, seed=None):
            seen.append(seed)
            return super().inductive_costs(graph, rows)

    InductiveSampledRankingRegime(num_negatives=3, seed=7).evaluate(
        Seeded(_truth(dataset)), dataset)
    assert set(seen) == {7}


def test_the_side_of_the_corrupted_entity_reaches_a_model_that_asks_for_it(dataset):
    seen = []

    class Sided(_Oracle):
        def inductive_costs(self, graph, rows, target=None):
            seen.append((target, len(rows)))
            return super().inductive_costs(graph, rows)

    InductiveSampledRankingRegime(num_negatives=3).evaluate(Sided(_truth(dataset)), dataset)
    assert [t for t, _ in seen] == ["tail", "head"] and {n for _, n in seen} == {8}


# -- RESHUFFLE ----------------------------------------------------------------------------

@pytest.fixture
def model(dataset):
    from mowl.models import Reshuffle
    th.manual_seed(0)
    return Reshuffle(dataset, l=6, k=8, num_layers=1, num_negs_per_pos=4, batch_size=16)


def test_reshuffle_scores_a_new_graph_and_runs_the_regime(model, dataset):
    model.train(epochs=2)
    regime = InductiveSampledRankingRegime(num_negatives=5)
    out = regime.evaluate(model, dataset)
    assert 0.0 <= out["all"]["hits@10"] <= 1.0 and out["all"]["mr"] >= 1.0
    assert regime.info["true_triples_without_a_score"] == 0     # all relations are known


def test_reshuffle_sees_the_facts_of_the_graph_it_is_given(dataset):
    """Without loop and with one layer, the tail of a fact is at least the head's features
    reshuffled by the relation. With the inverse matrices set to zero (so that nothing
    flows backwards) the only fact of a graph costs exactly 0 and its reverse does not."""
    from mowl.datasets.builtin.inductive_kg import InductiveGraph
    from mowl.models import Reshuffle
    th.manual_seed(0)
    model = Reshuffle(dataset, l=6, k=8, num_layers=1, num_negs_per_pos=4, batch_size=16,
                      kge_kwargs={"aggregation": "max_self_noLoop"})
    model.train(epochs=1)
    weights = model.module.relation_model.vectors.weight
    with th.no_grad():
        weights.view(weights.shape[0], 7, 6)[1::2, -1, :] = 1e4
    works = _r("works")
    graph = InductiveGraph((_e("m"), _e("n"), _e("o")), (works,),
                           th.tensor([[0, 0, 1]]))
    costs = model.inductive_costs(graph, th.tensor([[0, 0, 1], [1, 0, 0], [0, 0, 2]]))
    assert costs[0] == 0.0 and costs[1] > 0.0 and costs[2] > 0.0


def test_reshuffle_costs_inf_for_a_relation_it_never_saw(model, dataset):
    model.train(epochs=1)
    graph = dataset.inference_graph
    rows = th.tensor([[0, graph.relations.index(_r("partof")), 1],
                      [0, graph.relations.index(_r("works")), 1]])
    costs = model.inductive_costs(graph, rows)
    assert costs[0] == float("inf") and th.isfinite(costs[1])
    assert th.equal(costs, model.inductive_costs(graph, rows))       # same seed, same features


def test_with_inverse_triples_a_head_is_scored_from_the_tail_with_the_inverse_matrix(dataset):
    from mowl.models import Reshuffle
    from mowl.nn.graph.kge_interactions import reshuffle_interaction
    th.manual_seed(0)
    model = Reshuffle(dataset, l=6, k=8, num_layers=2, num_negs_per_pos=4, batch_size=16,
                      inverse_triples=True)
    assert model.triples_factory.create_inverse_triples
    model.train(epochs=2)
    graph = dataset.inference_graph
    rows = graph.index_triples(dataset.test_triples)
    heads, tails = model.inductive_costs(graph, rows, target="head"), \
        model.inductive_costs(graph, rows, target="tail")
    assert not th.allclose(heads, tails)
    x, rel = model.module.class_model.encode_graph(
        th.tensor([[graph.entities.index(h), model.triples_factory.relation_to_id[r],
                    graph.entities.index(t)]
                   for h, r, t in graph.iri_triples()
                   if r in model.triples_factory.relation_to_id]),
        len(graph.entities), seed=0, all_relations=True)
    r = model.triples_factory.relation_to_id[graph.relations[int(rows[0, 1])]]
    expected = -reshuffle_interaction(x[rows[0, 2]], rel[2 * r + 1], x[rows[0, 0]])
    assert th.isclose(heads[0], expected)
    # the factory trains 2R relations: the forward matrix of relation r is 2r
    costs = model.axiom_costs("gci2", th.tensor([[0, 0, 1]]))
    assert th.isfinite(costs).all()
    out = InductiveSampledRankingRegime(num_negatives=5).evaluate(model, dataset)
    assert out["head"]["mr"] >= 1.0


def test_a_transductive_embedding_is_refused_on_the_inductive_benchmark(dataset):
    from mowl.models import BoxE
    model = BoxE(dataset, embed_dim=4)
    model.train(epochs=1)
    with pytest.raises(UnsupportedInductiveModel, match="no vector"):
        InductiveSampledRankingRegime().evaluate(model, dataset)


def test_reshuffle_stops_early_on_the_validation_triples_of_the_training_graph(dataset):
    from mowl.models import Reshuffle
    th.manual_seed(0)
    model = Reshuffle(dataset, l=6, k=8, num_layers=1, num_negs_per_pos=4, batch_size=16,
                      early_stopping={"frequency": 1, "patience": 1, "relative_delta": 0.5})
    model.train(epochs=6)
    regime = model.training_regime
    assert regime.frequency == 1 and regime.patience == 1
    assert len(regime.history) >= 1
    assert len(regime._val) == len(VALID)
    assert Reshuffle(dataset, l=6, k=8, num_layers=1).training_regime is None


# -- registry -----------------------------------------------------------------------------

def test_benchmarks_published_numbers_and_recipes_are_registered():
    b = get_benchmark("kg-inductive-fb15k237-v1")
    assert b.evaluation_regime == "inductive-sampled-ranking"
    assert b.dataset_kwargs == {"name": "FB15k237", "version": "v1"}
    assert b.measures("link-prediction") and b.metrics[0] == "hits@10"
    assert len([k for k in (get_benchmark(f"kg-inductive-{d}-v{v}").key
                            for d in ("fb15k237", "wn18rr", "nell995")
                            for v in (1, 2, 3, 4))]) == 12
    assert b.assumption.value == "closed world over stated axioms"
    main = {r.benchmark: r.metrics["hits@10"] for r in published_results(method="reshuffle")
            if r.settings["variant"] == "reshuffle"}
    assert len(main) == 12 and main["kg-inductive-nell995-v1"] == 0.638
    assert main["kg-inductive-wn18rr-v3"] == 0.602 and main["kg-inductive-fb15k237-v4"] == 0.918
    square = {r.benchmark: r.metrics["hits@10"] for r in published_results(method="reshuffle")
              if r.settings["variant"] == "square"}
    assert square["kg-inductive-fb15k237-v4"] == 0.916
    assert square["kg-inductive-wn18rr-v3"] == 0.155
    nl = {r.benchmark: r.metrics["hits@10"] for r in published_results(method="reshuffle")
          if r.settings["variant"] == "no_loop"}
    assert nl["kg-inductive-nell995-v2"] == 0.738
    recipes = list_recipes(benchmark="kg-inductive-fb15k237-v3", method="reshuffle")
    assert {r.name for r in recipes} == {"reshuffle", "no-loop", "square"}
    main_recipe = [r for r in recipes if r.name == "reshuffle"][0]
    kw = main_recipe.model_kwargs
    assert (kw["num_layers"], kw["l"], kw["k"], kw["learning_rate"]) == (5, 25, 40, 0.005)
    assert kw["kge_kwargs"]["loss_kwargs"]["margin"] == 0.5 and main_recipe.epochs == 1000
    assert [r for r in recipes if r.name == "no-loop"][0].model_kwargs["kge_kwargs"][
        "aggregation"] == "max_self_noLoop"
    assert [r for r in recipes if r.name == "square"][0].model_kwargs["kge_kwargs"][
        "row_function"] == "square"
