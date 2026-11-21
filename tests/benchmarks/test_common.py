"""The joint benchmarks: one ranking regime for every model family."""

import os

import pytest
import torch as th

import mowl  # noqa: F401
from mowl.benchmarks import (Benchmark, Task, get_benchmark, list_benchmarks,
                             methods_for, supports)
from mowl.benchmarks.comparison import compare_methods, methods_table, runnable_methods
from mowl.benchmarks.entries._refs import JACKERMEIER2024
from mowl.evaluation.model_scoring import (PairClassifierHead, UnsupportedAxiomForm,
                                           UnsupportedScoring, axiom_scoring_of,
                                           resolve_model_scorer)
from tests.benchmarks.tiny import scratch_dir, tiny_dataset

METRICS = ("f_mrr", "f_hits@1", "f_hits@10", "f_hits@100", "f_mr", "f_med", "f_auc")


def _bench(gci_name="gci0", **protocol):
    return Benchmark(
        key=f"tiny-common-{gci_name}", name="tiny", task=Task.AXIOM_PREDICTION,
        description="tiny", dataset="tests.benchmarks.tiny.tiny_dataset",
        protocol="mowl.benchmarks.protocols.bindings.CommonRankingProtocol",
        protocol_kwargs={"gci_name": gci_name, **protocol}, metrics=METRICS,
        higher_is_better={m: m not in ("f_mr", "f_med") for m in METRICS},
        references=(JACKERMEIER2024,), evaluation_regime="common-filtered-ranking",
        completion_tasks=("subsumption-prediction",) if gci_name == "gci0"
        else ("existential-prediction",))


def test_joint_benchmarks_are_registered_and_linked():
    keys = {b.key for b in list_benchmarks()}
    for onto in ("galen", "go", "anatomy"):
        for kind in ("subsumption-prediction", "existential-prediction",
                     "subsumption-inference"):
            assert f"common-{onto}-{kind}" in keys
    assert {"common-yeast-ppi", "common-human-ppi"} <= keys
    inference = get_benchmark("common-go-subsumption-inference")
    assert inference.target.value == "deductive inference"
    assert get_benchmark("common-go-subsumption-prediction").target.value == \
        "inductive prediction"
    ppi = get_benchmark("common-yeast-ppi")
    assert ppi.completion_tasks == ("existential-prediction",)
    assert ppi.encodes == ("link-prediction",)
    for b in list_benchmarks():
        if b.key.startswith("common-"):
            assert b.evaluation_regime == "common-filtered-ranking"
            assert b.metrics == METRICS


def test_every_family_is_matched_to_the_joint_subsumption_benchmark():
    keys = {m.key for m in methods_for("common-go-subsumption-prediction")}
    # geometric EL, lattice, ALC, quantum logic, graph, language model, Word2Vec
    assert {"elem", "box2el", "cate", "falcon", "alcone", "e2r", "bcbox", "graph_kge",
            "hit", "ont", "dl2vec", "owl2vec_star", "onto2vec", "opa2vec"} <= keys
    # A taxonomy or name-only method does not rank fillers of ∃r.?.
    assert not supports("bcbox", "common-go-existential-prediction")
    assert not supports("e2r", "common-go-existential-prediction")
    assert supports("dl2vec", "common-go-existential-prediction")
    run, skipped = runnable_methods("common-go-subsumption-prediction")
    assert "elem" in run and "dl2vec" in run
    assert "exact construction" in skipped["canonical_model"]
    assert "gda_projection" in skipped


def test_learnable_methods_with_tasks_declare_how_they_score():
    """A method that the catalogue matches to ranking benchmarks must be rankable."""
    from mowl.catalog import list_methods
    from mowl.catalog.card import Status
    missing = []
    for card in list_methods():
        if card.status is not Status.LEARNABLE or not (card.tasks or card.head_tasks):
            continue
        if card.factory:
            module, _, attr = card.factory.rpartition(".")
            assert callable(getattr(__import__(module, fromlist=[attr]), attr)), card.key
            continue
        try:
            cls = card.load()
        except ImportError:        # optional dependency not installed
            continue
        if getattr(cls, "axiom_scoring", None) is None:
            missing.append(card.key)
    assert not missing, missing


@pytest.mark.parametrize("method,kwargs,scoring", [
    ("elem", {"embed_dim": 8}, "centre_distance"),
    ("octagon", {"embed_dim": 8}, "loss"),
    ("order_embeddings", {"embed_dim": 8}, "loss"),
    ("box_lattice", {"embed_dim": 8}, "loss"),
    ("gumbel_box", {"embed_dim": 8}, "loss"),
    ("cate", {"embed_dim": 8}, "loss"),
    ("alcone", {}, "model"),
    ("falcon", {"embed_dim": 8}, "model"),
    ("e2r", {"embed_dim": 8}, "model"),
    ("bcbox", {"embed_dim": 8}, "evaluation_model"),
    ("graph_kge", {"embed_dim": 8}, "model"),
    ("dl2vec", {"embed_dim": 8}, "pair_classifier"),
    ("onto2vec", {"embed_dim": 8}, "pair_classifier"),
])
def test_one_regime_ranks_every_family(method, kwargs, scoring):
    outcome = compare_methods(_bench("gci0"), methods=[method], epochs=2, seeds=(0,),
                              model_kwargs=kwargs, dataset=tiny_dataset(),
                              workdir=scratch_dir(), stop_on_error=True)
    (result,) = outcome.results
    metrics = result.metrics["all"]
    assert set(metrics) == set(METRICS)
    assert 0.0 < metrics["f_mrr"] <= 1.0
    assert result.protocol_info["scoring"] == scoring
    assert result.protocol_info["queries"] == 2
    assert result.evaluation_regime == "common-filtered-ranking"


def test_existential_candidates_and_unsupported_forms():
    ds = tiny_dataset()
    outcome = compare_methods(_bench("gci2"), methods=["elem", "falcon", "graph_kge",
                                                       "owl2vec_star", "bcbox"],
                              epochs=2, model_kwargs={"embed_dim": 8}, dataset=ds,
                              workdir=scratch_dir())
    done = {r.method for r in outcome.results}
    assert {"elem", "falcon", "graph_kge", "owl2vec_star"} <= done
    # bcbox is skipped by the data (no existential restrictions), not by a crash.
    assert "bcbox" in outcome.skipped and "∃r.C" in outcome.skipped["bcbox"]
    table = methods_table(outcome.results, benchmark=_bench("gci2"))
    assert "f_mrr" in table and "pair_classifier" in table and "elem" in table


def test_results_are_reused_from_the_output_directory():
    ds, out = tiny_dataset(), scratch_dir()
    first = compare_methods(_bench("gci0"), methods=["elem"], epochs=2, dataset=ds,
                            model_kwargs={"embed_dim": 8}, out_dir=out,
                            workdir=scratch_dir())
    assert os.path.exists(os.path.join(out, "tiny-common-gci0.elem.s0.json"))
    again = compare_methods(_bench("gci0"), methods=["elem"], epochs=999, dataset=ds,
                            model_kwargs={"embed_dim": 8}, out_dir=out)
    assert again.results[0].metrics == first.results[0].metrics
    assert again.results[0].epochs == 2


def test_failures_are_recorded_not_raised():
    outcome = compare_methods(_bench("gci0"), methods=["elem"], epochs=1,
                              model_kwargs={"no_such_argument": 1},
                              dataset=tiny_dataset(), workdir=scratch_dir())
    assert not outcome.results
    assert ("elem", 0) in outcome.failed


# --- scoring layer -----------------------------------------------------------------------

class _Vectors:
    """A vector-only model: 20 classes on a line, C_i ⊑ C_j stated for j = i + 1."""

    axiom_scoring = "vectors"

    def __init__(self):
        self.n = 20
        generator = th.Generator().manual_seed(0)
        self._emb = th.randn(self.n, 16, generator=generator)
        self.class_index_dict = {f"c{i}": i for i in range(self.n)}
        self.class_embeddings = {f"c{i}": self._emb[i].numpy() for i in range(self.n)}


def test_pair_classifier_head_learns_an_asymmetric_relation(monkeypatch):
    model = _Vectors()
    pairs = th.tensor([[i, i + 1] for i in range(model.n - 1)])
    head = PairClassifierHead(epochs=300, negatives=4, seed=0)
    monkeypatch.setattr(head, "_training_rows", lambda m: {"gci0": pairs})
    forward = head(model, "gci0", pairs)
    backward = head(model, "gci0", pairs.flip(1))
    # Stated pairs cost less than the reversed ones: cosine similarity could not tell.
    assert (forward < backward).float().mean() > 0.9
    assert head.history[("gci0", None)][-1] < head.history[("gci0", None)][0]
    # Deterministic for a seed.
    other = PairClassifierHead(epochs=300, negatives=4, seed=0)
    monkeypatch.setattr(other, "_training_rows", lambda m: {"gci0": pairs})
    assert th.allclose(other(model, "gci0", pairs), forward)
    with pytest.raises(UnsupportedAxiomForm):
        head(model, "gci1", th.zeros((1, 3), dtype=th.long))


def test_a_model_must_declare_its_scoring():
    class Plain:
        pass
    with pytest.raises(UnsupportedScoring):
        axiom_scoring_of(Plain())
    assert resolve_model_scorer(_Vectors())[1] == "pair_classifier"


def test_model_costs_agree_with_the_models_own_scores():
    from mowl.models import ALConeModel
    from mowl.models.e2r import E2R
    ds = tiny_dataset()
    cone = ALConeModel(ds)
    cone.train(2)
    index = cone.class_index_dict
    names = [n for n in index if n.endswith(("#A", "#B", "#C"))]
    rows = th.tensor([[index[a], index[b]] for a in names for b in names if a != b])
    costs = cone.axiom_costs("gci0", rows)
    expected = [-cone.score_subsumption(a, b) for a in names for b in names if a != b]
    assert th.allclose(costs.cpu(), th.tensor(expected), atol=1e-6)

    quantum = E2R(ds, embed_dim=8)
    quantum.train(epochs=2)
    index = quantum.class_index_dict
    rows = th.tensor([[index[a], index[b]] for a in names for b in names if a != b])
    costs = quantum.axiom_costs("gci0", rows)
    expected = [quantum.score_subsumption(a, b) for a in names for b in names if a != b]
    assert th.allclose(costs.cpu(), th.tensor(expected), atol=1e-6)
    with pytest.raises(UnsupportedAxiomForm):
        quantum.axiom_costs("gci2", th.zeros((1, 3), dtype=th.long))


def test_graph_embedding_scores_the_projected_edge():
    from mowl.benchmarks.families import graph_kge
    ds = tiny_dataset()
    model = graph_kge(ds, projector="taxonomy", embed_dim=8,
                      model_filepath=os.path.join(scratch_dir(), "m.pt"))
    model.train(epochs=1)
    index = model.class_index_dict
    a = index["http://mowl.borg/bench#A"]
    b = index["http://mowl.borg/bench#B"]
    rows = th.tensor([[a, b], [b, a]])
    costs = model.axiom_costs("gci0", rows)
    triples = model.triples_factory
    edge = th.tensor([[triples.entity_to_id["http://mowl.borg/bench#A"],
                       triples.relation_to_id["http://subclassof"],
                       triples.entity_to_id["http://mowl.borg/bench#B"]]])
    assert th.allclose(costs[0], -model._kge_method.score_hrt(edge).reshape(-1)[0])
    assert costs[0] != costs[1]
    # The taxonomy projection has no edge label for the role r: no score for C ⊑ ∃r.D.
    r = model.object_property_index_dict["http://mowl.borg/bench#r"]
    assert th.isinf(model.axiom_costs("gci2", th.tensor([[a, r, b]]))).all()


def test_every_runnable_method_has_default_settings_on_the_joint_benchmarks():
    from mowl.benchmarks import list_recipes
    for b in list_benchmarks():
        if not b.key.startswith("common-"):
            continue
        run, _ = runnable_methods(b)
        for method in run:
            found = list_recipes(b.key, method)
            assert len(found) == 1, (b.key, method, len(found))
            assert found[0].source.key == "mowl-common"
    # Where a publication trained on the same data, its settings are reused.
    (recipe,) = list_recipes("common-go-subsumption-prediction", "box2el")
    (original,) = list_recipes("box2el-go-prediction", "box2el")
    assert recipe.model_kwargs == original.model_kwargs
    assert recipe.training_regime == original.training_regime
    assert "jackermeier2024" in recipe.locator


def test_pair_classifier_head_falls_back_to_all_roles(monkeypatch):
    model = _Vectors()
    rows = th.tensor([[i, 0, i + 1] for i in range(model.n - 1)])
    head = PairClassifierHead(epochs=5, seed=0)
    monkeypatch.setattr(head, "_training_rows", lambda m: {"gci2": rows})
    unseen = th.tensor([[0, 7, 1], [1, 7, 0]])           # role 7 has no training axiom
    costs = head(model, "gci2", unseen)
    assert costs.shape == (2,) and th.isfinite(costs).all()
    assert head._nets[("gci2", 7)] is head._nets[("gci2", None)]
