"""OnT benchmarks, published results of Yang et al. (ISWC 2025) and the OnT evaluation
regimes applied to an EL model (no downloads)."""

import os

import pytest

import mowl  # noqa: F401
from mowl.benchmarks.registry import get_benchmark, published_results
from tests.benchmarks.tiny import scratch_dir, tiny_dataset

KEYS = [f"ont-{o}-{t}" for o in ("galen", "go", "anatomy")
        for t in ("prediction", "inference")]


@pytest.mark.parametrize("key", KEYS)
def test_benchmarks_registered(key):
    b = get_benchmark(key)
    assert b.dataset_kwargs["ontology"] in ("GALEN", "GO", "ANATOMY")
    expected = ("gci0", "gci1", "gci2", "gci3", "combined") if "prediction" in key \
        else ("all",)
    assert b.subsets == expected
    proto = b.load_protocol()
    assert proto.regime.key in ("ont-normal-form-ranking", "ont-inferred-subsumption-ranking")


def test_published_results():
    rows = [r for r in published_results() if r.source.key == "yang2025ont"]
    assert len(rows) == 10 * 3 * 2 + 8
    for r in rows:
        assert r.benchmark in KEYS
        assert "box2el-*" in r.note
        assert all(0 <= r.metrics[m] <= 1 for m in ("hits@1", "hits@10", "hits@100", "mrr"))
    methods = {r.method for r in rows}
    assert methods == {"elem", "elbe", "boxel", "box2el", "transbox", "opa2vec",
                       "owl2vec_star", "hit", "ont"}


def test_transcribed_spot_values():
    """Checked against the rendered pages of arXiv:2507.14334v1."""
    ont = [r for r in published_results(benchmark="ont-go-prediction", method="ont")
           if r.locator.startswith("Table 2") and "use_roles" not in r.settings][0]
    assert ont.metrics == {"hits@1": 0.37, "hits@10": 0.67, "hits@100": 0.81, "mrr": 0.46,
                           "mr": 1121}
    assert ont.subset == "combined"
    wo = [r for r in published_results(benchmark="ont-go-inference", method="ont")
          if r.settings.get("use_roles") is False and r.locator.startswith("Table 3")][0]
    assert wo.metrics["mr"] == 824 and wo.metrics["hits@100"] == 0.39
    hit = published_results(benchmark="ont-anatomy-inference", method="hit")[0]
    assert hit.metrics["mr"] == 441
    boxel = published_results(benchmark="ont-galen-prediction", method="boxel")[0]
    assert boxel.metrics["mr"] == 13824
    rows = published_results(benchmark="ont-galen-prediction", method="ont")
    mpnet = [r for r in rows if r.settings.get("encoder", "").endswith("mpnet-base-v2")]
    mpnet = [r for r in mpnet if "use_roles" not in r.settings][0]
    assert mpnet.metrics["mr"] == 630 and mpnet.locator.startswith("Table 4")


def test_cards_and_regimes():
    from mowl.catalog import get_method, get_regime
    for key in ("hit", "ont"):
        card = get_method(key)
        assert card.family.value == "language_model"
        assert "sentence-transformers" in card.optional_dependencies
        assert card.helly == "" and "not applicable" in card.notes
        for regime in card.evaluation:
            if regime.startswith("ont-"):
                assert get_regime(regime).kind.value == "evaluation"
    assert "FALCON" in get_method("ont").notes and "Section 5.1" in get_method("ont").notes


def test_ont_regime_on_el_model():
    """The OnT ranking regime also runs an EL model (geometric baselines under OnT's
    protocol): no trivial-answer exclusion, top and bottom not candidates."""
    import torch as th
    from mowl.models.elembeddings.model import ELEmbeddings
    from mowl.evaluation.ont_ranking import OnTRankingRegime
    from mowl.evaluation.normal_form_ranking import NormalFormRankingRegime
    ds = tiny_dataset()
    th.manual_seed(0)
    path = os.path.join(scratch_dir(), "elem_ont.pt")
    model = ELEmbeddings(ds, embed_dim=4, model_filepath=path)
    model.eval_gci_name = "gci0"
    model.train(epochs=2)
    ours = OnTRankingRegime().evaluate(model, ds)
    box2el = NormalFormRankingRegime().evaluate(model, ds)
    assert set(ours) == set(box2el)
    n = len([c for c in ds.classes.as_str if not c.endswith(("#Thing", "#Nothing"))])
    assert OnTRankingRegime().candidate_iris(ds).__len__() == n
    for subset in ours:
        assert 1 <= ours[subset]["mr"] <= n


def test_counting_ranks_equal_sorting_ranks():
    import torch as th
    from mowl.evaluation.ont_ranking import ordinal_ranks_by_counting
    from mowl.evaluation.ranking_metrics import ordinal_ranks
    g = th.Generator().manual_seed(0)
    scores = th.randint(0, 5, (50, 30), generator=g).double()   # many ties
    targets = th.randint(0, 30, (50,), generator=g)
    assert th.equal(ordinal_ranks_by_counting(scores, targets), ordinal_ranks(scores, targets))
