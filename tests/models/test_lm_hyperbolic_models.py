"""End-to-end tests of HiT and OnT with a small stand-in encoder (no language model, no
download), plus tests with sentence-transformers that are skipped when it is absent."""

import math
import os

import numpy as np
import pytest
import torch as th

import mowl.datasets  # noqa: F401
from mowl.datasets import Dataset
from mowl.owlapi import OWLAPIAdapter
from mowl.models.hit.model import HiT
from mowl.models.ont.model import OnT
from mowl.nn.lm.encoders import BagOfWordsEncoder

P = "http://example.org/zoo#"
LABELS = {"Animal": "animal", "Mammal": "mammal", "Dog": "dog", "Cat": "cat", "Bird": "bird",
          "Sparrow": "sparrow", "Vehicle": "vehicle", "Car": "car", "Bicycle": "bicycle",
          "Wheel": "wheel", "Person": "person", "PetOwner": "pet owner",
          "DogOwner": "dog owner", "hasPart": "has part", "owns": "owns"}


def _ontology(ad, iri, axioms, labels=None):
    from org.semanticweb.owlapi.model import IRI
    ont = ad.create_ontology(iri)
    df = ad.data_factory
    for k, v in (labels or {}).items():
        ad.owl_manager.addAxiom(ont, df.getOWLAnnotationAssertionAxiom(
            df.getRDFSLabel(), IRI.create(P + k), df.getOWLLiteral(v)))
    for ax in axioms:
        ad.owl_manager.addAxiom(ont, ax)
    return ont


@pytest.fixture(scope="module")
def zoo():
    ad = OWLAPIAdapter()
    c = {k: ad.create_class(P + k) for k in LABELS if k[0].isupper()}
    part, owns = ad.create_object_property(P + "hasPart"), ad.create_object_property(P + "owns")
    sub, ex, conj = ad.create_subclass_of, ad.create_object_some_values_from, \
        ad.create_object_intersection_of
    train = [sub(c["Dog"], c["Mammal"]), sub(c["Cat"], c["Mammal"]),
             sub(c["Mammal"], c["Animal"]), sub(c["Sparrow"], c["Bird"]),
             sub(c["Bird"], c["Animal"]), sub(c["Car"], c["Vehicle"]),
             sub(c["Bicycle"], c["Vehicle"]), sub(c["Car"], ex(part, c["Wheel"])),
             sub(c["Bicycle"], ex(part, c["Wheel"])), sub(c["PetOwner"], ex(owns, c["Animal"])),
             sub(conj(c["Person"], ex(owns, c["Dog"])), c["DogOwner"]),
             sub(ex(owns, c["Dog"]), c["PetOwner"]), sub(c["DogOwner"], c["Person"])]
    valid = [sub(c["Dog"], c["Animal"]), sub(c["Sparrow"], c["Animal"])]
    test = [sub(c["Cat"], c["Animal"]), sub(conj(c["Person"], c["PetOwner"]), c["DogOwner"]),
            sub(c["Bicycle"], ex(part, c["Wheel"])), sub(ex(owns, c["Cat"]), c["PetOwner"])]
    ds = Dataset(_ontology(ad, "http://example.org/zoo", train, LABELS),
                 validation=_ontology(ad, "http://example.org/zoo/valid", valid),
                 testing=_ontology(ad, "http://example.org/zoo/test", test))
    return ad, ds


def _model(cls, ds, tmp_path, **kw):
    th.manual_seed(0)
    defaults = dict(encoder=BagOfWordsEncoder(dim=16, num_buckets=256), batch_size=8,
                    warmup_steps=0, learning_rate=0.05, seed=0,
                    model_filepath=str(tmp_path / f"{cls.__name__}.pt"))
    defaults.update(kw)
    return cls(ds, **defaults)


def test_hit_end_to_end(zoo, tmp_path):
    ad, ds = zoo
    model = _model(HiT, ds, tmp_path, num_negatives=3)
    edges = model.taxonomy_edges()
    assert (P + "Dog", P + "Mammal") in edges and (P + "DogOwner", P + "Person") in edges
    assert all("Wheel" not in b for _, b in edges)   # no existential parents
    model.train(epochs=15)
    losses = [h["loss"] for h in model.history]
    assert losses[-1] < losses[0]
    assert model.centri_weight in model.centri_weight_grid
    assert model.best_validation is not None
    emb = model.class_embeddings
    assert np.isfinite(emb[P + "Dog"]).all() and emb[P + "Dog"].shape == (16,)
    s = model.score(ad.create_subclass_of(ad.create_class(P + "Dog"),
                                          ad.create_class(P + "Animal")))
    assert math.isfinite(s)
    with pytest.raises(NotImplementedError):
        model.object_property_embeddings


def test_hit_hard_negatives_are_siblings(zoo, tmp_path):
    _, ds = zoo
    model = _model(HiT, ds, tmp_path, num_negatives=1, hard_negatives=True)
    triplets = model.training_triplets()
    negs = {(c, n) for c, p, n in triplets if p == "mammal"}
    assert ("dog", "cat") in negs and ("cat", "dog") in negs
    for child, parent, neg in triplets:   # never an ancestor or the entity itself
        assert neg != child
        assert not (child == "dog" and neg in ("mammal", "animal"))


def test_ont_training_data(zoo, tmp_path):
    _, ds = zoo
    model = _model(OnT, ds, tmp_path)
    triplets = model.training_triplets()
    assert ("car", "something that has part some wheel") in {t[:2] for t in triplets}
    neg = [t[2] for t in triplets if t[0] == "car" and t[1].startswith("something")][0]
    assert neg.startswith("something that has part some ")
    assert ("something that has part some wheel", "has part", "wheel") in model.existentials
    texts = {c[0] for c in model.conjunctions}
    assert texts == {"person and something that owns some dog"}
    release = _model(OnT, ds, tmp_path, verbalisation_style="release")
    assert ("car", "has part some wheel") in {t[:2] for t in release.training_triplets()}


@pytest.mark.parametrize("use_roles", [True, False])
def test_ont_end_to_end(zoo, tmp_path, use_roles):
    from mowl.evaluation.ont_ranking import OnTRankingRegime, OnTInferenceRankingRegime
    ad, ds = zoo
    model = _model(OnT, ds, tmp_path, use_roles=use_roles)
    model.train(epochs=15)
    losses = [h["loss"] for h in model.history]
    assert losses[-1] < losses[0]
    regime = OnTRankingRegime()
    res = regime.evaluate(model, ds)
    assert set(res) == {"gci0", "gci1", "gci2", "gci3", "combined"}
    assert regime.info["num_candidates"] == len(LABELS) - 2   # classes, no top/bottom
    for subset, metrics in res.items():
        assert 1 <= metrics["mr"] <= regime.info["num_candidates"]
        assert 0 <= metrics["mrr"] <= 1
    assert res["combined"]["mr"] == pytest.approx(
        sum(res[k]["mr"] for k in ("gci0", "gci1", "gci2", "gci3")) / 4)
    inf = OnTInferenceRankingRegime().evaluate(model, ds)
    assert inf["all"] == res["gci0"]
    if use_roles:
        params = model.role_parameters()
        assert set(params) == {P + "hasPart", P + "owns"}


def test_ont_scores_with_role_images(zoo, tmp_path):
    _, ds = zoo
    model = _model(OnT, ds, tmp_path, existential_mode="role")
    model.train(epochs=2, validate=False)
    rows = th.tensor([[model.class_index_dict[P + "Bicycle"],
                       model.object_property_index_dict[P + "hasPart"],
                       model.class_index_dict[P + "Wheel"]]])
    cands = th.tensor(sorted(model.class_index_dict.values()))
    scores = model.score_candidates("gci2", rows, 0, cands)
    assert scores.shape == (1, len(cands)) and th.isfinite(scores).all()


def test_inductive_unseen_class(zoo, tmp_path):
    """A class that is not in the ontology is embedded from its label."""
    ad, ds = zoo
    model = _model(OnT, ds, tmp_path)
    model.train(epochs=2, validate=False)
    model.verbaliser.labels["http://example.org/new#Puppy"] = "dog puppy"
    puppy = ad.create_class("http://example.org/new#Puppy")
    x = model.embed([puppy, ad.create_class(P + "Dog")])
    assert x.shape == (2, 16) and th.isfinite(x).all()
    s = model.score(ad.create_subclass_of(puppy, ad.create_class(P + "Mammal")))
    assert math.isfinite(s)


def test_save_and_load(zoo, tmp_path):
    _, ds = zoo
    model = _model(OnT, ds, tmp_path)
    model.train(epochs=1, validate=False)
    path = str(tmp_path / "ont.pt")
    model.save(path)
    other = _model(OnT, ds, tmp_path)
    other.from_pretrained(path)
    a, b = model.class_embeddings, other.class_embeddings
    assert np.allclose(a[P + "Dog"], b[P + "Dog"])


def test_runner_family(zoo, tmp_path):
    from mowl.benchmarks import runner
    from mowl.benchmarks.protocols.ont import LanguageModelFamily
    from mowl.benchmarks.spec import Benchmark, Task
    from mowl.benchmarks.entries.ont import YANG2025ONT
    _, ds = zoo
    bench = Benchmark(key="tiny-ont", name="tiny", task=Task.AXIOM_PREDICTION,
                      description="tiny", dataset="unused", metrics=("mrr", "mr"),
                      protocol="mowl.benchmarks.protocols.ont.OnTPredictionProtocol",
                      subsets=("combined",), references=(YANG2025ONT,))
    # The runner registers the language-model family itself (lazily).
    assert isinstance(runner.family_for(OnT)._resolve(), LanguageModelFamily)
    result = runner.run_benchmark(
        bench, model_factory=lambda d: _model(OnT, d, tmp_path), dataset=ds, epochs=2,
        workdir=str(tmp_path), deterministic=False)
    assert set(result.metrics["combined"]) == {"mrr", "mr"}
    assert result.protocol_info["training"]["centri_weight"] is not None


# --------------------------------------------------------------------- real encoders

def _tiny_bert(path):
    """A randomly initialised two-layer BERT with a 60-word vocabulary, saved locally."""
    transformers = pytest.importorskip("transformers")
    words = {w for v in LABELS.values() for w in v.split()}
    words = sorted(words.union({"something", "that", "some", "and"}))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab) + "\n")
    tok = transformers.BertTokenizer(os.path.join(path, "vocab.txt"))
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2,
                                  num_attention_heads=2, intermediate_size=32,
                                  max_position_embeddings=64)
    th.manual_seed(0)
    transformers.BertModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def test_ont_with_tiny_random_transformer(zoo, tmp_path):
    pytest.importorskip("sentence_transformers")
    from mowl.nn.lm.encoders import SentenceTransformerEncoder
    _, ds = zoo
    enc = SentenceTransformerEncoder.from_pretrained(_tiny_bert(str(tmp_path / "bert")),
                                                     max_seq_length=32)
    assert enc.dim == 16
    model = _model(OnT, ds, tmp_path, encoder=enc, learning_rate=1e-3)
    model.train(epochs=2)
    assert len(model.history) == 2
    assert all(math.isfinite(h["loss"]) for h in model.history)
    out = enc(["dog", "pet owner"])
    out.sum().backward()   # gradients reach the transformer
    assert any(p.grad is not None for p in enc.parameters())


def test_encoder_without_dependency_raises():
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers is installed")
    except ImportError:
        pass
    from mowl.nn.lm.encoders import as_encoder
    with pytest.raises(ImportError, match="sentence-transformers"):
        as_encoder("sentence-transformers/all-MiniLM-L6-v2")


@pytest.mark.slow
def test_pretrained_smoke(zoo, tmp_path):
    """Loads a small pre-trained model from the Hugging Face hub (skipped offline)."""
    pytest.importorskip("sentence_transformers")
    from mowl.nn.lm.encoders import SentenceTransformerEncoder
    try:
        enc = SentenceTransformerEncoder.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2", max_seq_length=32)
    except Exception as e:  # no network or hub unavailable
        pytest.skip(f"Could not load the pre-trained model: {e}")
    _, ds = zoo
    model = _model(OnT, ds, tmp_path, encoder=enc, learning_rate=1e-5)
    model.train(epochs=1)
    x = model.embed([P + "Dog", P + "Animal"])
    assert x.shape == (2, 384) and th.isfinite(x).all()
