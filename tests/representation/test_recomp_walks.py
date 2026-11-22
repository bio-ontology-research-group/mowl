"""Walk and sentence methods as combinations of parts: what they are made of, swaps of every
part, the catalogue against the parts, and the defects fixed in the parts."""
import math
import os
import subprocess
import sys

import pytest
import torch as th

import mowl
mowl.init_jvm("1g")
pytest.importorskip("gensim")

from mowl.models import (ComposedSequenceModel, RandomWalkPlusW2VModel,  # noqa: E402
                         SyntacticPlusW2VModel)
from mowl.representation import METHODS, METHOD_KINDS, REPRESENTATIONS  # noqa: E402
from mowl.representation.sequence import (Entailment, SequenceGeometry,  # noqa: E402
                                          scratch_file)
from mowl.representation.library import walks_encoders as E  # noqa: E402
from mowl.representation.library import walks_views as V  # noqa: E402
from tests.baselines.pairs_walks import annotated_family  # noqa: E402

KEYS = ("dl2vec", "owl2vec_star", "taxonomy_walks", "onto2vec", "opa2vec")
SMALL = dict(embed_dim=8, num_walks=2, walk_length=5, window=3, workers=1, seed=0)


@pytest.fixture(scope="module")
def family():
    return annotated_family()


@pytest.fixture(scope="module")
def tiny():
    from tests.benchmarks.tiny import tiny_dataset
    return tiny_dataset()


def _finite(values):
    values = th.as_tensor(values).double()
    return bool(th.isfinite(values).all())


def _check_trained(model):
    rows = th.tensor([[i, j] for i in range(4) for j in range(4) if i != j])
    costs = model.axiom_costs("gci0", rows)
    assert costs.shape == (len(rows),) and _finite(costs)
    assert len(model.class_embeddings) > 0


def _evaluate(model, dataset, regime=None):
    from mowl.evaluation.filtered_ranking import common_ranking
    regime = regime or common_ranking("gci0")
    result = regime.evaluate(model, dataset)
    numbers = [v for subset in result.values() for v in subset.values()
               if isinstance(v, (int, float))]
    assert numbers and all(math.isfinite(v) for v in numbers), result
    return result


# -- (a) the named combinations, read back ---------------------------------------------------

EXPECTED = {
    "dl2vec": {"sequences": "walks", "projection": "DL2Vec projection",
               "walker": "DeepWalk walks"},
    "owl2vec_star": {"sequences": "walks", "projection": "OWL2Vec* projection",
                     "walker": "DeepWalk walks"},
    "taxonomy_walks": {"sequences": "walks", "projection": "taxonomy",
                       "walker": "DeepWalk walks"},
    "onto2vec": {"sequences": "sentences", "corpus": "axioms as sentences"},
    "opa2vec": {"sequences": "sentences",
                "corpus": "axioms as sentences + annotations as sentences"},
}
COMMON = {"classes": "point", "relations": "token", "individuals": "point",
          "measure": "similarity", "encoder": "word2vec skip-gram",
          "reading": "vector of the entity's token",
          "scorer": "classifier on pairs of vectors", "entailment": "none"}


@pytest.mark.parametrize("key", KEYS)
def test_the_named_combination_reads_back(family, key):
    assert METHOD_KINDS[key] == "sequences"
    model = ComposedSequenceModel(family, method=key, **SMALL)
    assert isinstance(model.module, SequenceGeometry)
    parts = model.module.parts
    for name, value in {**COMMON, **EXPECTED[key]}.items():
        assert parts[name] == value, (key, name, parts[name])
    assert isinstance(model.module.encoder, E.Word2VecEncoder)
    assert isinstance(model.module.reading, E.TokenVectors)
    assert isinstance(model.module.measure, E.DotSigmoid)
    assert isinstance(model.module.scorer, E.PairClassifier)


def test_the_released_classes_are_the_same_combinations(family):
    from mowl.projection import DL2VecProjector
    from mowl.walking import DeepWalk
    rw = RandomWalkPlusW2VModel(family)
    rw.set_projector(DL2VecProjector(True))
    rw.set_walker(DeepWalk(2, 5, 0.1, workers=1))
    rw.set_w2v_model(vector_size=8, sg=1)
    named = ComposedSequenceModel(family, method="dl2vec", **SMALL).module.parts
    assert rw.module.parts == named
    syn = SyntacticPlusW2VModel(family)
    syn.set_w2v_model(vector_size=8, sg=1)
    syn.generate_corpus(save=True, with_annotations=True)
    assert syn.module.parts == ComposedSequenceModel(family, method="opa2vec",
                                                     **SMALL).module.parts
    assert "mowl.baselines" not in type(rw).__module__


def test_query_and_axiom_trainers_refuse_sequence_methods(family):
    from mowl.models import ComposedEL
    with pytest.raises(TypeError, match="ComposedSequenceModel"):
        ComposedEL(family, method="dl2vec", embed_dim=8)
    with pytest.raises(ValueError, match="sequences"):
        ComposedSequenceModel(family, method="elbe")


# -- (b) swaps: two or more per method, one epoch, evaluated ---------------------------------

def test_dl2vec_with_node2vec_walks(family, tiny):
    model = ComposedSequenceModel(family, method="dl2vec", walker="node2vec", **SMALL)
    assert model.module.parts["walker"] == "node2vec walks"
    model.train(epochs=1)
    _check_trained(model)
    other = ComposedSequenceModel(tiny, method="dl2vec", walker="node2vec", **SMALL)
    other.train(epochs=1)
    _evaluate(other, tiny)


def test_dl2vec_with_cbow_and_a_refreshed_corpus(tiny):
    from mowl.training.sequences import RefreshSequences
    model = ComposedSequenceModel(tiny, method="dl2vec", sg=0, **SMALL)
    assert model.module.parts["encoder"] == "word2vec CBOW"
    regime = RefreshSequences(every=1)
    model.training_regime = regime
    model.train(epochs=2)
    assert regime.refreshed == [1]
    assert len(model.losses) == 2
    _evaluate(model, tiny)


def test_owl2vec_star_with_its_lexical_document(family):
    model = ComposedSequenceModel(family, method="owl2vec_star", lexical=True, **SMALL)
    parts = model.module.parts
    assert parts["sequences"] == "walks + sentences"
    assert parts["corpus"] == "annotations as sentences"
    model.train(epochs=1)
    _check_trained(model)
    with open(model.corpus_path) as f:
        text = f.read()
    assert "http://Male rdfs:label male person" in text


def test_owl2vec_star_scored_by_similarity_in_two_regimes(tiny):
    from mowl.evaluation.ont_ranking import OnTRankingRegime
    model = ComposedSequenceModel(tiny, method="owl2vec_star", scorer="similarity", **SMALL)
    assert model.module.parts["scorer"] == "similarity of the two vectors"
    model.train(epochs=1)
    _evaluate(model, tiny)
    regime = OnTRankingRegime(normal_forms=("gci0", "gci2"))
    _evaluate(model, tiny, regime)
    assert regime.info["scoring"] == "model"


def test_taxonomy_walks_with_a_corpus_added(family):
    model = ComposedSequenceModel(family, method="taxonomy_walks", corpus=("axioms",),
                                  **SMALL)
    parts = model.module.parts
    assert parts["projection"] == "taxonomy" and parts["corpus"] == "axioms as sentences"
    model.train(epochs=1)
    _check_trained(model)
    # every class occurs in an axiom sentence; the taxonomy alone misses some
    alone = ComposedSequenceModel(family, method="taxonomy_walks", **SMALL)
    alone.train(epochs=1)
    assert len(model.class_embeddings) >= len(alone.class_embeddings)


def test_taxonomy_walks_with_fasttext_and_relations(tiny):
    model = ComposedSequenceModel(tiny, method="taxonomy_walks", relations=True,
                                  encoder="fasttext", **SMALL)
    assert model.module.parts["encoder"] == "fastText skip-gram"
    assert model.module.parts["projection"] == "taxonomy with relations"
    model.train(epochs=1)
    _evaluate(model, tiny)


def test_onto2vec_with_entailed_axioms(family):
    model = ComposedSequenceModel(family, method="onto2vec",
                                  entailment=Entailment("elk"), **SMALL)
    assert model.module.parts["entailment"].startswith("entailed by elk")
    model.train(epochs=1)
    _check_trained(model)
    asserted = ComposedSequenceModel(family, method="onto2vec", **SMALL)
    asserted.train(epochs=1)
    with open(model.corpus_path) as a, open(asserted.corpus_path) as b:
        assert len(a.readlines()) > len(b.readlines())


def test_onto2vec_with_cosine_and_similarity_in_two_regimes(tiny):
    from mowl.evaluation.ont_ranking import OnTRankingRegime
    model = ComposedSequenceModel(tiny, method="onto2vec", measure="cosine",
                                  scorer="similarity", **SMALL)
    model.train(epochs=1)
    _evaluate(model, tiny)
    _evaluate(model, tiny, OnTRankingRegime(normal_forms=("gci0",)))
    a, b = model.module.classes(th.tensor([0, 1])), model.module.classes(th.tensor([1, 0]))
    assert th.allclose(model.module.inside(a, b), model.module.inside(b, a))


def test_opa2vec_with_normalised_vectors_and_another_window(family, tiny):
    model = ComposedSequenceModel(family, method="opa2vec", reading=E.TokenVectors(True),
                                  **{**SMALL, "window": 1})
    assert model.module.encoder.model.window == 1
    model.train(epochs=1)
    vectors, known = model.module.table("classes")
    assert th.allclose(vectors[known].norm(dim=1), th.ones(int(known.sum())), atol=1e-5)
    _check_trained(model)
    other = ComposedSequenceModel(tiny, method="opa2vec", reading=E.TokenVectors(True),
                                  **SMALL)
    other.train(epochs=1)
    _evaluate(other, tiny)


def test_free_combination_of_parts(tiny):
    """A source and an encoder chosen without a named method."""
    view = V.projection_view("dl2vec", bidirectional_taxonomy=True)
    source = V.Mixed(V.Walks(view, V.make_walker("deepwalk", 2, 5)),
                     V.Sentences(V.AxiomCorpus()))
    model = ComposedSequenceModel(tiny, source=source,
                                  encoder=E.Word2VecEncoder(vector_size=8, min_count=1,
                                                            workers=1, sg=1),
                                  scorer=E.SimilarityScorer())
    model.train(epochs=1)
    _evaluate(model, tiny)


# -- (c) the catalogue and the parts agree ----------------------------------------------------

@pytest.mark.parametrize("key", KEYS)
def test_catalogue_agrees_with_the_parts(family, key):
    from mowl.catalog import get_method
    card = get_method(key)
    rep = card.representation
    assert rep is REPRESENTATIONS[key]
    parts = METHODS[key](5, 2, 3, 8).parts
    values = {f: {v.value for v in getattr(rep, f)} for f in (
        "classes", "relations", "measure", "projection", "corpus", "walker", "encoder",
        "reading", "scorer")}
    for field in ("classes", "relations", "measure", "encoder", "reading", "scorer"):
        assert parts[field] in values[field], (key, field)
    if "projection" in parts:
        assert parts["projection"] in values["projection"]
        assert parts["walker"] in values["walker"]
    for corpus in parts.get("corpus", "none").split(" + "):
        if corpus != "none":
            assert corpus in values["corpus"]
    # the card's tasks: a pair classifier scores C ⊑ D (and C ⊑ ∃r.D)
    assert set(card.head_tasks) == {"subsumption"}
    assert "gci0" in ComposedSequenceModel.scored_forms
    rows = dict(rep.rows())
    assert rows["Sequences are encoded by"].startswith("word2vec skip-gram")


def test_listing_by_sequence_parts():
    from mowl.catalog import list_methods
    from mowl.representation.vocabulary import Corpus, Projection
    annotated = {c.key for c in list_methods(parts=Corpus.ANNOTATIONS)}
    assert annotated == {"opa2vec", "owl2vec_star"}
    assert {c.key for c in list_methods(parts=Projection.DL2VEC)} == {"dl2vec"}


# -- defects fixed in the parts ---------------------------------------------------------------

def test_a_second_walk_replaces_the_first(family):
    """The walkers append to their file; a second walk used to add to the first."""
    source = V.Walks(V.projection_view("taxonomy"), V.make_walker("deepwalk", 2, 5))
    source.write(family)
    with open(source.path) as f:
        first = f.readlines()
    source.write(family)
    with open(source.path) as f:
        assert f.readlines() == first
    appending = V.Walks(source.view, source.walker, fresh=False)
    appending.write(family)
    with open(source.path) as f:
        assert len(f.readlines()) == 2 * len(first)


def test_training_goes_on_from_pretrained_vectors(family):
    """from_pretrained followed by train discarded the loaded vectors."""
    path = scratch_file(".corpus")
    first = SyntacticPlusW2VModel(family, corpus_filepath=path)
    first.set_w2v_model(vector_size=8, min_count=1, workers=1, seed=0)
    first.generate_corpus()
    first.train(epochs=2)
    saved = scratch_file(".w2v")
    first.w2v_model.save(saved)
    vector = first.class_embeddings["http://Male"].copy()
    second = SyntacticPlusW2VModel(family, corpus_filepath=path)
    second.from_pretrained(saved)
    second.train(epochs=0)
    assert (second.class_embeddings["http://Male"] == vector).all()
    # the earlier behaviour, as an option: every train starts anew
    third = SyntacticPlusW2VModel(family, corpus_filepath=path)
    third.set_w2v_model(vector_size=8, min_count=1, workers=1, seed=0, restart=True)
    third.generate_corpus()
    third.train(epochs=2)
    before = third.class_embeddings["http://Male"].copy()
    third.train(epochs=0)
    assert not (third.class_embeddings["http://Male"] == before).all()


def test_annotation_sentences_name_the_entity_by_its_token(family):
    """The saved annotation corpus wrote <IRI>, a token no axiom sentence has."""
    fixed = V.AnnotationCorpus().read(family)
    old = V.AnnotationCorpus(entity_form="bracketed iri").read(family)
    assert "http://Male rdfs:label male person" in fixed
    assert "<http://Male> rdfs:label male person" in old
    model = ComposedSequenceModel(family, method="opa2vec", **SMALL)
    model.train(epochs=1)
    assert "<http://Male>" not in model.module.encoder.tokens()


_HASH_SCRIPT = """
import sys
import mowl
mowl.init_jvm("1g")
from mowl.representation.library.walks_encoders import Word2VecEncoder
encoder = Word2VecEncoder(vector_size=4, min_count=1, workers=1, seed=0)
encoder.prepare(sys.argv[1])
print(encoder.vector("b").tolist())
"""


def test_the_same_seed_gives_the_same_vectors_in_every_process():
    """Word2Vec seeded its start vectors with Python's hash, which changes between
    processes; the encoder uses a stable hash."""
    corpus = scratch_file(".corpus")
    with open(corpus, "w") as f:
        f.write("a b c\nb c d\n")
    outputs = []
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run([sys.executable, "-c", _HASH_SCRIPT, corpus], env=env,
                             capture_output=True, text=True, check=True)
        outputs.append(out.stdout.strip().splitlines()[-1])
    assert outputs[0] == outputs[1]


def test_node2vec_factory_builds(tiny):
    """mowl.benchmarks.families passed a seed that mowl.walking.Node2Vec does not take."""
    from mowl.benchmarks.families import dl2vec
    model = dl2vec(tiny, walker="node2vec", embed_dim=8, num_walks=2, walk_length=5,
                   workers=1)
    assert type(model.walker).__name__ == "Node2Vec"


def test_the_pair_classifier_is_the_same_for_the_same_seed(family):
    """The training axioms come from the normaliser in another order on every call; the
    head shuffled them as they came, so the same seed trained another classifier."""
    from mowl.evaluation.model_scoring import PairClassifierHead
    model = ComposedSequenceModel(family, method="onto2vec", **SMALL)
    model.train(epochs=1)
    rows = th.tensor([[i, j] for i in range(5) for j in range(5) if i != j])
    first = PairClassifierHead(epochs=5)(model, "gci0", rows)
    second = PairClassifierHead(epochs=5)(model, "gci0", rows)
    assert th.equal(first, second)
