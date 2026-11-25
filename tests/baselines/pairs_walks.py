"""Walk and sentence methods: the combinations of parts against the direct Word2Vec models.

Both sides run the same projector, walker, corpus extractor and gensim Word2Vec, with one
worker and fixed seeds, so the comparison is exact:

- "losses": Word2Vec has no loss of given parameters that could be evaluated on both sides
  (gensim reports only a running loss of its own training). What the loss is computed from
  is compared instead: the sequences the encoder is trained on (every token of every line,
  as CRC-32 codes, and the line lengths) and the vocabulary with its counts.
- "scores": the similarity of the ``evaluation_model`` (:math:`1 - \\sigma(x \\cdot y)`) on
  every pair of classes that have a vector (the direct implementation fills a class without a
  vector with unseeded random numbers, so those rows are left out), and the ranking scores of
  the regimes (``scoring="auto"``: the pair classifier head) for ``gci0`` and ``gci2``.
- "training": three passes from the seed on both sides; then the scores, the sequences and the
  class and relation vectors.

The combinations select two behaviours of the direct code that their parts no longer have by
default: ``restart=True`` (every ``train`` starts anew from the seed) and ``hashfxn=hash``
(Python's own hash seeds the start vectors; the parts use a hash that is the same in every
process), and for OPA2Vec ``annotation_entity_form="bracketed iri"`` (the direct code writes
``<IRI>`` in annotation sentences, a token that no axiom sentence contains).
"""
import copy
import zlib

import numpy as np
import torch as th

from tests.baselines.harness import Pair

W2V = dict(vector_size=8, window=3, min_count=1, workers=1, seed=0, sg=1)
WALKS = dict(num_walks=2, walk_length=5, alpha=0.1)
LABELS = {"Male": "male person", "Female": "female person", "Parent": "a parent",
          "Mother": "female parent", "Person": "a human being"}


def annotated_family():
    """The family ontology with a label on five classes (for the annotation sentences)."""
    from mowl.datasets.builtin import FamilyDataset
    from mowl.owlapi import OWLAPIAdapter
    dataset = FamilyDataset()
    adapter = OWLAPIAdapter()
    factory = adapter.data_factory
    for name, label in LABELS.items():
        iri = adapter.create_class(f"http://{name}").getIRI()
        axiom = factory.getOWLAnnotationAssertionAxiom(factory.getRDFSLabel(), iri,
                                                       factory.getOWLLiteral(label))
        adapter.owl_manager.addAxiom(dataset.ontology, axiom)
    return dataset


def _scratch(suffix):
    from mowl.representation.sequence import scratch_file
    return scratch_file(suffix)


# -- the direct implementation ------------------------------------------------------------

def _projector(name):
    from mowl.projection import DL2VecProjector, OWL2VecStarProjector, TaxonomyProjector
    return {"dl2vec": lambda: DL2VecProjector(True),
            "owl2vec_star": lambda: OWL2VecStarProjector(bidirectional_taxonomy=True),
            "taxonomy_walks": lambda: TaxonomyProjector(True)}[name]()


def _old_walks(name):
    def build(dataset):
        from mowl.baselines.models.graph_random_walk.random_walk_w2v_model import \
            RandomWalkPlusW2VModel
        from mowl.walking import DeepWalk
        model = RandomWalkPlusW2VModel(dataset)
        model.set_projector(_projector(name))
        model.set_walker(DeepWalk(WALKS["num_walks"], WALKS["walk_length"], WALKS["alpha"],
                                  outfile=_scratch(".walks"), workers=1, seed=0))
        model.set_w2v_model(**W2V)
        model.train(epochs=1)
        return model
    return build


def _old_corpus(with_annotations):
    def build(dataset):
        from mowl.baselines.models.syntactic.w2v_model import SyntacticPlusW2VModel
        model = SyntacticPlusW2VModel(dataset, corpus_filepath=_scratch(".corpus"))
        model.set_w2v_model(**W2V)
        model.generate_corpus(save=True, with_annotations=with_annotations)
        model.train(epochs=1)
        return model
    return build


# -- the combinations -----------------------------------------------------------------------

#: The behaviour of the direct code, selected explicitly (see the module docstring).
PUBLISHED = dict(restart=True, word2vec_options={"hashfxn": hash})
#: The methods that walk a graph (the others read sentences of the axioms).
WALK_METHODS = ("dl2vec", "owl2vec_star", "taxonomy_walks")


def _new_method(key, **options):
    def build(dataset):
        from mowl.models import ComposedSequenceModel
        w2v = {k: v for k, v in W2V.items() if k != "vector_size"}
        walks = WALKS if key in WALK_METHODS else {}   # (sentence methods take no walks)
        model = ComposedSequenceModel(dataset, method=key, embed_dim=W2V["vector_size"],
                                      **walks, **w2v, **PUBLISHED, **options)
        model.train(epochs=1)
        return model
    return build


def _new_walk_class(name):
    def build(dataset):
        from mowl.models import RandomWalkPlusW2VModel
        from mowl.walking import DeepWalk
        model = RandomWalkPlusW2VModel(dataset)
        model.set_projector(_projector(name))
        model.set_walker(DeepWalk(WALKS["num_walks"], WALKS["walk_length"], WALKS["alpha"],
                                  outfile=_scratch(".walks"), workers=1, seed=0))
        model.set_w2v_model(**W2V, hashfxn=hash, restart=True)
        model.train(epochs=1)
        return model
    return build


def _new_corpus_class(with_annotations):
    def build(dataset):
        from mowl.models import SyntacticPlusW2VModel
        model = SyntacticPlusW2VModel(dataset, corpus_filepath=_scratch(".corpus"))
        model.set_w2v_model(**W2V, hashfxn=hash, restart=True)
        model.generate_corpus(save=True, with_annotations=with_annotations,
                              annotation_entity_form="bracketed iri")
        model.train(epochs=1)
        return model
    return build


# -- checks -------------------------------------------------------------------------------

def copy_vectors(old, new):
    """The trained gensim model of the direct implementation becomes the combination's."""
    new.module.encoder.model = copy.deepcopy(old.w2v_model)
    new.module.refresh()


def _code(token):
    return zlib.crc32(token.encode("utf-8"))


def _corpus_path(model):
    path = getattr(model, "corpus_path", None)
    if path is None:
        walker = getattr(model, "walker", None)
        path = walker.outfile if walker is not None else model.corpus_filepath
    return path


def sequences(model):
    """What the encoder is trained on: the corpus and the vocabulary."""
    with open(_corpus_path(model)) as f:
        lines = [line.split() for line in f]
    wv = model.w2v_model.wv
    return {"corpus tokens": th.tensor([_code(t) for line in lines for t in line]),
            "corpus line lengths": th.tensor([len(line) for line in lines]),
            "vocabulary": th.tensor([_code(t) for t in wv.index_to_key]),
            "counts": th.tensor([wv.get_vecattr(t, "count") for t in wv.index_to_key])}


def _known_pairs(model):
    known = set(model.class_embeddings)
    ids = sorted(i for iri, i in model.class_index_dict.items() if iri in known)
    return th.tensor([[a, b] for a in ids for b in ids if a != b])


def scores(model):
    from mowl.evaluation.el_scoring import resolve_scorer
    rows = _known_pairs(model)
    scorer, _ = resolve_scorer(model, "auto")
    rel = th.zeros((len(rows), 1), dtype=th.long)
    with_relation = th.cat([rows[:, :1], rel, rows[:, 1:]], dim=1)
    return {"similarity": model.evaluation_model(rows).detach(),
            "ranking gci0": scorer(model, "gci0", rows).detach(),
            "ranking gci2": scorer(model, "gci2", with_relation).detach()}


def _table(names, embeddings, dim):
    out = np.zeros((len(names), dim), dtype=np.float32)
    for i, name in enumerate(names):
        if name in embeddings:
            out[i] = embeddings[name]
    return th.from_numpy(out)


def vectors(model):
    dim = model.w2v_model.vector_size
    return {"classes": _table(model.dataset.classes.as_str, model.class_embeddings, dim),
            "relations": _table(model.dataset.object_properties.as_str,
                                model.object_property_embeddings, dim)}


def _pair(key, new, old, **kwargs):
    return Pair(key=key, new=new, old=old, copy=copy_vectors, scores=scores,
                losses=sequences, parameters=vectors, dataset=annotated_family, **kwargs)


PAIRS = [
    _pair("dl2vec", _new_method("dl2vec"), _old_walks("dl2vec")),
    _pair("dl2vec/RandomWalkPlusW2VModel", _new_walk_class("dl2vec"), _old_walks("dl2vec")),
    _pair("owl2vec_star", _new_method("owl2vec_star"), _old_walks("owl2vec_star")),
    _pair("taxonomy_walks", _new_method("taxonomy_walks"), _old_walks("taxonomy_walks")),
    _pair("onto2vec", _new_method("onto2vec"), _old_corpus(False)),
    _pair("opa2vec", _new_method("opa2vec", annotation_entity_form="bracketed iri"),
          _old_corpus(True)),
    _pair("opa2vec/SyntacticPlusW2VModel", _new_corpus_class(True), _old_corpus(True)),
]

EXEMPT = {
    "composed_sequences": "ComposedSequenceModel is the generic combination of parts for walk "
                          "and sentence methods; it has no direct implementation to be "
                          "compared with (its named combinations are compared above).",
}
