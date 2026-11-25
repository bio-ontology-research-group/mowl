"""Views of the ontology and sources of sequences for the methods that learn from sequences
(``dl2vec``, ``owl2vec_star``, ``taxonomy_walks``, ``onto2vec``, ``opa2vec``).

Views (:class:`~mowl.representation.sequence.OntologyView`):

* :class:`ProjectionView`: a graph, from any projector of :mod:`mowl.projection`
  (:data:`PROJECTIONS` names them: taxonomy, taxonomy with relations, DL2Vec, OWL2Vec*, GDA);
* :class:`AxiomCorpus`: every axiom of the ontology as a sentence (Manchester syntax);
* :class:`AnnotationCorpus`: every literal annotation of a class or individual as a
  sentence.

Each takes an :class:`~mowl.representation.sequence.Entailment` to read the entailed axioms
too.

Sources (:class:`~mowl.representation.sequence.SequenceSource`):

* :class:`Walks`: a walker of :mod:`mowl.walking` (DeepWalk, node2vec) on a graph view;
* :class:`Sentences`: the sentences of a corpus view;
* :class:`Mixed`: several sources in one corpus, in order.

The projectors, walkers and corpus extractors are used as they are; nothing of
:mod:`mowl.projection`, :mod:`mowl.walking` or :mod:`mowl.corpus` is changed here.
"""

import os
import shutil

from mowl.representation.library.graph_projection import (  # noqa: F401
    PROJECTIONS, ProjectionView, projection_view)
from mowl.representation.sequence import (OntologyView, SequenceSource, scratch_file)
from mowl.representation.vocabulary import Corpus, Walker

__all__ = ["ProjectionView", "AxiomCorpus", "AnnotationCorpus", "Walks", "Sentences", "Mixed",
           "CorpusFile",
           "PROJECTIONS", "WALKERS", "projection_view", "make_walker"]


class AxiomCorpus(OntologyView):
    """Every axiom of the ontology as one sentence, in Manchester syntax, with full IRIs
    (:func:`mowl.corpus.extract_axiom_corpus`). Used by ``onto2vec`` and ``opa2vec``."""

    produces = "sentences"
    part = Corpus.AXIOMS.value

    def read(self, dataset):
        from mowl.corpus import extract_axiom_corpus
        return [str(s) for s in extract_axiom_corpus(self.ontology(dataset))]


class AnnotationCorpus(OntologyView):
    """Every literal annotation of a class or an individual as one sentence: the entity's
    IRI, the annotation property and the text. Used by ``opa2vec``.

    :param entity_form: How the entity is written. ``"iri"`` (default): the IRI, the same
        token as in the axiom sentences and the walks, so that the annotation is about the
        entity's vector. ``"bracketed iri"``: ``<IRI>`` for classes, a token of its own that
        no axiom sentence contains; this is what
        :func:`mowl.corpus.extract_and_save_annotation_corpus` wrote before its own
        ``entity_form`` option (default ``"iri"``, the same as this part), and so what the earlier
        ``SyntacticPlusW2VModel.generate_corpus(with_annotations=True)`` trained on (kept to
        reproduce it).
    """

    produces = "sentences"
    part = Corpus.ANNOTATIONS.value

    def __init__(self, entailment=None, entity_form="iri"):
        super().__init__(entailment)
        if entity_form not in ("iri", "bracketed iri"):
            raise ValueError("entity_form must be 'iri' or 'bracketed iri'.")
        self.entity_form = entity_form

    def read(self, dataset):
        ontology = self.ontology(dataset)
        if self.entity_form == "iri":
            from mowl.corpus import extract_annotation_corpus
            return [s.rstrip("\n") for s in extract_annotation_corpus(ontology)]
        from mowl.corpus import extract_and_save_annotation_corpus
        path = scratch_file(".annotations")
        try:
            extract_and_save_annotation_corpus(ontology, path, mode="w",
                                               entity_form="bracketed iri")
            with open(path) as f:
                return [line.rstrip("\n") for line in f]
        finally:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------------

def _walker_part(walker):
    name = type(walker).__name__
    if name == "DeepWalk":
        return Walker.DEEPWALK.value
    if name == "Node2Vec":
        return Walker.NODE2VEC.value
    return name


def make_walker(name="deepwalk", num_walks=10, walk_length=20, alpha=0.1, p=1.0, q=1.0,
                workers=1, seed=0, outfile=None):
    """A walker of :mod:`mowl.walking` by name (``"deepwalk"`` or ``"node2vec"``).

    node2vec walks take no seed (:class:`mowl.walking.Node2Vec` has none: the Scala source
    draws from an unseeded generator, ``walker.seeded`` is ``False``), so they differ from
    run to run; DeepWalk walks with one worker are the same for the same seed.
    """
    from mowl.walking import DeepWalk, Node2Vec
    outfile = outfile or scratch_file(".walks")
    if name == "deepwalk":
        return DeepWalk(num_walks, walk_length, float(alpha), outfile=outfile,
                        workers=workers, seed=seed)
    if name == "node2vec":
        return Node2Vec(num_walks, walk_length, p=p, q=q, outfile=outfile, workers=workers)
    raise ValueError(f"Unknown walker {name!r}; available: {sorted(WALKERS)}.")


#: Walker names accepted by :func:`make_walker`, with their vocabulary values.
WALKERS = {"deepwalk": Walker.DEEPWALK.value, "node2vec": Walker.NODE2VEC.value}


class Walks(SequenceSource):
    """Random walks on a graph view.

    :param view: A view that produces a graph (:class:`ProjectionView`), or ``None`` until
        one is set.
    :param walker: A :class:`mowl.walking.WalkingModel`, or ``None`` until one is set. The
        walks are written to its ``outfile``.
    :param fresh: ``True`` (default): :meth:`write` replaces the walks of an earlier
        :meth:`write` or :meth:`extend`. ``False``: :meth:`write` appends to the file, as the
        walkers did before they emptied their file themselves. :meth:`extend` always appends.

    Used by ``dl2vec``, ``owl2vec_star`` and ``taxonomy_walks``.
    """

    def __init__(self, view=None, walker=None, fresh=True):
        self.view = view
        self.walker = walker
        self.fresh = bool(fresh)
        self.edges = None

    @property
    def part(self):
        return "none" if self.walker is None else _walker_part(self.walker)

    @property
    def path(self):
        return self.walker.outfile

    def _edges(self, dataset):
        if self.view is None:
            raise AttributeError("The walks have no graph view (a projector) yet.")
        if self.view.produces != "graph":
            raise TypeError(f"Walks need a graph view, not {type(self.view).__name__}.")
        self.edges = self.view.read(dataset)
        return self.edges

    def write(self, dataset):
        edges = self._edges(dataset)
        if self.fresh and os.path.exists(self.path):
            open(self.path, "w").close()
        self._walk(edges, append=not self.fresh)
        return self.path

    def _walk(self, edges, append, **kwargs):
        """Walk; the walkers empty their file first unless they are told to append
        (``DeepWalk.append``; node2vec's Java code never appends)."""
        walker = self.walker
        old = getattr(walker, "append", None)
        if old is not None:
            walker.append = append
        try:
            walker.walk(edges, **kwargs)
        finally:
            if old is not None:
                walker.append = old

    def extend(self, dataset, entities):
        """Walks through the given entities on the graph of the dataset as it is now,
        appended to the walks there are."""
        edges = self._edges(dataset)
        self._walk(edges, append=True, nodes_of_interest=list(entities))
        return self.path

    def reseed(self, seed):
        if hasattr(self.walker, "seed"):
            self.walker.seed = int(seed)

    @property
    def parts(self):
        return {"sequences": "walks", "projection": "none" if self.view is None
                else self.view.part, "walker": self.part,
                "entailment": "none" if self.view is None or self.view.entailment is None
                else self.view.entailment.describe()}


class Sentences(SequenceSource):
    """The sentences of a corpus view, one per line.

    :param view: A view that produces sentences (:class:`AxiomCorpus`,
        :class:`AnnotationCorpus`).
    :param path: The file the sentences are written to (default: a new file in the
        temporary directory).

    Used by ``onto2vec`` and ``opa2vec``.
    """

    def __init__(self, view, path=None):
        if view.produces != "sentences":
            raise TypeError(f"Sentences need a corpus view, not {type(view).__name__}.")
        self.view = view
        self._path = path or scratch_file(".corpus")

    part = "sentences"

    @property
    def path(self):
        return self._path

    def sentences(self, dataset):
        """The sentences, as a list of strings."""
        return self.view.read(dataset)

    def write(self, dataset):
        with open(self.path, "w") as f:
            for sentence in self.sentences(dataset):
                f.write(f"{sentence}\n")
        return self.path

    @property
    def parts(self):
        return {"sequences": "sentences", "corpus": self.view.part,
                "entailment": "none" if self.view.entailment is None
                else self.view.entailment.describe()}


def _join(values):
    out = []
    for v in values:
        if v != "none" and v not in out:
            out.append(v)
    return " + ".join(out) if out else "none"


class Mixed(SequenceSource):
    """Several sources in one corpus: the sequences of each, in order.

    :param sources: The :class:`~mowl.representation.sequence.SequenceSource` objects.
    :param path: The file of the joint corpus (default: a new file in the temporary
        directory).

    OPA2Vec is axiom sentences followed by annotation sentences; OWL2Vec* with its lexical
    document is walks followed by annotation sentences. Used by ``opa2vec`` and
    ``owl2vec_star``.
    """

    part = "mixed"

    def __init__(self, *sources, path=None):
        if not sources:
            raise ValueError("Mixed needs at least one source.")
        self.sources = list(sources)
        self._path = path or scratch_file(".corpus")

    @property
    def path(self):
        return self._path

    def _concatenate(self, paths):
        with open(self.path, "w") as out:
            for path in paths:
                with open(path) as f:
                    shutil.copyfileobj(f, out)
        return self.path

    def write(self, dataset):
        return self._concatenate([s.write(dataset) for s in self.sources])

    def extend(self, dataset, entities):
        return self._concatenate([s.extend(dataset, entities) for s in self.sources])

    def reseed(self, seed):
        for source in self.sources:
            source.reseed(seed)

    @property
    def parts(self):
        each = [s.parts for s in self.sources]
        keys = []
        for p in each:
            keys += [k for k in p if k not in keys]
        return {k: _join([p.get(k, "none") for p in each]) for k in keys}


class CorpusFile(SequenceSource):
    """A corpus that is already written: the file ``path``, used as it is (one sequence per
    line). :meth:`write` checks that it exists and changes nothing. Used when a model is
    given a corpus file of its own (``SyntacticPlusW2VModel(corpus_filepath=...)``)."""

    part = "corpus file"

    def __init__(self, path):
        self._path = path

    @property
    def path(self):
        return self._path

    def write(self, dataset):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"The corpus file {self.path} does not exist.")
        return self.path

    @property
    def parts(self):
        return {"sequences": "corpus file"}
