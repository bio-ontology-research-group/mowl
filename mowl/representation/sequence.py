"""The parts of methods that learn from sequences: walks on a graph, sentences of a corpus.

DL2Vec, OWL2Vec*, Onto2Vec, OPA2Vec and random walks on the taxonomy have no region and no
relation model. They are a chain of five parts, each of which can be exchanged on its own:

1. an :class:`OntologyView`: what is read from the ontology, either a graph (a projection of
   :mod:`mowl.projection`) or sentences (axioms or annotations, :mod:`mowl.corpus`),
   optionally with the axioms a reasoner entails added first (:class:`Entailment`);
2. a :class:`SequenceSource`: the sequences made from the views. :class:`Walks` runs a
   walker of :mod:`mowl.walking` on a graph view, :class:`Sentences` writes the sentences of
   a corpus view, :class:`Mixed` puts several sources into one corpus (OWL2Vec* mixes walks
   and text, OPA2Vec axioms and annotations);
3. a :class:`SequenceEncoder`: what turns the sequences into one vector per token (Word2Vec
   today, :class:`~mowl.representation.library.walks_encoders.Word2VecEncoder`; any
   encoder with the same four methods fits);
4. an :class:`EntityReading`: which token is the vector of a class, an individual, a
   relation;
5. a :class:`PairMeasure` (how similar two entities are; it is the geometry's "inside") and
   a :class:`TaskScorer` (the score of a candidate axiom that the evaluation regimes rank:
   the similarity of the two entities, or a classifier on their vectors that is trained for
   each relation).

:class:`SequenceGeometry` holds the five parts and is a
:class:`~mowl.representation.Geometry`: it answers the same questions as the region
methods (the vector of a class, the cost of :math:`C \\sqsubseteq D`), and its
:meth:`~SequenceGeometry.score` has the signature of a scorer of the evaluation regimes
(:mod:`mowl.evaluation`), so a sequence method is evaluated by the same ranking as every
other method. :class:`~mowl.models.ComposedSequenceModel` trains it.

The concrete parts are in :mod:`mowl.representation.library.walks_views` (views and
sources), :mod:`mowl.representation.library.walks_encoders` (encoders, readings, measures,
scorers) and the named methods in :mod:`mowl.representation.library.walks_methods`.
"""

import os
import tempfile

import torch as th

from mowl.representation.geometry import Geometry
from mowl.representation.regions import ConstructorUnavailable, Region

__all__ = ["Tokens", "Entailment", "OntologyView", "SequenceSource", "SequenceEncoder",
           "EntityReading", "PairMeasure", "TaskScorer", "SequenceGeometry", "scratch_file"]


def scratch_file(suffix=""):
    """A fresh file name in the temporary directory (``$TMPDIR``); the file does not exist."""
    handle, path = tempfile.mkstemp(suffix=suffix)
    os.close(handle)
    os.remove(path)
    return path


class Tokens(Region):
    """The vectors of names, ``(B, d)``; ``known`` (``(B,)``, bool) marks the names that have
    a vector (a name that never occurs in the sequences has none)."""

    def __init__(self, vectors, known):
        self.vectors = vectors
        self.known = known

    @property
    def centre(self):
        return self.vectors

    def __len__(self):
        return self.vectors.shape[0]


# ---------------------------------------------------------------------------------------
# 1. View of the ontology
# ---------------------------------------------------------------------------------------

class Entailment:
    """The axioms a reasoner entails, added to the ontology before a view reads it.

    :param reasoner: A backend name of :mod:`mowl.reasoning` (``"elk"``, ``"hermit"``,
        ...) or ``"auto"`` (the cheapest backend complete for the ontology).
    :param kinds: Which entailed axioms are added: ``"subclass"`` (:math:`C \\sqsubseteq D`
        between named classes), ``"equivalent"``, ``"disjoint"``.
    :param direct: Only direct superclasses (for ``"subclass"``).

    The ontology of the dataset is not changed: :meth:`ontology` returns a new ontology with
    the axioms of the dataset and the entailed ones.
    """

    def __init__(self, reasoner="elk", kinds=("subclass",), direct=False):
        unknown = set(kinds) - {"subclass", "equivalent", "disjoint"}
        if unknown:
            raise ValueError(f"Unknown kinds of entailed axioms {sorted(unknown)}.")
        self.reasoner = reasoner
        self.kinds = tuple(kinds)
        self.direct = bool(direct)

    def axioms(self, ontology):
        """The entailed axioms of ``ontology`` (a list of OWL API axioms)."""
        from mowl.reasoning import MOWLReasoner
        from mowl.reasoning.backends import create_reasoner
        reasoner, _ = create_reasoner(ontology, self.reasoner, warn=False)
        try:
            owl = MOWLReasoner(reasoner)
            classes = list(ontology.getClassesInSignature())
            out = []
            if "subclass" in self.kinds:
                out += owl.infer_subclass_axioms(classes, direct=self.direct)
            if "equivalent" in self.kinds:
                out += owl.infer_equivalent_class_axioms(classes)
            if "disjoint" in self.kinds:
                out += owl.infer_disjoint_class_axioms(classes)
            return out
        finally:
            reasoner.dispose()

    def ontology(self, ontology):
        """A new ontology with the axioms of ``ontology`` and the entailed ones."""
        from java.util import HashSet
        from mowl.owlapi import OWLAPIAdapter
        manager = OWLAPIAdapter().owl_manager
        axioms = HashSet()
        axioms.addAll(ontology.getAxioms())
        for axiom in self.axioms(ontology):
            axioms.add(axiom)
        return manager.createOntology(axioms)

    def describe(self):
        return f"entailed by {self.reasoner} ({', '.join(self.kinds)})"


class OntologyView:
    """What a method reads from the ontology.

    :attr:`produces` is ``"graph"`` (:meth:`read` returns a list of
    :class:`mowl.projection.Edge`) or ``"sentences"`` (a list of strings, one sentence each).
    :attr:`part` is the value of the view in the vocabulary
    (:class:`~mowl.representation.vocabulary.Projection` or
    :class:`~mowl.representation.vocabulary.Corpus`).

    :param entailment: An :class:`Entailment`, or ``None`` to read the asserted axioms only.
    """

    produces = ""
    part = ""

    def __init__(self, entailment=None):
        self.entailment = entailment

    def ontology(self, dataset):
        """The ontology the view reads: the training ontology of ``dataset``, with the
        entailed axioms if the view has an :class:`Entailment`."""
        if self.entailment is None:
            return dataset.ontology
        return self.entailment.ontology(dataset.ontology)

    def read(self, dataset):
        raise NotImplementedError


# ---------------------------------------------------------------------------------------
# 2. Sequences
# ---------------------------------------------------------------------------------------

class SequenceSource:
    """Where the sequences come from. A source writes its sequences to a file, one sequence
    per line with tokens separated by white space, and says where (:attr:`path`).

    :meth:`write` makes the sequences anew (the file is replaced). :meth:`extend` adds the
    sequences that mention given entities, after axioms were added to the dataset (the
    sources of :meth:`mowl.base_models.Model.add_axioms`). :meth:`reseed` changes the seed
    of the random parts, so that :meth:`write` gives other sequences (a training regime
    that refreshes the corpus uses it).
    """

    part = ""

    @property
    def path(self):
        raise NotImplementedError

    def write(self, dataset):
        """Write the sequences of ``dataset`` to :attr:`path`; returns the path."""
        raise NotImplementedError

    def extend(self, dataset, entities):
        """Make the sequences of ``dataset`` that mention ``entities`` (names); by default
        all sequences are made anew."""
        return self.write(dataset)

    def reseed(self, seed):
        """Use ``seed`` for the random parts of the next :meth:`write` (default: none)."""

    @property
    def parts(self):
        """The vocabulary values of the source and of the views it reads."""
        return {"sequences": self.part}


# ---------------------------------------------------------------------------------------
# 3. Encoder
# ---------------------------------------------------------------------------------------

class SequenceEncoder:
    """What turns sequences into one vector per token.

    An encoder trains on a corpus file (:meth:`prepare` reads its vocabulary, :meth:`fit`
    trains) and then answers :meth:`vectors_of`. :attr:`part` is its value in
    :class:`~mowl.representation.vocabulary.Encoder`.
    """

    part = ""

    @property
    def vector_size(self):
        raise NotImplementedError

    @property
    def ready(self):
        """Whether the encoder has a vocabulary (it has been prepared or loaded)."""
        raise NotImplementedError

    def prepare(self, path, update=False):
        """Read the vocabulary of the corpus ``path``. ``update``: keep the vectors that
        exist and add the new tokens."""
        raise NotImplementedError

    def fit(self, path, epochs, start_alpha=None, end_alpha=None):
        """Train for ``epochs`` passes over the corpus ``path``; ``start_alpha`` and
        ``end_alpha`` set the learning rate of these passes (for training that is split
        into several calls). Returns the training loss if the encoder reports one."""
        raise NotImplementedError

    def vectors_of(self, tokens):
        """``(vectors, known)``: a float tensor ``(len(tokens), d)`` with the vector of every
        token (zeros for a token without one) and a bool tensor that marks the tokens that
        have one."""
        raise NotImplementedError

    def tokens(self):
        """The tokens that have a vector."""
        raise NotImplementedError

    def reset(self):
        """Forget the vocabulary and the vectors (the next :meth:`prepare` starts anew)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------------------
# 4. Entity reading
# ---------------------------------------------------------------------------------------

class EntityReading:
    """Which vector an entity (class, individual, relation) gets from the encoder."""

    part = ""

    def vectors(self, encoder, names):
        """``(vectors, known)`` for the entity names ``names`` (IRIs)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------------------
# 5. Measure and scorer
# ---------------------------------------------------------------------------------------

class PairMeasure:
    """How similar two entities are, from their vectors (:class:`Tokens`): the "inside" of
    a :class:`SequenceGeometry`. A cost of shape ``(B, 1)``, lower for more similar,
    ``inf`` when a name has no vector."""

    part = ""

    def cost(self, a, b):
        raise NotImplementedError

    def apart(self, a, b):
        """Cost of the two being unrelated; default: unavailable."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no cost of disjointness.")


class TaskScorer:
    """The score of candidate axioms that an evaluation regime ranks.

    A scorer is called like the scorers of :mod:`mowl.evaluation`:
    ``scorer(model, gci_name, rows)`` returns one cost per row (lower is more plausible),
    so it can also be passed as ``scoring=`` to any regime. ``model`` is the model the
    geometry belongs to (its index dictionaries, its training ontology).
    """

    part = ""

    def __call__(self, model, gci_name, rows):
        raise NotImplementedError

    def reset(self):
        """Forget what was fitted to the vectors (called after the encoder is trained)."""


# ---------------------------------------------------------------------------------------
# The geometry
# ---------------------------------------------------------------------------------------

class SequenceGeometry(Geometry):
    """A method that learns from sequences, as a combination of its parts.

    :param source: The :class:`SequenceSource` (it holds the views it reads).
    :param encoder: The :class:`SequenceEncoder`, or ``None`` until one is set.
    :param reading: The :class:`EntityReading`.
    :param measure: The :class:`PairMeasure` (the geometry's ``inside``).
    :param scorer: The :class:`TaskScorer` used by the evaluation regimes.

    Every name is a point (:class:`Tokens`); there is no region, no relation model and no
    constructor (``exists``, ``conjunction``, ... raise
    :class:`~mowl.representation.ConstructorUnavailable`). :meth:`bind` ties the geometry to
    the model that trains it (the names of classes, individuals and relations).
    """

    def __init__(self, source, encoder, reading, measure, scorer):
        self.source = source
        self.encoder = encoder
        self.reading = reading
        self.measure = measure
        self.scorer = scorer
        self.owner = None
        self._cache = {}

    # -- binding ------------------------------------------------------------------------
    def bind(self, model):
        """Tie the geometry to ``model``: class ``i`` is the ``i``-th name of
        ``model.class_index_dict`` (and so for individuals and relations)."""
        self.owner = model
        self.refresh()
        return self

    def refresh(self):
        """Forget the vectors read so far and what the scorer fitted to them (after
        training)."""
        self._cache = {}
        if self.scorer is not None:
            self.scorer.reset()

    def _names(self, kind):
        if self.owner is None:
            raise RuntimeError("The geometry is not bound to a model (SequenceGeometry.bind).")
        index = {"classes": self.owner.class_index_dict,
                 "individuals": self.owner.individual_index_dict,
                 "relations": self.owner.object_property_index_dict}[kind]
        names = [None] * len(index)
        for name, i in index.items():
            names[i] = name
        return names

    def table(self, kind):
        """``(vectors, known)`` of all names of ``kind`` (``"classes"``, ``"individuals"``,
        ``"relations"``) in index order."""
        if kind not in self._cache:
            if self.encoder is None or not self.encoder.ready:
                raise RuntimeError("The encoder is not trained yet.")
            self._cache[kind] = self.reading.vectors(self.encoder, self._names(kind))
        return self._cache[kind]

    def _tokens(self, kind, ids):
        vectors, known = self.table(kind)
        ids = th.as_tensor(ids).reshape(-1).cpu()
        inside = ids < len(known)
        rows = ids.clamp(max=max(len(known) - 1, 0))
        if len(known) == 0:
            return Tokens(th.zeros((len(ids), self.dimension)), th.zeros(len(ids), dtype=th.bool))
        v, k = vectors[rows], known[rows] & inside
        return Tokens(th.where(k[:, None], v, th.zeros_like(v)), k)

    @property
    def dimension(self):
        return 0 if self.encoder is None else self.encoder.vector_size

    # -- Geometry -----------------------------------------------------------------------
    def classes(self, ids):
        return self._tokens("classes", ids)

    def individuals(self, ids):
        return self._tokens("individuals", ids)

    def relations(self, ids):
        """The vectors of the relations ``ids`` (tokens; there is no relation model)."""
        return self._tokens("relations", ids)

    def inside(self, a, b):
        """The measure: how similar ``a`` and ``b`` are (a similarity, not an inclusion)."""
        return self.measure.cost(a, b)

    def apart(self, a, b):
        return self.measure.apart(a, b)

    def score(self, gci_name, rows):
        """Costs of the candidate axioms ``rows`` of form ``gci_name`` (the scorer)."""
        if self.owner is None:
            raise RuntimeError("The geometry is not bound to a model (SequenceGeometry.bind).")
        return self.scorer(self.owner, gci_name, rows)

    @property
    def num_classes(self):
        return 0 if self.owner is None else len(self.owner.class_index_dict)

    @property
    def num_individuals(self):
        return 0 if self.owner is None else len(self.owner.individual_index_dict)

    @property
    def parts(self):
        parts = {"classes": "point", "relations": "token", "individuals": "point",
                 "measure": self.measure.part, "conjunction": "none"}
        parts.update(self.source.parts)
        parts["encoder"] = "none" if self.encoder is None else self.encoder.part
        parts["reading"] = self.reading.part
        parts["scorer"] = self.scorer.part
        return parts
