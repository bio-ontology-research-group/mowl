"""Encoders, entity readings, measures and scorers of the methods that learn from sequences
(``dl2vec``, ``owl2vec_star``, ``taxonomy_walks``, ``onto2vec``, ``opa2vec``).

* Encoders (:class:`~mowl.representation.sequence.SequenceEncoder`):
  :class:`Word2VecEncoder` (skip-gram or CBOW, every option of
  :class:`gensim.models.Word2Vec`) and :class:`FastTextEncoder` (gensim's fastText, which
  also gives vectors to pieces of tokens).
* Reading (:class:`~mowl.representation.sequence.EntityReading`): :class:`TokenVectors`.
* Measures (:class:`~mowl.representation.sequence.PairMeasure`): :class:`DotSigmoid`
  (:math:`1 - \\sigma(x \\cdot y)`, the score of the earlier ``evaluation_model``) and
  :class:`Cosine`.
* Scorers (:class:`~mowl.representation.sequence.TaskScorer`): :class:`PairClassifier` (the
  scoring head :class:`mowl.evaluation.model_scoring.PairClassifierHead`, one classifier per
  relation) and :class:`SimilarityScorer` (the measure of the geometry).

:data:`ENCODERS`, :data:`READINGS`, :data:`PAIR_MEASURES` and :data:`SCORERS` name them.
"""

import zlib

import numpy as np
import torch as th

from mowl.representation.sequence import (SequenceEncoder, EntityReading, PairMeasure,
                                          TaskScorer)
from mowl.representation.vocabulary import Encoder, Reading, Measure, Scorer

__all__ = ["Word2VecEncoder", "FastTextEncoder", "TokenVectors", "DotSigmoid", "Cosine",
           "PairClassifier", "SimilarityScorer", "stable_hash", "ENCODERS", "READINGS",
           "PAIR_MEASURES", "SCORERS"]


def stable_hash(text):
    """A hash of a string that is the same in every Python process (CRC-32).

    Word2Vec seeds the start vector of every token with ``hashfxn(token + str(seed))``;
    its default, Python's ``hash``, changes from process to process unless
    ``PYTHONHASHSEED`` is set, so the same ``seed`` gave other vectors in every run.
    """
    return zlib.crc32(str(text).encode("utf-8"))


class Word2VecEncoder(SequenceEncoder):
    """Word2Vec (Mikolov et al. 2013) on the sequences, through :mod:`gensim`.

    :param args: Passed to :class:`gensim.models.Word2Vec`.
    :param restart: ``False`` (default): a model that already has a vocabulary (trained
        before, or loaded with :meth:`load`) keeps its vectors when it is prepared again;
        new tokens are added and training continues. ``True``: every :meth:`prepare` starts
        anew from the seed, which is what the earlier ``RandomWalkPlusW2VModel`` and
        ``SyntacticPlusW2VModel`` did on every ``train`` (so ``from_pretrained`` followed by
        ``train`` discarded the loaded vectors).
    :param model: An existing :class:`gensim.models.Word2Vec` to use instead of a new one.
    :param kwargs: Passed to :class:`gensim.models.Word2Vec` (``vector_size``, ``sg``,
        ``window``, ``negative``, ``min_count``, ``workers``, ``seed``, ``epochs``, ...).
        Without ``hashfxn`` the start vectors are seeded with :func:`stable_hash`, so that
        the same ``seed`` gives the same vectors in every process (with ``workers=1``;
        several workers train in a different order every time).

    Used by ``dl2vec``, ``owl2vec_star``, ``taxonomy_walks``, ``onto2vec`` and ``opa2vec``.
    """

    #: The gensim class.
    backend = "Word2Vec"

    def __init__(self, *args, restart=False, model=None, **kwargs):
        self.restart = bool(restart)
        if model is None:
            kwargs.setdefault("hashfxn", stable_hash)
            model = self._gensim_class()(*args, **kwargs)
        self.model = model

    def _gensim_class(self):
        import gensim.models
        return getattr(gensim.models, self.backend)

    # -- state --------------------------------------------------------------------------
    @property
    def part(self):
        skip = getattr(self.model, "sg", 0) == 1
        if self.backend == "FastText":
            return (Encoder.FASTTEXT_SKIPGRAM if skip else Encoder.FASTTEXT_CBOW).value
        return (Encoder.WORD2VEC_SKIPGRAM if skip else Encoder.WORD2VEC_CBOW).value

    @property
    def vector_size(self):
        return self.model.vector_size

    @property
    def epochs(self):
        """The number of passes of the gensim model (its ``epochs``)."""
        return self.model.epochs

    @property
    def ready(self):
        return len(self.model.wv) > 0

    @property
    def learning_rates(self):
        """``(alpha, min_alpha)``: the learning rate falls from the first to the second
        over the passes of one training."""
        return self.model.alpha, self.model.min_alpha

    def load(self, path):
        """Use the gensim model saved at ``path``."""
        self.model = self._gensim_class().load(path)
        return self

    def reset(self):
        """A new gensim model with the same settings and no vocabulary."""
        params = {k: getattr(self.model, k) for k in (
            "vector_size", "alpha", "window", "min_count", "max_vocab_size", "sample",
            "seed", "workers", "min_alpha", "sg", "hs", "negative", "ns_exponent",
            "cbow_mean", "hashfxn", "epochs", "sorted_vocab", "batch_words", "shrink_windows")
            if hasattr(self.model, k)}
        if hasattr(self.model, "max_final_vocab"):
            params["max_final_vocab"] = self.model.max_final_vocab
        self.model = self._gensim_class()(**params)

    # -- training -----------------------------------------------------------------------
    @staticmethod
    def _corpus(path):
        from gensim.models.word2vec import LineSentence
        return LineSentence(path)

    def prepare(self, path, update=False):
        """Read the vocabulary of the corpus file ``path``.

        With ``update`` (or, unless :attr:`restart`, whenever the model has a vocabulary
        already) the vectors there are stay and only new tokens are added; a corpus without
        new tokens leaves the vocabulary as it is.
        """
        corpus = self._corpus(path)
        keep = update or (self.ready and not self.restart)
        if not keep:
            self.model.build_vocab(corpus, update=False)
            return
        if not self.ready:
            self.model.build_vocab(corpus, update=False)
            return
        lines, new = 0, False
        known = self.model.wv.key_to_index
        for sentence in corpus:
            lines += 1
            new = new or any(token not in known for token in sentence)
        if new:
            self.model.build_vocab(corpus, update=True)
        self.model.corpus_count = lines

    def fit(self, path, epochs, start_alpha=None, end_alpha=None):
        if epochs <= 0:
            return None
        kwargs = {}
        if start_alpha is not None:
            kwargs.update(start_alpha=start_alpha, end_alpha=end_alpha)
        if getattr(self.model, "compute_loss", False):
            kwargs["compute_loss"] = True
        self.model.train(self._corpus(path), total_examples=self.model.corpus_count,
                         epochs=epochs, **kwargs)
        if getattr(self.model, "compute_loss", False):
            return float(self.model.get_latest_training_loss())
        return None

    # -- vectors ------------------------------------------------------------------------
    def tokens(self):
        return list(self.model.wv.index_to_key)

    def has(self, token):
        return token in self.model.wv.key_to_index

    def vector(self, token):
        return self.model.wv[token]

    def vectors_of(self, tokens):
        wv = self.model.wv
        rows = [wv.key_to_index.get(t, -1) for t in tokens]
        known = th.tensor([r >= 0 for r in rows], dtype=th.bool)
        out = np.zeros((len(rows), wv.vector_size), dtype=np.float32)
        for i, r in enumerate(rows):
            if r >= 0:
                out[i] = wv.vectors[r]
        return th.from_numpy(out), known


class FastTextEncoder(Word2VecEncoder):
    """fastText (Bojanowski et al. 2017) on the sequences, through :mod:`gensim`: Word2Vec
    whose token vectors are sums of vectors of character n-grams (an IRI shares pieces with
    IRIs of the same namespace). Same options as :class:`Word2VecEncoder`, plus those of
    :class:`gensim.models.FastText` (``min_n``, ``max_n``, ``bucket``).

    Only tokens of the corpus count as known (a fastText model has a vector for any
    string, but an entity that never occurs has no context).
    """

    backend = "FastText"

    def reset(self):
        params = {k: getattr(self.model, k) for k in (
            "vector_size", "alpha", "window", "min_count", "sample", "seed", "workers",
            "min_alpha", "sg", "hs", "negative", "cbow_mean", "hashfxn", "epochs")
            if hasattr(self.model, k)}
        wv = self.model.wv
        params.update(min_n=wv.min_n, max_n=wv.max_n, bucket=wv.bucket)
        self.model = self._gensim_class()(**params)


#: Encoder name -> class.
ENCODERS = {"word2vec": Word2VecEncoder, "fasttext": FastTextEncoder}


# ---------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------

class TokenVectors(EntityReading):
    """The vector of an entity is the vector of the token that is its IRI; an entity whose
    IRI is not a token of the corpus has none (``known`` is ``False``, the vector is zero).

    :param normalise: Scale every vector to unit length.

    Used by ``dl2vec``, ``owl2vec_star``, ``taxonomy_walks``, ``onto2vec`` and ``opa2vec``.
    """

    part = Reading.TOKEN.value

    def __init__(self, normalise=False):
        self.normalise = bool(normalise)

    def vectors(self, encoder, names):
        vectors, known = encoder.vectors_of(list(names))
        if self.normalise:
            norm = vectors.norm(dim=1, keepdim=True).clamp(min=1e-12)
            vectors = vectors / norm
        return vectors, known


#: Reading name -> class.
READINGS = {"token": TokenVectors}


# ---------------------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------------------

def _mask(cost, a, b, fill):
    known = (a.known & b.known)[:, None]
    return th.where(known, cost, th.full_like(cost, fill))


class DotSigmoid(PairMeasure):
    """:math:`1 - \\sigma(x \\cdot y)`: how unlike two entities are, from the dot product of
    their vectors; ``inf`` if one has no vector. The score of the ``evaluation_model`` of
    the word2vec models of mOWL. It is symmetric: a similarity, not an inclusion. ``apart``
    is :math:`\\sigma(x \\cdot y)`.

    Used by ``dl2vec``, ``owl2vec_star``, ``taxonomy_walks``, ``onto2vec`` and ``opa2vec``.
    """

    part = Measure.SIMILARITY.value

    def _similarity(self, a, b):
        return th.sigmoid((a.vectors * b.vectors).sum(dim=1, keepdim=True))

    def cost(self, a, b):
        return _mask(1 - self._similarity(a, b), a, b, float("inf"))

    def apart(self, a, b):
        return _mask(self._similarity(a, b), a, b, float("inf"))


class Cosine(PairMeasure):
    """:math:`(1 - \\cos(x, y)) / 2`, in :math:`[0, 1]`; ``inf`` if one has no vector.
    ``apart`` is :math:`(1 + \\cos(x, y)) / 2`."""

    part = Measure.SIMILARITY.value

    def _cos(self, a, b):
        return th.nn.functional.cosine_similarity(a.vectors, b.vectors, dim=1,
                                                  eps=1e-12)[:, None]

    def cost(self, a, b):
        return _mask((1 - self._cos(a, b)) / 2, a, b, float("inf"))

    def apart(self, a, b):
        return _mask((1 + self._cos(a, b)) / 2, a, b, float("inf"))


#: Measure name -> class.
PAIR_MEASURES = {"dot sigmoid": DotSigmoid, "cosine": Cosine}


# ---------------------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------------------

def _columns(gci_name):
    if gci_name in ("gci0", "class_assertion"):
        return 0, 1
    if gci_name in ("gci2", "object_property_assertion"):
        return 0, 2
    from mowl.evaluation.model_scoring import UnsupportedAxiomForm
    raise UnsupportedAxiomForm(
        f"A similarity of two entities scores 'gci0', 'gci2', 'class_assertion' and "
        f"'object_property_assertion' axioms, not {gci_name!r}.")


class SimilarityScorer(TaskScorer):
    """The cost of a candidate axiom is the measure of the geometry (its ``inside``) between
    its two entities: :math:`C` and :math:`D` of :math:`C \\sqsubseteq D` and of
    :math:`C \\sqsubseteq \\exists r.D` (the relation is not used), the individual and the
    class of :math:`C(a)`, the two individuals of :math:`r(a, b)`."""

    part = Scorer.SIMILARITY.value

    def __call__(self, model, gci_name, rows):
        geometry = model.module
        first, second = _columns(gci_name)
        if gci_name == "class_assertion":
            a, b = geometry.individuals(rows[:, 0]), geometry.classes(rows[:, 1])
        elif gci_name == "object_property_assertion":
            a, b = geometry.individuals(rows[:, 0]), geometry.individuals(rows[:, 2])
        else:
            a, b = geometry.classes(rows[:, first]), geometry.classes(rows[:, second])
        return geometry.inside(a, b).reshape(-1).float().to(rows.device)


class PairClassifier(TaskScorer):
    """A classifier on the pair of vectors, trained on the training axioms, one per relation
    (:class:`mowl.evaluation.model_scoring.PairClassifierHead`): the cost of
    :math:`C \\sqsubseteq D` is minus the logit of the classifier for ``gci0``, that of
    :math:`C \\sqsubseteq \\exists r.D` minus the logit of the classifier of :math:`r`. It is
    asymmetric and relation-aware, which a similarity is not; it is how vector-only methods
    are applied to subsumption prediction in the literature (OWL2Vec* trains such a
    classifier).

    :param head: A :class:`~mowl.evaluation.model_scoring.PairClassifierHead`, or ``None``
        for one with ``options``.
    :param options: Options of a new head (``hidden``, ``epochs``, ``negatives``,
        ``learning_rate``, ``batch_size``, ``seed``).

    The classifier is fitted the first time a form is scored, and fitted again after the
    encoder is trained (:meth:`reset`). Used by ``dl2vec``, ``owl2vec_star``,
    ``taxonomy_walks``, ``onto2vec`` and ``opa2vec``.
    """

    part = Scorer.PAIR_CLASSIFIER.value

    def __init__(self, head=None, **options):
        self.options = options
        self.head = head if head is not None else self._new_head()

    def _new_head(self):
        from mowl.evaluation.model_scoring import PairClassifierHead
        return PairClassifierHead(**self.options)

    def reset(self):
        head = self.head
        names = ("hidden", "epochs", "negatives", "learning_rate", "batch_size", "seed")
        options = {k: getattr(head, k) for k in names}
        self.head = type(head)(**options)

    def __call__(self, model, gci_name, rows):
        return self.head(model, gci_name, rows)


#: Scorer name -> class.
SCORERS = {"pair classifier": PairClassifier, "similarity": SimilarityScorer}
