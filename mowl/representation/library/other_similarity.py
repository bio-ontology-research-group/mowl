"""Walk and sentence methods (``dl2vec``, ``owl2vec_star``, ``taxonomy_walks``,
``gda_projection``, ``onto2vec``, ``opa2vec``) in the common interface.

In these methods every name (class, individual, role) is one word2vec vector; the only
thing the model defines is how similar two names are. :class:`SimilarityGeometry` says
exactly that and nothing more:

* ``classes`` and ``individuals`` are the vectors of the names (:class:`Tokens`);
* ``inside(a, b)`` and ``member(a, b)`` are the model's own score
  ``1 - sigmoid(a . b)`` (the one of ``evaluation_model``), which is a **similarity** (it
  is symmetric), not an inclusion: a low cost means the two names occur in similar
  contexts, not that one is a subset of the other. ``apart`` is the complement,
  ``sigmoid(a . b)``;
* ``exists``, ``conjunction``, ``negation`` and ``disjunction`` are unavailable
  (:class:`~mowl.representation.ConstructorUnavailable`).

This geometry describes a trained ``RandomWalkPlusW2VModel`` or ``SyntacticPlusW2VModel``
(and the direct implementations of the baseline area) from the outside, by reading its
word2vec model. The methods themselves are combinations of parts (view, sequences, encoder,
reading, measure, scorer): see :mod:`mowl.representation.sequence`; the ``module`` of those
classes is that combination.
"""

import torch as th

from mowl.representation.adapters import register_adapter
from mowl.representation.geometry import Geometry
from mowl.representation.sequence import Tokens

__all__ = ["Tokens", "SimilarityGeometry"]


class SimilarityGeometry(Geometry):
    """The geometry of a word2vec model on an ontology: points and a similarity.

    :param model: A trained ``RandomWalkPlusW2VModel`` or ``SyntacticPlusW2VModel``.

    The vectors are views on the word2vec array (``wv.vectors``), not copies; the model is
    not trained through this geometry. A name without a vector (it did not occur in the
    corpus) has a zero vector and the cost ``inf``.
    """

    def __init__(self, model):
        if model.w2v_model is None:
            raise AttributeError("The word2vec model is not set or trained.")
        self.model = model
        self.wv = model.w2v_model.wv
        self.vectors = th.from_numpy(self.wv.vectors)
        self._classes = self._rows(model.class_index_dict)
        self._individuals = self._rows(model.individual_index_dict)

    def _rows(self, index_dict):
        rows = th.full((max(index_dict.values(), default=-1) + 1,), -1, dtype=th.long)
        for iri, i in index_dict.items():
            rows[i] = self.wv.key_to_index.get(iri, -1)
        return rows

    def _tokens(self, rows):
        known = rows >= 0
        vectors = self.vectors[rows.clamp(min=0)]
        return Tokens(th.where(known[:, None], vectors, th.zeros_like(vectors)), known)

    @staticmethod
    def _lookup(table, ids):
        ids = ids.cpu()
        out = th.full(ids.shape, -1, dtype=th.long)
        valid = ids < len(table)
        out[valid] = table[ids[valid]]
        return out

    def classes(self, ids):
        return self._tokens(self._lookup(self._classes, ids))

    def individuals(self, ids):
        """Vectors of individuals; an ontology without individuals gives unknown names."""
        return self._tokens(self._lookup(self._individuals, ids))

    def similarity(self, a, b):
        """``sigmoid(a . b)`` in ``(0, 1)``, shape ``(B, 1)``; zero for unknown names."""
        s = th.sigmoid((a.vectors * b.vectors).sum(dim=1, keepdim=True))
        return th.where((a.known & b.known)[:, None], s, th.zeros_like(s))

    def inside(self, a, b):
        """``1 - sigmoid(a . b)``: a dissimilarity (the model's own score), *not* an
        inclusion; ``inf`` if a name has no vector."""
        cost = 1 - self.similarity(a, b)
        return th.where((a.known & b.known)[:, None], cost, th.full_like(cost, float("inf")))

    def apart(self, a, b):
        """``sigmoid(a . b)``: the similarity itself, as the cost of being far apart."""
        return self.similarity(a, b)

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def num_individuals(self):
        return len(self._individuals)

    @property
    def parts(self):
        return {"classes": "token", "relations": "none", "individuals": "token",
                "measure": "similarity", "conjunction": "none"}


register_adapter("RandomWalkPlusW2VModel", "SyntacticPlusW2VModel")(SimilarityGeometry)
