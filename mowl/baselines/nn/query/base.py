"""The interface of a query embedding geometry, independent of any method.

Every geometry is a :class:`QueryGeometry`: ``anchor(entity_ids)`` gives the state of the
query that is one entity, ``project``, ``intersect`` and ``negate`` are the operators and
``logit(state, entity_ids)`` is the score (higher: more likely an answer). A *state* is a
tuple of tensors with a leading batch dimension. The generic evaluation of a query tree
(:meth:`QueryGeometry.embed`) does not depend on the geometry. Unions are not operators:
the models take the best branch of the disjunctive normal form
(:func:`mowl.nn.query.expression.to_dnf`).
"""

import torch as th
import torch.nn as nn

__all__ = ["QueryGeometry", "UnsupportedOperator"]


class UnsupportedOperator(ValueError):
    """The geometry has no such operator (Query2Box has no negation)."""


def _clamp_positive(x):
    return th.clamp(x + 1.0, 0.05, 1e9)


class QueryGeometry(nn.Module):
    """Base class.

    :param num_entities: Number of entities (individuals and named classes).
    :param num_relations: Number of relations (roles, their inverses, ``isa`` and its
        inverse).
    :param dim: Embedding dimension.
    :param gamma: Margin of the score, ``logit = gamma - distance``.
    """

    #: Whether :meth:`negate` is defined.
    supports_negation = False

    def __init__(self, num_entities, num_relations, dim, gamma):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.register_buffer("gamma", th.tensor(float(gamma)))
        #: Range of the uniform initialisation (both reference codes).
        self.embedding_range = (float(gamma) + 2.0) / dim

    # -- to be provided -----------------------------------------------------------------
    def anchor(self, entity_ids):
        raise NotImplementedError

    def project(self, state, relation_ids):
        raise NotImplementedError

    def intersect(self, states):
        """``states`` is a list of states (at least two)."""
        raise NotImplementedError

    def negate(self, state):
        raise UnsupportedOperator(f"{type(self).__name__} has no negation operator.")

    def distance(self, state, entity_ids):
        """Distance of entities ``(B, K)`` to the queries ``(B,)`` in ``state``: ``(B, K)``."""
        raise NotImplementedError

    def entity_vectors(self):
        """One flat vector per entity (for inspection), ``(num_entities, *)``."""
        raise NotImplementedError

    def relation_vectors(self):
        raise NotImplementedError

    # -- shared -------------------------------------------------------------------------
    def logit(self, state, entity_ids):
        return self.gamma - self.distance(state, entity_ids)

    def embed(self, signature, id_matrix, _cursor=None):
        """State of a batch of union-free queries of one shape.

        :param signature: From :func:`mowl.nn.query.expression.signature`.
        :param id_matrix: ``LongTensor (B, n)`` with the ids of
            :func:`mowl.nn.query.expression.ids`, one row per query.
        """
        cursor = [0] if _cursor is None else _cursor
        kind = signature[0]
        children = [self.embed(s, id_matrix, cursor) for s in signature[1:]]
        if kind in ("a", "p"):
            col = id_matrix[:, cursor[0]]
            cursor[0] += 1
        if kind == "a":
            return self.anchor(col)
        if kind == "p":
            return self.project(children[0], col)
        if kind == "i":
            return self.intersect(children)
        if kind == "n":
            return self.negate(children[0])
        raise ValueError("Unions are not operators: use the disjunctive normal form.")
