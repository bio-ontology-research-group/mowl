r"""Which axioms an axis-parallel subspace embedding violates once it is made 0/1.

Embed2Reason-style models give every concept and relation an indicator of the coordinate
axes that span its subspace, and every entity a vector. After rounding the indicators at a
threshold the axioms can be decided with set operations on the axis supports, independently
of any loss function: an inclusion is a subset, a conjunction an intersection of supports, a
disjunction a union, a negation the complement among the real axes, and a vector belongs to a
subspace when what lies outside the supports is negligible against its norm.
:func:`discretised_violations` returns the rows of the training axioms for which that fails.
It applies to any model whose axioms are given as rows of the kinds listed in
:data:`KINDS`.
"""

import numpy as np

__all__ = ["KINDS", "discretised_violations"]

#: The kinds of rows decided here (the axiom kinds of the subspace logic).
KINDS = ("class_assertion", "role_assertion", "concept_inclusion", "concept_conjunction",
         "concept_disjunction", "concept_negation", "concept_disjointness",
         "role_inclusion", "range")


def discretised_violations(concepts, relations, vectors, rows, dim, tol=1e-3):
    """The training rows violated by the 0/1 embedding.

    :param concepts: ``(classes, 2 * dim)`` 0/1 indicators of the concept subspaces.
    :param relations: ``(relations, 2 * dim)`` 0/1 indicators of the relation subspaces.
    :param vectors: ``(entities, 2 * dim)`` entity vectors.
    :param rows: Iterable of ``(kind, rows)`` with ``kind`` one of :data:`KINDS` and ``rows``
        an integer array (or tensor) of index rows of that kind. Other kinds are skipped.
    :param dim: :math:`d`; vectors have :math:`2d` coordinates, the first :math:`d` real.
    :param tol: A vector is in a subspace if the part outside is at most ``tol`` times its
        norm (and it is not 0).
    :return: List of ``(kind, row)`` with ``row`` a tuple.
    """
    concept_support = [frozenset(np.flatnonzero(v)) for v in concepts]
    relation_support = [frozenset(np.flatnonzero(v)) for v in relations]

    def inside(support, v):
        norm = np.linalg.norm(v)
        outside = [t for t in range(len(v)) if t not in support]
        return norm > 0 and np.linalg.norm(v[outside]) <= tol * norm

    bad = []
    for kind, kind_rows in rows:
        for row in kind_rows.tolist():
            if kind == "class_assertion":
                ok = inside(concept_support[row[1]], vectors[row[0]])
            elif kind == "role_assertion":
                v = np.concatenate([vectors[row[0]][:dim], vectors[row[2]][:dim]])
                ok = inside(relation_support[row[1]], v)
            elif kind == "concept_inclusion":
                ok = concept_support[row[0]] <= concept_support[row[1]]
            elif kind == "concept_conjunction":
                ok = concept_support[row[0]] == frozenset.intersection(
                    *[concept_support[i] for i in row[1:]])
            elif kind == "concept_disjunction":
                ok = concept_support[row[0]] == frozenset.union(
                    *[concept_support[i] for i in row[1:]])
            elif kind == "concept_negation":
                ok = concept_support[row[0]] == frozenset(range(dim)) - concept_support[row[1]]
            elif kind == "concept_disjointness":
                ok = not (concept_support[row[0]] & concept_support[row[1]])
            elif kind == "role_inclusion":
                ok = relation_support[row[0]] <= relation_support[row[1]]
            elif kind == "range":
                ok = not ({t - dim for t in relation_support[row[0]] if t >= dim}
                          & concept_support[row[1]])
            else:
                continue
            if not ok:
                bad.append((kind, tuple(row)))
    return bad
