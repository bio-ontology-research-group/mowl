"""One shared layer between class expressions and query-embedding operators.

Query2Box, BetaE, ConE and AConE answer *queries*: trees of the operators projection
(``∃``), intersection (``⊓``), union (``⊔``) and negation (``¬``) over anchor entities.
This is the structure of an :math:`\\mathcal{ALC}` class expression (AConE states the
correspondence explicitly, He et al. 2024, Section 4), so the four methods share this
module, and each adds only its geometry of the four operators
(:mod:`mowl.nn.query.geometry`).

A query is a :class:`Node`. Its kinds are

``"a"``  anchor entity (``value`` is an entity id),
``"p"``  projection along a relation (``value`` is a relation id) of the one child,
``"i"``  intersection of the children,
``"u"``  union of the children,
``"n"``  negation of the one child.

Three front ends build trees:

* :func:`class_expression_to_query` reads an OWL API class expression;
* :func:`from_nested` reads the nested-tuple format of the released query datasets and of
  :mod:`mowl.evaluation.query_answering` (``("e", ("r", "r"))`` with the instance
  ``(a, (r1, r2))``; ``"n"`` in the relation tuple is negation, with the id ``-2``);
* the constructors :func:`anchor`, :func:`project`, :func:`intersect`, :func:`union` and
  :func:`negate`.

:func:`to_dnf` moves unions to the top (the models embed union-free trees and take the
best of the branches, Ren et al. 2020, Section 4.3), :func:`signature` and
:func:`ids` split a tree into its shape and its ids, so that queries of equal shape are
embedded as one batch, and :func:`answers` evaluates a tree on a graph exactly.
"""

from collections.abc import Set
from dataclasses import dataclass
from itertools import product

import numpy as np

__all__ = ["AnswerSet", "answer_mask", "Node", "Vocabulary", "UnsupportedClassExpression", "anchor", "project",
           "intersect", "union", "negate", "class_expression_to_query", "from_nested",
           "template_from_nested", "to_dnf", "signature", "ids", "from_signature",
           "contains_negation", "contains_union", "answers", "NEGATION_ID"]

#: The relation id the query datasets use for the negation operator in a query instance.
NEGATION_ID = -2


class UnsupportedClassExpression(ValueError):
    """The class expression has a constructor that is not a query operator."""


@dataclass(frozen=True)
class Node:
    """A query: ``kind`` is ``"a"``, ``"p"``, ``"i"``, ``"u"`` or ``"n"``."""

    kind: str
    children: tuple = ()
    value: int = None

    def __post_init__(self):
        if self.kind not in ("a", "p", "i", "u", "n"):
            raise ValueError(f"Unknown query node kind {self.kind!r}.")
        arity = {"a": 0, "p": 1, "n": 1}.get(self.kind)
        if arity is not None and len(self.children) != arity:
            raise ValueError(f"A {self.kind!r} node has {arity} child(ren).")
        if self.kind in ("i", "u") and len(self.children) < 1:
            raise ValueError("An intersection or union needs children.")
        if self.kind in ("a", "p") and self.value is not None:
            object.__setattr__(self, "value", int(self.value))

    def __repr__(self):
        if self.kind == "a":
            return f"e{self.value}"
        if self.kind == "p":
            return f"P[r{self.value}]({self.children[0]!r})"
        if self.kind == "n":
            return f"¬({self.children[0]!r})"
        sym = " ⊓ " if self.kind == "i" else " ⊔ "
        return "(" + sym.join(repr(c) for c in self.children) + ")"


def anchor(entity=None):
    """Anchor entity ``entity`` (``None`` in a template)."""
    return Node("a", (), entity)


def project(relation, child):
    """Relational projection ``∃relation⁻.child``: the entities reached from the
    answers of ``child`` along ``relation``."""
    return Node("p", (child,), relation)


def intersect(*children):
    return children[0] if len(children) == 1 else Node("i", tuple(children))


def union(*children):
    return children[0] if len(children) == 1 else Node("u", tuple(children))


def negate(child):
    return Node("n", (child,))


# ---------------------------------------------------------------------------------------
# Class expressions
# ---------------------------------------------------------------------------------------

@dataclass
class Vocabulary:
    """How names map to ids, for :func:`class_expression_to_query`.

    :param class_id: ``f(iri) -> entity id`` of a named class (``KeyError`` if unknown).
    :param individual_id: ``f(iri) -> entity id`` of a named individual.
    :param relation_id: ``f(iri, inverse) -> relation id`` of a role; ``inverse=True`` is
        the relation that goes from the filler to the subject.
    :param isa_inverse: Relation id that goes from a class to its subclasses and
        instances.
    :param extension: If ``True`` (default) a named class ``A`` is the query "everything
        that is below ``A``" (projection of the anchor ``A`` along ``isa_inverse``); if
        ``False`` it is the anchor ``A`` itself.
    """

    class_id: object
    individual_id: object
    relation_id: object
    isa_inverse: int
    extension: bool = True

    def concept(self, iri):
        node = anchor(self.class_id(iri))
        return project(self.isa_inverse, node) if self.extension else node


def class_expression_to_query(cexpr, vocabulary):
    """The query of an OWL API class expression.

    ``∃r.C`` is the projection of the query of ``C`` along the relation from fillers to
    subjects of ``r`` (the inverse of ``r``, or ``r`` for ``∃r⁻.C``), ``C ⊓ D``, ``C ⊔ D``
    and ``¬C`` the operators of the same name, ``∀r.C`` is ``¬∃r.¬C`` and ``{a}`` is the
    anchor ``a``. The answers of the query are the entities that fall under the class
    expression in the graph the model was trained on (closed world: the universe is the
    set of entities).

    :raises UnsupportedClassExpression: For number restrictions, ``∃r.Self``, data
        properties and the like.
    """
    from mowl.owlapi import ClassExpressionType as T
    kind = cexpr.getClassExpressionType()

    def role(prop, inverse):
        if prop.isAnonymous():                       # ObjectInverseOf(r)
            return vocabulary.relation_id(str(prop.getNamedProperty().toStringID()),
                                          not inverse)
        return vocabulary.relation_id(str(prop.asOWLObjectProperty().toStringID()), inverse)

    def rec(e):
        k = e.getClassExpressionType()
        if k == T.OWL_CLASS:
            return vocabulary.concept(str(e.asOWLClass().toStringID()))
        if k == T.OBJECT_SOME_VALUES_FROM:
            return project(role(e.getProperty(), True), rec(e.getFiller()))
        if k == T.OBJECT_ALL_VALUES_FROM:
            return negate(project(role(e.getProperty(), True), negate(rec(e.getFiller()))))
        if k == T.OBJECT_INTERSECTION_OF:
            return intersect(*[rec(o) for o in e.getOperandsAsList()])
        if k == T.OBJECT_UNION_OF:
            return union(*[rec(o) for o in e.getOperandsAsList()])
        if k == T.OBJECT_COMPLEMENT_OF:
            return negate(rec(e.getOperand()))
        if k == T.OBJECT_ONE_OF:
            return union(*[anchor(vocabulary.individual_id(str(i.toStringID())))
                           for i in e.getIndividuals()])
        if k == T.OBJECT_HAS_VALUE:
            return project(role(e.getProperty(), True),
                           anchor(vocabulary.individual_id(
                               str(e.getFiller().asOWLNamedIndividual().toStringID()))))
        raise UnsupportedClassExpression(
            f"{kind} is not a query operator (supported: named classes, nominals, ∃, ∀, "
            f"⊓, ⊔, ¬ and has-value).")

    return rec(cexpr)


# ---------------------------------------------------------------------------------------
# Nested-tuple format of the query datasets
# ---------------------------------------------------------------------------------------

def _is_chain(structure):
    return (isinstance(structure, tuple) and len(structure) == 2
            and isinstance(structure[1], tuple) and len(structure[1]) > 0
            and all(isinstance(x, str) and x in ("r", "n") for x in structure[1]))


def _is_union(structure):
    return isinstance(structure, tuple) and len(structure) > 0 and structure[-1] == ("u",)


def from_nested(structure, instance):
    """The query of a structure and an instance in the nested-tuple format.

    The structure ``("e", ("r", "r"))`` with the instance ``(a, (r1, r2))`` is the
    two-hop path; ``(q1, q2)`` an intersection; ``(q1, q2, ("u",))`` a union (instance
    ``(i1, i2, (-1,))``); ``(q, ("r", "n"))`` projects and negates the sub-query ``q``
    (the negation entry of the instance is :data:`NEGATION_ID`).
    """
    if _is_chain(structure):
        head, ops = structure
        node = (anchor(instance[0]) if head == "e"
                else from_nested(head, instance[0]))
        for op, value in zip(ops, instance[1]):
            if op == "n":
                node = negate(node)
            else:
                node = project(int(value), node)
        return node
    if _is_union(structure):
        return union(*[from_nested(s, q) for s, q in zip(structure[:-1], instance[:-1])])
    if isinstance(structure, tuple) and structure and structure != "e":
        return intersect(*[from_nested(s, q) for s, q in zip(structure, instance)])
    raise ValueError(f"Cannot read the query structure {structure!r}.")


def template_from_nested(structure):
    """The shape of a structure: a query whose anchors and relations are ``None``."""
    if _is_chain(structure):
        head, ops = structure
        node = anchor() if head == "e" else template_from_nested(head)
        for op in ops:
            node = negate(node) if op == "n" else project(None, node)
        return node
    if _is_union(structure):
        return union(*[template_from_nested(s) for s in structure[:-1]])
    return intersect(*[template_from_nested(s) for s in structure])


# ---------------------------------------------------------------------------------------
# Normal form, batching
# ---------------------------------------------------------------------------------------

def to_dnf(node):
    """A list of union-free queries whose union is ``node`` (unions moved to the top).

    ``∃r.(C ⊔ D) = ∃r.C ⊔ ∃r.D`` and ``C ⊓ (D ⊔ E) = (C ⊓ D) ⊔ (C ⊓ E)`` are used as
    such. A negation of a union is rewritten with De Morgan's law; a negation of an
    intersection stays an operator (the models negate any region).
    """
    k = node.kind
    if k == "a":
        return [node]
    if k == "p":
        return [project(node.value, c) for c in to_dnf(node.children[0])]
    if k == "u":
        out = []
        for c in node.children:
            out += to_dnf(c)
        return _unique(out)
    if k == "i":
        options = [to_dnf(c) for c in node.children]
        return _unique([intersect(*combo) for combo in product(*options)])
    inner = to_dnf(node.children[0])                           # negation
    if len(inner) == 1:
        c = inner[0]
        return [c.children[0] if c.kind == "n" else negate(c)]
    return to_dnf(intersect(*[negate(c) for c in inner]))


def _unique(nodes):
    seen, out = set(), []
    for n in nodes:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def contains_negation(node):
    return node.kind == "n" or any(contains_negation(c) for c in node.children)


def contains_union(node):
    return node.kind == "u" or any(contains_union(c) for c in node.children)


def signature(node):
    """The shape of a union-free query as a hashable nested tuple."""
    return (node.kind,) + tuple(signature(c) for c in node.children)


def ids(node):
    """The anchor and relation ids of a query in depth-first order (children first)."""
    out = []
    for c in node.children:
        out += ids(c)
    if node.kind in ("a", "p"):
        out.append(node.value)
    return out


def from_signature(sig, values):
    """Inverse of :func:`signature` and :func:`ids`: ``values`` is an iterator of ids."""
    children = tuple(from_signature(s, values) for s in sig[1:])
    value = next(values) if sig[0] in ("a", "p") else None
    return Node(sig[0], children, value)


# ---------------------------------------------------------------------------------------
# Exact answers
# ---------------------------------------------------------------------------------------

class AnswerSet(Set):
    """The answers of a query as a read-only set backed by a boolean mask over the entities
    (computed on first use by :func:`answer_mask`): membership is a lookup, the size is
    counted once and the elements are enumerated only when asked. It compares equal to a
    Python ``set`` with the same elements."""

    def __init__(self, node, graph):
        self.node = node
        self.graph = graph
        self._mask = None
        self._len = None

    @property
    def mask(self):
        """Boolean mask over all entities."""
        if self._mask is None:
            self._mask = answer_mask(self.node, self.graph)
        return self._mask

    def contains_many(self, entities):
        """Boolean array: which of ``entities`` (integers) are answers."""
        return self.mask[np.asarray(entities, dtype=np.int64)]

    def __contains__(self, entity):
        return 0 <= entity < self.graph.num_entities and bool(self.mask[entity])

    def __len__(self):
        if self._len is None:
            self._len = int(np.count_nonzero(self.mask))
        return self._len

    def __iter__(self):
        return iter(np.flatnonzero(self.mask).tolist())

    @classmethod
    def _from_iterable(cls, it):
        return set(it)

    def __repr__(self):
        return f"AnswerSet({self.node!r})"


def answer_mask(node, graph):
    """The exact answers of a query as a boolean mask over the entities of ``graph``, which
    must offer ``num_entities`` and ``project_mask(entities, relation)`` (a
    :class:`~mowl.models.query.graph.QueryGraph`). Same semantics as :func:`answers`; the cost
    is that of the array operations of the query, a few numpy passes over the entities and the
    edges leaving the intermediate answers."""
    k = node.kind
    n = graph.num_entities
    if k == "a":
        mask = np.zeros(n, dtype=bool)
        mask[node.value] = True
        return mask
    if k == "p":
        child = node.children[0]
        if child.kind == "a":
            return graph.project_mask(np.array([child.value], dtype=np.int64), node.value)
        entities = np.flatnonzero(answer_mask(child, graph))
        return graph.project_mask(entities, node.value)
    if k == "i":
        mask = answer_mask(node.children[0], graph)
        for c in node.children[1:]:
            mask &= answer_mask(c, graph)
        return mask
    if k == "u":
        mask = answer_mask(node.children[0], graph)
        for c in node.children[1:]:
            mask |= answer_mask(c, graph)
        return mask
    return ~answer_mask(node.children[0], graph)


def answers(node, graph, universe):
    """The exact answer set of a query on ``graph``.

    A graph with ``project_mask`` and ``expand`` (the query graph of the models) is evaluated
    with array operations and the result is an :class:`AnswerSet`, which computes the
    answers only when its size or elements are asked for; any other graph is evaluated with
    Python sets and the result is a ``set``.

    :param graph: An object with ``successors(entity, relation) -> set``.
    :param universe: All entities (negation is the complement in it).
    :rtype: set of int
    """
    if hasattr(graph, "project_mask") and hasattr(graph, "expand"):
        return AnswerSet(node, graph)
    k = node.kind
    if k == "a":
        return {node.value}
    if k == "p":
        out = set()
        for e in answers(node.children[0], graph, universe):
            out |= graph.successors(e, node.value)
        return out
    if k == "i":
        sets = [answers(c, graph, universe) for c in node.children]
        return set.intersection(*sets)
    if k == "u":
        out = set()
        for c in node.children:
            out |= answers(c, graph, universe)
        return out
    return set(universe) - answers(node.children[0], graph, universe)
