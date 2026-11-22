"""The knowledge graph that the query embedding models read from an ontology."""

from collections import defaultdict, deque

import numpy as np

#: Relation ids: ``isa`` (subclass of / instance of) and its inverse; role ``k`` is
#: ``2 + 2k`` and its inverse ``3 + 2k``.
ISA = 0
ISA_INV = 1


def role_relation(k, inverse=False):
    return 2 + 2 * k + (1 if inverse else 0)


class QueryGraph:
    """Triples ``(head, relation, tail)`` over integer entity and relation ids, indexed
    for traversal in both directions. ``triples`` must already contain the inverse edges
    (:func:`extract_graph` adds them).

    The index is a pair of sorted integer arrays (``relation * num_entities + head`` and the
    tails in the same order), so that the successors of a whole set of entities along a
    relation are gathered with one vectorised lookup (:meth:`project_mask`): the exact answer
    of a query is then computed with array operations over the entities (boolean masks)
    instead of Python sets. ``out`` and ``inc`` (the dictionaries of earlier versions) are
    built on first access only."""

    #: Relative cost of gathering one edge by index against one multiplication of the sparse
    #: matrix per edge of the relation (measured: about 6 to 10): a set whose outgoing edges
    #: are more than ``1 / MATRIX_EDGE_COST`` of the relation is projected with the matrix.
    MATRIX_EDGE_COST = 6

    def __init__(self, triples, num_entities, num_relations):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
        h, r, t = self.triples[:, 0], self.triples[:, 1], self.triples[:, 2]
        order = np.lexsort((t, h, r))
        self._keys = r[order] * num_entities + h[order]
        self._tails = t[order]
        # incoming edges of an entity (head, relation), sorted: same tuple order as sorting
        # the pairs
        by_tail = np.lexsort((r, h, t))
        self._in_tails = t[by_tail]
        self._in_pairs = np.stack([h[by_tail], r[by_tail]], axis=1)
        self.universe = set(range(num_entities))
        self._incoming = {}
        self._out = None
        self._inc = None
        self._matrices = None

    # ---- the dictionaries of earlier versions, built on demand -------------------------
    @property
    def out(self):
        if self._out is None:
            out = defaultdict(set)
            for h, r, t in self.triples.tolist():
                out[(h, r)].add(t)
            self._out = out
        return self._out

    @property
    def inc(self):
        if self._inc is None:
            inc = defaultdict(set)
            for h, r, t in self.triples.tolist():
                inc[t].add((h, r))
            self._inc = inc
        return self._inc

    def entities_with_incoming(self):
        """Sorted list of the entities that are the tail of some edge."""
        return np.unique(self.triples[:, 2]).tolist()

    def incoming(self, entity):
        """The ``(head, relation)`` pairs of the edges into ``entity``, sorted (computed once
        per entity: the training sampler draws from it in every step, and the classes near
        the root of a large taxonomy have hundreds of thousands)."""
        out = self._incoming.get(entity)
        if out is None:
            lo = np.searchsorted(self._in_tails, entity, side="left")
            hi = np.searchsorted(self._in_tails, entity, side="right")
            out = tuple(map(tuple, self._in_pairs[lo:hi].tolist()))
            self._incoming[entity] = out
        return out

    def successors(self, entity, relation):
        return set(self.successor_array(entity, relation).tolist())

    def successor_array(self, entity, relation):
        """Tails of the edges ``(entity, relation, *)`` as a sorted integer array."""
        key = relation * self.num_entities + entity
        lo = np.searchsorted(self._keys, key, side="left")
        hi = np.searchsorted(self._keys, key, side="right")
        return self._tails[lo:hi]

    def expand(self, entities, relation):
        """``(owner, tails)``: for every edge ``(entities[i], relation, t)`` the position
        ``i`` and the tail ``t``."""
        entities = np.asarray(entities, dtype=np.int64)
        keys = relation * self.num_entities + entities
        lo = np.searchsorted(self._keys, keys, side="left")
        hi = np.searchsorted(self._keys, keys, side="right")
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        offsets = np.cumsum(counts) - counts
        positions = np.arange(total) + np.repeat(lo - offsets, counts)
        return np.repeat(np.arange(len(entities)), counts), self._tails[positions]

    def _matrix(self, relation):
        """Sparse 0/1 matrix of a relation (rows: heads, columns: tails), built on first
        use; ``None`` if scipy is not installed."""
        if self._matrices is None:
            self._matrices = {}
        if relation not in self._matrices:
            try:
                from scipy.sparse import csr_matrix
            except ImportError:                                  # pragma: no cover
                self._matrices[relation] = None
                return None
            n = self.num_entities
            lo = np.searchsorted(self._keys, relation * n, side="left")
            hi = np.searchsorted(self._keys, (relation + 1) * n, side="left")
            heads = self._keys[lo:hi] - relation * n
            self._matrices[relation] = csr_matrix(
                (np.ones(hi - lo, dtype=np.int32), (heads, self._tails[lo:hi])), shape=(n, n))
        return self._matrices[relation]

    def project_mask(self, entities, relation):
        """Boolean mask over the entities of everything reached from ``entities`` (an
        integer array) along ``relation``. A few entities are expanded edge by edge; a large
        set (many outgoing edges) is multiplied with the sparse matrix of the relation, whose
        cost is that of the relation and not of the set."""
        n = self.num_entities
        mask = np.zeros(n, dtype=bool)
        if len(entities) == 0:
            return mask
        entities = np.asarray(entities, dtype=np.int64)
        keys = relation * n + entities
        edges = int((np.searchsorted(self._keys, keys, side="right")
                     - np.searchsorted(self._keys, keys, side="left")).sum())
        lo = np.searchsorted(self._keys, relation * n, side="left")
        hi = np.searchsorted(self._keys, (relation + 1) * n, side="left")
        if edges * self.MATRIX_EDGE_COST > hi - lo:
            matrix = self._matrix(relation)
            if matrix is not None:
                indicator = np.zeros(n, dtype=np.int32)
                indicator[entities] = 1
                return (matrix.T @ indicator) > 0
        mask[self.expand(entities, relation)[1]] = True
        return mask

    def __len__(self):
        return len(self.triples)


def _conjuncts(cexpr, T):
    if cexpr.getClassExpressionType() == T.OBJECT_INTERSECTION_OF:
        out = []
        for o in cexpr.getOperandsAsList():
            out += _conjuncts(o, T)
        return out
    return [cexpr]


def extract_graph(ontology, class_index, individual_index, role_index, closure=True):
    """Read the graph of an ontology.

    Entities are the named classes (ids ``class_index[iri]``) and the named individuals
    (``len(class_index) + individual_index[iri]``). Edges, each with its inverse:

    * ``A ⊑ B`` (named), and the same for ``A ≡ B`` in both directions, and
      ``a : B`` (class assertion): ``(A, isa, B)``;
    * ``A ⊑ ∃r.B`` (named ``A`` and ``B``) and ``r(a, b)``: ``(A, r, B)``;
    * a conjunction on the right-hand side is split (``A ⊑ B ⊓ ∃r.C`` gives both);
    * with ``closure``, ``isa`` is closed under transitivity and reflexive on classes
      (so that "everything under ``A``" contains ``A`` and all its descendants).

    Everything else (a complex left-hand side, ``⊥``, universal restrictions, disjunction,
    negation, role axioms) is skipped and counted.

    :return: ``(QueryGraph, skipped)`` with the number of skipped axioms.
    """
    from mowl.owlapi import (ClassExpressionType as T, Imports, OWLClassAssertionAxiom,
                             OWLEquivalentClassesAxiom, OWLObjectPropertyAssertionAxiom,
                             OWLSubClassOfAxiom)
    from mowl.owlapi.defaults import BOT, TOP
    n_classes = len(class_index)
    n_entities = n_classes + len(individual_index)
    n_relations = 2 + 2 * len(role_index)
    edges = set()
    skipped = 0

    def class_entity(e):
        if e.getClassExpressionType() != T.OWL_CLASS:
            return None
        iri = str(e.asOWLClass().toStringID())
        return None if iri in (TOP, BOT) else class_index.get(iri)

    def add_sub(entity, sup):
        nonlocal skipped
        ok = True
        for part in _conjuncts(sup, T):
            kind = part.getClassExpressionType()
            if kind == T.OWL_CLASS:
                iri = str(part.asOWLClass().toStringID())
                if iri == TOP:
                    continue
                target = class_index.get(iri)
                if target is None or iri == BOT:
                    ok = False
                else:
                    edges.add((entity, ISA, target))
            elif kind == T.OBJECT_SOME_VALUES_FROM and not part.getProperty().isAnonymous():
                target = class_entity(part.getFiller())
                role = role_index.get(str(part.getProperty().asOWLObjectProperty()
                                          .toStringID()))
                if target is None or role is None:
                    ok = False
                else:
                    edges.add((entity, role_relation(role), target))
            else:
                ok = False
        if not ok:
            skipped += 1

    for ax in ontology.getAxioms(Imports.INCLUDED):
        if isinstance(ax, OWLSubClassOfAxiom):
            sub = class_entity(ax.getSubClass())
            if sub is None:
                skipped += 1
            else:
                add_sub(sub, ax.getSuperClass())
        elif isinstance(ax, OWLEquivalentClassesAxiom):
            exprs = list(ax.getClassExpressions())
            for x in exprs:
                for y in exprs:
                    if x is y or x.equals(y):
                        continue
                    sub = class_entity(x)
                    if sub is not None:
                        add_sub(sub, y)
        elif isinstance(ax, OWLClassAssertionAxiom):
            ind = ax.getIndividual()
            iri = str(ind.toStringID()) if ind.isNamed() else None
            if iri not in individual_index:
                skipped += 1
            else:
                add_sub(n_classes + individual_index[iri], ax.getClassExpression())
        elif isinstance(ax, OWLObjectPropertyAssertionAxiom):
            s, o, p = ax.getSubject(), ax.getObject(), ax.getProperty()
            names = [str(i.toStringID()) if i.isNamed() else None for i in (s, o)]
            if (p.isAnonymous() or names[0] not in individual_index
                    or names[1] not in individual_index
                    or str(p.asOWLObjectProperty().toStringID()) not in role_index):
                skipped += 1
            else:
                k = role_index[str(p.asOWLObjectProperty().toStringID())]
                edges.add((n_classes + individual_index[names[0]], role_relation(k),
                           n_classes + individual_index[names[1]]))

    if closure:
        for c in range(n_classes):
            edges.add((c, ISA, c))
        parents = defaultdict(set)
        for h, r, t in edges:
            if r == ISA:
                parents[h].add(t)
        for h in list(parents):
            seen, queue = set(parents[h]), deque(parents[h])
            while queue:
                x = queue.popleft()
                for y in parents.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        queue.append(y)
            for y in seen:
                edges.add((h, ISA, y))
    both = set(edges)
    for h, r, t in edges:
        both.add((t, r ^ 1, h))                       # 0<->1, 2k+2<->2k+3
    return QueryGraph(sorted(both), n_entities, n_relations), skipped
