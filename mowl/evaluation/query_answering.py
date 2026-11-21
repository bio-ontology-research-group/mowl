"""Logical query answering: ranking the answers of one-hop and multi-hop queries.

The protocol of the query embedding literature (Query2Box, Ren et al. 2020; BetaE, Ren
and Leskovec 2020; ConE, Zhang et al. 2021): a query is a tree of role projections,
intersections and unions over anchor individuals, for example the two-hop query
:math:`?y\\,.\\,\\exists z\\,.\\,r_1(a, z) \\wedge r_2(z, y)`. Its answers are computed
on two graphs: the *known* one the model was trained on, and a *complete* one. The
answers found only on the complete graph (the "hard" answers) are what the model has to
predict; each is ranked among all individuals after removing the other answers of the
query (filtered setting).

What the complete graph is decides what is measured. With held-out assertions it is
inductive prediction, as in the literature. With the assertions *entailed* by an
ontology (a reasoner's realisation of the knowledge base) it is deductive inference: the
TBox acts as a constraint on the answers, and a query is a class expression
(:math:`\\exists r_2^-.\\exists r_1^-.\\{a\\}`) whose instances are asked for.

A query has a *structure* and an *instance*, in the nested-tuple format of the released
query datasets: the structure ``("e", ("r", "r"))`` with the instance ``(a, (r1, r2))``
is the two-hop query above; ``(q1, q2)`` is an intersection, ``(q1, q2, ("u",))`` a union
(instance ``(i1, i2, (-1,))``), and ``(q, ("r",))`` a projection of a sub-query.

A model takes part by providing ``score_query(structure, queries)``, returning one row
per query with one score per individual (higher is better).
:class:`mowl.models.GroundedEL` does (in translation mode).
"""

from collections import defaultdict

import numpy as np

from mowl.evaluation.regime import EvaluationRegime

_1P = ("e", ("r",))
_1PN = ("e", ("r", "n"))

#: The query structures of the Query2Box / BetaE benchmarks, by their usual names. In an
#: instance, ``"n"`` is a negation (id ``-2``) and ``("u",)`` a union (id ``-1``).
#: ``2u-DM`` and ``up-DM`` are the unions written with De Morgan's law (negations of
#: intersections of negations), the form BetaE's code evaluates for its "DM" variant; the
#: answers are those of ``2u`` and ``up``.
QUERY_STRUCTURES = {
    "1p": _1P,
    "2p": ("e", ("r", "r")),
    "3p": ("e", ("r", "r", "r")),
    "2i": (_1P, _1P),
    "3i": (_1P, _1P, _1P),
    "ip": ((_1P, _1P), ("r",)),
    "pi": (("e", ("r", "r")), _1P),
    "2u": (_1P, _1P, ("u",)),
    "up": ((_1P, _1P, ("u",)), ("r",)),
    "2in": (_1P, _1PN),
    "3in": (_1P, _1P, _1PN),
    "inp": ((_1P, _1PN), ("r",)),
    "pin": (("e", ("r", "r")), _1PN),
    "pni": (("e", ("r", "r", "n")), _1P),
    "2u-DM": ((_1PN, _1PN), ("n",)),
    "up-DM": ((_1PN, _1PN), ("n", "r")),
}

#: The groups of structures whose averages the literature reports separately: the nine
#: positive (existential positive first-order) structures and the five with negation.
STRUCTURE_GROUPS = {
    "epfo": ("1p", "2p", "3p", "2i", "3i", "ip", "pi", "2u", "up"),
    "negation": ("2in", "3in", "inp", "pin", "pni"),
}

TIES = ("average", "optimistic", "pessimistic")


def _is_chain(structure):
    return (isinstance(structure, tuple) and len(structure) == 2
            and isinstance(structure[1], tuple) and len(structure[1]) > 0
            and all(isinstance(x, str) and x in ("r", "n") for x in structure[1]))


def _is_union(structure):
    return isinstance(structure, tuple) and structure[-1] == ("u",)


class Graph:
    """A set of role assertions ``(head, role, tail)`` over integer ids, indexed for
    query answering and query sampling."""

    def __init__(self, triples):
        self.out = defaultdict(set)
        self.inc = defaultdict(set)
        self.triples = set()
        for h, r, t in np.asarray(triples, dtype=np.int64).reshape(-1, 3).tolist():
            self.triples.add((h, r, t))
            self.out[(h, r)].add(t)
            self.inc[t].add((h, r))

    def __len__(self):
        return len(self.triples)

    def project(self, entities, role):
        result = set()
        for e in entities:
            result |= self.out.get((e, role), set())
        return result


def answer_query(structure, query, graph, universe=None):
    """The exact answers of ``query`` (an instance of ``structure``) on ``graph``.

    A negation (``"n"`` in a chain, id ``-2`` in the instance) is the complement in
    ``universe``: closed world, an entity that no stated edge makes an answer is an answer of
    the negation.

    :param graph: A :class:`Graph` or an array of ``(head, role, tail)`` rows.
    :param universe: All entities, needed for structures with negation (default: the
        entities that occur in ``graph``).
    :rtype: set of int
    """
    if not isinstance(graph, Graph):
        graph = Graph(graph)
    if universe is None:
        universe = {h for h, _, _ in graph.triples} | {t for _, _, t in graph.triples}
    if _is_chain(structure):
        if structure[0] == "e":
            current = {int(query[0])}
        else:
            current = answer_query(structure[0], query[0], graph, universe)
        for op, role in zip(structure[1], query[1]):
            if op == "n":
                current = set(universe) - current
            else:
                current = graph.project(current, int(role))
        return current
    if _is_union(structure):
        result = set()
        for sub, q in zip(structure[:-1], query[:-1]):
            result |= answer_query(sub, q, graph, universe)
        return result
    parts = [answer_query(sub, q, graph, universe) for sub, q in zip(structure, query)]
    return set.intersection(*parts) if parts else set()


def _ground(structure, answer, graph, rng, entities):
    """Sample an instance of ``structure`` that has ``answer`` among its answers on
    ``graph`` (backwards from the answer), or ``None`` if a dead end is hit."""
    if _is_chain(structure):
        roles, current = [], answer
        for _ in structure[1]:
            incoming = graph.inc.get(current)
            if not incoming:
                return None
            incoming = sorted(incoming)
            current, role = incoming[int(rng.integers(len(incoming)))]
            roles.append(role)
        roles = tuple(reversed(roles))
        if structure[0] == "e":
            return (current, roles)
        sub = _ground(structure[0], current, graph, rng, entities)
        return None if sub is None else (sub, roles)
    if _is_union(structure):
        parts = []
        for i, sub in enumerate(structure[:-1]):
            # One disjunct has the answer; the others are grounded anywhere.
            target = answer if i == 0 else entities[int(rng.integers(len(entities)))]
            q = _ground(sub, target, graph, rng, entities)
            if q is None:
                return None
            parts.append(q)
        return tuple(parts) + ((-1,),)
    parts = []
    for sub in structure:
        q = _ground(sub, answer, graph, rng, entities)
        if q is None:
            return None
        parts.append(q)
    if len(set(parts)) < len(parts):
        return None                     # an intersection of a query with itself
    return tuple(parts)


def sample_queries(known, complete, structures=("1p", "2p", "2i"), num_queries=100,
                   seed=0, max_answers=None, max_tries=200):
    """Sample queries that have answers on ``complete`` which are not answers on
    ``known``.

    :param known: Role assertions the model was trained on, rows ``(head, role, tail)``
        of integer ids (for a mOWL EL model: the rows of
        ``model.training_datasets["object_property_assertion"]``).
    :param complete: All assertions that count as true: ``known`` together with held-out
        assertions (inductive prediction) or with the entailed ones (deductive
        inference). ``known`` is added if it is not contained.
    :param structures: Names from :data:`QUERY_STRUCTURES`, or ``{name: structure}``.
    :param num_queries: Queries per structure (fewer if the graph does not have them).
    :param max_answers: Skip queries with more answers on ``complete`` than this.
    :return: ``{name: [(query, easy_answers, hard_answers), ...]}`` where ``easy`` are
        the answers on ``known`` and ``hard`` the remaining answers on ``complete``.
    """
    known = np.asarray(known, dtype=np.int64).reshape(-1, 3)
    complete = np.asarray(complete, dtype=np.int64).reshape(-1, 3)
    known_graph = Graph(known)
    complete_graph = Graph(np.concatenate([known, complete], axis=0))
    if not isinstance(structures, dict):
        structures = {name: QUERY_STRUCTURES[name] for name in structures}
    negated = sorted(n for n, st in structures.items() if "'n'" in repr(st))
    if negated:
        raise ValueError(f"Queries with negation ({negated}) cannot be sampled here; read "
                         f"them from a released query dataset.")
    rng = np.random.default_rng(seed)
    entities = sorted({h for h, _, _ in complete_graph.triples}
                      | {t for _, _, t in complete_graph.triples})
    # Answers that can be hard: tails of assertions that are not known.
    new_tails = sorted({t for (h, r, t) in complete_graph.triples
                        if (h, r, t) not in known_graph.triples})
    result = {}
    for name, structure in structures.items():
        found, seen = [], set()
        tries = max_tries * num_queries
        while new_tails and len(found) < num_queries and tries > 0:
            tries -= 1
            answer = new_tails[int(rng.integers(len(new_tails)))]
            query = _ground(structure, answer, complete_graph, rng, entities)
            if query is None or query in seen:
                continue
            seen.add(query)
            all_answers = answer_query(structure, query, complete_graph)
            if max_answers is not None and len(all_answers) > max_answers:
                continue
            easy = answer_query(structure, query, known_graph)
            hard = all_answers - easy
            if hard:
                found.append((query, easy, hard))
        result[name] = found
    return result


class LogicalQueryAnswering(EvaluationRegime):
    """Filtered ranking of the hard answers of logical queries.

    For every query, each hard answer is ranked among all individuals except the other
    answers of the query; the reciprocal rank and Hits@k are averaged over the hard
    answers of a query and then over the queries of a structure. ``"average"`` is the
    mean over the structures that have queries.

    :param queries: ``{name: [(query, easy_answers, hard_answers), ...]}``, as returned
        by :func:`sample_queries` or read from a released query dataset.
    :param structures: ``{name: structure}`` for names that are not in
        :data:`QUERY_STRUCTURES`.
    :param hits: Values of k for Hits@k.
    :param ties: Rank of an answer among equally scored candidates: ``"average"``
        (default), ``"optimistic"`` or ``"pessimistic"``. Region-based scores tie often
        (all individuals outside the query region), so ``"optimistic"`` overstates.
    :param batch_size: Queries scored per call of ``model.score_query``.
    :param groups: ``{group: [structure names]}``; every group adds a subset
        ``"average_<group>"``, the mean over the group's structures that have queries.
        :data:`STRUCTURE_GROUPS` gives the groups of the literature (the "AVG" columns of the
        papers: ``epfo`` for the nine positive structures, ``negation`` for the five with
        negation).
    :param score_kwargs: Extra keyword arguments for ``model.score_query``.
    """

    key = "query-answering"

    def __init__(self, queries, structures=None, hits=(1, 3, 10), ties="average",
                 batch_size=256, score_kwargs=None, groups=None):
        if ties not in TIES:
            raise ValueError(f"ties must be one of {TIES}.")
        self.queries = {name: list(qs) for name, qs in queries.items()}
        self.structures = dict(QUERY_STRUCTURES)
        self.structures.update(structures or {})
        unknown = sorted(set(self.queries) - set(self.structures))
        if unknown:
            raise ValueError(f"No structure given for the query types {unknown}.")
        self.hits = tuple(int(k) for k in hits)
        self.ties = ties
        self.batch_size = int(batch_size)
        self.score_kwargs = dict(score_kwargs or {})
        self.metrics = ("mrr",) + tuple(f"hits@{k}" for k in self.hits)
        self.groups = {g: tuple(names) for g, names in (groups or {}).items()}
        self.subsets = (tuple(self.queries) + ("average",)
                        + tuple(f"average_{g}" for g in self.groups))

    def _ranks(self, scores, hard, answers):
        """Ranks of the hard answers among the non-answers and themselves (the other answers
        of the query are removed: filtered setting)."""
        mask = np.ones(len(scores), dtype=bool)
        mask[list(answers)] = False
        others = np.sort(scores[mask])
        s = scores[hard]
        greater = len(others) - np.searchsorted(others, s, side="right")
        equal = np.searchsorted(others, s, side="right") - np.searchsorted(others, s,
                                                                           side="left")
        if self.ties == "optimistic":
            return 1.0 + greater
        if self.ties == "pessimistic":
            return 1.0 + greater + equal
        return 1.0 + greater + equal / 2.0

    def evaluate(self, model, dataset=None):
        if not callable(getattr(model, "score_query", None)):
            raise TypeError(
                f"{type(model).__name__} has no 'score_query(structure, queries)': it "
                f"does not answer logical queries.")
        results = {}
        for name, items in self.queries.items():
            if not items:
                continue
            structure = self.structures[name]
            per_query = defaultdict(list)
            for start in range(0, len(items), self.batch_size):
                batch = items[start:start + self.batch_size]
                scores = model.score_query(structure, [q for q, _, _ in batch],
                                           **self.score_kwargs)
                scores = (scores.detach().cpu().numpy() if hasattr(scores, "detach")
                          else np.asarray(scores))
                for row, (_, easy, hard) in zip(scores, batch):
                    answers = set(easy) | set(hard)
                    ranks = self._ranks(row, np.array(sorted(hard), dtype=np.int64), answers)
                    per_query["mrr"].append(float((1.0 / ranks).mean()))
                    for k in self.hits:
                        per_query[f"hits@{k}"].append(float((ranks <= k).mean()))
            results[name] = {m: float(np.mean(per_query[m])) for m in self.metrics}
            results[name]["queries"] = len(items)
        if results:
            per_structure = dict(results)
            results["average"] = {m: float(np.mean([r[m] for r in per_structure.values()]))
                                  for m in self.metrics}
            for group, names in self.groups.items():
                found = [per_structure[n] for n in names if n in per_structure]
                if found:
                    results[f"average_{group}"] = {
                        m: float(np.mean([r[m] for r in found])) for m in self.metrics}
        return results


class ReleasedQueryAnswering(EvaluationRegime):
    """:class:`LogicalQueryAnswering` on the released validation or test queries of a query
    dataset (:class:`mowl.datasets.builtin.query_sets.QueryAnsweringDataset`).

    The queries are read from ``dataset.queries(...)`` at evaluation time with the entity
    and relation ids of the model (``model.entity_id``, ``model.relation_id``), so that any
    model with ``score_query`` and these two methods can be evaluated. Reports one subset
    per structure, ``"average"`` (all structures) and ``"average_epfo"`` /
    ``"average_negation"`` (the "AVG" columns of the papers).

    A model whose operators do not include negation (``model.module.supports_negation`` is
    false, e.g. Query2Box) is evaluated on the structures without negation only; the
    skipped names are in :attr:`info`.

    :param split: ``"valid"`` or ``"test"``.
    :param names: Structure names (default: all of the split except the De Morgan forms).
    :param limit: At most this many queries per structure (a seeded subset), for quick runs.
    :param seed: Seed of the subset.
    :param hits, ties, batch_size: See :class:`LogicalQueryAnswering`.
    """

    key = "query-answering"

    def __init__(self, split="test", names=None, limit=None, seed=0, hits=(1, 3, 10),
                 ties="average", batch_size=32):
        self.split = split
        self.names = None if names is None else tuple(names)
        self.limit = limit
        self.seed = seed
        self.hits = tuple(hits)
        self.ties = ties
        self.batch_size = batch_size
        self.metrics = ("mrr",) + tuple(f"hits@{k}" for k in self.hits)
        self.info = {}

    def evaluate(self, model, dataset):
        for attr in ("score_query", "entity_id", "relation_id"):
            if not callable(getattr(model, attr, None)):
                raise TypeError(f"{type(model).__name__} has no '{attr}': it cannot be "
                                f"evaluated on released query datasets.")
        names = self.names
        if names is None:
            names = [n for n in dataset.structure_names if not n.endswith("-DM")]
        skipped = []
        if getattr(getattr(model, "module", None), "supports_negation", True) is False:
            skipped = [n for n in names if "'n'" in repr(QUERY_STRUCTURES[n])]
            names = [n for n in names if n not in skipped]
        queries = dataset.queries(self.split, model.entity_id, model.relation_id,
                                  names=names, limit=self.limit, seed=self.seed)
        regime = LogicalQueryAnswering(queries, hits=self.hits, ties=self.ties,
                                       batch_size=self.batch_size, groups=STRUCTURE_GROUPS)
        results = regime.evaluate(model)
        self.info = {"split": self.split, "queries": {n: len(q) for n, q in queries.items()},
                     "skipped_structures": skipped, "ties": self.ties}
        return results
