"""Sampling of training queries from a :class:`~mowl.projection.query_graph.QueryGraph`."""

from mowl.nn.query.expression import (anchor, answers, intersect, negate, project,
                                      template_from_nested)

_1P = ("e", ("r",))
_1PN = ("e", ("r", "n"))

#: Query structures of the Query2Box and BetaE benchmarks, without union (unions are
#: answered with the disjunctive normal form and not trained).
POSITIVE_STRUCTURES = {
    "1p": _1P, "2p": ("e", ("r", "r")), "3p": ("e", ("r", "r", "r")),
    "2i": (_1P, _1P), "3i": (_1P, _1P, _1P), "ip": ((_1P, _1P), ("r",)),
    "pi": (("e", ("r", "r")), _1P),
}
#: The negation structures of BetaE.
NEGATION_STRUCTURES = {
    "2in": (_1P, _1PN), "3in": (_1P, _1P, _1PN), "inp": (((_1P, _1PN)), ("r",)),
    "pin": (("e", ("r", "r")), _1PN), "pni": (("e", ("r", "r", "n")), _1P),
}


def structures(negation, names=None):
    """``{name: template}`` of the training structures."""
    table = dict(POSITIVE_STRUCTURES)
    if negation:
        table.update(NEGATION_STRUCTURES)
    if names is not None:
        unknown = sorted(set(names) - set(table))
        if unknown:
            raise ValueError(f"Unknown or unsupported query structures {unknown}; "
                             f"available: {sorted(table)}.")
        table = {n: table[n] for n in names}
    return {n: template_from_nested(s) for n, s in table.items()}


def ground(template, answer, graph, rng, max_tries=20):
    """An instance of ``template`` that has ``answer`` among its answers on ``graph``
    (sampled backwards from the answer), or ``None`` if a dead end is hit."""
    k = template.kind
    if k == "a":
        return anchor(answer)
    if k == "p":
        incoming = (graph.incoming(answer) if hasattr(graph, "incoming")
                    else tuple(sorted(graph.inc.get(answer, ()))))
        if not incoming:
            return None
        head, relation = incoming[int(rng.integers(len(incoming)))]
        child = ground(template.children[0], head, graph, rng, max_tries)
        return None if child is None else project(relation, child)
    if k == "i":
        parts = []
        for c in template.children:
            if c.kind == "n":
                part = None
                for _ in range(max_tries):
                    other = int(rng.integers(graph.num_entities))
                    part = ground(c.children[0], other, graph, rng, max_tries)
                    if part is not None and answer not in answers(part, graph,
                                                                  graph.universe):
                        break
                    part = None
                parts.append(None if part is None else negate(part))
            else:
                parts.append(ground(c, answer, graph, rng, max_tries))
        if any(p is None for p in parts) or len(set(parts)) < len(parts):
            return None
        return intersect(*parts)
    if k == "n":                                        # a negation at the root
        for _ in range(max_tries):
            other = int(rng.integers(graph.num_entities))
            part = ground(template.children[0], other, graph, rng, max_tries)
            if part is not None and answer not in answers(part, graph, graph.universe):
                return negate(part)
        return None
    raise ValueError("Training templates are union-free.")


def sample_query(template, graph, rng, pool, tries=50):
    """``(query, answer_set, positive)`` or ``None``."""
    for _ in range(tries):
        answer = int(pool[int(rng.integers(len(pool)))])
        query = ground(template, answer, graph, rng)
        if query is None:
            continue
        found = answers(query, graph, graph.universe)
        if answer in found:
            return query, found, answer
    return None


def usable_structures(templates, graph, rng, pool):
    """The templates for which at least one query can be sampled on ``graph``."""
    return {n: t for n, t in templates.items()
            if sample_query(t, graph, rng, pool, tries=200) is not None}


def negative_entities(found, num_negatives, num_entities, rng):
    """Entities that are not answers (uniform; answers are redrawn once)."""
    neg = rng.integers(num_entities, size=num_negatives)
    if hasattr(found, "contains_many"):
        hit = found.contains_many(neg)               # one backward pass for all candidates
    else:
        hit = [int(e) in found for e in neg]
    for i in range(len(neg)):
        if hit[i]:
            neg[i] = rng.integers(num_entities)
    return neg
