"""Scoring of queries and axioms by a query embedding geometry.

The evaluation side of the query methods (Query2Box, BetaE, ConE, AConE, and any other
combination of the query parts), written in calls to the geometry's ``embed(signature,
rows)`` (the state of a batch of queries of one shape) and ``logit(state, entities)``:

* :func:`dnf_logits` scores queries against entities; a union is the best branch of the
  disjunctive normal form.
* :func:`axiom_query` builds, column-wise, the query of every candidate axiom of a normal
  form (a named class, or ``exists r.D``) so that no query object is needed per row.
* :func:`axiom_logits` scores the candidate axioms in chunks.
* :func:`top_answers` names the best entities of one score vector.
"""
import torch as th

from mowl.nn.query.expression import anchor, ids, project, signature, to_dnf
from mowl.projection.query_graph import ISA_INV

__all__ = ["dnf_logits", "axiom_query", "axiom_logits", "top_answers"]


def dnf_logits(geometry, nodes, num_entities, embed_dim, device, entity_ids=None,
               max_cells=2 ** 24):
    """Scores of queries against entities: ``(len(nodes), N)`` for all entities, or
    ``(len(nodes), K)`` for ``entity_ids`` of that shape. A union is the best branch of the
    disjunctive normal form; queries of one shape are embedded together, in chunks of at
    most ``max_cells`` cells."""
    branches = {}
    for q, node in enumerate(nodes):
        for b in to_dnf(node):
            branches.setdefault(signature(b), []).append((q, ids(b)))
    width = num_entities if entity_ids is None else entity_ids.shape[1]
    out = th.full((len(nodes), width), -float("inf"), device=device)
    chunk = max(1, max_cells // max(1, width * embed_dim))
    for sig, items in branches.items():
        for start in range(0, len(items), chunk):
            part = items[start:start + chunk]
            qs = th.tensor([q for q, _ in part], dtype=th.long, device=device)
            rows = th.tensor([r for _, r in part], dtype=th.long, device=device)
            state = geometry.embed(sig, rows)
            ent = (th.arange(num_entities, device=device).expand(len(part), -1)
                   if entity_ids is None else entity_ids[qs])
            out[qs] = th.maximum(out[qs], geometry.logit(state, ent))
    return out


def axiom_query(gci_name, rows, num_classes, extension):
    """``(signature, id_matrix, subjects)`` of candidate axioms: ``gci0`` ``[C, D]`` is
    "``C`` answers ``D``", ``gci2`` ``[C, r, D]`` "``C`` answers ``exists r.D``",
    ``class_assertion`` ``[a, C]`` "the individual ``a`` answers ``C``". Every axiom of one
    form is a query of the same shape, so the ids are columns of a matrix.

    :param extension: A named class means everything below it (``isa`` inverse) when true.
    """
    rows = rows.long()
    if gci_name == "class_assertion":
        subjects, classes = rows[:, 0] + num_classes, rows[:, 1]
    else:
        subjects, classes = rows[:, 0], rows[:, -1]
    cols = [classes]
    node = anchor(0)
    if extension:
        cols.append(th.full_like(classes, ISA_INV))
        node = project(ISA_INV, node)
    if gci_name == "gci2":
        cols.append(2 + 2 * rows[:, 1] + 1)            # role_relation(r, inverse=True)
        node = project(0, node)
    return signature(node), th.stack(cols, dim=1), subjects


def axiom_logits(geometry, sig, id_matrix, subjects, chunk):
    """The score of every subject against its own query, ``chunk`` queries at a time."""
    out = th.empty(len(id_matrix), device=id_matrix.device)
    step = max(1, int(chunk))
    for start in range(0, len(id_matrix), step):
        stop = start + step
        state = geometry.embed(sig, id_matrix[start:stop])
        out[start:stop] = geometry.logit(state, subjects[start:stop, None])[:, 0]
    return out


def top_answers(scores, names, top_k):
    """``(name, score)`` of the ``top_k`` best entities of a score vector, best first."""
    best = th.topk(scores, min(top_k, len(scores)))
    return [(names[int(i)], float(s)) for s, i in zip(best.values, best.indices)]
