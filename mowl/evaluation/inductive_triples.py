"""Scoring triples over a graph the model was not trained on, with the parts of the model.

In the inductive setting of knowledge graph completion the test graph has entities that the
training graph does not. A model whose entity model is a *graph encoder*
(:class:`~mowl.representation.library.other_graph.GraphEncodedPoints`) encodes the new graph
with fresh random features and the learned relation model, and the relation model scores the
triples; :class:`InductiveTripleCosts` does that for any geometry that has such an entity
model. It is the model side of
:class:`~mowl.evaluation.inductive_ranking.InductiveSampledRankingRegime`.
"""

import torch as th

from mowl.representation.library.shapes_regions import Point

__all__ = ["InductiveTripleCosts"]


class InductiveTripleCosts:
    """Cost of triples over another graph (lower is more plausible).

    The graph's entities get fresh random features (drawn once per seed and cached), the
    learned relation model is that of training; relations are matched by IRI. Facts with a
    relation the model has not seen are not used for message passing, and a row with such a
    relation costs ``inf``. With inverse triples and ``target="head"`` (the head of the rows
    is the corrupted entity), the triple is scored from the tail with the inverse relation.

    :param module: A geometry with ``class_model.encode_graph`` and a relation model with
        ``cost``.
    :param relation_ids: Relation IRI -> id of the training graph.
    :param inverse: Whether the model has inverse relations (relation ``r`` is ``2r``, its
        inverse ``2r + 1``).
    """

    def __init__(self, module, relation_ids, inverse=False):
        self.module = module
        self.relation_ids = dict(relation_ids)
        self.inverse = bool(inverse)
        self._cache = None

    @th.no_grad()
    def __call__(self, graph, rows, seed=0, target="tail"):
        """Costs of ``rows`` (``(n, 3)`` ``[head, relation, tail]`` indices into ``graph``,
        an :class:`~mowl.datasets.builtin.inductive_kg.InductiveGraph`); ``target`` is
        ``"tail"`` or ``"head"``: which entity of the rows is the candidate."""
        if target not in ("head", "tail"):
            raise ValueError("target must be 'head' or 'tail'.")
        module, inverse = self.module, self.inverse
        to_model = th.tensor([self.relation_ids.get(r, -1) for r in graph.relations],
                             dtype=th.long)
        key = (id(graph), int(seed), len(graph.triples))
        if self._cache is None or self._cache[0] != key:
            module.eval()
            facts = graph.triples
            keep = to_model[facts[:, 1]] >= 0
            facts = th.stack([facts[keep, 0], to_model[facts[keep, 1]], facts[keep, 2]], dim=1)
            x, _ = module.class_model.encode_graph(facts, len(graph.entities), seed=seed,
                                                   all_relations=inverse)
            self._cache = (key, x)
        x = self._cache[1]
        rows = rows.cpu()
        relation = to_model[rows[:, 1]]
        out = th.full((len(rows),), float("inf"))
        ok = relation >= 0
        if ok.any():
            device = x.device
            h, r, t = rows[ok, 0].to(device), relation[ok].to(device), rows[ok, 2].to(device)
            if inverse and target == "head":
                cost = module.relation_model.cost(2 * r + 1, _point(x[t]), _point(x[h]))
            else:
                cost = module.relation_model.cost(2 * r if inverse else r,
                                                  _point(x[h]), _point(x[t]))
            out[ok] = cost.reshape(-1).cpu()
        return out

    def reset(self):
        """Forget the cached encoding (call after the parameters changed)."""
        self._cache = None


def _point(matrices):
    return Point(matrices.reshape(len(matrices), -1))
