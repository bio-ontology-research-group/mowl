r"""Inductive link prediction with sampled negatives (the protocol of GraIL).

In inductive link prediction (Teru et al., ICML 2020) a model is trained on one graph and
tested on another graph whose entities it has never seen; the two graphs share the
relations. A model must therefore compute what it needs about an entity *from the facts
about that entity*, and it needs those facts at test time.

**The interface a model implements.** ``model.inductive_costs(graph, rows)``:

* ``graph`` is an :class:`mowl.datasets.builtin.inductive_kg.InductiveGraph`: entity and
  relation IRIs and the facts of the test graph as indices;
* ``rows`` is an ``(n, 3)`` long tensor of candidate triples ``[head, relation, tail]``
  (indices into ``graph.entities`` and ``graph.relations``), each the axiom
  :math:`h \sqsubseteq \exists r.t`;
* the result is one cost per row, *lower is more plausible*, as everywhere in
  :mod:`mowl.evaluation.model_scoring`. A row whose relation the model does not know gets
  ``inf``.

A model that has a vector per training entity (every transductive embedding) cannot
implement it, and the regime refuses it with :class:`UnsupportedInductiveModel`. If the
signature of the method has a ``seed`` parameter, the regime passes its own seed (a model
whose entity features are random draws, like RESHUFFLE, then encodes the graph with that
seed); if it has a ``target`` parameter, the regime passes ``"tail"`` or ``"head"``, the
column of the rows that holds the corrupted entity (a model that scores a head from the tail
with the inverse relation, like RESHUFFLE trained with inverse triples, needs to know).

**The protocol** is that of the authors of GraIL, which the RESHUFFLE paper follows
(Appendix, "Evaluation Protocol"; the code is ``Utils.analyze_checkpoints`` of
https://github.com/AleksVap/RESHUFFLE with PyKEEN's ``SampledRankBasedEvaluator``): every
test triple :math:`(h, r, t)` is ranked twice, against 50 corrupted triples
:math:`(h, r, t')` and against 50 corrupted triples :math:`(h', r, t)`. The corrupting
entities are drawn uniformly without replacement from the entities of the test graph that
do not make a known true triple (a fact of the test graph or a test triple); the rank of
the true triple among the 51 is the *realistic* rank (tied costs share the average rank).
Metrics are Hits@\ :math:`k`, MRR and MR over both directions.
"""

import inspect

import numpy as np
import torch as th

from mowl.evaluation.regime import EvaluationRegime
from mowl.evaluation.ranking_metrics import summarise_ranks

__all__ = ["UnsupportedInductiveModel", "InductiveSampledRankingRegime",
           "inductive_costs_of"]


class UnsupportedInductiveModel(TypeError):
    """The model cannot score triples over a graph it was not trained on."""


def inductive_costs_of(model):
    """``model.inductive_costs``, or raise :class:`UnsupportedInductiveModel` with the
    reason a model without it cannot be evaluated inductively."""
    method = getattr(model, "inductive_costs", None)
    if not callable(method):
        raise UnsupportedInductiveModel(
            f"{type(model).__name__} cannot be evaluated on an inductive benchmark: it has "
            f"no method inductive_costs(graph, rows). The test graph has entities the model "
            f"has not seen in training; a model with one vector per training entity (every "
            f"transductive embedding) has no vector for them. Only a model that computes "
            f"the representation of an entity from the facts about it can score triples "
            f"over a new graph (see mowl.evaluation.inductive_ranking).")
    return method


def _accepts(method, name):
    try:
        return name in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


class InductiveSampledRankingRegime(EvaluationRegime):
    """Rank every test triple against sampled corruptions of one side, on a graph the
    model was not trained on. See the module documentation for the protocol and for the
    interface a model needs.

    The dataset needs an ``inference_graph`` (:class:`InductiveGraph`) and
    ``test_triples`` (IRI triples), as
    :class:`mowl.datasets.builtin.inductive_kg.InductiveKnowledgeGraphDataset` has.

    :param num_negatives: Corrupted triples per test triple and side (the benchmark: 50).
    :param hits: Values of :math:`k` for Hits@\\ :math:`k`.
    :param seed: Seed of the negative sampling (and of the model's random features, if its
        method takes a ``seed``).
    :param ties: ``"average"`` (the *realistic* rank, as PyKEEN) or ``"optimistic"`` /
        ``"pessimistic"`` (ties ranked after none / all of the tied corruptions).
    :param chunk: Candidate triples scored per call of the model.
    """

    key = "inductive-sampled-ranking"

    def __init__(self, num_negatives=50, hits=(1, 3, 10), seed=0, ties="average",
                 chunk=2 ** 16, key=""):
        if ties not in ("average", "optimistic", "pessimistic"):
            raise ValueError("ties must be 'average', 'optimistic' or 'pessimistic'.")
        self.num_negatives = int(num_negatives)
        self.hits = tuple(hits)
        self.seed, self.ties, self.chunk = int(seed), ties, int(chunk)
        if key:
            self.key = key
        self.metrics = tuple([f"hits@{k}" for k in self.hits] + ["mrr", "mr"])
        self.subsets = ("all", "tail", "head")
        self.info = {}

    # -- negatives ----------------------------------------------------------------------
    def sample_negatives(self, graph, test_rows):
        """Corrupting entities for every test triple and side.

        :param test_rows: ``(m, 3)`` long tensor of the test triples in ``graph`` indices.
        :return: ``{"tail": (m, n), "head": (m, n)}`` long tensors of entity indices.
        """
        rng = np.random.default_rng(self.seed)
        num_entities = len(graph.entities)
        n = self.num_negatives
        facts = th.cat([graph.triples, test_rows]).tolist()
        known = {"tail": {}, "head": {}}
        for h, r, t in facts:
            known["tail"].setdefault((h, r), set()).add(t)
            known["head"].setdefault((r, t), set()).add(h)
        out = {}
        for side in ("tail", "head"):
            rows = []
            for h, r, t in test_rows.tolist():
                key = (h, r) if side == "tail" else (r, t)
                mask = np.ones(num_entities, dtype=bool)
                mask[list(known[side][key])] = False
                pool = np.flatnonzero(mask)
                if len(pool) == 0:
                    raise ValueError("Every entity of the graph is a true answer of a "
                                     "test triple; there is nothing to corrupt with.")
                if len(pool) < n:
                    # PyKEEN repeats the pool when it is too small
                    pool = np.tile(pool, int(np.ceil(n / len(pool))))
                rows.append(rng.choice(pool, size=n, replace=False))
            out[side] = th.as_tensor(np.stack(rows) if rows else np.zeros((0, n)),
                                     dtype=th.long).reshape(len(test_rows), n)
        return out

    # -- ranking ------------------------------------------------------------------------
    def _costs(self, method, graph, rows, kwargs):
        out = []
        for start in range(0, len(rows), self.chunk):
            out.append(method(graph, rows[start:start + self.chunk], **kwargs)
                       .detach().float().cpu().reshape(-1))
        return th.cat(out) if out else th.zeros(0)

    def _ranks(self, method, graph, test_rows, negatives, side, kwargs):
        m, n = negatives.shape
        column = 2 if side == "tail" else 0
        candidates = test_rows.unsqueeze(1).repeat(1, n + 1, 1)    # (m, n + 1, 3)
        candidates[:, 1:, column] = negatives
        costs = self._costs(method, graph, candidates.reshape(-1, 3), kwargs).reshape(m, n + 1)
        true, other = costs[:, :1], costs[:, 1:]
        better = (other < true).sum(1).float()
        tied = (other == true).sum(1).float()
        if self.ties == "average":
            rank = 1 + better + 0.5 * tied
        elif self.ties == "optimistic":
            rank = 1 + better
        else:
            rank = 1 + better + tied
        return rank, int(th.isinf(true).sum())

    def evaluate(self, model, dataset):
        method = inductive_costs_of(model)
        for attribute in ("inference_graph", "test_triples"):
            if not hasattr(dataset, attribute):
                raise ValueError(
                    f"{type(dataset).__name__} has no {attribute!r}; the inductive "
                    f"regime needs the test graph and the triples to predict (see "
                    f"mowl.datasets.builtin.inductive_kg.InductiveKnowledgeGraphDataset).")
        graph = dataset.inference_graph
        test_rows = graph.index_triples(dataset.test_triples)
        negatives = self.sample_negatives(graph, test_rows)
        wants_target = _accepts(method, "target")
        ranks, unscorable = {}, 0
        for side in ("tail", "head"):
            kwargs = {"seed": self.seed} if _accepts(method, "seed") else {}
            if wants_target:
                kwargs["target"] = side
            ranks[side], bad = self._ranks(method, graph, test_rows, negatives[side], side,
                                           kwargs)
            unscorable += bad
        n = self.num_negatives + 1
        out = {}
        for subset, values in (("tail", ranks["tail"]), ("head", ranks["head"]),
                               ("all", th.cat([ranks["tail"], ranks["head"]]))):
            summary = summarise_ranks(values.tolist(), n, hits=self.hits, auc="mowl")
            out[subset] = {name: summary[name] for name in self.metrics}
        self.info = {"num_negatives": self.num_negatives, "queries": int(len(test_rows)),
                     "directions": 2, "num_graph_entities": len(graph.entities),
                     "num_graph_facts": int(len(graph.triples)), "ties": self.ties,
                     "seed": self.seed, "true_triples_without_a_score": unscorable}
        return out
