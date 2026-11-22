"""Where the training queries of a query embedding model come from (the query sampler).

A sampler is a part of the training regime of the query methods, independent of the
geometry that is trained: it gives, for every optimisation update, a batch of queries
``{structure name: (signature, id rows, positive answers, negative entities[, weights])}``
that a :class:`~mowl.representation.library.query_objective.QueryObjective` scores.

* :class:`SampledQueries`: queries of the structures 1p, 2p, 3p, 2i, 3i, ip, pi (and the
  negation structures for geometries with negation) sampled from the graph, backwards from
  a random answer, afresh in every step.
* :class:`ReleasedQueries`: the training queries of a released query dataset, as in the four
  papers.
"""

import numpy as np
import torch as th

from mowl.training import query_structures as sampling
from mowl.nn.query.expression import ids, signature

__all__ = ["SampledQueries", "ReleasedQueries", "UpdateStream"]


class UpdateStream:
    """The updates of a sampler as a dataset of the shared training loop: every access
    (``stream[:]``) is the next update, in the order the sampler gives them (all the updates
    of a step, then the next step's)."""

    def __init__(self, sampler):
        self.sampler = sampler
        self._pending = []

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if not self._pending:
            self._pending = list(self.sampler.updates())
        return self._pending.pop(0)


class SampledQueries:
    """Queries sampled from the graph of the ontology.

    :param graph: The :class:`~mowl.projection.query_graph.QueryGraph`.
    :param templates: ``{name: template}`` of the structures the geometry can train
        (:func:`mowl.training.query_structures.structures`).
    :param rng: ``numpy`` random generator.
    :param batch_size: Queries per update (in total, or per structure with
        ``batch_per_structure``).
    :param negatives: Negative entities per query.
    :param batch_per_structure: ``batch_size`` queries of every structure (Query2Box,
        Appendix) instead of ``batch_size`` in total.
    """

    released = False

    def __init__(self, graph, templates, rng, batch_size=64, negatives=16,
                 batch_per_structure=False, device="cpu"):
        self.graph, self.rng, self.device = graph, rng, device
        self.batch_size, self.negatives = batch_size, negatives
        self.batch_per_structure = bool(batch_per_structure)
        self.pool = graph.entities_with_incoming()
        self.structures = (sampling.usable_structures(templates, graph, rng, self.pool)
                           if self.pool else {})

    def batch(self):
        names = list(self.structures)
        per = (self.batch_size if self.batch_per_structure
               else max(1, self.batch_size // len(names)))
        groups = {}
        for name in names:
            template = self.structures[name]
            rows, pos, negs = [], [], []
            for _ in range(per):
                sample = sampling.sample_query(template, self.graph, self.rng, self.pool)
                if sample is None:
                    continue
                query, found, answer = sample
                rows.append(ids(query))
                pos.append(answer)
                negs.append(sampling.negative_entities(
                    found, self.negatives, self.graph.num_entities, self.rng))
            if rows:
                dev = self.device
                groups[name] = (signature(template),
                                th.tensor(rows, dtype=th.long, device=dev),
                                th.tensor(pos, dtype=th.long, device=dev).unsqueeze(1),
                                th.tensor(np.array(negs), dtype=th.long, device=dev))
        return groups

    updates_per_step = 1

    def updates(self):
        """The batches of one step: ``[batch]``."""
        return [self.batch()]

    def check(self):
        if not self.structures:
            raise ValueError("No training query can be sampled from the graph of this "
                             "ontology: it has no subclass, role or assertion edges.")


class ReleasedQueries:
    """The released training queries of a query answering dataset (the dataset must have
    ``training_queries(entity_id, relation_id, names)``, e.g.
    :class:`~mowl.datasets.builtin.query_sets.QueryAnsweringDataset`).

    Every update draws ``batch_size`` queries uniformly from the pool of the path
    structures (1p, 2p, 3p) or of the other structures, so structures are drawn in
    proportion to their number in the release; the positive is a random answer on the
    training graph, the negatives are uniform non-answers, and every query is weighted by
    ``1/sqrt(number of answers)``. One step is an update on a batch of path queries, then one
    on a batch of the other queries, then again one of path queries (a single update if
    there are no other queries), the loop of the reference code (KGReasoning ``main.py``,
    which ConE and AConE copy).
    """

    released = True
    pool = ()
    structures = {}

    def __init__(self, dataset, graph, templates, rng, entity_id, relation_id,
                 structures=None, batch_size=64, negatives=16, device="cpu"):
        self.graph, self.rng, self.device = graph, rng, device
        self.batch_size, self.negatives = batch_size, negatives
        if not hasattr(dataset, "training_queries"):
            raise ValueError("queries='released' needs a dataset with training_queries(), "
                             "e.g. mowl.datasets.builtin.query_sets.QueryAnsweringDataset.")
        pool = dataset.training_queries(entity_id, relation_id)
        names = [n for n in pool if n in templates]
        if structures is not None:
            names = [n for n in names if n in structures]
        if not names:
            raise ValueError(f"None of the released training structures {sorted(pool)} can "
                             f"be trained by this geometry (it trains {sorted(templates)}).")
        self.queries = {n: pool[n] for n in names}
        self.templates = {n: templates[n] for n in names}
        self._columns = {}
        from mowl.datasets.builtin.query_sets import _kinds
        for n, q in self.queries.items():
            kinds = _kinds(q.structure)
            self._columns[n] = [i for i, k in enumerate(kinds) if k in ("e", "r")]
        path = [n for n in names if n in ("1p", "2p", "3p")]
        other = [n for n in names if n not in path]
        self.groups = {"path": path, "other": other}

    def _samples(self, q, local):
        """Positives, negatives and weights for the queries ``local`` of a
        :class:`~mowl.datasets.builtin.query_sets.ReleasedTrainingQueries` (vectorised): a
        uniformly drawn answer, ``negatives`` uniform entities that are not answers (the
        ones that are redrawn up to three times), and ``1/sqrt(number of answers)``."""
        n_entities = self.graph.num_entities
        counts = q.offsets[local + 1] - q.offsets[local]
        starts = q.offsets[local]
        pick = starts + (self.rng.random(len(local)) * counts).astype(np.int64)
        positives = q.answers[pick]
        total = int(counts.sum())
        owner = np.repeat(np.arange(len(local)), counts)
        within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        keys = owner * n_entities + q.answers[np.repeat(starts, counts) + within]
        is_answer = np.zeros(len(local) * n_entities, dtype=bool)
        is_answer[keys] = True
        rows = np.arange(len(local))[:, None]
        negatives = self.rng.integers(n_entities, size=(len(local), self.negatives))
        for _ in range(3):
            bad = is_answer[rows * n_entities + negatives]
            if not bad.any():
                break
            negatives[bad] = self.rng.integers(n_entities, size=int(bad.sum()))
        return positives, negatives, 1.0 / np.sqrt(counts)

    def batch(self, group):
        """``batch_size`` training queries drawn uniformly from the structures of a group
        (``"path"`` or ``"other"``), as ``{name: (signature, rows, positives, negatives,
        weights)}``."""
        names = self.groups[group]
        sizes = np.array([len(self.queries[n]) for n in names])
        draw = self.rng.integers(int(sizes.sum()), size=self.batch_size)
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        which = np.searchsorted(bounds, draw, side="right") - 1
        out = {}
        dev = self.device
        for j, name in enumerate(names):
            local = draw[which == j] - bounds[j]
            if len(local) == 0:
                continue
            q = self.queries[name]
            rows = q.rows[local][:, self._columns[name]]
            pos, negs, weights = self._samples(q, local)
            out[name] = (signature(self.templates[name]),
                         th.tensor(rows, dtype=th.long, device=dev),
                         th.tensor(pos, dtype=th.long, device=dev).unsqueeze(1),
                         th.tensor(negs, dtype=th.long, device=dev),
                         th.tensor(weights, dtype=th.float, device=dev))
        return out

    @property
    def updates_per_step(self):
        """Number of updates of a step (path, other, path, or the one group there is)."""
        return 3 if self.groups["other"] and self.groups["path"] else \
            sum(1 for g in ("path", "other") if self.groups[g])

    def updates(self):
        """The batches of one step: path, other, path (or the one group there is)."""
        both = self.groups["other"] and self.groups["path"]
        pattern = (["path", "other", "path"] if both
                   else [g for g in ("path", "other") if self.groups[g]])
        return [self.batch(g) for g in pattern]

    def check(self):
        pass
