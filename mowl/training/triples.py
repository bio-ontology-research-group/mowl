r"""Training and checking a model of the triples of a graph, as general parts.

A model that learns from the triples of a projected graph is trained the same way whatever
its parts are (entity model, relation model, interaction, loss, sampler); this module holds
what that takes, independent of any method:

* :func:`positive_triples`: the training triples in model ids, with the inverse triples when
  the model has them;
* :class:`TripleBatches`: the training regime of a triple geometry (shuffled batches,
  corrupted triples from the sampler of the geometry or the regime, loss of the geometry or
  the regime, regularisation); :func:`train_on_triples` trains a geometry with it in the
  shared loop of :mod:`mowl.training.loop` (the constraints applied after every step, every
  hook of :class:`~mowl.training.TrainingRegime`);
* :func:`carry_rows`: copy the table rows of the keys two graphs share, when the ontology
  changes;
* :func:`extend_embeddings`: table rows for the nodes of a new graph, from earlier embeddings
  where there are any and random otherwise;
* :class:`HitsEarlyStopping`: the training regime that stops on the Hits@k of validation
  triples (an evaluation regime used as the stopping rule).
"""

import copy

import numpy as np
import torch as th

from mowl.training.loop import Rows, TrainingLoop
from mowl.training.regime import ComposedRegime, TrainingRegime

__all__ = ["positive_triples", "train_on_triples", "TripleBatches", "carry_rows",
           "extend_embeddings", "forward_id", "HitsEarlyStopping"]


def positive_triples(mapped_triples, inverse=False):
    """The positive triples ``(E, 3)`` in model ids: ``mapped_triples`` themselves, or, with
    ``inverse``, ``(h, 2r, t)`` and ``(t, 2r + 1, h)`` for every triple (relation ``r`` is
    ``2r`` in the model and its inverse ``2r + 1``)."""
    if not inverse:
        return mapped_triples
    forward = mapped_triples.clone()
    forward[:, 1] = 2 * mapped_triples[:, 1]
    backward = mapped_triples.clone()
    backward[:, 1] = 2 * mapped_triples[:, 1] + 1
    return th.cat([forward, backward.flip(1)])


def _regularization(module, regime):
    term = module.regularization_loss()
    if regime is not None:
        out = regime.regularization(term)
        if out is not NotImplemented:
            term = out
    return term


class TripleBatches(TrainingRegime):
    """The training regime of a geometry of triples: every epoch visits the triples
    (form ``"triple"``) in random batches; the corrupted triples of a batch (the sampler of
    the geometry, or the ``negatives`` of the regimes in front of this one) are scored with
    the same triple cost, the objective of the geometry (or the regimes' ``loss``) turns the
    scores into the loss, plus the regularisation of the geometry (or the regimes'
    ``regularization``). It computes the loss of every step
    (:meth:`~mowl.training.TrainingRegime.step_loss`) of the shared loop.

    :param module: A geometry with ``triple_cost``, ``sampler``, ``loss``,
        ``regularization_loss``
        (:class:`~mowl.representation.library.other_graph.TripleGeometry`).
    :param num_entities, num_relations: Sizes of the graph (for the sampler).
    :param batch_size: Triples per batch.
    :param drop_last: Drop the last, smaller batch.
    :param device: Device of the batches.
    """

    key = "triple-batches"

    def __init__(self, module, num_entities, num_relations, batch_size, drop_last=False,
                 device="cpu"):
        super().__init__()
        self.module, self.num_entities, self.num_relations = module, num_entities, \
            num_relations
        self.batch_size, self.drop_last, self.device = batch_size, drop_last, device
        #: The regimes in front of this one (their negatives, loss and regularisation).
        self.outer = None
        self._batches, self._iterator = None, None

    def _sample(self, positives):
        outer = self.outer
        negatives = NotImplemented
        if outer is not None:
            negatives = outer.negatives("triple", positives)
        if negatives is NotImplemented:
            negatives = self.module.sampler(positives, self.num_entities, self.num_relations)
        if outer is not None:
            negatives = outer.filter_negatives("triple", positives, negatives)
        return negatives

    def training_datasets(self, datasets):
        from mowl.representation.library.other_graph import _TripleBatches
        self._batches = th.utils.data.DataLoader(
            dataset=_TripleBatches(datasets["triple"][:], self.batch_size, self.drop_last,
                                   self._sample),
            batch_size=None, batch_sampler=None)
        if len(self._batches) == 0:
            raise ValueError("There is no training batch: the graph has no triples.")
        return datasets

    @property
    def steps_per_epoch(self):
        return len(self._batches)

    def on_epoch_start(self, epoch, epochs):
        self._iterator = iter(self._batches)

    def _regularization(self):
        term = self.module.regularization_loss()
        if self.outer is not None:
            out = self.outer.regularization(term)
            if out is not NotImplemented:
                term = out
        return term

    def step_loss(self, step):
        module, outer = self.module, self.outer
        positives, negatives = next(self._iterator)
        positives = positives.to(self.device)
        negatives = negatives.to(self.device)
        shape = negatives.shape[:-1]
        positive_scores = -module.triple_cost(positives[:, 0], positives[:, 1], positives[:, 2])
        flat = negatives.reshape(-1, 3)
        negative_scores = -module.triple_cost(flat[:, 0], flat[:, 1],
                                              flat[:, 2]).reshape(*shape)
        loss = NotImplemented
        if outer is not None:
            loss = outer.loss("triple", positive_scores, negative_scores)
        if loss is NotImplemented:
            loss = module.loss(positive_scores, negative_scores)
        loss = loss + self._regularization()
        if not th.isfinite(loss):
            raise FloatingPointError("The loss is not finite.")
        return loss


class _TripleRun(TrainingLoop):
    """The shared loop configured for a geometry of triples: an epoch is a pass over the
    batches of :class:`TripleBatches`, the epoch hooks of the regimes see epochs and get the
    sum of the losses of the epoch, the constraints of the geometry are applied after every
    step (``module.project()``, then the regimes' ``after_step``); nothing is saved and the
    module stays in training mode."""

    epoch_hooks = True
    epoch_loss = "sum"
    eval_after_training = False

    def __init__(self, module, triples, regime, batches, optimizer, lr, device):
        self.module, self.device = module, device
        self.training_regime, self.batches = regime, batches
        self._triples, self._optimizer, self._lr = triples, optimizer, lr

    def epoch_steps(self, regime):
        return self.batches.steps_per_epoch

    @property
    def training_datasets(self):
        return {"triple": Rows(self._triples)}

    def get_optimizer(self):
        return self._optimizer(params=self.module.trainable_parameters(), lr=self._lr)

    def step_regularization_loss(self):
        return 0

    def after_optimizer_step(self):
        self.module.project()
        self.training_regime.after_step()


def train_on_triples(module, triples, num_entities, num_relations, optimizer, lr, batch_size,
                     epochs, regime=None, device="cpu", drop_last=False, reinitialise=False):
    """Train a geometry on the triples of a graph: :class:`TripleBatches` (with ``regime``
    in front of it) in the shared loop (:meth:`~mowl.training.loop.TrainingLoop.run_training`);
    after the optimiser step the geometry's constraints are applied (``module.project()``).

    :param module: A geometry with ``triple_cost``, ``sampler``, ``loss``,
        ``regularization_loss``, ``project``, ``trainable_parameters`` and ``reset_parameters``
        (:class:`~mowl.representation.library.other_graph.TripleGeometry`).
    :param triples: ``(E, 3)`` positive triples in model ids.
    :param optimizer: A class of :mod:`torch.optim` (the regime's ``optimizer`` replaces it).
    :param regime: A :class:`~mowl.training.TrainingRegime` or ``None``.
    :param reinitialise: Draw the parameters again before training.
    :return: The sum of the losses of every epoch.
    """
    if reinitialise:
        module.reset_parameters()
    batches = TripleBatches(module, num_entities, num_relations, batch_size, drop_last, device)
    batches.outer = regime
    run_regime = batches if regime is None else ComposedRegime(regime, batches)
    run_regime.model = getattr(regime, "model", None)
    return _TripleRun(module, triples, run_regime, batches, optimizer, lr,
                      device).run_training(epochs)


def carry_rows(old, new, before, after, doubled):
    """Copy the rows of the tables of the part ``old`` into ``new`` for the keys they share
    (``before``, ``after``: key -> row; ``doubled``: the rows of a relation are ``2 i`` and
    ``2 i + 1``)."""
    new_params = dict(new.named_parameters())
    for name, parameter in old.named_parameters():
        target = new_params.get(name)
        if target is None or parameter.shape[1:] != target.shape[1:]:
            continue
        if not parameter.requires_grad:
            continue
        with th.no_grad():
            for key, i in before.items():
                j = after.get(key)
                if j is None:
                    continue
                if doubled:
                    for d in (0, 1):
                        if 2 * i + d < len(parameter) and 2 * j + d < len(target):
                            target[2 * j + d] = parameter[2 * i + d]
                elif i < len(parameter) and j < len(target):
                    target[j] = parameter[i]


def extend_embeddings(previous, keys, size, rng=None):
    """Table rows ``(len(keys), size)`` for the keys of a new graph: the earlier embedding
    of a key that had one, a standard normal draw otherwise.

    :param previous: A mapping key -> earlier vector, or a list of mappings (the first one
        that has the key is read).
    :param keys: The keys, in the row order of the new tables.
    :param size: Length of a row.
    """
    mappings = previous if isinstance(previous, (list, tuple)) else [previous]
    rows = []
    for key in keys:
        for mapping in mappings:
            if key in mapping:
                rows.append(mapping[key])
                break
        else:
            rows.append((rng or np.random).normal(size=size))
    return th.tensor(np.asarray(rows))


class HitsEarlyStopping(TrainingRegime):
    """Early stopping on the Hits@k of the validation triples: the evaluation regime of the
    RESHUFFLE experiments (every ``frequency`` epochs, stop when ``patience`` evaluations in a
    row did not improve the best Hits@k by ``relative_delta``, then restore the best
    parameters). Hits@k is the share of validation triples, over both sides, whose true
    entity ranks among the first ``k`` of all entities (filtered: the other known triples of
    the graph are not counted; ties share their ranks).

    The validation triples are those of the validation ontology of the dataset projected like
    the training ontology (:meth:`TripleGraphModel.validation_triples`).
    """

    key = ""

    def __init__(self, frequency=10, patience=10, relative_delta=0.01, k=10, batch_size=64):
        super().__init__()
        self.frequency, self.patience = int(frequency), int(patience)
        self.relative_delta, self.k, self.batch_size = float(relative_delta), int(k), batch_size
        self.history = []
        self._best, self._best_state, self._bad, self._val = None, None, 0, None

    def attach(self, model):
        super().attach(model)
        self.history, self._best, self._best_state, self._bad = [], None, None, 0
        self._val = None

    def _ranks(self, module, queries, relations, candidates_are, truth, known, n):
        """Rank of ``truth`` among all entities as the ``candidates_are`` ("tail" or
        "head") of the queries."""
        ranks = []
        everyone = th.arange(n)
        for start in range(0, len(queries), self.batch_size):
            q = queries[start:start + self.batch_size]
            r = relations[start:start + self.batch_size]
            t = truth[start:start + self.batch_size]
            b = len(q)
            repeated = q.repeat_interleave(n)
            rels = r.repeat_interleave(n)
            cands = everyone.repeat(b)
            cost = module.triple_cost(repeated, rels, cands).reshape(b, n)
            true_cost = cost[th.arange(b), t].clone()
            for row in range(b):
                for other in known.get((int(q[row]), int(r[row])), ()):
                    if other != int(t[row]):
                        cost[row, other] = float("inf")
            optimistic = (cost < true_cost[:, None]).sum(dim=1) + 1
            pessimistic = (cost <= true_cost[:, None]).sum(dim=1)
            ranks.append((optimistic + pessimistic) / 2.0)
        return th.cat(ranks)

    @th.no_grad()
    def hits_at_k(self):
        """Hits@k of the validation triples, filtered, both sides, all entities."""
        model = self.model
        module, factory = model.module, model.triples_factory
        n = factory.num_entities
        if self._val is None:
            self._val = model.validation_triples()
        val, inverse = self._val, model.create_inverse_triples
        forward = 2 * val[:, 1] if inverse else val[:, 1]
        backward = 2 * val[:, 1] + 1 if inverse else val[:, 1]
        tails, heads = {}, {}
        for h, r, t in th.cat([factory.mapped_triples, val]).tolist():
            tails.setdefault((h, forward_id(r, inverse)), set()).add(t)
            if inverse:
                heads.setdefault((t, 2 * r + 1), set()).add(h)
            else:
                heads.setdefault((t, r), set()).add(h)
        module.eval()
        with module.encoding():
            tail_ranks = self._ranks(module, val[:, 0], forward, "tail", val[:, 2], tails, n)
            if inverse:
                head_ranks = self._ranks(module, val[:, 2], backward, "tail", val[:, 0],
                                         heads, n)
            else:
                head_ranks = self._head_ranks(module, val, n, heads)
        ranks = th.cat([tail_ranks, head_ranks])
        return float((ranks <= self.k).float().mean())

    def _head_ranks(self, module, val, n, known):
        ranks = []
        everyone = th.arange(n)
        for start in range(0, len(val), self.batch_size):
            v = val[start:start + self.batch_size]
            b = len(v)
            cost = module.triple_cost(everyone.repeat(b), v[:, 1].repeat_interleave(n),
                                      v[:, 2].repeat_interleave(n)).reshape(b, n)
            true_cost = cost[th.arange(b), v[:, 0]].clone()
            for row in range(b):
                for other in known.get((int(v[row, 2]), int(v[row, 1])), ()):
                    if other != int(v[row, 0]):
                        cost[row, other] = float("inf")
            optimistic = (cost < true_cost[:, None]).sum(dim=1) + 1
            pessimistic = (cost <= true_cost[:, None]).sum(dim=1)
            ranks.append((optimistic + pessimistic) / 2.0)
        return th.cat(ranks)

    def should_stop(self, epoch):
        if (epoch + 1) % self.frequency:
            return False
        hits = self.hits_at_k()
        self.history.append(hits)
        if self._best is None or hits > self._best * (1.0 + self.relative_delta):
            self._best, self._bad = hits, 0
            self._best_state = copy.deepcopy(self.model.module.state_dict())
        else:
            self._bad += 1
        return self._bad >= self.patience

    def on_train_end(self):
        if self._best_state is not None:
            self.model.module.load_state_dict(self._best_state)


def forward_id(relation, inverse):
    """The model id of the forward relation ``relation`` of the graph."""
    return 2 * relation if inverse else relation
