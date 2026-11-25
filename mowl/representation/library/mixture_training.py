"""Training regimes and negative samplers of the canonical mixture.

The trained part of the canonical mixture is the individual model, so its training goes over
*individuals*: :class:`IndividualBatches` (the method's own regime) draws a random batch of
individuals, takes their training assertions, samples one negative per assertion and makes
one optimiser step per batch, with the regularisers of the geometry evaluated on the batch.
:class:`AssertionBatches` goes over the assertions instead (minibatches of every form in
turn), as a swap.

Both are :class:`~mowl.training.TrainingRegime` objects that compute the loss of a step
themselves (:meth:`~mowl.training.TrainingRegime.step_loss`), and they train in the shared
loop of :mod:`mowl.training.loop` (:meth:`MixtureBatches.run`): losses and regularisers are
evaluated in chunks of bounded size (the support entries of the individuals,
:meth:`MixtureIndividuals.pair_chunks
<mowl.representation.library.mixture_individuals.MixtureIndividuals.pair_chunks>`) and their
gradients accumulated (:class:`~mowl.training.Accumulated`,
:func:`~mowl.training.loop.accumulate_backward`), so that a step never holds more than a
bounded number of entries; a batch with nothing to supervise takes no step. Other regimes
(schedules, early stopping, projections) can be put in front of them.

Negatives are parts of the regime: :class:`NonEntailedClasses` corrupts the class of a class
assertion with a class that the constructed model does not entail for the individual
(rejection sampling against its own element's classes), :class:`UniformTails` corrupts the
object of a role assertion with a uniformly drawn individual.
"""

import os

import torch as th

from mowl.training.loop import TrainingLoop, accumulate_backward
from mowl.training.regime import Accumulated, ComposedRegime, TrainingRegime

__all__ = ["NonEntailedClasses", "UniformTails", "MixtureBatches", "IndividualBatches",
           "AssertionBatches"]

_FORMS = (("class_assertion", 2), ("object_property_assertion", 3))


class NonEntailedClasses:
    """One negative per class assertion ``(a, C)``: ``(a, C')`` with a class :math:`C'` the
    constructed model does not entail for :math:`a`, drawn uniformly by rejection (``tries``
    draws in all), then the next class (cyclically, from a random start) that is not
    entailed. The membership table is that of the geometry's class model."""

    def __init__(self, tries=20):
        self.tries = int(tries)

    def sample(self, geometry, rows):
        table = geometry.class_model.table
        device = rows.device
        if rows.numel() == 0:
            return th.empty((0, 2), dtype=th.long, device=device)
        ind = rows[:, 0]
        n_classes = table.n_classes
        own = geometry.individual_model.individual_element.index_select(0, ind)
        neg = th.randint(0, n_classes, (len(ind),), device=device, dtype=th.long)
        bad = table.contains(own, neg)
        for _ in range(self.tries - 1):
            if not bool(bad.any()):
                break
            neg[bad] = th.randint(0, n_classes, (int(bad.sum()),), device=device,
                                  dtype=th.long)
            bad = table.contains(own, neg)
        for r in th.nonzero(bad, as_tuple=False).reshape(-1).tolist():
            el = int(own[r])
            if el < 0:
                continue
            entailed = set(table.types_of(el).tolist())
            if len(entailed) < n_classes:
                start = int(th.randint(0, n_classes, (1,), device=device, dtype=th.long))
                for step in range(n_classes):
                    cand = (start + step) % n_classes
                    if cand not in entailed:
                        neg[r] = cand
                        break
        return th.stack([ind, neg], dim=1)


class UniformTails:
    """One negative per role assertion ``(a, r, b)``: ``(a, r, b')`` with :math:`b'` a
    uniformly drawn individual (drawn in blocks of the rows of a chunk)."""

    def sample(self, geometry, rows):
        device = rows.device
        if rows.numel() == 0:
            return th.empty((0, 3), dtype=th.long, device=device)
        model = geometry.individual_model
        n, step = len(rows), model.row_chunk(len(rows))
        out = rows.clone()
        for s in range(0, n, step):
            e = min(n, s + step)
            out[s:e, 2] = th.randint(0, len(model), (e - s,), device=device, dtype=th.long)
        return out


def _data(model):
    out = {}
    for name, width in _FORMS:
        ds = model.training_datasets.get(name)
        out[name] = ds[:].to(model.device) if ds is not None and len(ds) else \
            th.empty((0, width), dtype=th.long, device=model.device)
    return out


def _backward(geometry, rows, form, neg=False):
    """The mean loss of ``rows``, its gradient accumulated chunk by chunk; returns its
    value."""
    n = int(rows.shape[0])
    if n == 0:
        return 0.0
    total = 0.0
    for s, e in geometry.individual_model.pair_chunks(rows):
        total += accumulate_backward(geometry(rows[s:e], form, neg=neg).mean(),
                                     float(e - s) / float(n))
    return total


def _penalties(geometry, batch):
    """The regularisers of the geometry on the individuals ``batch`` (chunk by chunk for
    those that read a batch), their gradients accumulated; returns their value."""
    total = 0.0
    for regularizer in geometry.regularizers:
        if getattr(regularizer, "kind", "penalty") != "penalty":
            continue
        if getattr(regularizer, "batch_penalty", False):
            n = max(int(batch.shape[0]), 1)
            for s, e in geometry.individual_model.chunks(batch):
                total += accumulate_backward(regularizer.penalty(geometry, batch[s:e]),
                                             float(e - s) / n)
        else:
            term = regularizer.penalty(geometry)
            if term is not None:
                total += accumulate_backward(term)
    return total


def _project(geometry):
    for regularizer in geometry.regularizers:
        if getattr(regularizer, "kind", None) == "projection":
            regularizer.project(geometry)
    geometry.project()


class _MixtureRun(TrainingLoop):
    """The shared loop configured for a model trained by a :class:`MixtureBatches` regime:
    an epoch is the regime's steps, a progress bar over the steps, the epoch hooks of the
    regimes see epochs, validation every ``validate_every`` epochs selects the best
    parameters (:meth:`MixtureBatches.select`), which are restored at the end."""

    loop_style = "epochs"
    epoch_hooks = True

    def __init__(self, model, regime, batches):
        self.model, self.batches = model, batches
        self.module, self.device = model.module, model.device
        self.training_regime = regime

    @property
    def training_datasets(self):
        return self.model.training_datasets

    def get_optimizer(self):
        return self.model.get_optimizer()

    def start_training(self, optimizer):
        self.model.training_optimizer = optimizer

    def epoch_steps(self, regime):
        return max(1, self.batches.steps_per_epoch)

    def step_regularization_loss(self):
        return 0

    def after_optimizer_step(self):
        self.training_regime.after_step()

    def record_epoch(self, epoch, mean_loss):
        self.model.loss_history.append(mean_loss)

    def validate_epoch(self, epoch, loss):
        self.batches.select(self.model, loss)

    def finish_training(self):
        self.batches.restore(self.model)


class MixtureBatches(TrainingRegime):
    """What the regimes of the canonical mixture share: negatives, the steps of an epoch
    (:meth:`steps`, a generator that does the draws of the epoch in order and yields one
    :class:`~mowl.training.Accumulated` loss per step, or ``None`` for a step with nothing
    to supervise), the projection after a step, validation and the best checkpoint.

    :param batch_size: Rows per step (default: the model's).
    :param class_negatives, role_negatives: The negative samplers.
    """

    key = ""

    def __init__(self, batch_size=None, class_negatives=None, role_negatives=None):
        super().__init__()
        self.batch_size = batch_size
        self.class_negatives = class_negatives if class_negatives is not None \
            else NonEntailedClasses()
        self.role_negatives = role_negatives if role_negatives is not None else UniformTails()
        self._data, self._epoch = None, None

    def sampler(self, form):
        return self.class_negatives if form == "class_assertion" else self.role_negatives

    @property
    def size(self):
        return self.batch_size or self.model.batch_size

    # -- the hooks of the shared loop -------------------------------------------------------
    def training_datasets(self, datasets):
        self._data = _data(self.model)
        return datasets

    @property
    def steps_per_epoch(self):
        raise NotImplementedError

    def steps(self, geometry, data, batch_size):
        raise NotImplementedError

    def on_epoch_start(self, epoch, epochs):
        self._epoch = self.steps(self.model.module, self._data, self.size)

    def step_loss(self, step):
        return next(self._epoch, None)

    def after_step(self):
        _project(self.model.module)

    # -- validation -------------------------------------------------------------------------
    @staticmethod
    def validation_loss(model):
        """Mean positive loss of the validation assertions of ``model.eval_gci_name`` (or
        of the first assertion form that has some), in batches."""
        datasets = model.validation_datasets
        if not datasets:
            return 0.0
        module = model.module
        with th.no_grad():
            module.eval()
            for form in (model.eval_gci_name, "class_assertion", "object_property_assertion"):
                if form in datasets and len(datasets[form]) > 0:
                    rows = datasets[form][:]
                    step = max(int(model.batch_size), 1)
                    total = sum(float(module(rows[s:s + step], form).detach().sum())
                                for s in range(0, len(rows), step))
                    return total / float(len(rows))
        return 0.0

    def select(self, model, loss):
        """After a validation epoch: the validation loss (or, without validation data, the
        epoch loss); the parameters are saved to ``model.model_filepath`` if it is the best
        so far."""
        value = self.validation_loss(model) if model.dataset.validation is not None else loss
        if value < self._best:
            self._best = value
            th.save(model.module.state_dict(), model.model_filepath)
            self._saved = True

    def restore(self, model):
        """At the end: load the best parameters, if any were saved."""
        if self._saved and model.model_filepath is not None and \
                os.path.exists(model.model_filepath):
            model.module.load_state_dict(th.load(model.model_filepath,
                                                 map_location=model.device, weights_only=True))

    def run(self, model, epochs, validate_every=1, epoch_callback=None):
        """Train ``model`` for ``epochs`` epochs in the shared loop
        (:meth:`~mowl.training.loop.TrainingLoop.run_training`), with the regime assigned to
        the model (``model.training_regime``, if any) in front of this one."""
        self.attach(model)
        self._best, self._saved = float("inf"), False
        model.loss_history = []
        assigned = getattr(model, "training_regime", None)
        regime = self if assigned is None else ComposedRegime(assigned, self)
        regime.model = model
        run = _MixtureRun(model, regime, self)
        callback = None if epoch_callback is None else \
            (lambda epoch, _: epoch_callback(epoch, model))
        run.run_training(epochs, validate_every=validate_every, epoch_callback=callback)
        return model.loss_history


class IndividualBatches(MixtureBatches):
    """The regime of the canonical mixture: per epoch a random order of the individuals, in
    batches of ``batch_size``; per batch the training assertions of its individuals (at most
    ``batch_size`` of every form, drawn at random if there are more) with one negative each,
    the regularisers on the batch, and one optimiser step if anything was supervised. The
    epoch loss is the mean of the losses of the steps. After every ``validate_every`` epochs
    the validation loss (or, without validation data, the epoch loss) selects the best
    parameters, which are saved to ``model.model_filepath`` and restored at the end.

    :param batch_size: Individuals per batch (default: the model's).
    :param class_negatives, role_negatives: The negative samplers.
    """

    key = "individual-batches"

    @property
    def steps_per_epoch(self):
        n_ind, b = len(self.model.module.individual_model), self.size
        return -(-n_ind // b)

    def steps(self, geometry, data, batch_size):
        device = self.model.device
        n_ind = len(geometry.individual_model)
        perm = th.randperm(n_ind, device=device)
        for start in range(0, n_ind, batch_size):
            batch = perm[start:start + batch_size]
            loss, supervised = 0.0, False
            for form, _ in _FORMS:
                rows = data[form]
                if rows.shape[0] == 0:
                    continue
                mask = th.isin(rows[:, 0], batch)
                if not bool(mask.any()):
                    continue
                pos = rows[mask]
                if pos.shape[0] > batch_size:
                    pos = pos.index_select(0, th.randperm(pos.shape[0],
                                                          device=device)[:batch_size])
                neg = self.sampler(form).sample(geometry, pos)
                loss += _backward(geometry, pos, form)
                loss += _backward(geometry, neg, form, neg=True)
                supervised = True
            penalised = any(getattr(r, "kind", "penalty") == "penalty"
                            for r in geometry.regularizers)
            if penalised:
                loss += _penalties(geometry, batch)
            yield Accumulated(loss) if supervised or penalised else None


class AssertionBatches(MixtureBatches):
    """A swap for :class:`IndividualBatches`: per epoch the class assertions and then the
    role assertions, each in a random order and in minibatches of ``batch_size``
    assertions, one negative each; one step per minibatch, the regularisers on the
    individuals of the minibatch."""

    key = "assertion-batches"

    @property
    def steps_per_epoch(self):
        b = self.size
        return sum(-(-int(rows.shape[0]) // b) for rows in self._data.values())

    def steps(self, geometry, data, batch_size):
        for form, _ in _FORMS:
            rows = data[form]
            perm = th.randperm(rows.shape[0], device=self.model.device)
            for s in range(0, rows.shape[0], batch_size):
                pos = rows.index_select(0, perm[s:s + batch_size])
                loss = _backward(geometry, pos, form)
                loss += _backward(geometry, self.sampler(form).sample(geometry, pos), form,
                                  neg=True)
                ids = th.unique(pos[:, 0] if form == "class_assertion"
                                else th.cat([pos[:, 0], pos[:, 2]]))
                loss += _penalties(geometry, ids)
                yield Accumulated(loss)
