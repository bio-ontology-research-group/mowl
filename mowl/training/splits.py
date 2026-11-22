"""An epoch as several optimisation steps on a shuffled split of the training data.

:class:`ShuffledSplits` is the training schedule of Volume MultiBoxEL (Aharoni 2024, Chapter 7):
at the start of every epoch the axioms of each normal form are shuffled and cut into
``num_batches`` parts, and the epoch takes one optimisation step on the ``b``-th part of every
normal form for ``b = 0 ... num_batches - 1``. Every axiom is used once per epoch, in a fresh
random batch. It differs from :class:`~mowl.training.PerNormalFormMiniBatches`, which draws
batches (with or without replacement) independently at every step.

One *step* is one epoch of the generic training loop, so train for ``epochs * num_batches``
steps (a model that names the regime as its default does this in ``train``);
:attr:`steps_per_epoch` is ``num_batches``.
"""

import logging

import torch as th

from mowl.training.batching import _BatchView
from mowl.training.regime import TrainingRegime

logger = logging.getLogger(__name__)


def canonical_order(data):
    """Rows sorted lexicographically: the order in which the normaliser produces axioms is
    not stable between runs, and sorting makes a run reproducible for a fixed seed."""
    for col in reversed(range(data.shape[1])):
        data = data[th.sort(data[:, col], stable=True).indices]
    return data


class ShuffledSplits(TrainingRegime):
    """Shuffled splits: an epoch is ``num_batches`` steps, each on one part of a fresh random
    split of every normal form.

    :param num_batches: Steps per epoch (5 in the thesis).
    :param seed: Seed of the shuffling generator (a generator of its own, created when
        training starts).
    :param forms: The normal forms that are trained (default: all with axioms); the others
        are left out with a warning. If none is left, training stops with an error.
    :param canonical_order: Sort the axioms of every form first (see :func:`canonical_order`).
    :param what: Named in the error and the warning (``"the thesis"``).

    The mean loss of every epoch is appended to ``model.training_losses`` (a list that is
    reset when training starts) if the model has that attribute.
    """

    key = "shuffled-splits"
    skips_empty_steps = True

    def __init__(self, num_batches=5, seed=0, forms=None, canonical_order=True,
                 what="the method"):
        super().__init__()
        if num_batches < 1:
            raise ValueError("num_batches must be at least 1.")
        self.num_batches = int(num_batches)
        self.seed = seed
        self.forms = None if forms is None else tuple(forms)
        self.canonical_order = bool(canonical_order)
        self.what = what
        self.views = {}
        self.epoch_losses = []
        self._splits = {}
        self._epoch_loss = 0.0
        self._generator = None

    @property
    def steps_per_epoch(self):
        return self.num_batches

    def training_datasets(self, datasets):
        usable, skipped = {}, []
        for name, ds in datasets.items():
            if len(ds) == 0:
                continue
            if self.forms is None or name in self.forms:
                usable[name] = ds
            else:
                skipped.append(name)
        if skipped:
            logger.warning("%s defines no loss for %s; these axioms are not used for "
                           "training.", self.what, skipped)
        if not usable:
            raise ValueError(f"No axioms with a loss defined by {self.what} were found.")
        self.views = {}
        for name, ds in usable.items():
            view = _BatchView(ds)
            if self.canonical_order:
                view.full = canonical_order(view.full)
            self.views[name] = view
        self._generator = th.Generator().manual_seed(self.seed)
        self.epoch_losses = []
        if hasattr(self.model, "training_losses"):
            self.model.training_losses = self.epoch_losses
        return dict(self.views)

    def on_epoch_start(self, epoch, epochs):
        b = epoch % self.num_batches
        if b == 0:
            self._epoch_loss = 0.0
            self._splits = {name: th.randperm(len(view.full), generator=self._generator)
                            .chunk(self.num_batches) for name, view in self.views.items()}
        for name, view in self.views.items():
            parts = self._splits[name]
            index = parts[b] if b < len(parts) else th.zeros(0, dtype=th.long)
            view.index = index.to(view.full.device)

    def on_epoch_end(self, epoch, losses):
        self._epoch_loss += losses["total"] / self.num_batches
        if (epoch + 1) % self.num_batches == 0:
            self.epoch_losses.append(self._epoch_loss)
