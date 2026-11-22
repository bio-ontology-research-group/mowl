"""Alternating minibatches of the assertion forms, as regimes (the loop of FaithEL).

The authors' code of FaithEL trains the ABox only: per epoch the concept assertions and then the
role assertions, each in a random order and in minibatches, one optimisation step per
minibatch. :class:`AlternatingMinibatches` is that batching: a step is the minibatch of one
form, the forms follow each other within an epoch. :class:`FormSelection` keeps only some forms
and takes everything at every step (:class:`FullBatch` for the assertion forms).
"""

import math

import torch as th

from mowl.training.batching import _BatchView
from mowl.training.loop import Rows
from mowl.training.regime import TrainingRegime

#: The forms of the ABox and the number of columns of their rows.
ASSERTION_FORMS = (("class_assertion", 2), ("object_property_assertion", 3))


class FormSelection(TrainingRegime):
    """Train on some forms only (the others are left out of the loop); every step takes all
    their rows.

    :param forms: ``{form: number of columns}`` or an iterable of form names; a form without
        axioms trains on nothing.
    """

    key = "form-selection"
    skips_empty_steps = True

    def __init__(self, forms=ASSERTION_FORMS):
        super().__init__()
        self.forms = dict(forms) if not isinstance(forms, dict) else dict(forms)

    def _rows(self, datasets, name, width):
        ds = datasets.get(name)
        if ds is not None and len(ds):
            return ds
        return Rows(th.empty((0, width), dtype=th.long, device=self.model.device))

    def training_datasets(self, datasets):
        return {name: self._rows(datasets, name, width) for name, width in self.forms.items()}


class FullBatch(FormSelection):
    """One step per epoch on all assertions of both forms (the sum of the mean losses): the
    generic loop of the other models restricted to the ABox, as a swap for
    :class:`AlternatingMinibatches`."""

    key = "full-batch"


class AlternatingMinibatches(FormSelection):
    """Per epoch, the rows of each form in turn in a fresh random order and in minibatches of
    ``batch_size`` rows; one step per minibatch (the loop of the authors' code of FaithEL).
    The permutation of a form is drawn from the global torch generator when its first
    minibatch starts.

    :param batch_size: Rows per step (default: the model's ``batch_size``).
    :param forms: As for :class:`FormSelection`; the order is the order of the epoch.
    """

    key = "alternating-minibatches"

    def __init__(self, batch_size=None, forms=ASSERTION_FORMS):
        super().__init__(forms)
        self.batch_size = batch_size
        self.views = {}
        self._perm = None

    @property
    def size(self):
        b = self.batch_size if self.batch_size is not None else self.model.batch_size
        return max(1, int(b))

    @property
    def steps_per_epoch(self):
        """The number of minibatches of all forms."""
        return sum(math.ceil(len(v.full) / self.size) for v in self.views.values())

    def training_datasets(self, datasets):
        chosen = super().training_datasets(datasets)
        self.views = {name: _BatchView(ds) for name, ds in chosen.items()}
        return dict(self.views)

    def on_epoch_start(self, epoch, epochs):
        steps, b = self.steps_per_epoch, self.size
        j = epoch % steps if steps else 0
        offset = 0
        for view in self.views.values():
            n = len(view.full)
            count = math.ceil(n / b)
            if offset <= j < offset + count:
                s = j - offset
                if s == 0:
                    self._perm = th.randperm(n, device=view.full.device)
                view.index = self._perm[s * b:(s + 1) * b]
            else:
                view.index = th.zeros(0, dtype=th.long, device=view.full.device)
            offset += count
