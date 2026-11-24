"""Per-normal-form mini-batches as a training regime.

mOWL's generic EL training loop is full-batch: every epoch scores all axioms of every
normal form and takes one optimisation step. Several papers instead take every step on a
*mini-batch per normal form*:

* ELEmbeddings (Kulmanov et al. 2019, Algorithm 1, lines 18-22): "Randomly sample a
  minibatch of size bs for each loss type", then update the embeddings with the sum of the
  losses; EmEL++ (Mondal et al., Section 3.3) does the same;
* the reference implementation of Box\\ :sup:`2`\\ EL (``model/BoxSquaredEL.py``,
  ``get_data_batch``): one step per epoch on ``batch_size`` axioms drawn with replacement
  from every normal form (all of them if there are at most ``batch_size``); Aharoni
  (2024, Section 7.1.3) reuses this code base;
* the reference implementation of the NeSy 2024 / DELE papers (``dataloader.py``): an
  epoch consists of :math:`\\max_t \\lceil n_t / b \\rceil` steps; step :math:`j` takes the
  :math:`(j \\bmod \\lceil n_t/b \\rceil)`-th consecutive batch of every normal form
  :math:`t` (smaller normal forms cycle).

:class:`PerNormalFormMiniBatches` replaces every training dataset by a view that serves a
new mini-batch at the start of every epoch, so that one epoch of the generic loop becomes
one optimisation *step* on one mini-batch per normal form. Positives, negatives (which are
generated from the view) and losses of a step all refer to the same mini-batch. To train
for :math:`E` epochs of :math:`S` steps each, as the DELE code does, call
``train(epochs=E * S)``; the regime's :attr:`steps_per_epoch` gives :math:`S`.

Place this regime *after* regimes that change the training data (for example
:class:`mowl.training.dele.ClosurePositives`) in a composition, so that batches are drawn
from the transformed data.
"""

import math

import numpy as np
import torch as th

from mowl.training.regime import TrainingRegime


class _BatchView(th.utils.data.Dataset):
    """The current mini-batch of a dataset."""

    def __init__(self, dataset):
        super().__init__()
        self.full = dataset[:]
        self.index = None

    @property
    def data(self):
        return self.full if self.index is None else self.full[self.index]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.full) if self.index is None else len(self.index)


class PerNormalFormMiniBatches(TrainingRegime):
    """One mini-batch per normal form per optimisation step.

    :param batch_size: Axioms per normal form and step (default: the model's
        ``batch_size``).
    :param order: ``"random"`` (draw ``batch_size`` axioms with replacement, all of them
        if there are at most ``batch_size``; ELEmbeddings, Box\\ :sup:`2`\\ EL code) or
        ``"cyclic"`` (consecutive batches, cycling; DELE code).
    :param shuffle: For ``order="cyclic"``, shuffle a normal form whenever its cycle
        restarts (the DELE code does not shuffle).
    :param normal_forms: Axiom types that are batched (default: all); the others are used
        in full at every step.
    :param sample_small: For ``order="random"``: also draw ``batch_size`` axioms with
        replacement from normal forms with at most ``batch_size`` axioms, instead of using
        all of them once (the Keras generators of the ELEmbeddings and EmEL++ code do
        this, so that every normal form contributes exactly ``batch_size`` rows).
    :param seed: Seed.
    """

    key = "per-normal-form-minibatch"

    def __init__(self, batch_size=None, order="random", shuffle=False, normal_forms=None,
                 sample_small=False, seed=None):
        super().__init__()
        if order not in ("random", "cyclic"):
            raise ValueError("order must be 'random' or 'cyclic'.")
        self.batch_size = batch_size
        self.order = order
        self.shuffle = bool(shuffle)
        self.sample_small = bool(sample_small)
        self.normal_forms = None if normal_forms is None else tuple(normal_forms)
        self.rng = np.random.default_rng(seed)
        self.views = {}
        self._cursor = {}
        self._perm = {}

    @property
    def size(self):
        b = self.batch_size if self.batch_size is not None else self.model.batch_size
        return max(1, int(b))

    @property
    def steps_per_epoch(self):
        """:math:`\\max_t \\lceil n_t / b \\rceil` over the batched normal forms."""
        if not self.views:
            return 0
        return max(math.ceil(len(v.full) / self.size) for v in self.views.values())

    def training_datasets(self, datasets):
        out = dict(datasets)
        self.views = {}
        for name, ds in datasets.items():
            if self.normal_forms is not None and name not in self.normal_forms:
                continue
            view = _BatchView(ds)
            self.views[name] = view
            self._cursor[name] = 0
            self._perm[name] = th.arange(len(view.full))
            out[name] = view
        return out

    def _next_index(self, name, view):
        n = len(view.full)
        b = self.size
        if n == 0:
            return th.zeros(0, dtype=th.long)
        if self.order == "random":
            if n <= b and not self.sample_small:
                return th.arange(n)
            return th.as_tensor(self.rng.integers(0, n, size=b), dtype=th.long)
        steps = math.ceil(n / b)
        j = self._cursor[name]
        if j == 0 and self.shuffle:
            self._perm[name] = th.as_tensor(self.rng.permutation(n), dtype=th.long)
        idx = self._perm[name][j * b:(j + 1) * b]
        self._cursor[name] = (j + 1) % steps
        return idx

    def on_epoch_start(self, epoch, epochs):
        for name, view in self.views.items():
            view.index = self._next_index(name, view).to(view.full.device)
