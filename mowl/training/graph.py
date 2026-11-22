"""Training regimes of the graph-style hierarchy methods (binary-code boxes, CatE), as parts.

The two methods train on the edges of a graph (or the morphisms of a lattice): epochs that are
passes over the edges in minibatches, several random corruptions of an edge, a mask that
removes corruptions that are edges themselves, and a loss that groups the corruptions of one
edge. Each piece is a regime of its own that other methods can use:

* :class:`EpochBatches`: an epoch is a pass over a fresh permutation of every form in
  minibatches; one step per minibatch (:attr:`~EpochBatches.steps_per_epoch`);
* :class:`EdgeCorruption`: ``k`` corruptions per row, one or two columns, drawn at once from
  the global torch generator;
* :class:`KnownRowMask`: marks corruptions that are training rows or reflexive, for the loss
  to leave out (through ``model.negative_masks``);
* :class:`SelfAdversarialBCE` and :class:`GroupedMarginLoss`: the losses of binary-code boxes
  (Eq. 10) and of CatE (Eq. 7) on the costs of a row and of its corruptions, calling the
  functions of :mod:`mowl.representation.library.hierarchy_objectives`;
* :class:`HeadRestrictedHinge`: the interaction term of the CatE code;
* :class:`ValidationPatience`: validation every ``every`` epochs, the best state kept and
  training stopped after ``patience`` evaluations without improvement.

Layout of corruptions follows :mod:`mowl.training.negatives`: ``k`` consecutive blocks of ``n``
rows, row ``j * n + i`` is the ``j``-th corruption of row ``i``.
"""

import math

import torch as th

from mowl.training.batching import _BatchView
from mowl.training.regime import TrainingRegime

_MASKS = "negative_masks"


def grouped(values, n):
    """``(k * n,)`` in the layout of the corruption regimes as ``(n, k)``."""
    return values.reshape(-1, n).t()


class EpochBatches(TrainingRegime):
    """An epoch is a pass over a fresh random permutation of every form, in minibatches of
    ``batch_size`` consecutive rows (the last one smaller); every step is one minibatch.

    It differs from :class:`~mowl.training.PerNormalFormMiniBatches`, which draws every batch
    independently, and from :class:`~mowl.training.ShuffledSplits`, which cuts the permutation
    into a fixed number of equal parts. A form with fewer minibatches than the longest one
    takes no part in the later steps of the epoch.

    :param batch_size: Rows per step and form (default: the model's ``batch_size``).
    :param forms: The forms that are batched (default: all).
    :param rng: ``None``: the permutation is drawn with ``torch.randperm`` (global generator,
        on the device of the rows); an object with ``shuffle(list)`` (a
        :class:`random.Random`, or the :mod:`random` module): the permutation is its shuffle
        of ``range(n)``.
    """

    key = "epoch-batches"

    def __init__(self, batch_size=None, forms=None, rng=None):
        super().__init__()
        self.batch_size = batch_size
        self.forms = None if forms is None else tuple(forms)
        self.rng = rng
        self.views = {}
        self._perm = {}

    @property
    def size(self):
        b = self.batch_size if self.batch_size is not None else self.model.batch_size
        return max(1, int(b))

    @property
    def steps_per_epoch(self):
        """:math:`\\max_f \\lceil n_f / b \\rceil` over the batched forms (at least 1)."""
        if not self.views:
            return 1
        return max(1, max(math.ceil(len(v.full) / self.size) for v in self.views.values()))

    def training_datasets(self, datasets):
        out = dict(datasets)
        self.views = {}
        for name, ds in datasets.items():
            if self.forms is not None and name not in self.forms:
                continue
            self.views[name] = out[name] = _BatchView(ds)
        return out

    def _permutation(self, view):
        n = len(view.full)
        if self.rng is None:
            return th.randperm(n, device=view.full.device)
        order = list(range(n))
        self.rng.shuffle(order)
        return th.as_tensor(order, dtype=th.long, device=view.full.device)

    def on_epoch_start(self, epoch, epochs):
        j = epoch % self.steps_per_epoch
        b = self.size
        for name, view in self.views.items():
            if j == 0:
                self._perm[name] = self._permutation(view)
            view.index = self._perm[name][j * b:(j + 1) * b]


class EdgeCorruption(TrainingRegime):
    """``num_negatives`` corruptions of every row, one entity replaced by a uniformly drawn
    one, all drawn at once from the global torch generator (so that ``torch.manual_seed``
    reproduces a run).

    :param num_negatives: Corruptions per row.
    :param columns: One column: that column is replaced. Two columns ``(a, b)``: a coin per
        corruption decides, ``a`` is replaced when it shows 0 and ``b`` when it shows 1
        ("either the source or the target node", Zhang et al. 2022, Section 4.4).
    :param num_entities: Size of the pool (entities ``0 ... n - 1``; default: the model's
        ``num_nodes``, else its named classes).
    :param forms: The forms that get corruptions (default: all; the others use the model's).
    :param draw_by: ``"row"``: the draws have shape ``(n, k)`` (the ``k`` corruptions of a row
        are neighbours in the stream); ``"block"``: shape ``(k * n,)`` (the corruptions of one
        block of the layout are). Only the random stream differs.
    :param coin_first: Draw the coins of two columns before the entities (default) or after.
    """

    key = "edge-corruption"

    def __init__(self, num_negatives=1, columns=(1,), num_entities=None, forms=None,
                 draw_by="row", coin_first=True):
        super().__init__()
        if int(num_negatives) < 1:
            raise ValueError("num_negatives must be at least 1.")
        columns = tuple(int(c) for c in columns)
        if len(columns) not in (1, 2):
            raise ValueError("columns must name one or two columns.")
        if draw_by not in ("row", "block"):
            raise ValueError("draw_by must be 'row' or 'block'.")
        self.num_negatives = int(num_negatives)
        self.columns = columns
        self.num_entities = num_entities
        self.forms = None if forms is None else tuple(forms)
        self.draw_by = draw_by
        self.coin_first = bool(coin_first)

    def pool_size(self):
        if self.num_entities is not None:
            return int(self.num_entities)
        nodes = getattr(self.model, "num_nodes", None)
        return int(nodes) if nodes is not None else len(self.model.named_class_index_dict)

    def _draw(self, high, n, device):
        k = self.num_negatives
        if self.draw_by == "block":
            return th.randint(0, high, (k * n,), device=device)
        return th.randint(0, high, (n, k), device=device).t().reshape(-1)

    def negatives(self, form, dataset):
        if self.forms is not None and form not in self.forms:
            return NotImplemented
        pos = dataset[:]
        n, k = len(pos), self.num_negatives
        if n == 0:
            return None
        neg = pos.repeat(k, 1)
        if len(self.columns) == 2:
            first, second = self.columns
            if self.coin_first:
                coin = self._draw(2, n, pos.device)
                entity = self._draw(self.pool_size(), n, pos.device)
            else:
                entity = self._draw(self.pool_size(), n, pos.device)
                coin = self._draw(2, n, pos.device)
            neg[:, first] = th.where(coin == 0, entity, neg[:, first])
            neg[:, second] = th.where(coin == 1, entity, neg[:, second])
        else:
            neg[:, self.columns[0]] = self._draw(self.pool_size(), n, pos.device)
        return neg


class KnownRowMask(TrainingRegime):
    """Mark the corruptions that are no negatives: a corruption that is a row of the training
    data of its form, or a reflexive pair (equal first and second column). The loss regimes
    of this module leave a marked corruption out of their means; the marks are in
    ``model.negative_masks[form]`` for the step (a boolean per corrupted row, ``False`` =
    leave out). The negatives themselves are not changed.

    :param forms: The forms to mark (default: all with corruptions).
    :param reflexive: Also mark corruptions whose first two columns are equal.
    :param drop: Remove the marked corruptions from the negatives instead of marking them (the
        losses that take a plain mean over the corruptions, such as E2R's; ``None`` if none
        is left).
    """

    key = "known-row-mask"

    def __init__(self, forms=None, reflexive=True, drop=False):
        super().__init__()
        self.forms = None if forms is None else tuple(forms)
        self.reflexive = bool(reflexive)
        self.drop = bool(drop)
        self._known = {}

    def training_datasets(self, datasets):
        self._known = {}
        return datasets

    def _codes(self, rows, width):
        code = th.zeros(len(rows), dtype=th.long, device=rows.device)
        for j in range(rows.shape[1]):
            code = code * width + rows[:, j]
        return code

    def filter_negatives(self, form, dataset, negatives):
        if negatives is None or (self.forms is not None and form not in self.forms):
            return negatives
        known = self.model.training_datasets[form][:].to(negatives.device)
        width = int(max(known.max().item(), negatives.max().item())) + 1
        mask = ~th.isin(self._codes(negatives, width), self._codes(known, width))
        if self.reflexive:
            mask = mask & (negatives[:, 0] != negatives[:, 1])
        if self.drop:
            kept = negatives[mask]
            return kept if len(kept) > 0 else None
        masks = getattr(self.model, _MASKS, None)
        if masks is None:
            masks = self.model.negative_masks = {}
        masks[form] = mask
        return negatives


class _GroupedLoss(TrainingRegime):
    """Base of the losses over a row and its corruptions: the costs of the corruptions are the
    costs of the rows read like positives (``(n, k)`` per row), minus the marked ones."""

    score_negatives_like_positives = True

    def __init__(self, forms=None):
        super().__init__()
        self.forms = None if forms is None else tuple(forms)

    def loss(self, form, pos_scores, neg_scores):
        if self.forms is not None and form not in self.forms:
            return NotImplemented
        pos = pos_scores.reshape(len(pos_scores), -1).mean(dim=1)
        if neg_scores is None:
            return pos.mean()
        n = len(pos)
        if len(neg_scores) % n:
            raise ValueError(f"{type(self).__name__} needs a whole number of corruptions per "
                             f"row ({len(neg_scores)} for {n} rows).")
        neg = grouped(neg_scores.reshape(len(neg_scores), -1).mean(dim=1), n)
        mask = getattr(self.model, _MASKS, {}).get(form)
        if mask is not None:
            mask = grouped(mask, n)
        return self.combine(pos, neg, mask)

    def combine(self, pos, neg, mask):
        raise NotImplementedError


class SelfAdversarialBCE(_GroupedLoss):
    """The binary cross entropy of binary-code boxes (Zhang et al. 2022, Eq. 10): the energy
    :math:`E^+` of a row plus the weighted :math:`-\\log(1 - e^{-E^-})` of its corruptions,
    the weights a softmax over the corruptions (RotatE, Sun et al. 2019) or all 1; the mean over
    the rows. Marked corruptions get weight 0. See
    :func:`~mowl.representation.library.hierarchy_objectives.bce_edge_loss`.

    :param self_adversarial: Softmax weights (the paper) instead of the plain sum.
    :param temperature: Temperature of the softmax.
    """

    key = "self-adversarial-bce"

    def __init__(self, self_adversarial=True, temperature=1.0, forms=None):
        super().__init__(forms)
        self.self_adversarial = bool(self_adversarial)
        self.temperature = temperature

    def combine(self, pos, neg, mask):
        from mowl.representation.library.hierarchy_objectives import bce_edge_loss
        return bce_edge_loss(pos, neg, mask=mask, self_adversarial=self.self_adversarial,
                             temperature=self.temperature).mean()


class GroupedMarginLoss(_GroupedLoss):
    """The loss of CatE (Zhapa-Camacho and Hoehndorf, Eq. 7): the mean cost of the rows plus a
    hinge of margin :math:`\\gamma` on the cost of their corruptions. With
    ``aggregation="per_negative"`` every corruption has a hinge of its own (the paper); with
    ``"mean"`` the costs of the corruptions of a row are averaged first (the reference code).
    Marked corruptions are left out of the means. See
    :func:`~mowl.representation.library.hierarchy_objectives.margin_loss`.

    :param margin: :math:`\\gamma` (default: the model's ``margin``).
    :param aggregation: ``"per_negative"`` or ``"mean"``.
    """

    key = "grouped-margin"

    def __init__(self, margin=None, aggregation="per_negative", forms=None):
        super().__init__(forms)
        if aggregation not in ("per_negative", "mean"):
            raise ValueError("aggregation must be 'per_negative' or 'mean'.")
        self.margin = margin
        self.aggregation = aggregation

    def combine(self, pos, neg, mask):
        from mowl.representation.library.hierarchy_objectives import margin_loss
        margin = self.margin if self.margin is not None else self.model.margin
        return margin_loss(pos, neg, margin, aggregation=self.aggregation, mask=mask)


class HeadRestrictedHinge(TrainingRegime):
    """The interaction term of the CatE code (``src/cat_ppi_evaluator.py``): on the rows of
    every step whose first entity is one of ``heads`` (the proteins), the mean cost of the
    rows plus a hinge of margin :math:`\\gamma` on one corruption whose second entity is drawn
    uniformly from ``candidates`` (the elements :math:`\\exists r.p`). Added to the loss of the
    step.

    :param heads, candidates: Index tensors, or a function ``model -> (heads, candidates)``
        (called when training starts).
    :param margin: :math:`\\gamma` (default: the model's ``margin``).
    :param form: The form whose rows the term reads.
    """

    key = "head-restricted-hinge"

    def __init__(self, heads=None, candidates=None, margin=None, form="edge"):
        super().__init__()
        self.heads, self.candidates = heads, candidates
        self.margin = margin
        self.form = form
        self._rows = None

    def _ids(self):
        if callable(self.heads):
            return self.heads(self.model)
        return self.heads, self.candidates

    def training_datasets(self, datasets):
        self._rows = None
        self._resolved = self._ids()
        return datasets

    def filter_negatives(self, form, dataset, negatives):
        if form == self.form:
            self._rows = dataset[:]
        return negatives

    def additional_loss(self, epoch):
        rows, self._rows = self._rows, None
        if rows is None:
            return None
        heads, candidates = self._resolved
        heads, candidates = heads.to(rows.device), candidates.to(rows.device)
        model = self.model
        pos = model.score_rows(self.form, rows)
        sel = th.isin(rows[:, 0], heads)
        if not sel.any():
            return pos.sum() * 0.0
        pick = candidates[th.randint(0, len(candidates), (int(sel.sum()),),
                                     device=candidates.device)]
        neg = model.score_rows(self.form, th.stack([rows[sel, 0], pick], dim=1))
        margin = self.margin if self.margin is not None else model.margin
        return pos[sel].mean() + th.relu(margin - neg).mean()


class ValidationPatience(TrainingRegime):
    """Model selection with patience: every ``every`` epochs (and after the last one) a
    quantity of the model is evaluated, the parameters are saved to ``model_filepath`` when it
    improves, training stops after ``patience`` evaluations without improvement, and the best
    parameters are loaded at the end. It replaces the validation of the loop. The CatE code
    validates this way (Zhapa-Camacho and Hoehndorf: mean reciprocal rank on the validation
    queries); it differs from :class:`~mowl.training.ValidationSelection` (best state kept in
    memory, evaluated at the start of epochs) and from :class:`~mowl.training.EarlyStopping`
    (which does not select).

    :param metric: ``f(model) -> float`` (``higher=True``: larger is better, the default).
    :param every: Epochs between evaluations; ``None``: the ``validate_every`` of ``train``
        (no evaluation without one).
    :param patience: Evaluations without improvement that end training (``None``: never).
    :param history: Name of the list on the model that receives ``(epoch, value)`` (default
        ``"validation_history"``).
    :param usable: ``f(model) -> bool``: whether there is anything to evaluate (default:
        the model has validation data).
    :param on_best: ``f(model, value, epoch)`` called when an evaluation improves (``epoch``
        counts from 1).
    :param save_final: Without anything to evaluate, save the final parameters to
        ``model_filepath`` (default); ``False``: nothing is saved then.
    """

    key = "validation-patience"
    replaces_validation = True

    def __init__(self, metric, every=None, patience=10, higher=True,
                 history="validation_history", usable=None, on_best=None, save_final=True):
        super().__init__()
        self.metric = metric
        self.every = every
        self.patience = patience
        self.higher = bool(higher)
        self.history_name = history
        self.usable = usable
        self.on_best = on_best
        self.save_final = save_final
        self.steps = 1
        self.best = None
        self._bad = 0
        self._stop = False
        self._active = False

    def training_datasets(self, datasets):
        from mowl.training.loop import steps_per_epoch
        self.steps = steps_per_epoch(self.model.training_regime)
        every = self.every if self.every is not None else getattr(
            self.model, "_train_validate_every", None)
        self._every = None if every is None else max(1, int(every))
        self._active = self._every is not None and (
            self.usable(self.model) if self.usable is not None
            else self.model.dataset.validation is not None)
        self.best, self._bad, self._stop = None, 0, False
        setattr(self.model, self.history_name, [])
        return datasets

    def _better(self, value):
        return self.best is None or (value > self.best if self.higher else value < self.best)

    def on_epoch_end(self, step, losses):
        if not self._active or (step + 1) % self.steps:
            return
        epoch = (step + 1) // self.steps
        total = getattr(self.model, "training_steps", 0) // self.steps
        if epoch % self._every and epoch != total:
            return
        value = float(self.metric(self.model))
        getattr(self.model, self.history_name).append((epoch, value))
        if self._better(value):
            self.best, self._bad = value, 0
            th.save(self.model.module.state_dict(), self.model.model_filepath)
            if self.on_best is not None:
                self.on_best(self.model, value, epoch)
        else:
            self._bad += 1
            if self.patience is not None and self._bad >= self.patience:
                self._stop = True

    def should_stop(self, step):
        return self._stop

    def on_train_end(self):
        if self._active and self.best is not None:
            self.model.module.load_state_dict(
                th.load(self.model.model_filepath, weights_only=True))
        elif not self._active and self.save_final:
            path = getattr(self.model, "_model_filepath", None)
            if path is not None:
                th.save(self.model.module.state_dict(), path)
