"""The generic training loop for models whose data are not the EL normal forms.

:class:`~mowl.base_models.EmbeddingELModel` has a loop that consults every hook of a
:class:`~mowl.training.TrainingRegime`, but it needs the normalised EL axioms of an ontology.
Several methods train on other data: the edges of a projected graph (binary-code boxes), the
morphisms of a lattice (CatE), the typed rows of E2R, the verbalised triplets of a text
encoder (HiT, OnT). :class:`RegimeTrained` gives them the same loop and the same hooks, so
that batching, negatives, the combination of losses, schedules, model selection and
projections are regimes (:mod:`mowl.training`) that any of these models, and the EL models,
can use. The model only says *what* its data and its score are:

* :attr:`training_datasets`: ``{form: dataset}`` where a dataset has ``len`` and ``[:]``
  returning a tensor of rows (:class:`Rows`);
* :meth:`score_rows`: the per-row cost of rows of a form (default: ``module(rows, form)``,
  and ``module(rows, form, neg=True)`` for corruptions);
* :meth:`generate_negatives`: the model's own corruptions of a form (default: none; a
  regime of :mod:`mowl.training.negatives` or :mod:`mowl.training.graph` supplies them);
* :meth:`get_optimizer`.

A regime may compute the loss of a step itself (:meth:`~mowl.training.TrainingRegime.step_loss`):
a method whose step is one objective over all its forms at once (a sampled objective), or
one that accumulates the gradient of a step chunk by chunk to bound its memory
(:class:`~mowl.training.Accumulated`, :func:`accumulate_backward`; the canonical mixture,
:class:`MicroBatchAccumulation` for any model) and skips steps with nothing to supervise.
The loop still takes the optimiser step, calls the hooks and records the losses, so every
optimisation step of mOWL is taken here (the inner solvers of a constructed or solved method
-- a gradient fit of a solver, the fit of an evaluation head -- are not training loops).

One *step* of the loop is one optimisation step. A batching regime may declare
``steps_per_epoch`` (:class:`~mowl.training.PerNormalFormMiniBatches`,
:class:`~mowl.training.EpochBatches`); :meth:`train` then runs ``epochs * steps_per_epoch``
steps and reports one mean loss per epoch, as the papers count. Regimes see steps (the
``epoch`` argument of their hooks is the step), as in the loop of
:class:`~mowl.base_models.EmbeddingELModel`.
"""

import logging

import torch as th
from tqdm import trange

from mowl.training.regime import HOOKS, Accumulated, check_compatible, TrainingRegime

logger = logging.getLogger(__name__)

__all__ = ["Rows", "TrainingLoop", "RegimeTrained", "steps_per_epoch", "accumulate_backward",
           "MicroBatchAccumulation"]

_UNSET = object()


class Rows:
    """A tensor of rows as a training dataset (``len`` and ``[:]``)."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accumulate_backward(loss, weight=1.0):
    """Back-propagate ``weight * loss`` now (if it needs a gradient) and return its value as a
    float: one chunk of a step whose gradient a regime accumulates
    (:meth:`~mowl.training.TrainingRegime.step_loss` returning
    :class:`~mowl.training.Accumulated`). With ``weight = rows of the chunk / rows of the
    step`` and ``loss`` the mean over the chunk, the accumulated gradient is that of the mean
    over the step, while only one chunk is held in memory."""
    term = loss * weight if weight != 1.0 else loss
    if getattr(term, "requires_grad", False):
        term.backward()
    return float(term.detach()) if hasattr(term, "detach") else float(term)


def _as_float(value):
    return float(value.detach().item()) if hasattr(value, "detach") else float(value)


def steps_per_epoch(regime):
    """Steps per epoch that a regime (or one part of a composition) declares; 1 if none."""
    for part in getattr(regime, "regimes", (regime,)):
        steps = getattr(part, "steps_per_epoch", None)
        if isinstance(steps, int) and steps >= 1:
            return steps
    return 1


class TrainingLoop:
    """The one training loop of mOWL: what :class:`RegimeTrained` models and
    :class:`~mowl.base_models.EmbeddingELModel` share. :meth:`run_training` consults every hook
    of the :class:`~mowl.training.TrainingRegime` and of the model (negatives, scores, the
    combination of losses, schedules, projections, selection).

    A model provides ``module``, ``training_datasets``, ``training_regime``, ``device``,
    ``score_rows``, ``generate_negatives``, ``combine_scores``, ``get_optimizer``,
    ``get_regularization_loss`` and the hooks below (defaults here do nothing).

    :attr:`loop_style` names how an epoch is counted:

    * ``"steps"`` (default): an epoch is ``steps_per_epoch`` optimisation steps of the
      batching regime and a mean loss is recorded per epoch; the regimes of the method are
      attached if none is; validation is the business of a selection regime; the parameters
      are saved at the end and the module is left in evaluation mode;
    * ``"epochs"`` (:class:`~mowl.base_models.EmbeddingELModel`): an epoch is one step, there
      is a progress bar, the model validates itself every ``validate_every`` epochs
      (:meth:`validate_epoch`), nothing is saved at the end and the module stays in training
      mode.
    """

    loop_style = "steps"

    #: ``True``: the epoch hooks of a regime (``on_epoch_start``, ``additional_loss``,
    #: ``on_epoch_end``, ``should_stop``) are called once per *epoch* with the epoch index,
    #: although every step is one optimisation step (the query models: an epoch is a number of
    #: updates, and a learning-rate schedule counts epochs). ``False``: they see steps.
    epoch_hooks = False

    def epoch_steps(self, regime):
        """Optimisation steps of one epoch: with ``loop_style = "steps"`` what the batching
        regime declares (one without); with ``"epochs"`` one."""
        return steps_per_epoch(regime) if self.loop_style == "steps" else 1

    #: ``validate_every`` of :meth:`run_training` when the call gives none (``None``: no
    #: validation).
    default_validate_every = None

    #: How the losses of the steps of an epoch make the loss of the epoch: ``"mean"`` (over
    #: the steps taken) or ``"sum"``.
    epoch_loss = "mean"

    #: ``loop_style = "steps"``: put the module in evaluation mode after training.
    eval_after_training = True

    #: Show a progress bar over the steps (``None``: with ``loop_style = "epochs"``).
    progress_bar = None

    def start_training(self, optimizer):
        """Called once with the optimiser of the run, before the first step (a model that
        lets regimes reset the optimiser keeps it here)."""

    def on_epoch_start(self, step, steps):
        """Called at the start of every step before the regime's hook."""

    def additional_training_loss(self, step):
        return None

    def record_epoch(self, epoch, mean_loss):
        """Called with the mean loss of every epoch before the regime's ``on_epoch_end`` of
        its last step (a model keeps its history here)."""

    def finish_training(self):
        """Called once after the regime's ``on_train_end`` (a model finalises what the
        regimes selected)."""

    def validate_epoch(self, epoch, loss):
        """Validation of the model itself after an epoch (``loop_style = "epochs"``)."""

    def log_training_datasets(self, datasets):
        """Called once with the datasets that the run trains on."""

    #: Save the parameters after training even when no ``model_filepath`` was given (a
    #: temporary file is made).
    save_always = False

    def save_state(self):
        """Save the parameters to ``model_filepath`` (after training, unless a regime selects
        the checkpoint)."""
        path = self.model_filepath if self.save_always else getattr(self, "_model_filepath",
                                                                    None)
        if path is not None:
            th.save(self.module.state_dict(), path)

    # -------------------------------------------------------------- one step
    def batch_loss(self, batches, step=0, negatives=True):
        """The loss of the batches ``{form: rows}`` as the loop computes it (negatives,
        scores and combination of the regime, then the additional and regularisation terms).
        The model's hooks are used where the regime does not decide.

        :param negatives: ``False``: no corruptions."""
        data = {form: Rows(rows) if isinstance(rows, th.Tensor) else rows
                for form, rows in batches.items()}
        return self._step_loss(data, step, negatives)

    def _step_loss(self, datasets, step, negatives=True):
        loss = self.forms_loss(datasets, step, negatives)
        loss = self._add_extra_terms(loss, step)
        return loss + self.step_regularization_loss()

    def forms_loss(self, datasets, step=0, negatives=True):
        """The loss of the rows of every form of ``datasets`` (``{form: dataset}``):
        negatives, scores and the combination of the regime (the model's hooks where the
        regime does not decide), without the additional and regularisation terms of a
        step."""
        regime = self.training_regime
        self.negative_masks = {}
        loss = th.tensor(0.0, device=self.device)
        for form, dataset in datasets.items():
            if len(dataset) == 0:
                continue
            rows = dataset[:]
            pos_scores = self.score_rows(form, rows)
            neg_data = None
            if negatives:
                neg_data = NotImplemented
                if regime is not None:
                    neg_data = regime.negatives(form, dataset)
                if neg_data is NotImplemented:
                    neg_data = self.generate_negatives(form, dataset)
                if regime is not None:
                    neg_data = regime.filter_negatives(form, dataset, neg_data)
            neg_scores = None
            if neg_data is not None:
                like_pos = regime is not None and regime.scores_negatives_like_positives(form)
                neg_scores = self.score_rows(form, neg_data, neg=not like_pos)
            form_loss = NotImplemented
            if regime is not None:
                form_loss = regime.loss(form, pos_scores, neg_scores)
            if form_loss is NotImplemented:
                form_loss = self.combine_scores(form, pos_scores, neg_scores)
            loss = loss + form_loss
        return loss

    def _extra_terms(self, step):
        """The additional loss terms of a step: the model's, then the regime's."""
        out = [self.additional_training_loss(step)]
        regime = self.training_regime
        if regime is not None:
            out.append(regime.additional_loss(step))
        return [extra for extra in out if extra is not None]

    def _add_extra_terms(self, loss, step):
        for extra in self._extra_terms(step):
            loss = loss + extra
        return loss

    # ------------------------------------------------------------------ train
    def run_training(self, epochs=100, validate_every=None, verbose=False, epoch_callback=None,
                     validate=None):
        """Train with the training regime of the model (or the one assigned).

        :param epochs: Number of epochs; with ``loop_style = "steps"`` an epoch has
            ``steps_per_epoch`` optimisation steps of the batching regime (one without).
        :param validate_every: Epochs between the evaluations of a model selection regime
            (``None``: the default of the model, usually no validation).
        :param verbose: Print the mean loss of every epoch.
        :param epoch_callback: ``callable(epoch, model)`` called after every epoch.
        :param validate: ``False`` switches validation off; ``True`` asks for the model's
            default interval (or every epoch).
        :return: The mean loss of every epoch (also ``model.training_losses``).
        :rtype: list of float
        """
        by_steps = self.loop_style == "steps"
        if by_steps and self.training_regime is None and \
                hasattr(self, "_attach_default_regimes"):
            self._attach_default_regimes()
        regime = self.training_regime
        if by_steps:
            if validate is False:
                validate_every = None
            elif validate_every is None:
                validate_every = self.default_validate_every if validate is None \
                    else (self.default_validate_every or 1)
        self._train_validate_every = validate_every
        history, running, counted = [], 0.0, 0
        if not by_steps:
            # a regime may keep the losses itself (ShuffledSplits), after this
            self.training_losses = history
        datasets = self.training_datasets
        if regime is not None:
            datasets = regime.training_datasets(datasets)
        self.log_training_datasets(datasets)
        per_epoch = self.epoch_steps(regime)
        total = epochs * per_epoch
        self.training_steps = total
        optimizer = NotImplemented
        if regime is not None:
            optimizer = regime.optimizer()
        if optimizer is NotImplemented:
            optimizer = self.get_optimizer()
        #: The optimiser of the run (regimes that reset its state find it here).
        self.training_optimizer = optimizer
        self.start_training(optimizer)
        if by_steps:
            self.training_losses = history
        # a regime that computes the loss of a step (a sampled objective, accumulated chunks)
        own_steps = regime is not None and "step_loss" in regime.overridden_hooks()
        progress = (not by_steps) if self.progress_bar is None else self.progress_bar
        for step in (trange(total) if progress else range(total)):
            self.module.train()
            self.on_epoch_start(step, total)
            if regime is not None and (not self.epoch_hooks or step % per_epoch == 0):
                if self.epoch_hooks:
                    regime.on_epoch_start(step // per_epoch, epochs)
                else:
                    regime.on_epoch_start(step, total)
            hook_step = step // per_epoch if self.epoch_hooks else step
            value, terms, taken = self._one_step(optimizer, datasets, hook_step, regime,
                                                 own_steps)
            if taken:
                running += value
                counted += 1
            ends_epoch = (step + 1) % per_epoch == 0
            if ends_epoch:
                epoch = step // per_epoch
                history.append(running if self.epoch_loss == "sum"
                               else running / max(counted, 1))
                running, counted = 0.0, 0
                self.record_epoch(epoch, history[-1])
            if regime is not None and not self.epoch_hooks:
                regime.on_epoch_end(step, terms if terms is not None else {"total": value})
            elif regime is not None and ends_epoch:
                regime.on_epoch_end(epoch, {"total": history[-1]})
            if ends_epoch:
                if verbose:
                    print(f"Epoch {epoch + 1}: loss {history[-1]:.6f}")
                if epoch_callback is not None:
                    epoch_callback(epoch, self)
                if not by_steps and validate_every and (epoch + 1) % validate_every == 0 \
                        and not (regime is not None and regime.replaces_validation):
                    self.validate_epoch(epoch, history[-1])
            if regime is not None and (
                    regime.should_stop(epoch) if self.epoch_hooks and ends_epoch
                    else (not self.epoch_hooks and regime.should_stop(step))):
                break
        if regime is not None:
            regime.on_train_end()
        self.finish_training()
        if by_steps:
            if not (regime is not None and regime.replaces_validation):
                self.save_state()
            if self.eval_after_training:
                self.module.eval()
        return history

    def _one_step(self, optimizer, datasets, step, regime, own_steps):
        """One optimisation step; returns ``(loss value, terms or None, taken)``.

        The loss is the regime's (:meth:`~mowl.training.TrainingRegime.step_loss`) when it
        computes one, else the loop's own (:meth:`_step_loss`). ``taken`` is ``False`` when
        no optimiser step was taken (nothing to supervise)."""
        outcome = NotImplemented
        if own_steps:
            optimizer.zero_grad()
            outcome = regime.step_loss(step)
        if outcome is NotImplemented:
            loss = self._step_loss(datasets, step)
            if regime is not None and regime.skips_empty_steps and not loss.requires_grad:
                return 0.0, None, True
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.after_optimizer_step()
            return loss.detach().item(), None, True
        if outcome is None:
            return 0.0, None, False
        if isinstance(outcome, Accumulated):
            value = outcome.value
            for extra in self._extra_terms(step):
                value += accumulate_backward(extra)
            optimizer.step()
            self.after_optimizer_step()
            terms = None
            if outcome.terms is not None:
                terms = dict(outcome.terms)
                terms.setdefault("total", value)
            return value, terms, True
        terms = None
        if isinstance(outcome, th.Tensor):
            loss = outcome
        else:
            loss = outcome["total"]
            terms = {k: _as_float(v) for k, v in outcome.items()}
        loss = self._add_extra_terms(loss, step)
        loss.backward()
        optimizer.step()
        self.after_optimizer_step()
        return loss.detach().item(), terms, True


class MicroBatchAccumulation(TrainingRegime):
    """Gradient accumulation over micro-batches: the step of the loop computed chunk by
    chunk, with the memory of one chunk.

    Every form of the step is split into chunks of at most ``chunk_size`` rows; the loss of
    a chunk (negatives, scores and combination of the loop, :meth:`TrainingLoop.forms_loss`)
    is back-propagated at once with the weight ``rows of the chunk / rows of the form``, then
    the regularisation term of the step; the loop takes one optimiser step
    (:meth:`~mowl.training.TrainingRegime.step_loss`, :class:`~mowl.training.Accumulated`).
    For a loss that is a mean over the rows of a form (and negatives that do not depend on
    the other rows) the gradient is the gradient of the whole step; otherwise it is the usual
    micro-batch approximation. A step whose forms are all empty is not taken.

    Place it *last* in a composition (after the regimes that change or batch the data), so
    that it reads the data of the step.

    :param chunk_size: Rows per chunk.
    :param negatives: Score corruptions (default ``True``).
    """

    key = "micro-batch-accumulation"

    def __init__(self, chunk_size=1024, negatives=True):
        super().__init__()
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1.")
        self.chunk_size = int(chunk_size)
        self.negatives_on = bool(negatives)
        self.datasets = {}

    def training_datasets(self, datasets):
        self.datasets = datasets
        return datasets

    def step_loss(self, step):
        model = self.model
        value, taken = 0.0, False
        for form, dataset in self.datasets.items():
            n = len(dataset)
            if n == 0:
                continue
            rows = dataset[:]
            for start in range(0, n, self.chunk_size):
                chunk = rows[start:start + self.chunk_size]
                loss = model.forms_loss({form: Rows(chunk)}, step, self.negatives_on)
                value += accumulate_backward(loss, float(len(chunk)) / float(n))
                taken = True
        if not taken:
            return None
        value += accumulate_backward(model.step_regularization_loss())
        return Accumulated(value)


class RegimeTrained(TrainingLoop):
    """Mixin (before the model base class) that trains with the hooks of a
    :class:`~mowl.training.TrainingRegime`. See the module docstring for what the model
    provides. Models name the regimes that belong to the method by setting
    ``module.default_regimes`` (a tuple of functions that make one regime each) and calling
    :meth:`_attach_default_regimes`; a model without a regime gets them again when it trains.
    """

    #: The generic loop consults every hook.
    training_hooks = frozenset(HOOKS)

    #: Counted by steps, whatever the other base classes of the model say.
    loop_style = "steps"

    # ------------------------------------------------------------------ regime
    @property
    def training_regime(self):
        """The :class:`~mowl.training.TrainingRegime` that :meth:`train` uses: the regimes of
        the method (:attr:`default_regimes`) until another is assigned; with ``None`` assigned
        :meth:`train` takes the regimes of the method again. See :mod:`mowl.training.loop`
        for the hooks it consults."""
        if self.__dict__.get("_training_regime", _UNSET) is _UNSET:
            self._training_regime = None
            self._attach_default_regimes()
        return self._training_regime

    @training_regime.setter
    def training_regime(self, regime):
        if regime is not None:
            if not isinstance(regime, TrainingRegime):
                raise TypeError("training_regime must be a mowl.training.TrainingRegime "
                                "or None.")
            check_compatible(regime, self)
            regime.attach(self)
        self._training_regime = regime

    #: The regimes that make the training of the method: a tuple of functions that make one
    #: regime each (or ``None`` when the option that asks for it is off), read from the model's
    #: options. A model without it takes ``module.default_regimes`` (what a named method of
    #: :mod:`mowl.representation.library` declares).
    default_regimes = ()

    def _attach_default_regimes(self):
        """Attach the regimes of the method unless a regime is attached already."""
        if getattr(self, "_training_regime", None) is not None:
            return
        factories = self.default_regimes or getattr(self.module, "default_regimes", ())
        if not factories:
            return
        from mowl.training.regime import ComposedRegime
        regimes = [r for r in (make() for make in factories) if r is not None]
        if not regimes:
            return
        self.training_regime = regimes[0] if len(regimes) == 1 else ComposedRegime(*regimes)

    # ------------------------------------------------------------------- data
    def score_rows(self, form, rows, neg=False):
        """The per-row cost of ``rows`` of ``form``; ``neg=True`` for corruptions scored with
        the module's negative cost."""
        return self.module(rows, form, neg=True) if neg else self.module(rows, form)

    def generate_negatives(self, form, dataset):
        """The model's own corruptions of ``dataset`` (rows, ``None``: none). A regime that
        gives negatives takes precedence."""
        return None

    def combine_scores(self, form, pos_scores, neg_scores=None):
        """The loss of a form when no regime decides it: mean cost of the positives plus mean
        cost of the negatives."""
        loss = th.mean(pos_scores)
        if neg_scores is not None:
            loss = loss + th.mean(neg_scores)
        return loss

    # ------------------------------------------------------------- loop hooks
    def get_optimizer(self):
        return th.optim.Adam(self.module.parameters(), lr=self.learning_rate)

    def get_regularization_loss(self):
        module = self.module
        return module.regularization_loss() if hasattr(module, "regularization_loss") else 0

    def step_regularization_loss(self):
        """The regularisation term of a step: the module's, unless the regime replaces or
        extends it (:class:`~mowl.training.Regularization`)."""
        reg = self.get_regularization_loss()
        regime = self.training_regime
        if regime is not None:
            out = regime.regularization(reg)
            if out is not NotImplemented:
                reg = out
        return reg

    def after_optimizer_step(self):
        """Called after every optimisation step; lets the regime project the parameters."""
        regime = self.training_regime
        if regime is not None:
            regime.after_step()

    def train(self, epochs=100, validate_every=None, verbose=False, epoch_callback=None,
              validate=None):
        """Train with the training regime of the method (or the one assigned): the loop of
        :meth:`TrainingLoop.run_training`."""
        return self.run_training(epochs, validate_every, verbose, epoch_callback, validate)
