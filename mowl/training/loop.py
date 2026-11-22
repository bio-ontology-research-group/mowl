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

One *step* of the loop is one optimisation step. A batching regime may declare
``steps_per_epoch`` (:class:`~mowl.training.PerNormalFormMiniBatches`,
:class:`~mowl.training.EpochBatches`); :meth:`train` then runs ``epochs * steps_per_epoch``
steps and reports one mean loss per epoch, as the papers count. Regimes see steps (the
``epoch`` argument of their hooks is the step), as in the loop of
:class:`~mowl.base_models.EmbeddingELModel`.
"""

import logging

import torch as th

from mowl.training.regime import HOOKS, check_compatible, TrainingRegime

logger = logging.getLogger(__name__)

__all__ = ["Rows", "RegimeTrained", "steps_per_epoch"]


class Rows:
    """A tensor of rows as a training dataset (``len`` and ``[:]``)."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def steps_per_epoch(regime):
    """Steps per epoch that a regime (or one part of a composition) declares; 1 if none."""
    for part in getattr(regime, "regimes", (regime,)):
        steps = getattr(part, "steps_per_epoch", None)
        if isinstance(steps, int) and steps >= 1:
            return steps
    return 1


class RegimeTrained:
    """Mixin (before the model base class) that trains with the hooks of a
    :class:`~mowl.training.TrainingRegime`. See the module docstring for what the model
    provides. Models name the regimes that belong to the method by setting
    ``module.default_regimes`` (a tuple of functions that make one regime each) and calling
    :meth:`_attach_default_regimes`; a model without a regime gets them again when it trains.
    """

    #: The generic loop consults every hook.
    training_hooks = frozenset(HOOKS)

    #: ``validate_every`` of :meth:`train` when the call gives none (``None``: no validation).
    default_validate_every = None

    # ------------------------------------------------------------------ regime
    @property
    def training_regime(self):
        """The :class:`~mowl.training.TrainingRegime` that :meth:`train` uses: the regimes of
        the method (:attr:`default_regimes`) until another is assigned; assigning ``None``
        gives the regimes of the method again. See :mod:`mowl.training.loop` for the hooks it
        consults."""
        if getattr(self, "_training_regime", None) is None:
            self._attach_default_regimes()
        return getattr(self, "_training_regime", None)

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
    @property
    def training_datasets(self):
        """``{form: dataset}`` of the training data (:class:`Rows`)."""
        raise NotImplementedError(f"{type(self).__name__} must define training_datasets.")

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

    def on_epoch_start(self, step, steps):
        """Called at the start of every step before the regime's hook."""

    def additional_training_loss(self, step):
        return None

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

    def record_epoch(self, epoch, mean_loss):
        """Called with the mean loss of every epoch (a model keeps its history here)."""

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
    def batch_loss(self, batches, step=0):
        """The loss of the batches ``{form: rows}`` as the loop computes it (negatives,
        scores and combination of the regime, then the additional and regularisation terms).
        The model's hooks are used where the regime does not decide."""
        data = {form: Rows(rows) if isinstance(rows, th.Tensor) else rows
                for form, rows in batches.items()}
        return self._step_loss(data, step)

    def _step_loss(self, datasets, step):
        regime = self.training_regime
        self.negative_masks = {}
        loss = th.tensor(0.0, device=self.device)
        for form, dataset in datasets.items():
            if len(dataset) == 0:
                continue
            rows = dataset[:]
            pos_scores = self.score_rows(form, rows)
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
        extra = self.additional_training_loss(step)
        if extra is not None:
            loss = loss + extra
        if regime is not None:
            extra = regime.additional_loss(step)
            if extra is not None:
                loss = loss + extra
        return loss + self.step_regularization_loss()

    # ------------------------------------------------------------------ train
    def train(self, epochs=100, validate_every=None, verbose=False, epoch_callback=None,
              validate=None):
        """Train with the training regime of the method (or the one assigned).

        :param epochs: Number of epochs; an epoch has ``steps_per_epoch`` optimisation steps
            of the batching regime (one without).
        :param validate_every: Epochs between the evaluations of a model selection regime
            (``None``: the default of the model, usually no validation).
        :param verbose: Print the mean loss of every epoch.
        :param epoch_callback: ``callable(epoch, model)`` called after every epoch.
        :param validate: ``False`` switches validation off; ``True`` asks for the model's
            default interval (or every epoch).
        :return: The mean loss of every epoch.
        :rtype: list of float
        """
        regime = self.training_regime
        if validate is False:
            validate_every = None
        elif validate_every is None:
            validate_every = self.default_validate_every if validate is None \
                else (self.default_validate_every or 1)
        self._train_validate_every = validate_every
        datasets = self.training_datasets
        if regime is not None:
            datasets = regime.training_datasets(datasets)
        per_epoch = steps_per_epoch(regime)
        total = epochs * per_epoch
        self.training_steps = total
        optimizer = NotImplemented
        if regime is not None:
            optimizer = regime.optimizer()
        if optimizer is NotImplemented:
            optimizer = self.get_optimizer()
        history, running, counted = [], 0.0, 0
        for step in range(total):
            self.module.train()
            self.on_epoch_start(step, total)
            if regime is not None:
                regime.on_epoch_start(step, total)
            loss = self._step_loss(datasets, step)
            value = 0.0
            if regime is not None and regime.skips_empty_steps and not loss.requires_grad:
                pass
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.after_optimizer_step()
                value = loss.detach().item()
            running += value
            counted += 1
            if regime is not None:
                regime.on_epoch_end(step, {"total": value})
            if (step + 1) % per_epoch == 0:
                epoch = step // per_epoch
                history.append(running / max(counted, 1))
                running, counted = 0.0, 0
                self.record_epoch(epoch, history[-1])
                if verbose:
                    print(f"Epoch {epoch + 1}: loss {history[-1]:.6f}")
                if epoch_callback is not None:
                    epoch_callback(epoch, self)
            if regime is not None and regime.should_stop(step):
                break
        if regime is not None:
            regime.on_train_end()
        if not (regime is not None and regime.replaces_validation):
            self.save_state()
        self.module.eval()
        return history
