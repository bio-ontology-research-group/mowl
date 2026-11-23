"""Training a method whose step is one sampled objective over all its axioms.

Some methods do not score rows of axioms one form at a time: one step draws elements and
points for every axiom form at once and returns the terms of one objective (GroundedEL's
grounded objective, :class:`~mowl.representation.library.grounded_objective.GroundedObjective`).
:class:`SampledObjectiveTraining` is the training regime of such a method: it computes the
loss of a step (:meth:`~mowl.training.TrainingRegime.step_loss`) and the method trains in the
shared loop of :mod:`mowl.training.loop`, with one optimisation step per epoch:

1. ``regime.on_epoch_start(epoch, epochs)`` of the regimes in front of it;
2. the terms ``model.compute_batch_losses(generator=...)`` (a dict with ``"total"``); the
   loop adds ``regime.additional_loss(epoch)`` and takes one step of ``regime.optimizer()``
   (default Adam at the model's learning rate);
3. ``model.project_parameters()`` (a projection of the parameters onto their domain, if the
   model has one), then the ``after_step`` of the other regimes;
4. the record of the epoch (the terms as floats, ``model.epoch_record(rec)`` adds the
   model's own fields, the temperature of a :class:`~mowl.training.TemperatureAnnealing` in
   the regime as ``"tau"``) after the ``on_epoch_end`` of the other regimes,
   ``model.log_epoch(...)`` every ``log_every`` epochs, ``epoch_callback(epoch, model)``;
5. every ``validate_every`` epochs the validation loss of ``model.eval_gci_name``
   (:func:`~mowl.training.schedules.validation_loss`) and a checkpoint of the best; the
   module is then left in evaluation mode (``module.train()`` is called at the start of
   every epoch);
6. ``regime.should_stop(epoch)``; at the end ``regime.on_train_end()`` and a final
   checkpoint when none was written.

The generator of the samples is seeded with ``model.seed + seed_offset`` and is
``model._train_generator`` during the run (regimes that draw, such as re-seeding, use it).
"""
from __future__ import annotations

import os

import torch as th

from mowl.training.loop import TrainingLoop
from mowl.training.regime import ComposedRegime, TrainingRegime
from mowl.training.schedules import TemperatureAnnealing, validation_loss

__all__ = ["SampledObjectiveTraining", "find_regime"]


def find_regime(regime, kind):
    """The first part of ``regime`` (itself, or a part of a composition, depth first) that is
    a ``kind``; ``None`` if there is none."""
    if isinstance(regime, kind):
        return regime
    for sub in getattr(regime, "regimes", ()):
        found = find_regime(sub, kind)
        if found is not None:
            return found
    return None


def _has_validation(model):
    dataset, form = model.dataset, model.eval_gci_name
    return (dataset is not None and dataset.validation is not None and form is not None
            and form in model.validation_datasets and len(model.validation_datasets[form]) > 0)


class _SampledRun(TrainingLoop):
    """The shared loop configured for a model trained by :class:`SampledObjectiveTraining`:
    an epoch is one step, no progress bar, validation and the checkpoints of the regime."""

    loop_style = "epochs"
    progress_bar = False

    def __init__(self, model, regime, objective):
        self.model, self.objective = model, objective
        self.module = model.module
        self.device = getattr(model, "device", "cpu")
        self.training_regime = regime

    @property
    def training_datasets(self):
        return getattr(self.model, "training_datasets", None) or {}

    def get_optimizer(self):
        return th.optim.Adam(self.model.module.parameters(), lr=self.model.learning_rate)

    def start_training(self, optimizer):
        # regimes that reset the optimiser (re-seeding) find it on the model
        self.model.optimizer = optimizer

    def step_regularization_loss(self):
        return 0

    def after_optimizer_step(self):
        self.training_regime.after_step()

    def validate_epoch(self, epoch, loss):
        self.objective.validate(self.model)

    def finish_training(self):
        self.objective.final_checkpoint(self.model)


class SampledObjectiveTraining(TrainingRegime):
    """The training regime of a method whose step is one sampled objective (see the module
    docstring); :meth:`run` trains a model with it in the shared loop.

    The model provides ``module``, ``seed``, ``learning_rate``, ``model_filepath``,
    ``dataset``, ``eval_gci_name`` and ``compute_batch_losses(generator=...)``, optionally
    ``project_parameters()``, ``epoch_record(rec)`` and ``log_epoch(epoch, epochs, rec)``.

    :param seed_offset: Offset of the seed of the sample generator from ``model.seed``.
    """

    key = "sampled-objective"

    def __init__(self, seed_offset=1337):
        super().__init__()
        self.seed_offset = int(seed_offset)
        self._generator, self._annealer, self._tau = None, None, None
        self._epochs, self._log_every, self._validate_every = 0, 0, 1
        self._best, self._saved_best = float("inf"), False

    # -- the hooks of the shared loop -------------------------------------------------------
    def step_loss(self, step):
        model = self.model
        # at full precision when a schedule set it (a module may keep it in float32)
        self._tau = self._annealer.history[-1] if self._annealer is not None \
            else float(model.module.tau)
        return model.compute_batch_losses(generator=self._generator)

    def after_step(self):
        project = getattr(self.model, "project_parameters", None)
        if project is not None:
            project()

    def on_epoch_end(self, epoch, losses):
        model, rec = self.model, losses
        record = getattr(model, "epoch_record", None)
        if record is not None:
            record(rec)
        rec["tau"] = float(self._tau)
        model.loss_history.append(rec)
        epochs, every = self._epochs, self._log_every
        if every > 0 and ((epoch + 1) % every == 0 or epoch == 0 or epoch + 1 == epochs):
            log = getattr(model, "log_epoch", None)
            if log is not None:
                log(epoch, epochs, rec)

    # -- validation and checkpoints ---------------------------------------------------------
    def validate(self, model):
        """The validation loss of ``model.eval_gci_name`` (if there are validation axioms)
        and a checkpoint of the best parameters; the module is left in evaluation mode."""
        if not _has_validation(model):
            return
        value = validation_loss(model)
        # the module stays in evaluation mode until the next epoch starts, as the loop of
        # GroundedEL always left it after a validation
        model.module.eval()
        if value < self._best:
            self._best = value
            if model.model_filepath is not None:
                th.save(model.module.state_dict(), model.model_filepath)
                self._saved_best = True

    def final_checkpoint(self, model):
        """At the end: save the parameters when no validation checkpoint was written."""
        if model.model_filepath is not None and not self._saved_best:
            if self._validate_every <= self._epochs or not os.path.exists(model.model_filepath):
                th.save(model.module.state_dict(), model.model_filepath)

    # -- a run ------------------------------------------------------------------------------
    def run(self, model, regime, epochs, validate_every=1, epoch_callback=None, log_every=0):
        """Train ``model`` for ``epochs`` steps in the shared loop
        (:meth:`~mowl.training.loop.TrainingLoop.run_training`) with ``regime`` in front of
        this one; returns the list of epoch records (also ``model.loss_history``)."""
        model.loss_history = []
        if epochs == 0:
            return model.loss_history
        self.attach(model)
        self._annealer = find_regime(regime, TemperatureAnnealing)
        self._epochs, self._log_every = int(epochs), int(log_every)
        self._validate_every = int(validate_every)
        self._best, self._saved_best = float("inf"), False
        gen = th.Generator(device="cpu")
        gen.manual_seed(model.seed + self.seed_offset)
        self._generator = gen
        composed = ComposedRegime(regime, self)
        composed.model = model
        model._train_generator = gen
        model._active_regime = regime
        callback = None if epoch_callback is None else \
            (lambda epoch, _: epoch_callback(epoch, model))
        try:
            _SampledRun(model, composed, self).run_training(
                epochs, validate_every=validate_every, epoch_callback=callback)
        finally:
            model._train_generator = None
            model._active_regime = None
            self._generator = None
        return model.loss_history
