"""The training loop of a method whose step is one sampled objective over all its axioms.

Some methods do not score rows of axioms one form at a time: one step draws elements and
points for every axiom form at once and returns the terms of one objective (GroundedEL's
grounded objective, :class:`~mowl.representation.library.grounded_objective.GroundedObjective`).
:class:`SampledObjectiveTraining` trains such a model with the hooks of a
:class:`~mowl.training.TrainingRegime`, one optimisation step per epoch:

1. ``regime.on_epoch_start(epoch, epochs)``;
2. the terms ``model.compute_batch_losses(generator=...)`` (a dict with ``"total"``), plus
   ``regime.additional_loss(epoch)``; one step of ``regime.optimizer()`` (default Adam at the
   model's learning rate);
3. ``model.project_parameters()`` (a projection of the parameters onto their domain, if the
   model has one);
4. the record of the epoch (the terms as floats, ``model.epoch_record(rec)`` adds the
   model's own fields, the temperature of a :class:`~mowl.training.TemperatureAnnealing` in
   the regime as ``"tau"``), ``regime.on_epoch_end``, ``model.log_epoch(...)`` every
   ``log_every`` epochs, ``epoch_callback(epoch, model)``;
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


class SampledObjectiveTraining:
    """The loop of the module docstring.

    :param seed_offset: Offset of the seed of the sample generator from ``model.seed``.
    """

    def __init__(self, seed_offset=1337):
        self.seed_offset = int(seed_offset)

    def run(self, model, regime, epochs, validate_every=1, epoch_callback=None, log_every=0):
        """Train ``model`` for ``epochs`` steps; returns the list of epoch records (also
        ``model.loss_history``)."""
        model.loss_history = []
        if epochs == 0:
            return model.loss_history
        annealer = find_regime(regime, TemperatureAnnealing)
        optimizer = regime.optimizer()
        if optimizer is NotImplemented:
            optimizer = th.optim.Adam(model.module.parameters(), lr=model.learning_rate)
        model.optimizer = optimizer
        gen = th.Generator(device="cpu")
        gen.manual_seed(model.seed + self.seed_offset)
        model._train_generator = gen
        model._active_regime = regime
        best, saved_best = float("inf"), False
        try:
            for epoch in range(epochs):
                model.module.train()
                regime.on_epoch_start(epoch, epochs)
                # at full precision when a schedule set it (a module may keep it in float32)
                tau = annealer.history[-1] if annealer is not None else float(model.module.tau)
                terms = model.compute_batch_losses(generator=gen)
                total = terms["total"]
                extra = regime.additional_loss(epoch)
                if extra is not None:
                    total = total + extra
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
                project = getattr(model, "project_parameters", None)
                if project is not None:
                    project()
                rec = {k: float(v.detach().item()) for k, v in terms.items()}
                regime.on_epoch_end(epoch, rec)
                record = getattr(model, "epoch_record", None)
                if record is not None:
                    record(rec)
                rec["tau"] = float(tau)
                model.loss_history.append(rec)
                if log_every > 0 and ((epoch + 1) % log_every == 0 or epoch == 0
                                      or epoch + 1 == epochs):
                    log = getattr(model, "log_epoch", None)
                    if log is not None:
                        log(epoch, epochs, rec)
                if epoch_callback is not None:
                    epoch_callback(epoch, model)
                if (epoch + 1) % validate_every == 0 and _has_validation(model):
                    value = validation_loss(model)
                    # the module stays in evaluation mode until the next epoch starts, as
                    # the loop of GroundedEL always left it after a validation
                    model.module.eval()
                    if value < best:
                        best = value
                        if model.model_filepath is not None:
                            th.save(model.module.state_dict(), model.model_filepath)
                            saved_best = True
                if regime.should_stop(epoch):
                    break
        finally:
            model._train_generator = None
            model._active_regime = None
        regime.on_train_end()
        if model.model_filepath is not None and not saved_best:
            if validate_every <= epochs or not os.path.exists(model.model_filepath):
                th.save(model.module.state_dict(), model.model_filepath)
        return model.loss_history
