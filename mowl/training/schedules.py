"""Learning-rate schedules as training regimes.

The generic EL training loop takes one optimisation step per epoch. A schedule regime
builds the optimiser (the model's own, from
:meth:`~mowl.base_models.elmodel.EmbeddingELModel.get_optimizer`, or one returned by
``base_optimizer``) and changes its learning rate at the start of every epoch, i.e. after
the step of the previous epoch -- which is where ``scheduler.step()`` sits in the
reference loops of the papers. A schedule regime must be the regime that provides the
optimiser: in a :class:`~mowl.training.ComposedRegime`, no earlier regime may return an
optimiser.

* :class:`StepDecay`: multiply the learning rate by ``gamma`` at given epochs --
  Box\\ :sup:`2`\\ EL (Appendix H: "decay the learning rate with a factor of 0.1 after 2000
  training epochs"), ELK-Em (Section 5.1: "decayed by 0.1 after 5,000 epochs").
* :class:`ConstantThenCosine`: constant for a fraction of training, then cosine decay to
  a floor -- ELK-Em (Section 5.2).
* :class:`LinearDecay`: linear decay to a floor -- GroundedEL (``lr_schedule="linear"``).
* :class:`TemperatureAnnealing`: not a learning-rate schedule but the same kind of
  per-epoch schedule, for the temperature of soft memberships -- GroundedEL, MultiBoxEL.
* :class:`CyclicLearningRate`: triangular cyclical learning rate (Smith 2017) -- CatE
  (Section 4.1).
* :class:`ReduceOnPlateau`: divide the learning rate when a validation quantity stops
  improving -- DELE (Section 6.3) and its NeSy 2024 predecessor (Section 4.4).
"""

import math

import torch as th

from mowl.training.regime import TrainingRegime


class _Schedule(TrainingRegime):
    def __init__(self, base_optimizer=None):
        super().__init__()
        self.base_optimizer = base_optimizer
        self.opt = None
        self.history = []

    def optimizer(self):
        if self.base_optimizer is not None:
            self.opt = self.base_optimizer(self.model)
        else:
            self.opt = self.model.get_optimizer()
        self.setup(self.opt)
        return self.opt

    def setup(self, opt):
        """Called once with the optimiser."""

    def on_epoch_start(self, epoch, epochs):
        if self.opt is None:
            raise RuntimeError(
                f"{type(self).__name__} has no optimiser: another regime provided it. A "
                f"schedule must be the regime whose optimizer() hook is used.")
        self.update(epoch, epochs)
        self.history.append(self.opt.param_groups[0]["lr"])

    def update(self, epoch, epochs):
        raise NotImplementedError


class StepDecay(_Schedule):
    """Multiply the learning rate by ``gamma`` after each epoch in ``milestones``.

    Epochs before a milestone ``m`` use the initial rate, epoch ``m`` onwards the decayed
    one.

    :param milestones: Epoch numbers, or a single one.
    :param gamma: Decay factor.
    """

    key = "step-lr-decay"

    def __init__(self, milestones, gamma=0.1, base_optimizer=None):
        super().__init__(base_optimizer)
        if isinstance(milestones, int):
            milestones = [milestones]
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = float(gamma)

    def setup(self, opt):
        self.base_lrs = [g["lr"] for g in opt.param_groups]

    def update(self, epoch, epochs):
        factor = self.gamma ** sum(1 for m in self.milestones if epoch >= m)
        for g, lr in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = lr * factor


class ConstantThenCosine(_Schedule):
    """Constant learning rate for the first ``constant_fraction`` of the epochs, then a
    cosine decay to ``min_lr`` at the last epoch.

    ELK-Em (Singh et al. 2026, Section 5.2): "learning rate 0.01, held constant for the
    first third of training and then cosine-decayed to :math:`10^{-4}`".

    :param constant_fraction: Fraction of the epochs at the initial rate.
    :param min_lr: Final learning rate.
    """

    key = "cosine-lr-decay"

    def __init__(self, constant_fraction=1.0 / 3.0, min_lr=1e-4, base_optimizer=None):
        super().__init__(base_optimizer)
        if not 0.0 <= constant_fraction <= 1.0:
            raise ValueError("constant_fraction must be in [0, 1].")
        self.constant_fraction = float(constant_fraction)
        self.min_lr = float(min_lr)

    def setup(self, opt):
        self.base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_at(self, base_lr, epoch, epochs):
        start = int(round(self.constant_fraction * epochs))
        if epoch < start or epochs - start <= 1:
            return base_lr
        progress = (epoch - start) / (epochs - start - 1)
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def update(self, epoch, epochs):
        for g, lr in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = self.lr_at(lr, epoch, epochs)


class LinearDecay(_Schedule):
    """Linear decay of the learning rate from its initial value at the first epoch to
    ``min_lr`` at the last one.

    :param min_lr: Final learning rate.
    """

    key = "linear-lr-decay"

    def __init__(self, min_lr=1e-4, base_optimizer=None):
        super().__init__(base_optimizer)
        self.min_lr = float(min_lr)

    def setup(self, opt):
        self.base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_at(self, base_lr, epoch, epochs):
        if epochs <= 1:
            return base_lr
        t = epoch / (epochs - 1)
        return (1.0 - t) * base_lr + t * self.min_lr

    def update(self, epoch, epochs):
        for g, lr in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = self.lr_at(lr, epoch, epochs)


class TemperatureAnnealing(TrainingRegime):
    """Anneal the temperature of a model with soft (sigmoid or softmin) memberships from
    ``start`` at the first epoch to ``end`` at the last one.

    A high temperature gives smooth memberships and useful gradients far from a region's
    boundary; a low one makes the trained memberships agree with the crisp regions that
    are read off the model afterwards. Used by GroundedEL (always) and MultiBoxEL
    (``tau_anneal=True``); it applies to every model whose module has a temperature: a
    ``set_tau(value)`` method, or a ``tau`` attribute.

    :param start: Temperature of the first epoch (> 0).
    :param end: Temperature of the last epoch (> 0).
    :param shape: ``"geometric"`` (``start * (end / start) ** t``, the default) or
        ``"linear"``, with ``t = epoch / (epochs - 1)``.
    """

    key = "temperature-annealing"

    def __init__(self, start=1.0, end=0.01, shape="geometric"):
        super().__init__()
        if not (start > 0 and end > 0):
            raise ValueError("start and end must be > 0.")
        if shape not in ("geometric", "linear"):
            raise ValueError("shape must be 'geometric' or 'linear'.")
        self.start, self.end, self.shape = float(start), float(end), shape
        self.history = []

    def attach(self, model):
        module = getattr(model, "module", None)
        if not (hasattr(module, "set_tau") or hasattr(module, "tau")
                or hasattr(module, "temperature")):
            raise TypeError(
                f"TemperatureAnnealing needs a model whose module has a temperature "
                f"('set_tau', 'tau' or 'temperature'); {type(model).__name__} has none.")
        super().attach(model)

    def value_at(self, epoch, epochs):
        """The temperature of ``epoch`` (0-based) in a run of ``epochs`` epochs."""
        if epochs <= 1:
            return self.start
        t = min(max(epoch / (epochs - 1), 0.0), 1.0)
        if self.shape == "linear":
            return (1.0 - t) * self.start + t * self.end
        return self.start * ((self.end / self.start) ** t)

    def on_epoch_start(self, epoch, epochs):
        tau = float(self.value_at(epoch, epochs))
        module = self.model.module
        if hasattr(module, "set_tau"):
            module.set_tau(tau)
        elif hasattr(module, "tau"):
            module.tau = tau
            if hasattr(self.model, "tau") and not callable(self.model.tau):
                self.model.tau = tau
        else:
            module.temperature = tau
        self.history.append(tau)


class CyclicLearningRate(_Schedule):
    """Triangular cyclical learning rate between ``base_lr`` and ``max_lr``
    (:class:`torch.optim.lr_scheduler.CyclicLR`, Smith 2017), stepped once per epoch.

    CatE (Zhapa-Camacho and Hoehndorf, Section 4.1) trains with "the Adam optimizer with a
    Cyclic Learning Rate scheduler"; the paper gives neither the bounds nor the cycle
    length, so they are parameters here.

    :param base_lr: Lower bound (default: the optimiser's learning rate).
    :param max_lr: Upper bound (default: ten times ``base_lr``).
    :param step_size_up: Epochs from the lower to the upper bound.
    :param mode: ``"triangular"``, ``"triangular2"`` or ``"exp_range"``.
    """

    key = "cyclic-lr"

    def __init__(self, base_lr=None, max_lr=None, step_size_up=10, mode="triangular",
                 base_optimizer=None):
        super().__init__(base_optimizer)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = int(step_size_up)
        self.mode = mode
        self.scheduler = None

    def setup(self, opt):
        base = self.base_lr if self.base_lr is not None else opt.param_groups[0]["lr"]
        top = self.max_lr if self.max_lr is not None else 10 * base
        self.scheduler = th.optim.lr_scheduler.CyclicLR(
            opt, base_lr=base, max_lr=top, step_size_up=self.step_size_up, mode=self.mode,
            cycle_momentum=False)

    def update(self, epoch, epochs):
        if epoch > 0:
            self.scheduler.step()


def validation_loss(model, gci_name=None):
    """Mean loss of the validation axioms of ``gci_name`` (default:
    ``model.eval_gci_name``) -- the quantity mOWL's training loop and DELE monitor."""
    gci_name = gci_name or model.eval_gci_name
    if gci_name is None:
        raise ValueError("Set model.eval_gci_name or pass gci_name.")
    module = model.module
    was_training = module.training
    module.eval()
    with th.no_grad():
        value = float(th.mean(module(model.validation_datasets[gci_name][:], gci_name)))
    if was_training:
        module.train()
    return value


class ReduceOnPlateau(_Schedule):
    """Reduce the learning rate by ``factor`` when a monitored quantity has not improved
    for ``patience`` epochs (:class:`torch.optim.lr_scheduler.ReduceLROnPlateau`).

    DELE (Mashkova et al., Section 6.3) and the NeSy 2024 paper (Section 4.4) use "Adam
    optimizer and ReduceLROnPlateau scheduler with patience parameter 10", monitoring the
    validation loss of the predicted normal form (reference implementation), and stop
    early "if validation loss does not improve for 20 epochs". With
    :class:`~mowl.training.PerNormalFormMiniBatches` one epoch of mOWL is one *step*, so
    pass ``every`` (or ``every=None`` to take the steps per epoch of the batching regime
    of the same composition) to reduce and stop once per paper epoch, and ``stop=True``
    to end training when ``stop_patience`` paper epochs passed without improvement (the
    epoch at which it would stop is recorded in :attr:`stop_epoch` either way).

    :param metric: ``f(model) -> float`` to minimise (default: :func:`validation_loss`).
    :param patience: Epochs without improvement before the rate is reduced.
    :param factor: Reduction factor (PyTorch's default 0.1, as in the reference code).
    :param stop_patience: Epochs without improvement after which early stopping would stop.
    :param every: Monitor (and step the scheduler) every ``every`` epochs (mOWL epochs,
        i.e. optimisation steps when a batching regime is composed). ``1`` (the default)
        monitors every epoch; ``None`` takes the steps per epoch of a
        :class:`~mowl.training.PerNormalFormMiniBatches` regime of the same composition.
    :param stop: End training (:meth:`should_stop`) once ``stop_patience`` monitored
        epochs passed without improvement. The default ``False`` only records
        :attr:`stop_epoch`, as before.
    :param early_stop: Alias of ``stop`` (the reference code of the faithfulness paper,
        Mashkova & Hoehndorf 2026, ``box2el.py``: ``epochs_no_improve``).
    """

    key = "reduce-lr-on-plateau"

    def __init__(self, metric=None, patience=10, factor=0.1, stop_patience=20,
                 every=1, stop=False, base_optimizer=None, early_stop=None):
        super().__init__(base_optimizer)
        if early_stop is not None:
            stop = early_stop
        self.early_stop = bool(stop)
        self.metric = metric or validation_loss
        self.patience = int(patience)
        self.factor = float(factor)
        self.stop_patience = stop_patience
        self.stop = bool(stop)
        self.every = None if every is None else max(1, int(every))
        self.scheduler = None
        self.values = []
        self.stop_epoch = None
        self._best = math.inf
        self._bad = 0
        self._interval = None

    def setup(self, opt):
        self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.patience, factor=self.factor)

    def _steps_per_epoch(self):
        """The steps per epoch of a batching regime attached to the same model."""
        from mowl.training.batching import PerNormalFormMiniBatches
        regime = getattr(self.model, "training_regime", None)
        for part in getattr(regime, "regimes", None) or [regime]:
            if isinstance(part, PerNormalFormMiniBatches):
                return max(1, part.steps_per_epoch)
        return 1

    def _monitor_every(self):
        if self._interval is None:
            self._interval = self.every or self._steps_per_epoch()
        return self._interval

    def update(self, epoch, epochs):
        if epoch == 0:
            return
        every = self._monitor_every()
        if epoch % every != 0:
            return
        value = float(self.metric(self.model))
        self.values.append(value)
        self.scheduler.step(value)
        if value < self._best:
            self._best, self._bad = value, 0
        else:
            self._bad += 1
            waiting = self.stop_patience is not None and self.stop_epoch is None
            if waiting and self._bad >= self.stop_patience:
                self.stop_epoch = epoch - 1

    def should_stop(self, epoch):
        return self.stop and self.stop_epoch is not None


class EarlyStopping(ReduceOnPlateau):
    """End training when a validation quantity has not improved for ``patience`` monitored
    epochs, without changing the learning rate ("early stopping if validation loss does
    not improve for 20 epochs", Mashkova et al., 2024).

    This is :class:`ReduceOnPlateau` with ``stop=True`` and a learning rate that is never
    reduced; use :class:`ReduceOnPlateau` directly to combine both.

    :param metric: ``metric(model) -> float`` to minimise. Defaults to the validation loss.
    :param patience: Monitored epochs without improvement after which training ends.
    :param every: Monitor every this many epochs (see :class:`ReduceOnPlateau`).
    :param base_optimizer: Optimizer class, see :class:`ReduceOnPlateau`.
    """

    key = "early-stopping"

    def __init__(self, metric=None, patience=20, every=1, base_optimizer=None):
        super().__init__(metric=metric, patience=10, factor=1.0, stop_patience=patience,
                         every=every, stop=True, base_optimizer=base_optimizer)

    def setup(self, opt):
        # No learning-rate reduction: a constant schedule that accepts ``step(value)``.
        self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=10 ** 9, factor=0.999999)
