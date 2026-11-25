"""Regimes that change the structure of a model during training.

Models that represent a concept by several components (a union of boxes) can get stuck
with components that cover nothing: no gradient reaches them. Re-seeding moves such
components, between optimisation steps, to places where they are needed.
"""

from mowl.baselines.training.regime import TrainingRegime


class IdleComponentReseeding(TrainingRegime):
    """Every ``every`` epochs, after the optimisation step, relocate the idle components
    of the model and (by default) forget the optimiser's moment estimates, which no
    longer describe the moved parameters.

    The model decides what "idle" means and where a component goes: it must provide
    ``reseed_components(epoch)`` returning the number of components moved. GroundedEL
    (uncovered own elements get a component; components containing no element are
    switched off) and MultiBoxEL (dead or idle primitive components) provide it.

    :param every: Period in epochs; re-seeding runs at the end of epochs ``every``,
        ``2 * every``, ... (1-based). ``0`` disables it.
    :param reset_optimizer: Clear the optimiser state (``model.optimizer``) after a
        re-seeding step.
    """

    key = "idle-component-reseeding"

    def __init__(self, every=100, reset_optimizer=True):
        super().__init__()
        if int(every) < 0:
            raise ValueError("every must be >= 0.")
        self.every = int(every)
        self.reset_optimizer = bool(reset_optimizer)
        #: ``(epoch, number of components moved)`` for every re-seeding step.
        self.history = []

    def attach(self, model):
        if not callable(getattr(model, "reseed_components", None)):
            raise TypeError(
                f"IdleComponentReseeding needs a model with 'reseed_components(epoch)'; "
                f"{type(model).__name__} has none.")
        super().attach(model)

    def on_epoch_end(self, epoch, losses):
        if self.every <= 0 or (epoch + 1) % self.every != 0:
            return
        moved = self.model.reseed_components(epoch)
        self.history.append((epoch, moved))
        optimizer = getattr(self.model, "optimizer", None)
        if self.reset_optimizer and optimizer is not None:
            optimizer.state.clear()
