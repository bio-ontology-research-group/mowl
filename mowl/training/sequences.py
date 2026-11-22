"""Training regimes for models that learn from sequences
(:class:`~mowl.models.ComposedSequenceModel`, ``RandomWalkPlusW2VModel``,
``SyntacticPlusW2VModel``).

With a regime attached, such a model trains its encoder one pass at a time (the learning rate
falls over all passes as in one call) and consults ``on_epoch_start``, ``on_epoch_end``,
``should_stop`` and ``on_train_end``. Schedule regimes of :mod:`mowl.training` that use only
these hooks (early stopping on the loss, for example) work with them too.
"""

from mowl.training.regime import TrainingRegime

__all__ = ["RefreshSequences"]


class RefreshSequences(TrainingRegime):
    """New sequences every ``every`` passes: the walks are drawn again with another seed
    (``seed + epoch``) before the pass, so that the encoder sees other walks of the same
    graph instead of the same ones again (sentence corpora have no random part and are
    written again unchanged). The vectors are kept; new tokens are added.

    :param every: Passes between two refreshes.
    :param seed: Seed of the first refresh (default: 0).

    :attr:`refreshed` lists the passes before which the sequences were made anew.
    """

    def __init__(self, every=1, seed=0):
        super().__init__()
        if int(every) < 1:
            raise ValueError("every must be at least 1.")
        self.every = int(every)
        self.seed = int(seed)
        self.refreshed = []

    def attach(self, model):
        if not callable(getattr(model, "refresh_sequences", None)):
            raise TypeError(f"{type(model).__name__} does not learn from sequences "
                            f"(it has no refresh_sequences).")
        super().attach(model)

    def on_epoch_start(self, epoch, epochs):
        if epoch > 0 and epoch % self.every == 0:
            self.model.refresh_sequences(seed=self.seed + epoch)
            self.refreshed.append(epoch)
