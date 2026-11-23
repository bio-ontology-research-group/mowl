"""Training a geometry with a joint objective (:mod:`mowl.representation.joint_objective`).

:class:`JointObjectiveSteps` computes the loss of every step of the shared loop
(:meth:`~mowl.training.TrainingRegime.step_loss`) with the joint objective of the geometry of
the model (``model.module.joint_objective``): the rows of every form of the training data at
once, read at one draw of a generator seeded with ``seed``. The terms of the objective reach
``on_epoch_end`` of the regimes; the negatives of a regime in front of this one (its
``negatives`` and ``filter_negatives`` of ``"object_property_assertion"``) replace the
objective's own corrupted role assertions.
"""

import torch as th

from mowl.training.regime import TrainingRegime

__all__ = ["JointObjectiveSteps"]


class JointObjectiveSteps(TrainingRegime):
    """The regime of the module docstring.

    :param seed: Seed of the generator of the samples of the steps (default: the model's
        ``seed`` if it has one, else 0).
    """

    key = "joint-objective-steps"

    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        self.generator = None
        #: The regime of the run whose negatives are used (set by the model).
        self.outer = None

    def attach(self, model):
        objective = getattr(getattr(model, "module", None), "joint_objective", None)
        if objective is None:
            raise TypeError(f"JointObjectiveSteps needs a geometry with a joint objective; "
                            f"{type(getattr(model, 'module', None)).__name__} has none.")
        super().attach(model)

    def training_datasets(self, datasets):
        seed = self.seed if self.seed is not None else getattr(self.model, "seed", 0) or 0
        self.generator = th.Generator().manual_seed(int(seed))
        self._datasets = datasets
        return datasets

    def _negatives(self):
        regime = getattr(self.model, "training_regime", None)
        if regime is None or regime is self:
            return None

        def corrupted(table, generator):
            rows = regime.negatives("object_property_assertion", table)
            if rows is NotImplemented or rows is None:
                return None
            filtered = regime.filter_negatives("object_property_assertion", table, rows)
            return rows if filtered is None else filtered
        return corrupted

    def step_loss(self, step):
        module = self.model.module
        return module.joint_objective.losses(module, self._datasets, self.generator,
                                             negatives=self._negatives())
