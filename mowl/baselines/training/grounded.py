"""The training design of GroundedEL as a composition of shared regimes.

GroundedEL's loss (memberships of domain elements, sampled points of regions) is part of
the method. Everything else about how it is trained is a regime that other models can
use; :class:`GroundedELTraining` is the combination GroundedEL uses when no regime is
assigned to the model.
"""

from mowl.baselines.training.constraints import LagrangianDualWeights
from mowl.baselines.training.regime import ComposedRegime
from mowl.baselines.training.schedules import ConstantThenCosine, LinearDecay, TemperatureAnnealing
from mowl.baselines.training.structure import IdleComponentReseeding


class GroundedELTraining(ComposedRegime):
    """How GroundedEL trains by default:

    * the membership temperature is annealed geometrically from ``tau0`` to ``tau1``
      (:class:`~mowl.baselines.training.TemperatureAnnealing`);
    * optionally (``lr_final``), the learning rate decays from the model's learning rate
      to ``lr_final`` along a cosine (:class:`~mowl.baselines.training.ConstantThenCosine` without
      a constant phase) or a line (:class:`~mowl.baselines.training.LinearDecay`);
    * optionally (``reseed_every > 0``), idle components are re-seeded every
      ``reseed_every`` epochs (:class:`~mowl.baselines.training.IdleComponentReseeding`);
    * optionally (``constraint_mode="lagrangian"``), the fact, axiom and region terms
      are weighted by dual variables (:class:`~mowl.baselines.training.LagrangianDualWeights`).

    To change one part, compose the others yourself, or put a regime in front:
    ``ComposedRegime(KnownPositiveFilter(), GroundedELTraining.for_model(model))``
    filters GroundedEL's corrupted facts against the stated ones.

    :param extra: Regimes placed before the ones above (they take precedence).
    """

    key = "grounded-el-training"

    def __init__(self, tau0=1.0, tau1=0.01, lr_final=None, lr_schedule="cosine",
                 reseed_every=0, constraint_mode="fixed", dual_rho=1.0, dual_max=1e3,
                 dual_target=0.0, extra=()):
        if lr_schedule not in ("cosine", "linear"):
            raise ValueError(f"lr_schedule must be 'cosine' or 'linear', got {lr_schedule!r}")
        if constraint_mode not in ("fixed", "lagrangian"):
            raise ValueError("constraint_mode must be 'fixed' or 'lagrangian'.")
        parts = list(extra)
        if lr_final is not None:
            parts.append(ConstantThenCosine(constant_fraction=0.0, min_lr=lr_final)
                         if lr_schedule == "cosine" else LinearDecay(min_lr=lr_final))
        parts.append(TemperatureAnnealing(start=tau0, end=tau1))
        if reseed_every > 0:
            parts.append(IdleComponentReseeding(every=reseed_every))
        if constraint_mode == "lagrangian":
            parts.append(LagrangianDualWeights(rho=dual_rho, max_weight=dual_max,
                                               target=dual_target))
        super().__init__(*parts)
        self.key = type(self).key

    @classmethod
    def for_model(cls, model, lr_final=None, lr_schedule="cosine", extra=()):
        """The composition given by the hyperparameters of a GroundedEL ``model``
        (``tau0``, ``tau1``, ``reseed_every``, ``constraint_mode`` and the dual
        settings)."""
        return cls(tau0=model.tau0, tau1=model.tau1, lr_final=lr_final,
                   lr_schedule=lr_schedule, reseed_every=model.reseed_every,
                   constraint_mode=model.constraint_mode, dual_rho=model.dual_rho,
                   dual_max=model.dual_max, dual_target=model.dual_target, extra=extra)
