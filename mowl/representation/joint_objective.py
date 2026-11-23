"""Joint objectives: one loss over all axiom forms of a step at once.

The objective of a :class:`~mowl.representation.ComposedGeometry`
(:class:`~mowl.representation.Objective`) scores the rows of one axiom form at a time, and the
training loop adds the forms. Some methods read every form of a step together: they draw the
elements (or points) a step works on once and evaluate every axiom on the same draw, so the
terms of different forms share samples (GroundedEL's grounded constraints). A *joint
objective* is the part for these: it receives the rows of every form of a step and returns
the terms of the step.

A geometry declares one as ``geometry.joint_objective``; a general model
(:class:`~mowl.models.ComposedEL`) then trains it with the regime
:class:`~mowl.training.joint.JointObjectiveSteps` (the loss of a step is the joint
objective's, :meth:`~mowl.training.TrainingRegime.step_loss`) instead of scoring forms one at a
time. Any geometry can have one: a method whose axioms are read at a shared sample of the
domain, a joint likelihood, a loss that couples forms.

The protocol (:class:`JointObjective`):

* ``losses(geometry, datasets, generator, negatives=None)``: the terms of one step, a mapping
  with a ``"total"`` tensor (the other entries, tensors or numbers, are reported per epoch);
* ``probe(geometry, datasets)``: evaluate one step without gradient and with a generator of
  its own, so that a combination whose parts do not fit raises
  :class:`~mowl.representation.ConstructorUnavailable` when it is built;
* ``part``: its name in the vocabulary of parts.
"""

import torch as th

__all__ = ["JointObjective"]


class JointObjective:
    """Base class of joint objectives (see the module docstring)."""

    part = "joint objective"

    def losses(self, geometry, datasets, generator, negatives=None):
        raise NotImplementedError

    def probe(self, geometry, datasets, seed=0):
        """One step without gradient (and a generator of its own: the run is not touched)."""
        generator = th.Generator().manual_seed(int(seed))
        with th.no_grad():
            return self.losses(geometry, datasets, generator)

    def replace(self, **options):
        """A copy with ``options`` changed (the options are the attributes of the part)."""
        import copy
        other = copy.copy(self)
        for name, value in options.items():
            if not hasattr(other, name):
                raise TypeError(f"{type(self).__name__} has no option {name!r}.")
            setattr(other, name, value)
        return other
