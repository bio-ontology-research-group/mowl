"""One shape for the per-axiom losses of a module: a column of shape ``(B, 1)``."""

import functools

LOSS_METHODS = (
    "gci0_loss", "gci0_bot_loss", "gci1_loss", "gci1_bot_loss", "gci2_loss", "gci3_loss",
    "gci3_bot_loss", "class_assertion_loss", "object_property_assertion_loss",
    "role_inclusion_loss", "role_chain_loss",
)


def _as_column(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        out = method(self, *args, **kwargs)
        if self.per_axiom_column and getattr(out, "ndim", None) == 1:
            return out.unsqueeze(-1)
        return out

    wrapper._as_column = True
    return wrapper


class PerAxiomColumn:
    """Mixin for :class:`~mowl.nn.ELModule` subclasses: every per-axiom loss method defined
    on the class returns a column ``(B, 1)`` like the losses of
    :class:`~mowl.nn.ELBEModule`, never a flat ``(B,)`` vector.

    Set the attribute ``per_axiom_column`` to ``False`` (on the instance or the class) to get
    the earlier shapes back, where each loss returned what its formula produced: ``(B,)``
    for the Box\\ :sup:`2`\\ EL and MultiBoxEL losses, a mix of ``(B,)`` and ``(B, 1)`` for
    MultiELBE. Losses that are not one value per row (more than one axis) are unchanged.
    The mean, the sum and any ``reshape(n, -1)`` of a loss are the same in both shapes;
    only code that adds a loss to a flat ``(B,)`` vector sees a difference (it broadcasts to
    ``(B, B)``).
    """

    per_axiom_column = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in LOSS_METHODS:
            method = cls.__dict__.get(name)
            if method is not None and not getattr(method, "_as_column", False):
                setattr(cls, name, _as_column(method))
