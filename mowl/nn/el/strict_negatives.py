"""One rule for every :class:`~mowl.nn.ELModule`: asking for the negative loss of an axiom form
the module has no negative loss for is an error."""

import functools

from mowl.nn.el.per_axiom_column import LOSS_METHODS

_FORM_OF = {name: name[:-len("_loss")] for name in LOSS_METHODS}


def _guarded(method, form):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        neg = kwargs.get("neg", args[1] if len(args) > 1 else False)
        if neg and self.strict_negatives and form not in self.neg_capable_gcis:
            raise NotImplementedError(
                f"{type(self).__name__} has no negative loss for '{form}' (it has one for "
                f"{sorted(self.neg_capable_gcis) or 'no form'}), so neg=True is refused. "
                f"Do not ask for a negative of this form (negative sampling is restricted "
                f"to 'neg_capable_gcis' by default), or set 'strict_negatives = False' on "
                f"the module to get the positive loss back instead (the earlier "
                f"behaviour).")
        return method(self, *args, **kwargs)

    wrapper._neg_guard = True
    return wrapper


class StrictNegatives:
    """Mixin of :class:`~mowl.nn.ELModule` (so of every subclass): ``neg=True`` on a per-axiom
    loss whose form is not in :attr:`~mowl.nn.ELModule.neg_capable_gcis` raises
    :class:`NotImplementedError` instead of returning the positive loss.

    The attribute ``strict_negatives`` (default ``True``; on the instance or the class) turns
    the check off: ``False`` gives the earlier behaviour, where a loss without a negative
    version ignored ``neg=True`` and returned the positive loss. The check reads
    ``neg_capable_gcis`` when the loss is called, so modules that extend it at run time (the
    DELE negatives of ELEm, ELBE and BoxSquaredEL) are covered.
    """

    strict_negatives = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, form in _FORM_OF.items():
            # Looked up through the MRO, not in cls.__dict__: a loss that comes from a plain
            # mixin (as in the RegD modules) is not defined on an ELModule subclass itself.
            method = getattr(cls, name, None)
            if method is None or getattr(method, "_neg_guard", False):
                continue
            if getattr(method, "__qualname__", "").startswith("ELModule."):
                continue  # the base class stub: not implemented at all
            setattr(cls, name, _guarded(method, form))
