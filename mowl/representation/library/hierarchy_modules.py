"""Helpers for the public module classes of the methods that embed a hierarchy.

:class:`mowl.nn.OrderEmbeddingsModule`, :class:`~mowl.nn.BoxLatticeModule`,
:class:`~mowl.nn.GumbelBoxModule`, :class:`~mowl.nn.BinaryCodeBoxModule` and
:class:`~mowl.nn.CatEModule` are geometries built from parts: each one calls the builder of
its method (:mod:`mowl.representation.library.hierarchy_methods`, the one place where the
combination is written down) and keeps the parts it returns. :func:`adopt` does the second
step. The classes themselves live next to the earlier modules (``mowl.nn.el.<method>`` and
``mowl.nn.graph.<method>``), with the constructor arguments those had.
"""

from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.hierarchy_geometry import LatticeGeometry

__all__ = ["adopt", "FlatLossOption"]


def adopt(module, built):
    """Initialise ``module`` (a not yet initialised subclass of
    :class:`~mowl.representation.ComposedGeometry` or
    :class:`~mowl.representation.library.hierarchy_geometry.LatticeGeometry`) with the parts
    of the geometry ``built``, which a method builder returned. The parameters are those of
    ``built``'s parts (not copies)."""
    if isinstance(built, LatticeGeometry):
        LatticeGeometry.__init__(
            module, built.class_model, built.measure, built.objective,
            existentials=built._existential_ids, conjunctions=built._conjunction_ids,
            regularizers=built.regularizers)
    else:
        ComposedGeometry.__init__(
            module, built.class_model, built.relation_model, built.measure, built.objective,
            individuals=built.individual_model, regularizers=built.regularizers,
            supported_forms=built.supported_forms,
            neg_capable_gcis=built.neg_capable_gcis)
    if isinstance(module, FlatLossOption):
        module.margin_semantics = None
    return module


class FlatLossOption:
    """Mixin (before the geometry class) of the public module classes of the taxonomy
    methods, which keep the interface of an :class:`~mowl.nn.ELModule` that reads ``gci0``
    only:

    * a form the geometry does not read raises :class:`NotImplementedError`, as every
      module does for a form it does not implement (the part-built geometry itself raises
      :class:`~mowl.representation.ConstructorUnavailable`, which says which part is
      missing);
    * ``module.per_axiom_column = False`` gives the flat ``(B,)`` vector of the formula
      instead of the column ``(B, 1)`` (the values are the same);
    * the margin does not enter the cost of a stated pair (it is the margin of the negative
      cost), so ``margin_semantics`` is ``None``.
    """

    margin_semantics = None

    def cost(self, form, rows, neg=False):
        if form not in self.supported_forms:
            raise NotImplementedError(
                f"{type(self).__name__} reads {sorted(self.supported_forms)} only, not "
                f"{form!r}.")
        out = super().cost(form, rows, neg)
        if not self.per_axiom_column and out.dim() == 2 and out.shape[-1] == 1:
            return out.squeeze(-1)
        return out
