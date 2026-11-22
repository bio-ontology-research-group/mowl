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

__all__ = ["adopt"]


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
    return module
