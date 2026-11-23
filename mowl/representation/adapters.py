"""The geometry of an existing model (:func:`geometry_of`).

An adapter describes an existing module as a combination of parts that *share its
parameters*: the returned :class:`~mowl.representation.ComposedGeometry` reads and trains
the same tables, and its losses equal the module's. Modules that are not a combination of
parts get a geometry of their own.
"""

import inspect

from mowl.representation.geometry import Geometry

__all__ = ["geometry_of", "register_adapter", "adapter_for", "ADAPTERS"]

#: Qualified class name (``"package.module.Class"``) -> function from a module (or model) to
#: its :class:`Geometry`. The classes are identified by where they are defined, so that a
#: class of another module with the same name (a direct implementation of the same method) is
#: not matched by accident; the adapter of such a class is registered for it explicitly.
ADAPTERS = {}


def qualified_name(cls):
    """``"module.Class"`` of a class (the key of :data:`ADAPTERS`)."""
    return f"{cls.__module__}.{cls.__qualname__}"


def register_adapter(*classes):
    """Decorator: use the function as the adapter of these classes: module classes, or, for
    models that have no usable ``.module`` (the models of FaithEL, of the knowledge-graph and
    word2vec families, BoxLitE), model classes. Each is the class or its qualified name
    (``"mowl.models.faithel.model.FaithEL"``; a name is for a class that cannot be imported
    where the adapter is registered). The function takes the module (the model itself where
    the adapter is registered under a model class); if it has a parameter ``model``, it also
    gets the model the module belongs to (``None`` when only a module was given)."""
    def decorate(function):
        for cls in classes:
            ADAPTERS[cls if isinstance(cls, str) else qualified_name(cls)] = function
        return function
    return decorate


def adapter_for(cls):
    """The adapter of the class ``cls`` or of its nearest base class that has one, or
    ``None``."""
    for base in cls.__mro__:
        adapter = ADAPTERS.get(qualified_name(base))
        if adapter is not None:
            return adapter
    return None


def _mirror_strict_negatives(geometry, module):
    """A geometry that describes ``module`` refuses a negative loss of a form without one
    exactly when the module does (every :class:`~mowl.nn.ELModule` has the attribute
    ``strict_negatives``)."""
    objective = getattr(geometry, "objective", None)
    strict = getattr(module, "strict_negatives", None)
    if objective is not None and isinstance(strict, bool):
        objective.strict_negatives = strict


def geometry_of(model):
    """The :class:`~mowl.representation.Geometry` of a model or module.

    :raises NotImplementedError: If the module has no adapter yet.
    """
    if isinstance(model, Geometry):
        return model
    geometry = _geometry_of(model)
    module = getattr(model, "module", model)
    if geometry is module:      # the module is a geometry (built from parts): nothing to link
        return geometry
    _mirror_strict_negatives(geometry, module)
    return _link(geometry, module)


def _link(geometry, module):
    """Make the margin of a geometry that describes ``module`` the module's margin, so that
    the two cannot drift apart (when the adapter has given the geometry the margin of the
    module: otherwise the margin of the geometry is a number of its own). A module that is
    the geometry itself (ComposedEL, a constructed model) is not linked to itself: its
    margin would be read through itself without end."""
    if module is geometry:
        return geometry
    attribute = getattr(module, "margin_attribute", "margin")
    value = getattr(module, attribute, None)
    link = getattr(geometry, "link_margin", None)
    if link is not None and isinstance(value, (int, float)) and not isinstance(value, bool) \
            and abs(geometry.margin - float(value)) <= 1e-12 \
            and getattr(module, "margin_semantics", None) in (None, geometry.margin_semantics):
        link(module)
    return geometry


def _geometry_of(model):
    # The model's own class first (also its base classes): a model without a usable
    # ``.module`` is described by its class. Only then the module.
    adapter = adapter_for(type(model))
    if adapter is not None:
        return _call(adapter, model, model)
    module = getattr(model, "module", model)
    if isinstance(module, Geometry):
        return module
    adapter = adapter_for(type(module))
    if adapter is not None:
        return _call(adapter, module, model)
    raise NotImplementedError(
        f"{type(model).__name__} does not expose a geometry yet. Classes that do: "
        f"{sorted(name.rsplit('.', 1)[-1] for name in ADAPTERS)}.")


def _call(adapter, obj, model):
    # An adapter may take the model as well (texts, sizes and options that live there).
    parameters = list(inspect.signature(adapter).parameters)
    if "model" in parameters[1:]:
        return adapter(obj, model=model if model is not obj else None)
    return adapter(obj)
