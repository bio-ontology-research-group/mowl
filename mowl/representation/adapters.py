"""The geometry of an existing model (:func:`geometry_of`).

An adapter describes an existing module as a combination of parts that *share its
parameters*: the returned :class:`~mowl.representation.ComposedGeometry` reads and trains
the same tables, and its losses equal the module's. Modules that are not a combination of
parts get a geometry of their own.
"""

import inspect

from mowl.representation import methods
from mowl.representation.geometry import Geometry

__all__ = ["geometry_of", "register_adapter", "ADAPTERS"]

#: Class name -> function from a module (or model) to its :class:`Geometry`.
ADAPTERS = {}


def register_adapter(*class_names):
    """Decorator: use the function as the adapter of the classes with these names: module
    classes, or, for models that have no usable ``.module`` (the models of FaithEL, of the
    knowledge-graph and word2vec families, BoxLitE), model classes. The function takes the
    module (the model itself where the adapter is registered under a model class); if it has
    a parameter ``model``, it also gets the model the module belongs to (``None`` when only
    a module was given)."""
    def decorate(function):
        for name in class_names:
            ADAPTERS[name] = function
        return function
    return decorate


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
    for cls in type(model).__mro__:
        adapter = ADAPTERS.get(cls.__name__)
        if adapter is not None:
            return _call(adapter, model, model)
    module = getattr(model, "module", model)
    if isinstance(module, Geometry):
        return module
    for cls in type(module).__mro__:
        adapter = ADAPTERS.get(cls.__name__)
        if adapter is not None:
            return _call(adapter, module, model)
    raise NotImplementedError(
        f"{type(model).__name__} does not expose a geometry yet. Classes that do: "
        f"{sorted(ADAPTERS)}.")


def _call(adapter, obj, model):
    # An adapter may take the model as well (texts, sizes and options that live there).
    parameters = list(inspect.signature(adapter).parameters)
    if "model" in parameters[1:]:
        return adapter(obj, model=model if model is not obj else None)
    return adapter(obj)


@register_adapter("ELEmModule")
def _elem(module):
    return methods.elem(
        None, None, embed_dim=module.embed_dim, margin=module.margin,
        reg_norm=module.reg_norm, all_negatives=module.all_negative_losses,
        linear_negative=module.negative_loss_form == "linear", epsilon=module.epsilon,
        gci1_radius_term=module.gci1_radius_term, tables=module)


@register_adapter("ELBEModule")
def _elbe(module):
    return methods.elbe(None, None, margin=module.margin,
                        all_negatives=module.all_negative_losses, epsilon=module.epsilon,
                        gci3=module.gci3_form, tables=module)
