"""The geometry of an existing model (:func:`geometry_of`).

An adapter describes an existing module as a combination of parts that *share its
parameters*: the returned :class:`~mowl.representation.ComposedGeometry` reads and trains
the same tables, and its losses equal the module's. Modules that are not a combination of
parts get a geometry of their own.
"""

from mowl.representation import methods
from mowl.representation.geometry import Geometry

__all__ = ["geometry_of", "register_adapter", "ADAPTERS"]

#: Module class name -> function from a module to its :class:`Geometry`.
ADAPTERS = {}


def register_adapter(*module_class_names):
    """Decorator: use the function as the adapter of the module classes with these names."""
    def decorate(function):
        for name in module_class_names:
            ADAPTERS[name] = function
        return function
    return decorate


def geometry_of(model):
    """The :class:`~mowl.representation.Geometry` of a model or module.

    :raises NotImplementedError: If the module has no adapter yet.
    """
    if isinstance(model, Geometry):
        return model
    module = getattr(model, "module", model)
    if isinstance(module, Geometry):
        return module
    adapter = ADAPTERS.get(type(module).__name__)
    if adapter is None:
        raise NotImplementedError(
            f"{type(module).__name__} does not expose a geometry yet. Modules that do: "
            f"{sorted(ADAPTERS)}.")
    return adapter(module)


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
