"""The geometry of an existing model (:func:`geometry_of`).

An adapter describes an existing module as a combination of parts that *share its
parameters*: the returned :class:`~mowl.representation.ComposedGeometry` reads and trains
the same tables, and its losses equal the module's. Modules that are not a combination of
parts get a geometry of their own.
"""

from mowl.representation.class_models import BallClasses, BoxClasses
from mowl.representation.geometry import ComposedGeometry, Geometry
from mowl.representation.measures import CentreDistance, Bounds
from mowl.representation.objectives import Objective
from mowl.representation.relation_models import Translation

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


def _individuals(module, cls, sizes):
    if getattr(module, "ind_embed", None) is None:
        return None
    return cls(centres=module.ind_embed, **{sizes[0]: getattr(module, sizes[1])})


@register_adapter("ELEmModule")
def _elem(module):
    objective = Objective.elem(
        module.margin, all_negatives=module.all_negative_losses,
        linear_negative=module.negative_loss_form == "linear", epsilon=module.epsilon,
        gci3="inclusion" if getattr(module, "gci3_form", "overlap") == "inclusion"
        else "overlap")
    return ComposedGeometry(
        BallClasses(centres=module.class_embed, radii=module.class_rad),
        Translation(vectors=module.rel_embed),
        CentreDistance(radius_term=module.gci1_radius_term), objective,
        individuals=_individuals(module, BallClasses, ("radii", "ind_rad")))


@register_adapter("ELBEModule")
def _elbe(module):
    objective = Objective.elbe(module.margin, all_negatives=module.all_negative_losses,
                               epsilon=module.epsilon, gci3=module.gci3_form)
    return ComposedGeometry(
        BoxClasses(centres=module.class_embed, offsets=module.class_offset),
        Translation(vectors=module.rel_embed), Bounds(separation="every"), objective,
        individuals=_individuals(module, BoxClasses, ("offsets", "ind_offset")))
