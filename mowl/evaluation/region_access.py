"""Reading the class regions of a trained model through the common interface.

Evaluation regimes that look at regions (inclusion accuracy, sampled F1, the exact-model
check, the empty-intersection rule) used to find out whether a module has boxes or balls
by looking for attributes such as ``class_offset`` or ``class_rad``. They ask the
geometry of the model instead (:func:`mowl.representation.geometry_of`), so that every
method that has a geometry of balls or boxes works, whatever it names its parameters.
"""

import torch as th

from mowl.representation import geometry_of
from mowl.representation.regions import Ball, Box

__all__ = ["geometry_for", "centre_and_size", "all_class_regions", "device_of"]


def device_of(model):
    """The device of the parameters of a model or module (CPU if it has none)."""
    module = getattr(model, "module", model)
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return th.device("cpu")


def geometry_for(model, purpose):
    """The geometry of ``model``; a clear error naming ``purpose`` if it has none.

    :raises NotImplementedError: If the module has no adapter (no geometry yet).
    """
    try:
        return geometry_of(model)
    except NotImplementedError as error:
        raise NotImplementedError(
            f"{purpose} needs the regions of the classes, which come from the geometry of "
            f"the model, and {type(getattr(model, 'module', model)).__name__} has none: "
            f"{error}") from error


def centre_and_size(region, purpose):
    """``(centres, sizes, is_ball)`` of a region of balls or boxes: sizes are the
    non-negative radii ``(B,)`` for balls and half-widths ``(B, d)`` for boxes.

    :raises TypeError: For any other region (unions of boxes, octagons, ...).
    """
    if isinstance(region, Ball):
        return region.centre, region.radius.reshape(-1), True
    if isinstance(region, Box):
        return region.centre, region.offset, False
    raise TypeError(
        f"{purpose} compares balls or boxes; the classes of this model are "
        f"{type(region).__name__} regions.")


def all_class_regions(model, purpose):
    """The region of every named class of ``model``: ``(region, geometry)``."""
    geometry = geometry_for(model, purpose)
    ids = th.arange(geometry.num_classes, device=device_of(model))
    return geometry.classes(ids), geometry
