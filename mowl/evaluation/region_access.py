"""Reading the class regions of a trained model through the common interface.

Evaluation regimes that look at regions (inclusion accuracy, sampled F1, the exact-model
check, the empty-intersection rule) used to find out whether a module has boxes or balls
by looking for attributes such as ``class_offset`` or ``class_rad``. They ask the
geometry of the model instead (:func:`mowl.representation.geometry_of`), so that every
method that has a geometry of balls or boxes works, whatever it names its parameters.
"""

from functools import singledispatch

import torch as th

from mowl.representation import geometry_of
from mowl.representation.regions import Ball, Box, CornerBox

__all__ = ["geometry_for", "centre_and_size", "all_class_regions", "device_of",
           "evaluation_regions", "geometry_evaluation_regions"]


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


# -- from the regions of the geometry to the regions of the exact-model check -----------------

def _np(t):
    return t.detach().cpu().double().numpy()


@singledispatch
def evaluation_regions(region, purpose="The exact-model check"):
    """The regions of a batch ``region`` of the geometry as a list of
    :class:`mowl.evaluation.regions.Region`, one per row, for the exact decision procedures
    of :mod:`mowl.evaluation.regions`.

    Boxes (also soft boxes: their location box), balls, unions of boxes, prisms and
    products of octagons are converted. A region that says it is empty
    (:meth:`~mowl.representation.Region.is_empty`) gives an empty evaluation region. The
    arithmetic is that of the region: a box that keeps its corners
    (:class:`~mowl.representation.CornerBox`: BoxEL, ELK-Em) gives them as they are, a box
    given by centre and half-width gives ``centre -+ offset`` computed in double precision.

    :raises TypeError: For any other region family.
    """
    raise TypeError(f"{purpose} cannot read regions of the family "
                    f"{type(region).__name__}: no converter to the evaluation regions.")


@evaluation_regions.register(Ball)
def _balls(region, purpose="The exact-model check"):
    from mowl.evaluation.regions import BallRegion
    return [BallRegion(c, r) for c, r in zip(_np(region.centre), _np(region.radius).reshape(-1))]


@evaluation_regions.register(Box)
def _boxes(region, purpose="The exact-model check"):
    from mowl.evaluation.regions import BoxRegion
    if isinstance(region, CornerBox):
        low, high = _np(region.low), _corner_high(region)
    else:
        centre, offset = _np(region.centre), _np(region.offset)
        low, high = centre - offset, centre + offset
    return [BoxRegion(lo, hi) for lo, hi in zip(low, high)]


def _corner_high(region):
    """The upper corner: where the class model gives the width of the box (BoxEL:
    ``min + exp(delta)``), the sum is done in double precision, as the exact reading of the
    parameters, not in the precision of the model."""
    width = getattr(region, "width", None)
    if width is None:
        return _np(region.high)
    return _np(region.low) + _np(width)


def _union_boxes(centre, offset, active=None):
    from mowl.evaluation.regions import BoxRegion, UnionRegion
    out = []
    for i in range(centre.shape[0]):
        pieces = [BoxRegion(centre[i, k] - offset[i, k], centre[i, k] + offset[i, k])
                  for k in range(centre.shape[1]) if active is None or active[i, k]]
        out.append(UnionRegion(pieces, dim=centre.shape[2]))
    return out


def _register_library_regions():
    from mowl.representation.library.multibox_parts import UnionOfBoxes
    from mowl.representation.library.other_regions import MaskedUnionOfBoxes
    from mowl.representation.library.shapes_regions import OctagonProduct, Prism

    @evaluation_regions.register(UnionOfBoxes)
    def _unions(region, purpose="The exact-model check"):
        return _union_boxes(_np(region.centre), _np(region.offset))

    @evaluation_regions.register(MaskedUnionOfBoxes)
    def _grounded(region, purpose="The exact-model check"):
        return _union_boxes(_np(region.cen), _np(region.w), _np(region.active).astype(bool))

    @evaluation_regions.register(Prism)
    def _prisms(region, purpose="The exact-model check"):
        from mowl.evaluation.regions import PrismRegion
        h, centre, offset = _np(region.supports), _np(region.centre), _np(region.offset)
        return [PrismRegion(h[i], centre[i] - offset[i], centre[i] + offset[i])
                for i in range(h.shape[0])]

    @evaluation_regions.register(OctagonProduct)
    def _octagons(region, purpose="The exact-model check"):
        from mowl.evaluation.regions import OctagonRegion
        return [OctagonRegion(h) for h in _np(region.bounds)]


_register_library_regions()


def geometry_evaluation_regions(model, purpose="The exact-model check"):
    """One :class:`mowl.evaluation.regions.Region` per class index of ``model``, read through
    its geometry (:func:`evaluation_regions` of the region of every class)."""
    with th.no_grad():
        region, _ = all_class_regions(model, purpose)
        return evaluation_regions(region, purpose)
