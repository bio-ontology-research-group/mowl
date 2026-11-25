"""Element and region samplers of a grounded geometry, as functions.

The samplers are the parts :class:`~mowl.representation.library.grounded_sampling.ElementSampler`
and :class:`~mowl.representation.library.grounded_sampling.RegionSampler`; these functions call
them on a geometry with a finite domain of elements (:data:`GROUNDED_INTERFACE`), such as
:class:`~mowl.nn.GroundedELModule`:

- :func:`sample_elements`, :func:`sample_filler_elements`, :func:`sample_inside_elements`:
  candidate elements of a row (its own, random, of the batch, currently inside its classes);
- :func:`sample_active_component`, :func:`sample_component_intersection`: one point in a
  random non-empty active box of a union of boxes (of an intersection of two unions);
- :func:`sample_class_region`, :func:`sample_pair_region`, :func:`sample_role_region`,
  :func:`sample_exists_region`: points of the regions of the class model and the relation
  model (with corners of the box and signed-gradient ascent, :func:`adversarial_ascent`);
- :func:`component_extremes`: the corners of boxes.

These functions keep the published gradient of the first point of a row
(``RegionSampler(doubled_gradient=True)``). Every random draw uses the generator passed in.
Rows without a non-empty active box have
``valid == False`` and zero points.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch as th

from mowl.representation.library.grounded_parts import index_tensor
from mowl.representation.library.grounded_sampling import (ElementSampler, RegionSample,
                                                           RegionSampler, _draw, ascend,
                                                           box_corners)

#: What the samplers read of a geometry: a finite domain of elements with positions, classes
#: and relations as unions of boxes (the class model and the relation model of the parts).
GROUNDED_INTERFACE = ("positions", "class_regions", "role_regions", "depth")

__all__ = [
    "RegionSample", "sample_elements", "sample_inside_elements", "sample_filler_elements",
    "adversarial_ascent", "component_extremes", "sample_active_component",
    "sample_component_intersection", "sample_class_region", "sample_pair_region",
    "sample_role_region", "sample_exists_region",
]

Ids = Union[int, Sequence[int], th.Tensor]
Hinge = Optional[Callable[[th.Tensor], th.Tensor]]


def is_grounded_geometry(module) -> bool:
    """Whether ``module`` answers what the samplers read (:data:`GROUNDED_INTERFACE`)."""
    return all(callable(getattr(module, name, None)) for name in GROUNDED_INTERFACE)


def _randperm_with_generator(n, device, generator):
    """A permutation of ``range(n)`` drawn from ``generator`` (on ``device``)."""
    return _draw("randperm", generator, device, int(n))


def _generator(generator):
    if not isinstance(generator, th.Generator):
        raise TypeError(
            f"generator must be a torch.Generator instance, got {type(generator).__name__}")
    return generator


def _count(name, value, low):
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{name} must be an int >= {low}, got {value!r}")
    return value


def _tensor(name, value):
    if not isinstance(value, th.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {tuple(value.shape)}")
    return value


def _rows_of(name, extra, bsz):
    if extra is not None and extra.ndim == 2 and extra.shape[0] != bsz:
        raise ValueError(f"{name} must have shape (M,) or ({bsz}, M), got {tuple(extra.shape)}")
    return extra


# -- elements -----------------------------------------------------------------------------------
def sample_elements(lhs_elements, nb_elements, n_random, generator, batch_elements=None,
                    inside_elements=None) -> th.Tensor:
    """``(B, L + n_random + M + I)`` candidate elements: the row's ``lhs_elements`` (first),
    ``n_random`` uniform elements of ``[0, nb_elements)``, ``batch_elements`` and
    ``inside_elements`` (``(M,)`` for every row or ``(B, M)``). Not deduplicated."""
    _generator(generator)
    lhs = _tensor("lhs_elements", lhs_elements)
    _count("nb_elements", nb_elements, 1)
    _count("n_random", n_random, 0)
    bsz = int(lhs.shape[0])
    extra = (_rows_of("batch_elements", batch_elements, bsz),
             _rows_of("inside_elements", inside_elements, bsz))
    return ElementSampler(n_random=n_random).candidates(lhs, nb_elements, generator, extra)


def sample_inside_elements(module, lhs_classes, n_inside_elements, generator,
                           pos_all=None) -> th.Tensor:
    """``(B, I)``, ``I <= n_inside_elements``: elements currently crisp inside all the
    classes of the row (first) or at least one, read from the geometry (no reasoner); short
    rows are padded with the row's first class."""
    _generator(generator)
    if not is_grounded_geometry(module):
        raise TypeError(f"module must be a GroundedELModule, got {type(module).__name__}")
    lhs = _tensor("lhs_classes", lhs_classes).to(dtype=th.long)
    _count("n_inside_elements", n_inside_elements, 0)
    if n_inside_elements == 0 or lhs.shape[0] == 0:
        return th.empty((lhs.shape[0], 0), dtype=th.long, device=lhs.device)
    positions = module.positions() if pos_all is None else pos_all
    sampler = ElementSampler(n_inside=n_inside_elements, whole_domain=0)
    return sampler.inside(module.class_model, lhs, positions, generator)


def sample_filler_elements(filler_elements, nb_elements, n_random, generator,
                           batch_size=1) -> th.Tensor:
    """``(B, F + n_random)``: the ``filler_elements`` (``(F,)`` for ``batch_size`` rows or
    ``(B, F)``) and ``n_random`` uniform elements per row."""
    _generator(generator)
    fe = _tensor("filler_elements", filler_elements).to(dtype=th.long)
    _count("batch_size", batch_size, 1)
    if fe.ndim == 1:
        fe = fe.unsqueeze(0).expand(batch_size, fe.shape[0])
    if n_random == 0:
        return fe.clone()
    # the fillers are the "own" elements of the rows, then the random ones
    return ElementSampler(n_random=n_random).candidates(fe, nb_elements, generator)


# -- points of boxes ----------------------------------------------------------------------------
def adversarial_ascent(points, lower, upper, hinge_fn, n_adv=1, step_size=None,
                       valid=None) -> th.Tensor:
    """``n_adv`` signed-gradient ascent steps on ``hinge_fn`` inside ``[lower, upper]``
    (:func:`~mowl.representation.library.grounded_sampling.ascend`)."""
    _count("n_adv", n_adv, 0)
    if points.shape != lower.shape or points.shape != upper.shape:
        raise ValueError(
            f"points, lower, and upper must have identical shapes, got {tuple(points.shape)}, "
            f"{tuple(lower.shape)}, {tuple(upper.shape)}")
    if step_size is not None and not isinstance(step_size, th.Tensor) and (
            isinstance(step_size, bool) or float(step_size) <= 0.0):
        raise ValueError(f"step_size must be > 0, got {step_size!r}")
    return ascend(points, lower, upper, hinge_fn, n_adv, step_size, valid)


def component_extremes(lower, upper, valid=None) -> th.Tensor:
    """``(B, M, d)`` corners (``d <= 4``) or face centres of the boxes, just inside; zero for
    rows that are not ``valid``."""
    if lower.ndim != 2 or upper.shape != lower.shape:
        raise ValueError(
            f"lower and upper must be 2-D tensors of identical shape, got "
            f"{tuple(lower.shape)} and {tuple(upper.shape)}")
    return box_corners(lower, upper, valid=valid)


def sample_active_component(regions, generator, n_adv=2, hinge_fn: Hinge = None,
                            step_size=None) -> RegionSample:
    """One point per row in a random active non-empty box of ``regions`` (``(B, K, d)``)."""
    _generator(generator)
    _count("n_adv", n_adv, 0)
    cen, w, active = regions.cen, regions.w, regions.active
    if cen.ndim != 3 or w.shape != cen.shape or active.shape != cen.shape[:2]:
        raise ValueError(f"Invalid Regions shapes: cen={tuple(cen.shape)}, w={tuple(w.shape)}, "
                         f"active={tuple(active.shape)}")
    return RegionSampler(1, n_adv, step_size=step_size, doubled_gradient=True).one(
        regions, generator, hinge_fn, n_adv)


def sample_component_intersection(regions_a, regions_b, generator, n_adv=2,
                                  hinge_fn: Hinge = None, step_size=None) -> RegionSample:
    """One point per row in a random non-empty intersection of an active box of
    ``regions_a`` and one of ``regions_b``."""
    _generator(generator)
    from mowl.representation.library.other_regions import MaskedUnionOfBoxes
    if regions_a.cen.ndim != 3 or regions_b.cen.ndim != 3 or \
            regions_a.cen.shape[0] != regions_b.cen.shape[0] or \
            regions_a.cen.shape[2] != regions_b.cen.shape[2]:
        raise ValueError(f"Incompatible region shapes {tuple(regions_a.cen.shape)} and "
                         f"{tuple(regions_b.cen.shape)}")
    both, _ = MaskedUnionOfBoxes.of(regions_a).intersect(MaskedUnionOfBoxes.of(regions_b))
    return sample_active_component(both, generator, n_adv, hinge_fn, step_size)


def _sampler(n_samples, n_adv, step_size, include_corners):
    _count("n_samples", n_samples, 1)
    _count("n_adv", n_adv, 0)
    return RegionSampler(n_samples, n_adv, include_corners, step_size, doubled_gradient=True)


def sample_class_region(module, class_ids: Ids, generator, n_samples=1, n_adv=2,
                        hinge_fn: Hinge = None, step_size=None,
                        include_corners=True) -> RegionSample:
    """``n_samples`` points (sample-major) of each class ``class_ids`` (the class model)."""
    sampler = _sampler(n_samples, n_adv, step_size, include_corners)
    idx = index_tensor(class_ids, module.nb_classes, "class_ids", module.elements.device)
    return sampler.classes(module.class_model, idx, _generator(generator), hinge_fn)


def sample_pair_region(module, class_a_ids: Ids, class_b_ids: Ids, generator, n_samples=1,
                       n_adv=2, hinge_fn: Hinge = None, step_size=None,
                       include_corners=True) -> RegionSample:
    """``n_samples`` points of each intersection of ``class_a_ids`` and ``class_b_ids``."""
    sampler = _sampler(n_samples, n_adv, step_size, include_corners)
    device = module.elements.device
    a = index_tensor(class_a_ids, module.nb_classes, "class_a_ids", device)
    b = index_tensor(class_b_ids, module.nb_classes, "class_b_ids", device)
    if a.shape != b.shape:
        raise ValueError(f"class_a_ids and class_b_ids must have the same shape, got "
                         f"{tuple(a.shape)} vs {tuple(b.shape)}")
    return sampler.conjunction(module.class_model, a, b, _generator(generator), hinge_fn)


def sample_role_region(module, rel_ids: Ids, generator, n_samples=1, n_adv=2,
                       hinge_fn: Hinge = None, step_size=None, include_corners=True):
    """Points of the regions of pairs of the relations ``rel_ids`` (the relation model):
    ``(heads, tails)`` for a relation of products of boxes, the translations for a relation
    of translations."""
    _generator(generator)
    sampler = _sampler(n_samples, n_adv, step_size, include_corners)
    relations = module.relation_model
    idx = index_tensor(rel_ids, module.nb_rels, "rel_ids", module.elements.device)
    sample = sampler.pairs(relations, idx, generator, hinge_fn)
    if getattr(relations, "mode", None) == "translation":
        return sample
    halves = [relations.split_pair(t) for t in (sample.points, sample.lower, sample.upper)]
    return tuple(RegionSample(halves[0][i], sample.valid, halves[1][i], halves[2][i])
                 for i in (0, 1))


def sample_exists_region(module, rel_ids: Ids, class_ids: Ids, generator, n_samples=1,
                         n_adv=2, hinge_fn: Hinge = None, step_size=None,
                         include_corners=True) -> RegionSample:
    """Points of :math:`\\exists r.B` (needs a relation model of translations)."""
    _generator(generator)
    sampler = _sampler(n_samples, n_adv, step_size, include_corners)
    relations = module._translation_roles("exists_box")
    r_idx, c_idx = relations.align_with_classes(rel_ids, class_ids, module.nb_classes)
    return sampler.exists(relations, module.class_model, r_idx, c_idx, generator, hinge_fn)
