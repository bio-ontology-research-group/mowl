"""Regions and class models of the structural reading: unions of boxes with signed
half-widths and inactive slots, and the wrapper "defined classes are intersections".

Regions
    :class:`SlotBoxes`: ``K`` slots per class, each a box of signed half-width (a negative
    half-width in a coordinate means the slot is empty along it) and an ``active`` flag.
    The intersection of two of them is the exact set of pairs (:func:`pair_slots` of the
    structural module), truncated to the ``K`` slots of the first argument.

Class models
    :class:`SlotClasses`: a union of ``k`` boxes per class padded with inactive slots to
    ``K``. :class:`DefinedClasses`: a wrapper that works with **any** class model whose
    regions intersect: a class that a definition table says is :math:`A \\equiv X \\sqcap Y`
    has the intersection of the regions of :math:`X` and :math:`Y` (recursively) instead of
    its own row; the row stays where a measure needs one of its own (the bump vector of
    Box2EL, the triangle of a prism is *not* kept: the whole region is the intersection).
"""

import inspect
from dataclasses import dataclass, fields, replace

import torch as th

from mowl.nn.el.structural import Regions, nonempty_measure, pair_slots
from mowl.representation.class_models import Classes
from mowl.representation.library.multibox_parts import UnionClasses, UnionOfBoxes
from mowl.representation.methods import register_class_model
from mowl.representation.regions import Box, ConstructorUnavailable

__all__ = ["SlotBoxes", "to_slots", "slots", "intersect_signed", "SlotClasses",
           "DefinedClasses"]

#: The gap of two regions without a single pair of active slots.
_BIG = 1e6


@dataclass(frozen=True)
class SlotBoxes(UnionOfBoxes):
    """A union of ``K`` boxes per row with signed half-widths ``w`` and an ``active`` mask
    (the regions of :class:`~mowl.nn.MultiBoxELModule` with the structural objective).

    It is a :class:`~mowl.representation.library.multibox_parts.UnionOfBoxes` whose
    ``offset`` is the absolute value of ``w``, so every measure of unions of boxes reads it
    (without the mask); the measures that know the mask
    (:class:`~mowl.representation.library.structural_measure.SlotMeasure`) read ``gate``.

    :param centre: ``(B, K, d)``.
    :param offset: ``(B, K, d)``, ``abs(w)``.
    :param bump: ``(B, d)`` or ``None``: the bump vector of the class (defined classes keep
        the one of their own row).
    :param w: ``(B, K, d)``, signed half-widths.
    :param active: ``(B, K)``, boolean.
    """

    w: th.Tensor = None
    active: th.Tensor = None

    @classmethod
    def make(cls, centre, w, active, bump=None):
        return cls(centre, th.abs(w), bump, w, active)

    @property
    def gate(self):
        """``(B, K)``: active and with positive extent in every coordinate (the slots that
        the structural losses take into account)."""
        return self.active & (nonempty_measure(self.w) > 0.0)

    @property
    def slots(self):
        """The structural module's :class:`~mowl.nn.el.structural.Regions`."""
        return Regions(cen=self.centre, w=self.w, active=self.active)

    @property
    def signed_offset(self):
        """``(B, K, d)``: the signed half-widths, with every inactive slot read as an empty
        box (half-width -1). The region is the union of these boxes: a box of negative
        half-width in some coordinate is empty."""
        return th.where(self.active.unsqueeze(-1), self.w, -th.ones_like(self.w))

    def contains(self, points):
        """Whether the points ``(P, d)`` lie in an active slot of non-negative half-width
        in every coordinate, ``(B, P)``. (The inherited union test would read the inactive
        slots, whose ``offset`` is ``abs(w)``, as boxes.)"""
        batch, k, d = self.centre.shape
        boxes = Box(self.centre.reshape(-1, d), self.signed_offset.reshape(-1, d))
        return boxes.contains(points).reshape(batch, k, -1).any(dim=1)

    def is_empty(self):
        """No active slot has a non-negative half-width in every coordinate."""
        return ~(self.active & (self.w >= 0).all(dim=-1)).any(dim=-1)

    def scale(self, factor):
        w = self.w * factor.abs().unsqueeze(1)
        return SlotBoxes.make(self.centre * factor.unsqueeze(1), w, self.active, self.bump)

    def translate(self, vector):
        centre = th.where(self.active.unsqueeze(-1), self.centre + vector.unsqueeze(1),
                          self.centre)
        return replace(self, centre=centre)

    def intersect(self, other, signed=True):
        """The exact set intersection: all pairs of slots, an intersection slot active iff
        both are, its half-width signed (negative where the slots are apart), the active
        ones first and at most as many as ``self`` has slots. The gap is zero iff some
        pair of active slots intersects (and large where there is none)."""
        a, b = self.slots, to_slots(other).slots
        out = pair_slots(a, b, self.centre.shape[1])
        pair_active = (a.active.unsqueeze(2) & b.active.unsqueeze(1))
        lo = th.maximum((a.cen - a.w).unsqueeze(2), (b.cen - b.w).unsqueeze(1))
        hi = th.minimum((a.cen + a.w).unsqueeze(2), (b.cen + b.w).unsqueeze(1))
        gap = th.linalg.norm(th.relu(lo - hi), dim=-1).flatten(1)
        gap = th.where(pair_active.flatten(1), gap, th.full_like(gap, _BIG))
        gap = gap.amin(dim=1, keepdim=True)
        meets = (out.active & (nonempty_measure(out.w) > 0)).any(dim=1, keepdim=True)
        gap = th.where(meets, th.zeros_like(gap), gap)
        return SlotBoxes.make(out.cen, out.w, out.active), gap


def to_slots(region):
    """A box, a union of boxes or slots as :class:`SlotBoxes` (all slots active)."""
    if isinstance(region, SlotBoxes):
        return region
    if isinstance(region, UnionOfBoxes):
        return SlotBoxes.make(region.centre, region.offset,
                              th.ones(region.centre.shape[:2], dtype=th.bool,
                                      device=region.centre.device), region.bump)
    if isinstance(region, Box):
        return SlotBoxes.make(region.centre.unsqueeze(1), region.offset.unsqueeze(1),
                              th.ones(region.centre.shape[0], 1, dtype=th.bool,
                                      device=region.centre.device),
                              getattr(region, "bump", None))
    raise TypeError(f"{type(region).__name__} is not a box, a union of boxes or slots.")


def slots(region):
    """The :class:`~mowl.nn.el.structural.Regions` of a box, union of boxes or slots."""
    return to_slots(region).slots


def intersect_signed(a, b):
    """``(region, gap)``: the intersection of two regions with *signed* extent where the
    family has it (the half-widths are negative where the regions are apart: a box, a prism,
    slots), else the intersection of the family."""
    intersect = a.intersect
    if "signed" in inspect.signature(intersect).parameters:
        return intersect(b, signed=True)
    return intersect(b)


# ====================================================================================
# class models
# ====================================================================================

@register_class_model("union of boxes with slots")
class SlotClasses(UnionClasses):
    """A class is a union of ``k`` boxes in ``max_components`` slots: the others are
    inactive (centre 0, half-width -1), so that a defined class, whose intersection has up to
    :math:`k^2` boxes, has room. Tables as :class:`UnionClasses` (and the same objects when
    given); the region is :class:`SlotBoxes`.

    :param components: ``k``.
    :param max_components: ``K``, default :math:`k^2` (as the module).

    Part (vocabulary value): ``union of boxes``. Used by: ``multiboxel(reading=structural)``.
    """

    region = SlotBoxes
    part = "union of boxes"
    #: The intersection of two classes keeps every pair of slots (it raises where the
    #: pairs do not fit into ``max_components`` slots), so the conjunction is exact.
    exact_conjunction = True

    def __init__(self, size=None, embed_dim=None, components=2, max_components=None,
                 bump=True, centres=None, offsets=None, bumps=None):
        super().__init__(size, embed_dim, components=components, bump=bump, centres=centres,
                         offsets=offsets, bumps=bumps)
        self.max_components = int(components ** 2 if max_components is None
                                  else max_components)
        if self.max_components < self.components:
            raise ValueError("max_components must be at least components.")

    def forward(self, ids, **unused):
        union = super().forward(ids)
        batch, k, dim = union.centre.shape
        pad = self.max_components - k
        centre, w = union.centre, union.offset
        active = th.ones(batch, k, dtype=th.bool, device=centre.device)
        if pad:
            centre = th.cat([centre, centre.new_zeros(batch, pad, dim)], dim=1)
            w = th.cat([w, -th.ones(batch, pad, dim, dtype=w.dtype, device=w.device)], dim=1)
            active = th.cat([active, th.zeros(batch, pad, dtype=th.bool,
                                              device=centre.device)], dim=1)
        return SlotBoxes.make(centre, w, active, union.bump)


class DefinedClasses(Classes):
    """A class model wrapper: *defined classes are the intersections of their definers*.

    ``base`` is any class model whose regions intersect (boxes, slots of boxes, prisms,
    products of octagons, ...). For a class :math:`A` with a definition
    :math:`A \\equiv X \\sqcap Y` the region is the intersection of the regions of :math:`X`
    and :math:`Y` (those are resolved the same way, so definitions may be nested), with
    signed extent where the family has it (a defined class can be empty, and says by how
    much). Fields of the region that the intersection does not produce (the bump vector of a
    class, the centre of a point-like region) stay those of the class's own row.

    The definitions are the :class:`~mowl.nn.el.structural.DefinitionTable` of the
    structural module, or a mapping ``{A: (X, Y)}`` (acyclic). Rows that *state* a
    definition (:math:`A \\sqsubseteq X`, :math:`A \\sqsubseteq Y`,
    :math:`X \\sqcap Y \\sqsubseteq A`) are true by construction:
    :meth:`masked_gci0` and :meth:`masked_gci1` mark them, for objectives that skip them.

    Everything else (``tables()``, ``centres``, ``bumps``, ...) is the base model's: the
    wrapper has no parameters of its own.

    :param base: The class model.
    :param table: A ``DefinitionTable``, or
    :param definitions: a mapping ``{defined class id: (X id, Y id)}``.
    """

    def __init__(self, base, table=None, definitions=None):
        super().__init__()
        self.base = base
        self.region = base.region
        self.part = base.part
        # the definition is the exact intersection; it is as exact as the base's intersection
        self.exact_conjunction = bool(getattr(base, "exact_conjunction", False))
        self.table = table
        n = len(base)
        if table is not None:
            mask, x, y = table.def_mask, table.def_x, table.def_y
        else:
            mask = th.zeros(n, dtype=th.bool)
            x = th.full((n,), -1, dtype=th.long)
            y = th.full((n,), -1, dtype=th.long)
            for a, (p, q) in (definitions or {}).items():
                mask[a], x[a], y[a] = True, p, q
        self.register_buffer("def_mask", mask.clone().bool())
        self.register_buffer("def_x", x.clone().long())
        self.register_buffer("def_y", y.clone().long())

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base":
                raise
            return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    @property
    def embed_dim(self):
        return self.base.embed_dim

    @property
    def n_defined(self):
        return int(self.def_mask.sum())

    def tables(self):
        return self.base.tables()

    def param_groups(self):
        return self.base.param_groups()

    def primitive_mask(self, exclude=None):
        """``(N,)`` boolean: the classes that are neither defined nor ``exclude`` (the
        top class)."""
        mask = ~self.def_mask
        if exclude is not None and 0 <= exclude < mask.shape[0]:
            mask = mask.clone()
            mask[exclude] = False
        return mask

    def forward(self, ids, **options):
        region = self.base(ids, **options)
        is_defined = self.def_mask[ids]
        if not bool(is_defined.any()):
            return region
        rows = th.nonzero(is_defined, as_tuple=False).squeeze(1)
        chosen = ids[rows]
        rx = self(self.def_x[chosen], **options)
        ry = self(self.def_y[chosen], **options)
        inter, _ = intersect_signed(rx, ry)
        return _replace_rows(region, rows, inter)

    # -- the rows that state a definition ------------------------------------------------
    def masked_gci0(self, rows):
        """``(B,)`` boolean: ``gci0`` rows :math:`A \\sqsubseteq X`, :math:`A \\sqsubseteq Y`
        of a definition."""
        if self.table is not None:
            return self.table.masked_gci0(rows)
        rows = rows.to(self.def_x.device)
        a, c = rows[:, 0], rows[:, 1]
        return self.def_mask[a] & ((self.def_x[a] == c) | (self.def_y[a] == c))

    def masked_gci1(self, rows):
        """``(B,)`` boolean: ``gci1`` rows :math:`X \\sqcap Y \\sqsubseteq A` of a definition
        (in either order)."""
        if self.table is not None:
            return self.table.masked_gci1(rows)
        rows = rows.to(self.def_x.device)
        a = rows[:, 2]
        both = ((self.def_x[a] == rows[:, 0]) & (self.def_y[a] == rows[:, 1])) | \
            ((self.def_x[a] == rows[:, 1]) & (self.def_y[a] == rows[:, 0]))
        return self.def_mask[a] & both


def _replace_rows(region, rows, new):
    """``region`` with the rows ``rows`` replaced by those of ``new``, field by field (the
    tensors ``new`` has)."""
    updates = {}
    for f in fields(new):
        replacement = getattr(new, f.name)
        current = getattr(region, f.name, None)
        if not isinstance(replacement, th.Tensor) or current is None:
            continue
        if current.shape[1:] != replacement.shape[1:]:
            raise ConstructorUnavailable(
                f"The intersection of two {type(new).__name__} regions has shape "
                f"{tuple(replacement.shape[1:])} per class, the class model's own "
                f"{tuple(current.shape[1:])}: definitions need a class model whose "
                f"intersection keeps the shape (a box, a prism, slots of boxes: "
                f"SlotClasses).")
        merged = current.clone()
        merged[rows] = replacement
        updates[f.name] = merged
    if isinstance(region, SlotBoxes) and "w" in updates:
        updates["offset"] = th.abs(updates["w"])
    return replace(region, **updates)
