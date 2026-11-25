"""The measure of unions of boxes that knows which slots count.

:class:`SlotMeasure` is :class:`~mowl.representation.library.multibox_parts.UnionMeasure`
(hard aggregation) for :class:`~mowl.representation.library.structural_regions.SlotBoxes`,
in two readings (``reading``):

``"exact"`` (default)
    An inactive slot and a slot of negative half-width are *empty boxes*: the region is the
    union of its non-empty slots and every relation is that of the union, so a class that is
    empty is included in everything and meets nothing, and nothing non-empty is included in
    it. Zero cost exactly when the relation holds (as the other measures of unions of boxes).
``"structural"``
    The reading of the structural objective of MultiBoxEL and Prism (published): a slot that
    does not count is neither a source (it adds nothing to the mean over the slots of the
    first argument) nor a target; a row without any counting target costs zero (an
    unsatisfiable target makes the inclusion vacuous; its non-emptiness is the existence
    penalty's business). Unsound as an inclusion into an empty class: ``multiboxel(reading=
    "structural")`` selects it, as the module has it.

Boxes and unions without a mask count in every slot.
"""

import torch as th

from mowl.representation.library.multibox_parts import UnionMeasure, UnionOfBoxes, _union
from mowl.representation.library.structural_regions import SlotBoxes
from mowl.representation.measures import Bounds
from mowl.representation.methods import register_measure
from mowl.representation.regions import Box

__all__ = ["SlotMeasure", "gate_of"]


def gate_of(region):
    """``(B, K)`` boolean: the slots of ``region`` that count (all of them for a region
    without a mask)."""
    gate = getattr(region, "gate", None)
    if gate is not None:
        return gate
    region = _union(region)
    return th.ones(region.centre.shape[:2], dtype=th.bool, device=region.centre.device)


@register_measure("bounds per coordinate over slots", default_for=(SlotBoxes,))
class SlotMeasure(UnionMeasure):
    """:class:`UnionMeasure` for slots of boxes (hard aggregation only), see the module
    docstring for ``reading``. Part (vocabulary value): ``bounds per coordinate``. Used by:
    ``multiboxel`` with ``reading="structural"`` (the measure's ``reading="structural"``).

    In the structural reading:

    * ``inside(A, B)``: the mean over the counting slots of ``A`` of the minimum over the
      counting slots of ``B`` of the cost of the pair (zero where ``B`` has none).
    * ``inside_exists``: the minimum over the counting slots of ``A`` (zero where none).
    * ``meets``: as ``inside`` with the cost ``meets`` of the base measure.

    ``apart``, ``empty``, ``nonempty``, ``meets_all`` and ``intersection_inside`` do not read
    the mask there (the structural reading does not use them on slots).
    """

    accepts = (UnionOfBoxes, Box)

    def __init__(self, base=None, reading="exact"):
        if reading not in ("exact", "structural"):
            raise ValueError("reading must be 'exact' or 'structural'.")
        super().__init__(base if base is not None else Bounds(), "hard", 1.0)
        self.reading = reading

    @property
    def exact_intersection(self):
        return self.reading == "exact" and getattr(self.base, "conjunction", None) == "exact"

    # -- the exact reading: the union of the non-empty slots -------------------------------
    def _signed(self, region):
        if self.reading == "exact" and isinstance(region, SlotBoxes):
            return UnionOfBoxes(region.centre, region.signed_offset, region.bump)
        return region

    def _masked_costs(self, function, a, b, margin):
        costs = self._pairs(function, a, b, margin)
        gate_a, gate_b = gate_of(a), gate_of(b)
        inf = float("inf")
        per = costs.masked_fill(~gate_b.unsqueeze(1), inf).amin(dim=2)
        per = th.where(gate_b.any(dim=1, keepdim=True), per, th.zeros_like(per))
        return per, gate_a

    def inside(self, a, b, margin=0.0):
        if self.reading == "exact":
            return super().inside(self._signed(a), self._signed(b), margin)
        per, gate = self._masked_costs(self.base.inside, a, b, margin)
        total = th.where(gate, per, th.zeros_like(per)).sum(dim=1, keepdim=True)
        return total / gate.sum(dim=1, keepdim=True).clamp_min(1)

    def meets(self, a, b, margin=0.0):
        if self.reading == "exact":
            return super().meets(self._signed(a), self._signed(b), margin)
        per, gate = self._masked_costs(self.base.meets, a, b, margin)
        total = th.where(gate, per, th.zeros_like(per)).sum(dim=1, keepdim=True)
        return total / gate.sum(dim=1, keepdim=True).clamp_min(1)

    def inside_exists(self, a, b, margin=0.0):
        if self.reading == "exact":
            return super().inside_exists(self._signed(a), self._signed(b), margin)
        per, gate = self._masked_costs(self.base.inside, a, b, margin)
        least = per.masked_fill(~gate, float("inf")).amin(dim=1, keepdim=True)
        return th.where(gate.any(dim=1, keepdim=True), least, th.zeros_like(least))

    def meets_each(self, a, b, margin=0.0):
        if self.reading == "exact":
            return super().meets_each(self._signed(a), self._signed(b), margin)
        per, _ = self._masked_costs(self.base.meets, a, b, margin)
        return per

    def meets_all(self, a, b, margin=0.0):
        return super().meets_all(self._signed(a), self._signed(b), margin)

    def apart(self, a, b, margin=0.0):
        return super().apart(self._signed(a), self._signed(b), margin)

    def empty(self, a, margin=0.0):
        return super().empty(self._signed(a), margin)

    def nonempty(self, a, epsilon=0.0):
        return super().nonempty(self._signed(a), epsilon)

    def intersection_inside(self, a, b, c, margin=0.0):
        return super().intersection_inside(self._signed(a), self._signed(b), self._signed(c),
                                           margin)
