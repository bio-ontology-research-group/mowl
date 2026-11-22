"""The measure of unions of boxes that knows which slots count.

:class:`SlotMeasure` is :class:`~mowl.representation.library.multibox_parts.UnionMeasure`
(hard aggregation) for regions with a ``gate`` (:class:`SlotBoxes`: active and non-empty
slots): a slot that does not count is neither a source (it adds nothing to the mean over the
slots of the first argument) nor a target (it is not one of the candidates of the minimum).
A row without any counting target costs zero, as in the structural module (an unsatisfiable
target makes the inclusion vacuous; its non-emptiness is the existence penalty's business).
Boxes and unions without a mask count in every slot.
"""

import torch as th

from mowl.representation.library.multibox_parts import UnionMeasure, UnionOfBoxes, _union
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


@register_measure("bounds per coordinate over slots")
class SlotMeasure(UnionMeasure):
    """:class:`UnionMeasure` with masks (hard aggregation only).

    * ``inside(A, B)``: the mean over the counting slots of ``A`` of the minimum over the
      counting slots of ``B`` of the cost of the pair (zero where ``B`` has none).
    * ``inside_exists``: the minimum over the counting slots of ``A`` (zero where none).
    * ``meets``: as ``inside`` with the cost ``meets`` of the base measure.

    ``apart``, ``empty``, ``nonempty``, ``meets_all`` and ``intersection_inside`` are those of
    :class:`UnionMeasure` and do not read the mask: the structural reading does not use them
    on slots (see :class:`~mowl.representation.library.structural_reading.StructuralReading`).
    """

    accepts = (UnionOfBoxes, Box)

    def __init__(self, base=None):
        super().__init__(base if base is not None else Bounds(), "hard", 1.0)

    def _masked_costs(self, function, a, b, margin):
        costs = self._pairs(function, a, b, margin)
        gate_a, gate_b = gate_of(a), gate_of(b)
        inf = float("inf")
        per = costs.masked_fill(~gate_b.unsqueeze(1), inf).amin(dim=2)
        per = th.where(gate_b.any(dim=1, keepdim=True), per, th.zeros_like(per))
        return per, gate_a

    def inside(self, a, b, margin=0.0):
        per, gate = self._masked_costs(self.base.inside, a, b, margin)
        total = th.where(gate, per, th.zeros_like(per)).sum(dim=1, keepdim=True)
        return total / gate.sum(dim=1, keepdim=True).clamp_min(1)

    def meets(self, a, b, margin=0.0):
        per, gate = self._masked_costs(self.base.meets, a, b, margin)
        total = th.where(gate, per, th.zeros_like(per)).sum(dim=1, keepdim=True)
        return total / gate.sum(dim=1, keepdim=True).clamp_min(1)

    def inside_exists(self, a, b, margin=0.0):
        per, gate = self._masked_costs(self.base.inside, a, b, margin)
        least = per.masked_fill(~gate, float("inf")).amin(dim=1, keepdim=True)
        return th.where(gate.any(dim=1, keepdim=True), least, th.zeros_like(least))

    def meets_each(self, a, b, margin=0.0):
        per, _ = self._masked_costs(self.base.meets, a, b, margin)
        return per
