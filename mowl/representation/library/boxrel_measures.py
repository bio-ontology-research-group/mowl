"""Measures of the boxes group: a (smoothed) volume ratio (BoxEL, BoxSEL), bounds averaged
over the coordinates (ELK-Em) and bounds that skip the empty coordinates of an
intersection (TransBox)."""

import torch as th
import torch.nn.functional as F

from mowl.representation.library.boxrel_boxes import signed_intersection
from mowl.representation.measures import Bounds
from mowl.representation.methods import register_measure

__all__ = ["VolumeRatio", "AveragedBounds", "MaskedBounds"]


@register_measure("smoothed volume ratio")
class VolumeRatio(Bounds):
    """Boxes compared by volume: :math:`A \\subseteq B` costs
    :math:`1 - \\mathrm{Vol}(A \\cap B) / \\mathrm{Vol}(A)`, the share of :math:`A` outside
    :math:`B`; :math:`A \\cap B = \\emptyset` costs the share inside. Zero iff the relation
    holds when the volume is exact (``temperature=None``); with the softplus volume of
    BoxEL (``temperature`` :math:`\\beta`) the cost of ``apart`` is never exactly zero and
    approaches the exact one as :math:`\\beta` grows.

    A ratio of volumes cannot exceed one, so a positive (strict) margin has no meaning and
    is ignored; a negative margin is a slack: ``inside`` is free once the ratio reaches
    :math:`1 - |\\gamma|`, and ``apart`` once it falls below :math:`|\\gamma|`. ``meets``,
    ``empty`` and ``nonempty`` are those of :class:`~mowl.representation.Bounds` (they are
    about extent, not volume). The intersection of two boxes in ``intersection_inside`` is
    kept with its signed widths, which the smoothed volume prices, so no separate
    non-emptiness term is added.

    :param temperature: :math:`\\beta` of the softplus volume
        :math:`\\mathrm{softplus}_\\beta(\\text{width})`; ``None`` for the exact volume
        :math:`\\max(0, \\text{width})`.
    :param floor, ceiling: The volumes are clipped to this range before their logarithm
        is taken (as in the BoxEL code).
    """

    part = "smoothed volume ratio"

    def __init__(self, temperature=1.0, floor=1e-10, ceiling=1e4):
        super().__init__()
        self.temperature = temperature
        self.floor = floor
        self.ceiling = ceiling
        self.part = "volume ratio" if temperature is None else "smoothed volume ratio"

    # -- volumes ----------------------------------------------------------------------------
    def volume(self, region):
        """Volume of the boxes, ``(B, 1)`` (the product of the widths)."""
        width = region.upper - region.lower
        width = F.relu(width) if self.temperature is None \
            else F.softplus(width, beta=self.temperature)
        return width.prod(1, keepdim=True)

    def _log(self, volume):
        return th.log(th.clamp(volume, self.floor, self.ceiling))

    def ratio(self, a, b):
        """:math:`\\mathrm{Vol}(A \\cap B) / \\mathrm{Vol}(A)` per row, ``(B, 1)``."""
        self.check(a, b)
        log_inter = self._log(self.volume(signed_intersection(a, b)))
        return th.exp(log_inter - self._log(self.volume(a)))

    def product_overlap(self, a, b):
        """:math:`\\mathrm{Vol}(A \\cap B) / (\\mathrm{Vol}(A)\\,\\mathrm{Vol}(B))`: the
        quantity the BoxEL code minimises for :math:`A \\sqcap B \\sqsubseteq \\bot`. It is
        not scale invariant; the reading of disjointness is ``apart``."""
        self.check(a, b)
        log_inter = self._log(self.volume(signed_intersection(a, b)))
        return th.exp(log_inter - self._log(self.volume(a)) - self._log(self.volume(b)))

    # -- relations --------------------------------------------------------------------------
    @staticmethod
    def _slack(margin):
        return max(-float(margin), 0.0)

    def inside(self, a, b, margin=0.0):
        return th.relu(1 - self.ratio(a, b) - self._slack(margin))

    def apart(self, a, b, margin=0.0):
        return th.relu(self.ratio(a, b) - self._slack(margin))

    def intersection_inside(self, a, b, c, margin=0.0):
        return self.inside(signed_intersection(a, b), c, margin)

    def intersection_apart(self, a, b, c, margin=0.0):
        return self.apart(signed_intersection(a, b), c, margin)

    # -- statements about probabilities (BoxSEL) ---------------------------------------------
    def conditional(self, premise, conclusion, lower, upper):
        """Cost of :math:`(D \\mid C)[l, u]` with :math:`C` the ``premise`` and :math:`D`
        the ``conclusion``: :math:`[l\\,\\mathrm{Vol}(C) - \\mathrm{Vol}(D \\cap C)]_+ +
        [\\mathrm{Vol}(D \\cap C) - u\\,\\mathrm{Vol}(C)]_+`, ``(B, 1)``. ``lower`` and ``upper``
        are numbers or one per row. Volumes are unclipped and smoothed as the measure is."""
        self.check(premise, conclusion)
        vc = self.volume(premise)
        vdc = self.volume(signed_intersection(premise, conclusion))
        lower, upper = (th.as_tensor(x, dtype=vc.dtype, device=vc.device)
                        for x in (lower, upper))
        lower, upper = (x.unsqueeze(1) if x.dim() == 1 else x for x in (lower, upper))
        return th.relu(lower * vc - vdc) + th.relu(vdc - upper * vc)

    @th.no_grad()
    def probability(self, premise, conclusion):
        """The proportion of the premise that lies in the conclusion, with exact volumes
        (``nan`` where the premise has no volume), ``(B, 1)``."""
        self.check(premise, conclusion)
        exact = VolumeRatio(temperature=None)
        return exact.volume(signed_intersection(premise, conclusion)) / exact.volume(premise)


@register_measure("averaged bounds")
class AveragedBounds(Bounds):
    """Bounds per coordinate, with the hinge averaged over the coordinates instead of
    taken as a norm, as in ELK-Em: :math:`\\frac1n\\lVert\\max(0, |c_A - c_B| + o_A -
    o_B + \\gamma)\\rVert_1`. A region with a negative half-width in some coordinate is
    *empty* (ELK-Em represents :math:`\\bot` this way) and lies inside everything: its
    ``inside`` cost is zero."""

    part = "bounds per coordinate"

    @staticmethod
    def is_empty(a):
        """``(B, 1)`` boolean: the region has a negative half-width in some coordinate."""
        return (a.lower > a.upper).any(dim=1, keepdim=True)

    @staticmethod
    def _mean(x):
        return th.relu(x).mean(dim=1, keepdim=True)

    def inside(self, a, b, margin=0.0):
        self.check(a, b)
        cost = self._mean(th.abs(a.centre - b.centre) + a.offset - b.offset + margin)
        return th.where(self.is_empty(a), th.zeros_like(cost), cost)

    def meets(self, a, b, margin=0.0):
        self.check(a, b)
        return self._mean(th.abs(a.centre - b.centre) - a.offset - b.offset + margin)

    def empty(self, a, margin=0.0):
        """Zero iff ``a`` is empty (negative half-width, with ``margin`` room)."""
        self.check(a)
        return self._mean(a.offset + margin)


@register_measure("masked bounds")
class MaskedBounds(Bounds):
    """Bounds per coordinate whose intersection treatment is that of TransBox: in
    :math:`C_1 \\sqcap C_2 \\sqsubseteq D` the coordinates in which the two boxes are apart
    are left out of the inclusion and charged as a penalty on the size of the gap.

    :param empty_weight: Factor of that penalty, the norm of the gaps
        (``intersection_inside`` adds it with factor 1 in :class:`Bounds`; mOWL's TransBox
        uses 0.5, the code of its authors 1).
    :param gci1: ``"inclusion"`` (default): the intersection lies inside :math:`D`.
        ``"overlap"``: it only has to overlap :math:`D`, which is what the authors' code
        computes (it uses the signed half-width of the intersection).
    """

    part = "bounds per coordinate"

    def __init__(self, separation="any", empty_weight=1.0, gci1="inclusion"):
        super().__init__(separation)
        if gci1 not in ("inclusion", "overlap"):
            raise ValueError("gci1 must be 'inclusion' or 'overlap'.")
        self.empty_weight = float(empty_weight)
        self.gci1 = gci1

    def intersection_inside(self, a, b, c, margin=0.0):
        self.check(a, b, c)
        start = th.maximum(a.lower, b.lower)
        end = th.minimum(a.upper, b.upper)
        mask = (start <= end).to(start.dtype)
        gap = th.linalg.norm(th.relu(start - end), dim=1, keepdim=True)
        centre, half = (start + end) / 2, (end - start) / 2
        if self.gci1 == "overlap":
            half = -half
        s = (th.abs(centre - c.centre) + half - c.offset) * mask
        return self.empty_weight * gap + self._norm(s + margin * mask)


@register_measure("volume ratio")
def _exact_volume_ratio():
    """The volume ratio with exact (not smoothed) volumes."""
    return VolumeRatio(temperature=None)
