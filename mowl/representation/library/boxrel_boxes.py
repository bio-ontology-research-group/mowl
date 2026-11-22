"""Class models of the boxes group: boxes stored as lower corner and log-width (BoxEL,
BoxSEL). Points are :class:`~mowl.representation.library.shapes_classes.Points` of the
family ``"box"``."""

from dataclasses import dataclass

import torch as th
import torch.nn as nn

from mowl.representation.class_models import Classes
from mowl.representation.methods import register_class_model
from mowl.representation.regions import Box, CornerBox

__all__ = ["MinDeltaBoxes", "FixableBoxes", "WidthBox", "signed_intersection"]


def signed_intersection(a, b):
    """The intersection of two boxes as a :class:`~mowl.representation.regions.Box` whose
    half-widths are *signed*: negative in the coordinates in which the boxes are apart.
    Unlike :meth:`Box.intersect`, the result keeps track of how far apart they are, which
    volume-based measures need; intersecting it again is associative."""
    lower = th.maximum(a.lower, b.lower)
    upper = th.minimum(a.upper, b.upper)
    return Box((lower + upper) / 2, (upper - lower) / 2)


def _uniform(num, dim, low, high):
    table = nn.Embedding(num, dim)
    nn.init.uniform_(table.weight, low, high)
    return table


@dataclass(frozen=True)
class WidthBox(CornerBox):
    """A box that keeps its corners and the width :math:`e^{\\delta}` that its upper corner is
    the sum of, so that the exact-model check can add them in double precision.

    :param width: ``(B, d)``.
    """

    width: th.Tensor = None


@register_class_model("box with lower corner")
class MinDeltaBoxes(Classes):
    """A class is a box stored as its lower corner and the logarithm of its widths
    (BoxEL, Xiong et al. 2022): :math:`[m, m + e^{\\delta}]`. The width is positive by
    construction. The first table (the lower corner) plays the role of the ``centres`` of
    other box models, so that code that looks for ``class_embed`` finds it.

    :param mins, deltas: Existing tables to use instead of new ones.
    :param min_bounds, delta_bounds: Ranges of the uniform initialisation.
    """

    region = Box
    part = "box"
    exact_conjunction = True

    def __init__(self, size=None, embed_dim=None, mins=None, deltas=None,
                 min_bounds=(1e-4, 0.2), delta_bounds=(-0.1, 0.0)):
        super().__init__()
        self.mins = mins if mins is not None else _uniform(size, embed_dim, *min_bounds)
        self.deltas = deltas if deltas is not None else _uniform(
            len(self.mins.weight), self.mins.embedding_dim, *delta_bounds)
        self.centres = self.mins

    def forward(self, ids):
        low, width = self.mins(ids), th.exp(self.deltas(ids))
        half = width / 2
        return WidthBox(low + half, half, low, low + width, width)


@register_class_model("box with fixed classes")
class FixableBoxes(Classes):
    """A class is a box :math:`[c - o, c + o]` with :math:`o = |\\tilde o| + \\epsilon`, and
    some classes are *fixed* (ELK-Em, Singh et al. 2026): to the empty box (a class that is
    unsatisfiable, :math:`\\bot`), which has half-width :math:`-B` in every coordinate, or to
    the full box (:math:`\\top`), half-width :math:`B` with a very large :math:`B`. Fixed
    classes are not trained. An empty box is *inside* every box (see
    :class:`~mowl.representation.library.boxrel_measures.AveragedBounds`).

    :param centres, offsets: Existing tables to use instead of new ones.
    :param bot_mask, top_mask: ``(size,)`` boolean tensors of the fixed classes (shared
        with the module that owns them, if any).
    :param epsilon: Lower bound of the half-widths of trained boxes.
    :param big: :math:`B`.
    """

    region = Box
    part = "box"
    exact_conjunction = True

    def __init__(self, size=None, embed_dim=None, centres=None, offsets=None,
                 bot_mask=None, top_mask=None, epsilon=1e-5, big=1e6):
        super().__init__()
        self.centres = centres if centres is not None else _uniform(size, embed_dim, -0.5, 0.5)
        self.offsets = offsets if offsets is not None else _uniform(
            len(self.centres.weight), self.centres.embedding_dim, -0.1, 0.1)
        n = len(self.centres.weight)
        self.register_buffer("bot_mask", bot_mask if bot_mask is not None
                             else th.zeros(n, dtype=th.bool))
        self.register_buffer("top_mask", top_mask if top_mask is not None
                             else th.zeros(n, dtype=th.bool))
        self.epsilon = epsilon
        self.big = big

    def fix(self, bot_ids=(), top_ids=()):
        """Fix classes to the empty box (``bot_ids``) or the full box (``top_ids``)."""
        self.bot_mask.zero_()
        self.top_mask.zero_()
        self.bot_mask[th.as_tensor(list(bot_ids), dtype=th.long)] = True
        self.top_mask[th.as_tensor(list(top_ids), dtype=th.long)] = True
        if (self.bot_mask & self.top_mask).any():
            raise ValueError("A class cannot be fixed to both the empty and the full box.")

    def forward(self, ids):
        centre = self.centres(ids)
        offset = th.abs(self.offsets(ids)) + self.epsilon
        bot = self.bot_mask[ids].unsqueeze(-1)
        top = self.top_mask[ids].unsqueeze(-1)
        big = th.full_like(centre, self.big)
        offset = th.where(bot, -big, th.where(top, big, offset))
        centre = th.where(bot | top, th.zeros_like(centre), centre)
        return CornerBox(centre, offset, centre - offset, centre + offset)
