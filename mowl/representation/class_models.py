"""Class models: what a class is. A class model is a table of parameters and the rule that
turns a row into a region (:mod:`mowl.representation.regions`). The same classes serve as
individual models when individuals are small regions of the same kind.
"""

import torch as th
import torch.nn as nn

from mowl.representation.regions import Ball, Box

__all__ = ["Classes", "BallClasses", "BoxClasses"]


def _unit_rows(table):
    with th.no_grad():
        nn.init.uniform_(table.weight, a=-1, b=1)
        table.weight /= th.linalg.norm(table.weight, dim=1, keepdim=True)
    return table


class Classes(nn.Module):
    """Base class of class models.

    :cvar region: The region class :meth:`forward` returns.
    :cvar part: The vocabulary entry (:class:`mowl.representation.ClassModel` value).
    :cvar exact_conjunction: Whether the regions are closed under intersection.
    """

    region = None
    part = ""
    exact_conjunction = False

    def forward(self, ids):
        """The regions of the classes ``ids`` (``LongTensor (B,)``)."""
        raise NotImplementedError

    def __len__(self):
        return self.centres.num_embeddings

    @property
    def embed_dim(self):
        return self.centres.embedding_dim

    def all(self):
        """The regions of all classes."""
        return self(th.arange(len(self), device=self.centres.weight.device))


class BallClasses(Classes):
    """A class is a ball: a centre and a radius (stored signed, used through its absolute
    value).

    :param size: Number of classes.
    :param embed_dim: Dimension.
    :param centres, radii: Existing tables to use instead of new ones (the tables of
        another module, which then shares its parameters with this one).
    """

    region = Ball
    part = "ball"
    exact_conjunction = False

    def __init__(self, size=None, embed_dim=None, centres=None, radii=None):
        super().__init__()
        self.centres = centres if centres is not None else _unit_rows(
            nn.Embedding(size, embed_dim))
        self.radii = radii if radii is not None else _unit_rows(
            nn.Embedding(len(self), 1))

    def forward(self, ids):
        raw = self.radii(ids)
        return Ball(self.centres(ids), th.abs(raw), raw)


class BoxClasses(Classes):
    """A class is an axis-parallel box: a centre and half-widths (stored signed, used
    through their absolute value).

    :param centres, offsets: Existing tables to use instead of new ones.
    """

    region = Box
    part = "box"
    exact_conjunction = True

    def __init__(self, size=None, embed_dim=None, centres=None, offsets=None):
        super().__init__()
        self.centres = centres if centres is not None else _unit_rows(
            nn.Embedding(size, embed_dim))
        self.offsets = offsets if offsets is not None else _unit_rows(
            nn.Embedding(len(self), self.embed_dim))

    def forward(self, ids):
        return Box(self.centres(ids), th.abs(self.offsets(ids)))
