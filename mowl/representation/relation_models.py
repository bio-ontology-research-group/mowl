"""Relation models: what a relation is. A relation model maps regions to regions:
:meth:`~Relations.image` gives the things related *from* a region,
:meth:`~Relations.preimage` the things related *to* it, which is the region of
:math:`\\exists r.C`.
"""

import torch as th
import torch.nn as nn

from mowl.representation.regions import ConstructorUnavailable

__all__ = ["Relations", "Translation"]


class Relations(nn.Module):
    """Base class of relation models.

    :cvar part: The vocabulary entry (:class:`mowl.representation.RelationModel` value).
    """

    part = ""

    def image(self, ids, region):
        """:math:`\\{y : (x, y) \\in r, x \\in \\text{region}\\}` for the relations
        ``ids``."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no image.")

    def preimage(self, ids, region):
        """:math:`\\{x : (x, y) \\in r, y \\in \\text{region}\\}`: the region of
        :math:`\\exists r.C`."""
        raise ConstructorUnavailable(f"{type(self).__name__} has no preimage.")

    def __len__(self):
        return self.vectors.num_embeddings


class Translation(Relations):
    """A relation is one shift vector: :math:`(x, y) \\in r` iff :math:`y = x + v_r`
    (TransE; ELEmbeddings, ELBE and many others). A function: every point has exactly one
    successor, so one-to-many relations cannot be represented.

    :param vectors: An existing table to use instead of a new one.
    """

    part = "translation"

    def __init__(self, size=None, embed_dim=None, vectors=None):
        super().__init__()
        if vectors is None:
            vectors = nn.Embedding(size, embed_dim)
            with th.no_grad():
                nn.init.uniform_(vectors.weight, a=-1, b=1)
                vectors.weight /= th.linalg.norm(vectors.weight, dim=1, keepdim=True)
        self.vectors = vectors

    def image(self, ids, region):
        return region.translate(self.vectors(ids))

    def preimage(self, ids, region):
        return region.translate(-self.vectors(ids))
