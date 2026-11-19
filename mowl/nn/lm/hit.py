"""Losses and module of HiT (He, Yuan, Chen, Horrocks, "Language Models as Hierarchy
Encoders", NeurIPS 2024, Section 3).

Entities are embedded by a text encoder; the embeddings are read as points of the Poincaré
ball of curvature :math:`c = 1/d` (radius :math:`\\sqrt{d}`). For a training triplet
:math:`(e, e^+, e^-)` of an entity, a parent and a negative parent:

* hyperbolic clustering loss (Eq. 3):
  :math:`\\max(d_c(e, e^+) - d_c(e, e^-) + \\alpha, 0)`,
* hyperbolic centripetal loss (Eq. 4): :math:`\\max(\\lVert e^+\\rVert_c -
  \\lVert e\\rVert_c + \\beta, 0)`,
* :math:`L_{HiT} = L_{cluster} + L_{centri}` (Eq. 5),
* subsumption score (Eq. 6): :math:`s(e_1 \\sqsubseteq e_2) = -(d_c(e_1, e_2) +
  \\lambda(\\lVert e_2\\rVert_c - \\lVert e_1\\rVert_c))`.

All functions work on plain tensors and return one value per row.
"""

import torch as th
from torch import nn

from mowl.nn.lm import hyperbolic as hb


def clustering_loss(child, parent, negative, c, margin):
    """Hyperbolic clustering (triplet) loss, HiT Eq. 3, one value per row.

    :param child: Embeddings of the entities :math:`e`, shape ``(n, d)``.
    :param parent: Embeddings of the parents :math:`e^+`, shape ``(n, d)``.
    :param negative: Embeddings of the negative parents :math:`e^-`, shape ``(n, d)``.
    :param c: Curvature parameter.
    :param margin: Distance margin :math:`\\alpha`.
    """
    return th.relu(hb.distance(child, parent, c) - hb.distance(child, negative, c) + margin)


def centripetal_loss(child, parent, c, margin):
    """Hyperbolic centripetal loss, HiT Eq. 4, one value per row: zero iff the parent is
    at least ``margin`` closer to the origin (in hyperbolic norm) than the child."""
    return th.relu(hb.hyperbolic_norm(parent, c) - hb.hyperbolic_norm(child, c) + margin)


def hierarchy_loss(child, parent, negative, c, clustering_margin, centripetal_margin,
                   clustering_weight=1.0, centripetal_weight=1.0):
    """:math:`L_{HiT}` (HiT Eq. 5; OnT's :math:`L_\\prec`, Eq. 5), one value per row:
    ``clustering_weight * clustering + centripetal_weight * centripetal``. The papers use
    unit weights; the reference implementations expose the weights."""
    cluster = clustering_loss(child, parent, negative, c, clustering_margin)
    centri = centripetal_loss(child, parent, c, centripetal_margin)
    return clustering_weight * cluster + centripetal_weight * centri


def subsumption_score(child, parent, c, centri_weight):
    """HiT Eq. 6 / OnT Eq. 8: :math:`-(d_c(x, y) + \\lambda(\\lVert y\\rVert_c -
    \\lVert x\\rVert_c))` for :math:`x \\sqsubseteq y`; higher means more likely.
    Broadcasts over leading dimensions."""
    norm_gap = hb.hyperbolic_norm(parent, c) - hb.hyperbolic_norm(child, c)
    return -(hb.distance(child, parent, c) + centri_weight * norm_gap)


class HiTModule(nn.Module):
    """A text encoder whose outputs are points of a Poincaré ball, with the HiT losses.

    :param encoder: An encoder module (see :mod:`mowl.nn.lm.encoders`).
    :param curvature: Curvature parameter :math:`c`; default :math:`1/d` (HiT, Section 3).
    :param clustering_margin: :math:`\\alpha` (Eq. 3).
    :param centripetal_margin: :math:`\\beta` (Eq. 4).
    :param clustering_weight: Weight of the clustering loss (1 in the paper).
    :param centripetal_weight: Weight of the centripetal loss (1 in the paper).
    """

    def __init__(self, encoder, curvature=None, clustering_margin=5.0, centripetal_margin=0.1,
                 clustering_weight=1.0, centripetal_weight=1.0):
        super().__init__()
        self.encoder = encoder
        self.dim = int(encoder.dim)
        self.c = float(curvature) if curvature is not None else hb.curvature_for_dim(self.dim)
        self.clustering_margin = clustering_margin
        self.centripetal_margin = centripetal_margin
        self.clustering_weight = clustering_weight
        self.centripetal_weight = centripetal_weight

    def embed(self, texts):
        """Encode a list of texts into points of the ball, shape ``(len(texts), d)``."""
        return self.encoder(list(texts))

    def hierarchy_loss(self, child, parent, negative):
        """Per-row :math:`L_\\prec` on embeddings (see :func:`hierarchy_loss`)."""
        return hierarchy_loss(child, parent, negative, self.c, self.clustering_margin,
                              self.centripetal_margin, self.clustering_weight,
                              self.centripetal_weight)

    def forward(self, child_texts, parent_texts, negative_texts):
        """Per-row HiT loss of triplets of texts."""
        return self.hierarchy_loss(self.embed(child_texts), self.embed(parent_texts),
                                   self.embed(negative_texts))

    def score(self, child, parent, centri_weight):
        """Subsumption score (Eq. 6) of embeddings; higher is better."""
        return subsumption_score(child, parent, self.c, centri_weight)
