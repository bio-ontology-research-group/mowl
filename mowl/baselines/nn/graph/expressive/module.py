r"""ExpressivE [pavlovic2023]_: relations as hyper-parallelograms in the virtual triple space.

An entity is a point ``e`` in :math:`R^d`. A relation has, for each argument position
``p`` in ``{h, t}``, a slope vector ``s^p``, a centre ``c^p`` and a non-negative width
``d^p``. A triple ``r(h, t)`` is *captured* when

.. math::

    |e_h - c^h - s^t \circ e_t| \preceq d^h \quad\text{and}\quad
    |e_t - c^t - s^h \circ e_h| \preceq d^t,

that is, when the point :math:`(e_h \Vert e_t)` of the virtual triple space
:math:`R^{2d}` lies in the hyper-parallelogram of ``r``. The score is the negative norm of
the distance of Equation 3 of the paper, which is small inside the parallelogram.

This module follows the authors' code (https://github.com/AleksVap/ExpressivE): ``tanh``
on entities, centres, widths, and the minimal denominator ``min_denom`` that bounds
:math:`|1 - s^h s^t|` from below. Only the base model is implemented (not the ablations
Functional, EqSlopes, NoCenter, OneBand).

The model is a PyKEEN :class:`pykeen.models.ERModel`, so it is trained by the existing
``GraphPlusPyKEENModel`` machinery.
"""

import torch as th
from pykeen.losses import NSSALoss
from pykeen.models import ERModel
from mowl.nn.graph.kge_interactions import (ExpressivEInteraction,  # noqa: F401
    captured, expressive_distance, expressive_relation_parameters)


class ExpressivE(ERModel):
    """ExpressivE as a PyKEEN model: entities have dimension ``embedding_dim``, relations
    ``6 * embedding_dim`` (``d_h, d_t, c_h, c_t, s_h, s_t``). The default loss is the
    self-adversarial negative sampling loss with margin 3 and temperature 2, as in the
    authors' code."""

    loss_default = NSSALoss
    loss_default_kwargs = dict(margin=3, adversarial_temperature=2.0, reduction="sum")

    def __init__(self, embedding_dim=50, p=2, min_denom=0.5, tanh_map=True, **kwargs):
        super().__init__(
            interaction=ExpressivEInteraction(p=p, tanh_map=tanh_map, min_denom=min_denom),
            entity_representations_kwargs=dict(shape=embedding_dim),
            relation_representations_kwargs=dict(shape=6 * embedding_dim),
            **kwargs)
