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
from pykeen.nn.modules import Interaction


def _positive_sign(x):
    s = th.sign(x)
    return th.where(s == 0, th.ones_like(s), s)


def expressive_relation_parameters(r, tanh_map=True, min_denom=0.5):
    """Split a raw relation vector of size ``6 d`` into
    ``(d_h, d_t, c_h, c_t, s_h, s_t)`` and apply the constraints of the authors' code.

    Widths are made non-negative (``abs`` then ``tanh``, or ``relu`` without ``tanh``),
    centres go through ``tanh``, and the head slope is shifted so that
    :math:`|1 - s^h s^t| \\ge` ``min_denom``.
    """
    d_h, d_t, c_h, c_t, s_h, s_t = r.tensor_split(6, dim=-1)
    d_h, d_t = d_h.abs(), d_t.abs()
    if tanh_map:
        d_h, d_t, c_h, c_t = th.tanh(d_h), th.tanh(d_t), th.tanh(c_h), th.tanh(c_t)
    else:
        d_h, d_t = th.relu(d_h), th.relu(d_t)
    # the tail slope must not be 0: it divides below
    s_t = s_t + _positive_sign(s_t) * 1e-4
    denominator = 1 - s_h * s_t
    adjusted = denominator - _positive_sign(denominator) * min_denom
    s_h = th.where(denominator.abs() < min_denom, s_h + adjusted / s_t, s_h)
    return d_h, d_t, c_h, c_t, s_h, s_t


def _contextualised(h, t, d_h, d_t, c_h, c_t, s_h, s_t):
    d = th.cat((d_h, d_t), dim=-1)
    c = th.cat((c_h, c_t), dim=-1)
    s = th.cat((s_t, s_h), dim=-1)
    h, t = th.broadcast_tensors(h, t)
    ht = th.cat((h, t), dim=-1)
    th_ = th.cat((t, h), dim=-1)
    return (ht - c - s * th_).abs(), d


def captured(h, t, d_h, d_t, c_h, c_t, s_h, s_t):
    """``True`` where the entity pair lies inside the hyper-parallelogram (both
    inequalities of the paper hold in every dimension). Inputs are already constrained
    (see :func:`expressive_relation_parameters`) and entities already mapped by tanh."""
    tau, d = _contextualised(h, t, d_h, d_t, c_h, c_t, s_h, s_t)
    return (tau <= d).all(dim=-1)


def expressive_distance(h, t, d_h, d_t, c_h, c_t, s_h, s_t):
    """The distance :math:`D(h, r, t)` of Equation 3 of the paper, shape ``(..., 2d)``."""
    tau, d = _contextualised(h, t, d_h, d_t, c_h, c_t, s_h, s_t)
    inside = (tau <= d).all(dim=-1, keepdim=True)
    w = 2 * d + 1
    k = 0.5 * (w - 1) * (w - 1 / w)
    return th.where(inside, tau / w, tau * w - k)


class ExpressivEInteraction(Interaction):
    """The interaction function of ExpressivE: ``-||D(h, r, t)||_p``.

    :param p: Norm of the score.
    :param tanh_map: Map entities, centres and widths into ``[-1, 1]`` with ``tanh``.
    :param min_denom: Lower bound of :math:`|1 - s^h s^t|` (``Dmin`` of the paper).
    """

    relation_shape = ("e",)
    entity_shape = ("d",)

    def __init__(self, p=2, tanh_map=True, min_denom=0.5):
        super().__init__()
        self.p = p
        self.tanh_map = tanh_map
        self.min_denom = min_denom

    def forward(self, h, r, t):
        params = expressive_relation_parameters(r, self.tanh_map, self.min_denom)
        if self.tanh_map:
            h, t = th.tanh(h), th.tanh(t)
        return -expressive_distance(h, t, *params).norm(p=self.p, dim=-1)


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
