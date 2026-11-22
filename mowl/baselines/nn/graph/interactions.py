"""Interaction functions of the knowledge graph embeddings of a projection (neutral maths).

The scores of ExpressivE (Pavlovic and Sallinger 2023) and RESHUFFLE (Pavlovic et al. 2025) as
functions of tensors, with PyKEEN's :class:`~pykeen.nn.modules.Interaction` interface. They
hold no loss and no training: the parts of :mod:`mowl.representation.library.other_graph`
use them as the score of a relation model, and the direct implementations in
:mod:`mowl.baselines` use them too.

ExpressivE. An entity is a point ``e`` in :math:`R^d`. A relation has, for each argument
position ``p`` in ``{h, t}``, a slope vector ``s^p``, a centre ``c^p`` and a non-negative
width ``d^p``. A triple ``r(h, t)`` is *captured* when

.. math::

    |e_h - c^h - s^t \\circ e_t| \\preceq d^h \\quad\\text{and}\\quad
    |e_t - c^t - s^h \\circ e_h| \\preceq d^t,

that is, when the point :math:`(e_h \\Vert e_t)` of the virtual triple space
:math:`R^{2d}` lies in the hyper-parallelogram of ``r``. The score is the negative norm of
the distance of Equation 3 of the paper, small inside the parallelogram. It follows the
authors' code (https://github.com/AleksVap/ExpressivE): ``tanh`` on entities, centres,
widths, and the minimal denominator ``min_denom`` that bounds :math:`|1 - s^h s^t|` from
below. Only the base model.

RESHUFFLE. Every entity ``a`` has a feature matrix :math:`x_a \\in \\{0, 1\\}^{k \\times l}`;
a relation ``r`` is a matrix :math:`R_r \\in [0, 1]^{l \\times l}` obtained from a raw
:math:`(l + 1) \\times l` parameter by a column-wise softmax whose last row is dropped. A
message passing layer takes the coordinate-wise maximum of an entity and the reshuffled
features of its neighbours (:func:`propagate`); a triple is scored by
:math:`-\\lVert \\mathrm{ReLU}(x_h R_r - x_t) \\rVert_2` (:func:`reshuffle_interaction`).
See :class:`mowl.baselines.nn.graph.reshuffle.Reshuffle` for the differences from the
authors' code.
"""

import torch as th
from pykeen.nn.modules import Interaction

__all__ = ["expressive_relation_parameters", "captured", "expressive_distance",
           "ExpressivEInteraction", "relation_matrices", "random_features", "propagate",
           "reshuffle_interaction", "ReshuffleInteraction"]


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


def relation_matrices(raw, row_function="softmax"):
    """Soft reshuffling matrices from raw ``(..., l + 1, l)`` parameters: softmax over the
    rows of every column, then drop the last row. Shape ``(..., l, l)``. With
    ``row_function="square"`` the entries are squared instead (the RESHUFFLE2 ablation of
    the paper, which allows arbitrary non-negative values)."""
    if row_function == "softmax":
        return th.softmax(raw, dim=-2)[..., :-1, :]
    if row_function == "square":
        return raw.square()[..., :-1, :]
    raise ValueError("row_function must be 'softmax' or 'square'.")


def random_features(num_entities, k, l,  # noqa: E741
                    density=0.3085, generator=None):
    """Random binary features of shape ``(n, k, l)``. The default density is the
    probability that a standard normal exceeds 0.5, which is how the authors' code draws
    them."""
    return (th.rand(num_entities, k, l, generator=generator) < density).float()


def propagate(x, rel, loop, edge_index, edge_type, num_layers,
              aggregation="max_self_and_selfLoop"):
    """Run ``num_layers`` message passing layers.

    :param x: Features ``(n, k, l)``.
    :param rel: Relation matrices ``(2R, l, l)``; relation ``i`` is ``rel[2i]`` and its
        inverse ``rel[2i + 1]``.
    :param loop: The loop matrix ``(l, l)``.
    :param edge_index: ``(2, E)`` source and target of every triple.
    :param edge_type: ``(E,)`` relation index of every triple (0 .. R-1).
    """
    if aggregation not in ("max_self_and_selfLoop", "max_self_noLoop"):
        raise ValueError("aggregation must be 'max_self_and_selfLoop' or 'max_self_noLoop'.")
    src, dst = edge_index
    for _ in range(num_layers):
        def message(source, target, matrices):
            m = th.bmm(x[source], matrices)
            index = target.view(-1, 1, 1).expand_as(m)
            return th.zeros_like(x).scatter_reduce(0, index, m, reduce="amax")

        new = th.maximum(x, th.maximum(
            message(src, dst, rel[2 * edge_type]),
            message(dst, src, rel[2 * edge_type + 1])))
        if aggregation == "max_self_and_selfLoop":
            new = th.maximum(new, x @ loop)
        x = new
    return x


def reshuffle_interaction(h, r, t):
    """:math:`-\\lVert \\mathrm{ReLU}(h R - t) \\rVert_2` over the ``k x l`` entries."""
    return -th.norm(th.relu(h @ r - t), dim=(-2, -1), p=2)


class ReshuffleInteraction(Interaction):
    """The scoring function of RESHUFFLE. Relations arrive as softmaxed matrices."""

    entity_shape = ("kl",)
    relation_shape = ("dl",)

    def forward(self, h, r, t):
        return reshuffle_interaction(h, r, t)
