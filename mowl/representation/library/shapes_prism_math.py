"""The triangle of a prism: its three unit normals and the support values of a triangle.

A prism is a triangle times a box. The triangle is
:math:`T(h) = \\{p \\in \\mathbb{R}^2 : n_k \\cdot p \\le h_k\\}` for three fixed unit normals that
sum to zero, so it is determined by its three support values :math:`h_k = n_k \\cdot c + r`
(centre :math:`c`, inradius :math:`r`); Federico, Leemhuis and Kutz (DL 2026), Definition 4.
Used by :class:`~mowl.representation.library.shapes_classes.PrismClasses` and
:class:`~mowl.representation.library.shapes_regions.Prism` (the baseline
:mod:`mowl.baselines.nn.el.prism.losses` has its own copy).
"""
import math

import torch as th

NORMALS = th.tensor(
    [
        [0.0, -1.0],
        [math.sqrt(3.0) / 2.0, 0.5],
        [-math.sqrt(3.0) / 2.0, 0.5],
    ],
    dtype=th.float32,
)


def supports(center, inradius):
    """Triangle support values :math:`h_k = n_k \\cdot c + r` for :math:`k=1,2,3`.

    :param center: Triangle centres of shape ``(..., 2)``.
    :param inradius: Inradii of shape ``(..., 1)`` or ``(...,)`` (not clamped: pass the
        absolute value of a raw table).
    :return: Support values of shape ``(..., 3)``.
    """
    normals = NORMALS.to(device=center.device, dtype=center.dtype)
    if isinstance(inradius, th.Tensor) and inradius.ndim == center.ndim - 1:
        inradius = inradius.unsqueeze(-1)
    return center @ normals.T + inradius
