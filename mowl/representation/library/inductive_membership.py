"""Degrees of membership of points that are not elements of the domain, and their
cross-entropy against labels (an objective part for inductive data).

A geometry whose classes are regions with a signed depth (positive inside) can score any
point, not only its own elements: a protein encoded from its features, say, against the
classes of an ontology. :class:`SoftClassScores` gives the degree

.. math:: s(p, C) = \\sigma\\big(\\operatorname{dep}(p, C) / \\tau_s\\big)

of every point in every class of a list (crisp depth: the largest depth over the active
components; ``measure.soft`` of the geometry), in chunks of classes so that a batch of
``B`` points against ``n`` classes of ``K`` components in dimension ``d`` never holds more
than ``max_elements`` numbers. :class:`MembershipCrossEntropy` is the objective: the mean
binary cross-entropy of those degrees against a ``(B, n)`` table of labels.

Used by GroundedEL (:meth:`mowl.models.GroundedEL.compute_data_loss`); any geometry with
``depth(points, class_ids, crisp=True)``, a measure with ``soft`` and ``n_components``
can use it.
"""
from __future__ import annotations

import torch as th
import torch.nn.functional as F

__all__ = ["SoftClassScores", "MembershipCrossEntropy"]


class SoftClassScores:
    """Degrees ``(B, n)`` of ``B`` points in ``n`` classes (module docstring).

    :param max_elements: Largest ``B * classes_per_chunk * K * d`` of one chunk.
    """

    def __init__(self, max_elements=8_388_608):
        self.max_elements = int(max_elements)

    def __call__(self, geometry, points, class_ids, tau):
        bsz, n_terms = int(points.shape[0]), int(class_ids.numel())
        tau = tau.to(dtype=points.dtype)
        if bsz == 0 or n_terms == 0:
            return th.empty((bsz, n_terms), dtype=points.dtype, device=points.device)
        dim = int(points.shape[1])
        per_class = max(1, bsz * max(1, geometry.n_components) * max(1, dim))
        chunk = max(1, self.max_elements // per_class)
        out = []
        for start in range(0, n_terms, chunk):
            ids = class_ids[start:start + chunk]
            c = int(ids.numel())
            rep = points.unsqueeze(1).expand(bsz, c, dim).reshape(bsz * c, dim)
            depth = geometry.depth(rep, ids.unsqueeze(0).expand(bsz, c).reshape(bsz * c),
                                   crisp=True).view(bsz, c)
            out.append(geometry.measure.soft(depth, tau))
        return out[0] if len(out) == 1 else th.cat(out, dim=1)


class MembershipCrossEntropy:
    """Mean binary cross-entropy of the degrees of :class:`SoftClassScores` against labels.

    :param scores: The scoring part (default :class:`SoftClassScores`).
    """

    def __init__(self, scores=None):
        self.scores = scores if scores is not None else SoftClassScores()

    def __call__(self, geometry, points, class_ids, targets, tau):
        """``(loss, scores)`` of ``points`` ``(B, d)`` against ``targets`` ``(B, n)``."""
        scores = self.scores(geometry, points, class_ids, tau)
        return F.binary_cross_entropy(scores, targets.to(dtype=scores.dtype),
                                      reduction="mean"), scores
