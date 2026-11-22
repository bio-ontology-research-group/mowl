"""Loss terms that are not tied to one method: functions of scores or of points, to be added
to the loss of any geometry (named parts of the training side, next to the regimes of
:mod:`mowl.training.losses` and the penalties of :mod:`mowl.training.regularization`).

* :func:`pearson_correlation` and :func:`correlation_contrastive_loss`: a *correlation-based
  contrastive term*. Instead of pushing scores of true axioms up and of corrupted ones down
  one by one, it asks the scores to correlate with the labels over the whole batch (ELK-Em,
  Singh et al. 2026, Eq. 5). :class:`~mowl.training.PearsonCorrelationLoss` applies it per
  normal form inside a training regime.
* :func:`lexical_similarity_loss`: a *lexical-similarity term* that asks the geometry of a set
  of points to follow an external similarity of the same entities, for instance of their
  labels (ELK-Em, Eq. 6).
"""

import torch as th
import torch.nn.functional as F

__all__ = ["pearson_correlation", "correlation_contrastive_loss", "lexical_similarity_loss"]


def pearson_correlation(scores, labels, eps=1e-8):
    r""":math:`\frac{(\sigma - \bar\sigma)^\top (t - \bar t)}{\lVert\sigma - \bar\sigma\rVert
    \lVert t - \bar t\rVert + \varepsilon}`: the Pearson correlation of ``scores``
    :math:`\sigma` with ``labels`` :math:`t` (flat tensors of one length; ``eps`` keeps the
    value defined when one of them is constant, where it is 0)."""
    s, t = scores - scores.mean(), labels - labels.mean()
    return (s * t).sum() / (s.norm() * t.norm() + eps)


def correlation_contrastive_loss(pos_scores, neg_scores, eps=1e-8):
    r"""One minus the correlation of the scores of the axioms and of their corruptions with
    the labels 1 (axioms) and 0 (corruptions): :math:`1 - \rho(\sigma, t)`. The scores are
    those where *higher is more plausible* (for a loss :math:`\ell`, ELK-Em passes
    :math:`e^{-\ell}`). The two score tensors may have different lengths."""
    scores = th.cat([pos_scores.reshape(-1), neg_scores.reshape(-1)])
    labels = th.cat([th.ones(pos_scores.numel(), dtype=scores.dtype, device=scores.device),
                     th.zeros(neg_scores.numel(), dtype=scores.dtype, device=scores.device)])
    return 1 - pearson_correlation(scores, labels, eps)


def lexical_similarity_loss(points, features, decay):
    r""":math:`\binom{m}{2}^{-1} \sum_{i<j} (\bar s^{ext}_{ij} - s^{geo}_{ij})^2`: the
    external similarity :math:`\bar s^{ext}_{ij} = (1 + \cos(z_i, z_j)) / 2` of the features
    :math:`z_i` of :math:`m` entities should equal the geometric similarity
    :math:`s^{geo}_{ij} = \exp(-\frac{\lambda}{n}\lVert e_i - e_j\rVert_2^2)` of their points
    :math:`e_i \in \mathbb{R}^n`. Zero (with a graph) for fewer than two entities.

    :param points: ``(m, n)``.
    :param features: ``(m, d)``, any ``d``.
    :param decay: :math:`\lambda > 0`.
    """
    m, n = points.shape
    if m < 2:
        return points.sum() * 0.0
    # cosine similarities of the features, and squared distances of the points from the Gram
    # matrix (the distance function has an undefined gradient at 0)
    unit = F.normalize(features, dim=1)
    gram = points @ points.t()
    squared = (gram.diagonal().unsqueeze(1) + gram.diagonal().unsqueeze(0) - 2 * gram)
    gap = (1 + unit @ unit.t()) / 2 - th.exp(-(decay / n) * squared.clamp_min(0))
    # each unordered pair once: the strict upper triangle
    return th.triu(gap.pow(2), diagonal=1).sum() / (m * (m - 1) / 2)
