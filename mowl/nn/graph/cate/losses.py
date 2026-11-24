"""Loss of CatE (Zhapa-Camacho and Hoehndorf, arXiv:2305.07163v3, Eq. 7)."""

import torch as th


def order_margin_loss(pos_scores, neg_scores, margin, negative_aggregation="per_negative",
                      neg_mask=None):
    """Margin loss on order-violation scores.

    Eq. 7 of the paper sums, over the morphisms :math:`A \\sqsubseteq B` and their
    corruptions :math:`A \\sqsubseteq B'`, the term
    :math:`s(A, B) + \\max(0, \\gamma - s(A, B'))`. This function returns the mean of
    those terms over the batch (a rescaling of the sum).

    :param pos_scores: :math:`s(A, B)` of the positive morphisms, shape ``(b,)``.
    :param neg_scores: :math:`s(A, B')` of the corruptions, shape ``(b, k)``.
    :param margin: :math:`\\gamma`.
    :param negative_aggregation: ``"per_negative"`` (Eq. 7: one hinge per corruption) or
        ``"mean"`` (the reference implementation: the scores of the ``k`` corruptions of a
        morphism are averaged before the hinge, :math:`\\max(0, \\gamma - \\frac1k\\sum_j
        s(A, B'_j))`).
    :param neg_mask: Optional boolean tensor of the shape of ``neg_scores``; ``False``
        marks corruptions that are not negatives (Eq. 7 sums only over
        :math:`A \\sqsubseteq B' \\notin (\\tilde C, \\sqsubseteq)`). Masked corruptions are
        left out of the means.
    :return: Scalar loss.
    """
    if neg_scores.dim() == 1:
        neg_scores = neg_scores.unsqueeze(1)
    if neg_mask is None:
        neg_mask = th.ones_like(neg_scores, dtype=th.bool)
    elif neg_mask.dim() == 1:
        neg_mask = neg_mask.unsqueeze(1)
    w = neg_mask.to(neg_scores.dtype)
    if negative_aggregation == "per_negative":
        neg_term = (th.relu(margin - neg_scores) * w).sum() / w.sum().clamp(min=1)
    elif negative_aggregation == "mean":
        count = w.sum(dim=1)
        mean_score = (neg_scores * w).sum(dim=1) / count.clamp(min=1)
        valid = (count > 0).to(neg_scores.dtype)
        neg_term = (th.relu(margin - mean_score) * valid).sum() / valid.sum().clamp(min=1)
    else:
        raise ValueError("negative_aggregation must be 'per_negative' or 'mean'.")
    return pos_scores.mean() + neg_term
