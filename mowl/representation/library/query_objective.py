"""The objective of the query embedding methods: negative sampling on sampled queries.

A query method is trained on batches of queries of one structure each. For a batch the
geometry gives the state of the queries (:meth:`~mowl.nn.query.base.QueryGeometry.embed`)
and the score ``logit = gamma - distance`` of a positive answer and of ``negatives``
entities that are not answers. :class:`QueryObjective` turns those scores into a loss. It
knows nothing about boxes, distributions or cones, so every query method (and every
combination of their parts) can use any of its options.
"""

import torch.nn.functional as F

__all__ = ["QueryObjective"]


class QueryObjective:
    """Negative sampling loss of a batch of queries.

    The default is the loss of the four reference codes (Query2Box, BetaE, ConE, AConE),
    :math:`-\\log\\sigma(s^+) - \\frac1K\\sum_k \\log\\sigma(-s^-_k)`, averaged with factor
    :math:`\\tfrac12` over the queries.

    :param kind: ``"logsigmoid"`` (default) or ``"hinge"``: :math:`[m - s^+]_+ + \\frac1K
        \\sum_k [m + s^-_k]_+`.
    :param margin: The margin :math:`m` of ``"hinge"``.
    :param adversarial_temperature: ``None`` (default): the negatives of a query count
        equally. A number :math:`\\alpha`: the negatives are weighted by
        :math:`\\mathrm{softmax}_k(\\alpha s^-_k)` (self-adversarial negative sampling of
        RotatE, the option ``negative_adversarial_sampling`` of the KGReasoning code; the
        weights carry no gradient).
    :param use_weights: Whether the weights that a batch comes with (the released query
        datasets weight a query by ``1 / sqrt(number of answers)``) are applied (default) or
        every query counts once.
    """

    def __init__(self, kind="logsigmoid", margin=1.0, adversarial_temperature=None,
                 use_weights=True):
        if kind not in ("logsigmoid", "hinge"):
            raise ValueError("kind must be 'logsigmoid' or 'hinge'.")
        self.kind = kind
        self.margin = float(margin)
        self.adversarial_temperature = adversarial_temperature
        self.use_weights = bool(use_weights)

    def _terms(self, pos, neg):
        """Per-query positive and negative terms, ``(B,)`` each; ``pos`` ``(B, 1)``,
        ``neg`` ``(B, K)``."""
        p = pos.squeeze(1)
        if self.kind == "logsigmoid":
            positive = -F.logsigmoid(p)
            per_negative = -F.logsigmoid(-neg)
        else:
            positive = F.relu(self.margin - p)
            per_negative = F.relu(self.margin + neg)
        if self.adversarial_temperature is None:
            return positive, per_negative.mean(dim=1)
        weights = F.softmax(self.adversarial_temperature * neg, dim=1).detach()
        return positive, (weights * per_negative).sum(dim=1)

    def __call__(self, geometry, groups):
        """The loss of ``groups``: ``{name: (signature, rows, positives, negatives[,
        weights])}`` (what a query sampler gives), scored by ``geometry``."""
        pos_loss, neg_loss, total = 0.0, 0.0, 0.0
        for group in groups.values():
            sig, rows, pos, neg = group[:4]
            state = geometry.embed(sig, rows)
            p = geometry.logit(state, pos)
            n = geometry.logit(state, neg)
            positive, negative = self._terms(p, n)
            if len(group) > 4 and self.use_weights:
                w = group[4]
                positive, negative = w * positive, w * negative
                total = total + w.sum()
            else:
                total = total + len(rows)
            pos_loss = pos_loss + positive.sum()
            neg_loss = neg_loss + negative.sum()
        return (pos_loss / total + neg_loss / total) / 2
