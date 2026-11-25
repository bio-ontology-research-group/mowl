"""The objective of the query embedding methods: negative sampling on sampled queries.

A query method is trained on batches of queries of one structure each. For a batch the
geometry gives the state of the queries (:meth:`~mowl.nn.query.base.QueryGeometry.embed`)
and the score ``logit = gamma - distance`` of a positive answer and of ``negatives``
entities that are not answers. :class:`QueryObjective` turns those scores into a loss. It
knows nothing about boxes, distributions or cones, so every query method (and every
combination of their parts) can use any of its options.
"""

import torch as th
import torch.nn.functional as F

__all__ = ["QueryObjective", "AnswerMarginLoss", "AnswerMarginObjective"]


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


class AnswerMarginLoss:
    """Margin loss of the scores of answers and of non-answers of queries (scores positive
    inside the answer region):

    .. math:: \\operatorname{mean}[m - s^+]_+ + \\operatorname{mean}[m + s^-]_+ ,

    the second term only when non-answers are given. Used by GroundedEL's witness-chain
    queries (:func:`mowl.nn.el.grounded.queries.query_witness_loss`) and by the query
    embedding methods as :class:`AnswerMarginObjective`; any scorer of targets can use it.

    :param margin: The margin :math:`m`.
    """

    def __init__(self, margin=0.0):
        self.margin = float(margin)

    @staticmethod
    def targets(answers, negatives=None):
        """The targets to score: the answers, then the non-answers (``(B, M+)``, ``(B, M-)``)."""
        if negatives is None or negatives.shape[1] == 0:
            return answers
        return th.cat([answers, negatives], dim=1)

    def __call__(self, scores, n_answers, with_negatives, reference):
        """The loss of ``scores`` of :meth:`targets` (the first ``n_answers`` columns are
        answers). ``reference`` (a parameter) gives a zero loss with a gradient path when
        there are no queries."""
        if scores.shape[0] == 0:
            return reference.sum() * 0.0
        loss = F.relu(self.margin - scores[:, :n_answers]).mean()
        if with_negatives and scores.shape[1] > n_answers:
            loss = loss + F.relu(self.margin + scores[:, n_answers:]).mean()
        return loss


class AnswerMarginObjective:
    r"""The objective of a query embedding model as the margin loss of answers
    (:class:`AnswerMarginLoss`): for every batch of queries of one structure, the scores
    ``logit = gamma - distance`` of the positive answer and of the sampled non-answers, the
    loss :math:`\operatorname{mean}[m - s^+]_+ + \operatorname{mean}[m + s^-]_+`; the
    batches count by their number of queries. It replaces
    :class:`QueryObjective` (``QueryEmbeddingModel(..., objective=AnswerMarginObjective())``).

    :param margin: The margin :math:`m`.
    """

    kind = "answer margin"

    def __init__(self, margin=0.0):
        self.loss = AnswerMarginLoss(margin)

    @property
    def margin(self):
        return self.loss.margin

    def __call__(self, geometry, groups):
        total, count = 0.0, 0
        for group in groups.values():
            sig, rows, pos, neg = group[:4]
            state = geometry.embed(sig, rows)
            p = geometry.logit(state, pos)
            n = geometry.logit(state, neg)
            scores = th.cat([p.reshape(len(rows), -1), n.reshape(len(rows), -1)], dim=1)
            total = total + self.loss(scores, p.reshape(len(rows), -1).shape[1], True, p) \
                * len(rows)
            count += len(rows)
        return total / max(count, 1)
