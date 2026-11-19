"""Rank computations and rank-based metrics, exactly as the evaluation regimes of the
literature compute them.

Publications that report "the same" metric often compute it differently: ties are broken
by position or averaged, the ROC AUC is a trapezoid over false positive rates or a
Riemann sum over ranks. The functions here implement each variant once, so that every
evaluation regime in :mod:`mowl.evaluation` states which one it uses. They operate on
plain tensors and lists and do not need the JVM.
"""

from collections import Counter

import numpy as np
import torch as th


def ordinal_ranks(scores, targets):
    """1-based ranks of the targets when candidates are sorted by increasing score, ties
    broken by candidate position (``argsort(argsort(scores))``), as in the reference
    implementation of Box\\ :sup:`2`\\ EL (``dists_to_ranks``).

    :param scores: ``(n, num_candidates)`` tensor, lower is better.
    :param targets: ``(n,)`` tensor of candidate positions of the true answers.
    :rtype: torch.Tensor
    """
    order = th.argsort(scores, dim=1, stable=True).argsort(dim=1) + 1
    return th.take_along_dim(order, targets.reshape(-1, 1), dim=1).flatten()


def average_ranks(scores, targets):
    """1-based ranks of the targets with ties averaged (``scipy.stats.rankdata(...,
    method="average")``), as in the reference implementation of DELE.

    :param scores: ``(n, num_candidates)`` tensor, lower is better.
    :param targets: ``(n,)`` tensor of candidate positions of the true answers.
    :rtype: torch.Tensor (float)
    """
    target_scores = th.take_along_dim(scores, targets.reshape(-1, 1), dim=1)
    less = (scores < target_scores).sum(dim=1).double()
    equal = (scores == target_scores).sum(dim=1).double()
    # Ranks less+1, ..., less+equal share their average.
    return less + (equal + 1) / 2


def compute_ranks(scores, targets, ties="ordinal"):
    """Dispatch to :func:`ordinal_ranks` (``ties="ordinal"``) or :func:`average_ranks`
    (``ties="average"``)."""
    if ties == "ordinal":
        return ordinal_ranks(scores, targets)
    if ties == "average":
        return average_ranks(scores, targets)
    raise ValueError(f"Unknown tie handling {ties!r}; use 'ordinal' or 'average'.")


def auc_box2el(ranks, num_candidates):
    """ROC AUC of the reference implementation of Box\\ :sup:`2`\\ EL
    (``evaluate.py``, ``compute_rank_roc``).

    The model is read as a binary classifier that accepts a candidate iff its rank is at
    most a threshold :math:`k`; the curve of true positive rate against false positive
    rate (negatives: the ``num_candidates - 1`` other candidates of every query) is
    integrated with the trapezoid rule.

    :param ranks: Iterable of ranks (1-based).
    :param num_candidates: Number of candidates per query.
    """
    counts = Counter(float(r) for r in ranks)
    if not counts:
        return float("nan")
    num_queries = sum(counts.values())
    num_negatives = (num_candidates - 1) * num_queries
    if num_negatives <= 0:
        return float("nan")
    tprs, fprs = [0.0], [0.0]
    tp = 0
    for x in sorted(counts):
        tp += counts[x]
        tprs.append(tp / num_queries)
        fp = sum((x - 1) * v if k <= x else x * v for k, v in counts.items())
        fprs.append(fp / num_negatives)
    tprs.append(1.0)
    fprs.append(1.0)
    return float(np.trapezoid(tprs, fprs) if hasattr(np, "trapezoid") else np.trapz(tprs, fprs))


def auc_rank_curve(ranks, num_candidates):
    """ROC AUC of the PPI evaluation of the reference implementation of
    Box\\ :sup:`2`\\ EL (``evaluate_ppi.py``, ``compute_rank_roc``) and of ELEmbeddings:
    the cumulative fraction of queries with rank at most :math:`x`, integrated over
    :math:`x \\in [r_{min}, N]` with the trapezoid rule and divided by :math:`N`.
    """
    counts = Counter(float(r) for r in ranks)
    if not counts:
        return float("nan")
    xs = sorted(counts)
    total = sum(counts.values())
    ys, tp = [], 0
    for x in xs:
        tp += counts[x]
        ys.append(tp / total)
    xs = xs + [float(num_candidates)]
    ys = ys + [1.0]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(ys, xs) / num_candidates)


def auc_mowl(ranks, num_candidates):
    """ROC AUC of :func:`mowl.evaluation.base.compute_rank_roc` (left Riemann sum of the
    cumulative rank curve, used by mOWL's evaluators and by the reference implementation
    of DELE), in closed form: :math:`\\frac{1}{n}\\sum_i \\max(N - r_i, 0) / (N - 1)`.
    For integer ranks this equals ``compute_rank_roc``; the closed form also accepts the
    fractional ranks of averaged ties."""
    r = np.asarray([float(x) for x in ranks], dtype=float)
    if len(r) == 0 or num_candidates < 2:
        return float("nan")
    return float(np.mean(np.maximum(num_candidates - r, 0.0)) / (num_candidates - 1))


AUC_FUNCTIONS = {"box2el": auc_box2el, "rank_curve": auc_rank_curve, "mowl": auc_mowl}


def summarise_ranks(ranks, num_candidates, hits=(1, 10, 100), auc="box2el", prefix=""):
    """Standard rank metrics of a list of ranks.

    :param ranks: Iterable of 1-based ranks.
    :param num_candidates: Number of candidates per query (for the AUC).
    :param hits: The :math:`k` of the reported Hits@\\ :math:`k`.
    :param auc: AUC variant, a key of :data:`AUC_FUNCTIONS`.
    :param prefix: Prefix of every metric name (``"f_"`` for filtered metrics).
    :return: ``{"hits@k": ..., "med": ..., "mrr": ..., "mr": ..., "auc": ...}`` with
        ``prefix`` prepended. Empty input gives ``nan`` values.
    """
    r = np.asarray([float(x) for x in ranks], dtype=float)
    out = {}
    if len(r) == 0:
        for k in hits:
            out[f"{prefix}hits@{k}"] = float("nan")
        for name in ("med", "mrr", "mr", "auc"):
            out[f"{prefix}{name}"] = float("nan")
        return out
    for k in hits:
        out[f"{prefix}hits@{k}"] = float(np.mean(r <= k))
    out[f"{prefix}med"] = float(np.median(r))
    out[f"{prefix}mrr"] = float(np.mean(1.0 / r))
    out[f"{prefix}mr"] = float(np.mean(r))
    out[f"{prefix}auc"] = AUC_FUNCTIONS[auc](r.tolist(), num_candidates)
    return out


def micro_metrics(ranks, groups, num_candidates, auc="mowl", prefix=""):
    """Per-query-entity ("micro") mean rank and AUC as defined by DELE (Mashkova et al.,
    Eqs. 19-22): the metric is computed for the test axioms of each head entity
    separately and then averaged over the head entities.

    :param ranks: Iterable of ranks.
    :param groups: Iterable of the same length with the head entity of every query.
    :return: ``{prefix + "micro_mr": ..., prefix + "micro_auc": ...}``
    """
    by_group = {}
    for r, g in zip(ranks, groups):
        by_group.setdefault(g, []).append(float(r))
    if not by_group:
        return {f"{prefix}micro_mr": float("nan"), f"{prefix}micro_auc": float("nan")}
    fn = AUC_FUNCTIONS[auc]
    mrs = [float(np.mean(v)) for v in by_group.values()]
    aucs = [fn(v, num_candidates) for v in by_group.values()]
    return {f"{prefix}micro_mr": float(np.mean(mrs)),
            f"{prefix}micro_auc": float(np.mean(aucs))}


def binary_f1(y_true, y_pred):
    """F1 score of binary predictions (positive class ``1``)."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum(~y_true & y_pred))
    fn = float(np.sum(y_true & ~y_pred))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def best_threshold(scores, labels):
    """Threshold :math:`t` maximising the F1 score of the rule "positive iff
    ``score <= t``" (lower scores are better), and that F1 score.

    Every distinct score is tried as a threshold.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    best_t, best_f1 = float("inf"), -1.0
    for t in np.unique(scores):
        f1 = binary_f1(labels, scores <= t)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1
