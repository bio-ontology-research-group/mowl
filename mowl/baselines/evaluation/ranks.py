"""``ordinal_ranks_by_counting`` of ``mowl/evaluation/ont_ranking.py`` at 2d0ed505 (the HiT/OnT
validation ranks), frozen for the baselines."""
import torch as th


def ordinal_ranks_by_counting(scores, targets):
    """The ranks of :func:`mowl.evaluation.ranking_metrics.ordinal_ranks` (lower score is
    better, ties broken by candidate position) computed by counting instead of sorting:
    :math:`1 + |\\{j : s_j < s_t\\}| + |\\{j < t : s_j = s_t\\}|`. Linear in the number of
    candidates and computed on the device of ``scores``.

    :param scores: ``(n, num_candidates)`` tensor.
    :param targets: ``(n,)`` tensor of candidate positions.
    """
    targets = targets.to(scores.device).reshape(-1, 1)
    target_scores = th.take_along_dim(scores, targets, dim=1)
    position = th.arange(scores.shape[1], device=scores.device).unsqueeze(0)
    better = (scores < target_scores).sum(dim=1)
    tied_before = ((scores == target_scores) & (position < targets)).sum(dim=1)
    return better + tied_before + 1
