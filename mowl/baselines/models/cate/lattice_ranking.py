"""Scoring and validating the elements of a lattice by their order violation.

A geometry that embeds the elements of a lattice (CatE) scores a pair ``(A, B)`` by how much
the order is violated, ``module.order_violation(A, B)``, lower is better. Two parts use only
that: :func:`score_columns` (rows of an axiom form from columns of ids) and
:func:`order_mean_reciprocal_rank` (the raw mean reciprocal rank of validation queries, the
model-selection score of CatE).
"""

import torch as th

__all__ = ["score_columns", "order_mean_reciprocal_rank"]


def score_columns(module, form, columns, device="cpu"):
    """Score the axioms of ``form`` whose rows are the ``columns`` (sequences of ids of equal
    length): ``module(rows, form)``, one score per row, lower is better."""
    rows = th.stack([th.as_tensor(c, dtype=th.long, device=device).reshape(-1)
                     for c in columns], dim=1)
    return module(rows, form)


@th.no_grad()
def order_mean_reciprocal_rank(module, queries, device="cpu"):
    """Raw mean reciprocal rank of validation queries ``(head, true, candidates)`` (ties:
    average rank): the candidates are scored by ``module.order_violation(head, candidate)``
    (the head itself is not a candidate) and the true one is ranked among them. The module
    is left in training mode. ``0.0`` without queries."""
    module.eval()
    reciprocal = []
    for head, true, candidates in queries:
        candidates = candidates.to(device)
        scores = module.order_violation(th.full_like(candidates, head), candidates)
        keep = candidates != head
        scores = scores[keep]
        true_score = module.order_violation(th.tensor([head], device=device),
                                            th.tensor([true], device=device))[0]
        better = (scores < true_score).sum().item()
        ties = (scores == true_score).sum().item()
        rank = 1 + better + max(ties - 1, 0) / 2
        reciprocal.append(1.0 / rank)
    module.train()
    return sum(reciprocal) / len(reciprocal) if reciprocal else 0.0
