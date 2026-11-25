"""Scoring and ranking for models that embed texts (class labels, verbalised expressions).

The helpers do the tensor work of an evaluation of such a model, so that the model class
only names its parts:

* :func:`encode_batched` encodes texts in batches without gradients.
* :func:`encode_cached` encodes the texts that are not cached yet, once each.
* :func:`embed_unique_pairs` embeds ``(first, second)`` index pairs once per distinct pair.
* :func:`candidate_scores` scores every query against every candidate, in chunks of candidates.
* :func:`best_on_grid` chooses the value of a score parameter (a grid) with the best mean
  reciprocal rank of the targets.
"""
import numpy as np
import torch as th

from mowl.evaluation.ont_ranking import ordinal_ranks_by_counting

__all__ = ["encode_batched", "encode_cached", "embed_unique_pairs", "candidate_scores",
           "best_on_grid"]


def encode_batched(encode, texts, batch_size, module, dim, device):
    """Encode ``texts`` with ``encode(batch)`` (points, one row per text) in batches, without
    gradients, with ``module`` in evaluation mode (its mode is restored); shape ``(n, dim)``
    in single precision. ``dim`` is the width of the points, or a function that returns it
    (asked only when there are no texts)."""
    was_training = module.training
    module.eval()
    out = []
    with th.no_grad():
        for i in range(0, len(texts), batch_size):
            out.append(encode(texts[i:i + batch_size]).float())
    if was_training:
        module.train()
    if not out:
        return th.zeros((0, dim() if callable(dim) else dim), device=device)
    return th.cat(out)


def encode_cached(encode, texts, cache):
    """Rows for ``texts`` from ``cache`` (a dict text -> row), encoding the missing texts
    first with ``encode(list_of_texts)``, each once."""
    missing = sorted({t for t in texts if t not in cache})
    if missing:
        for t, e in zip(missing, encode(missing)):
            cache[t] = e
    return th.stack([cache[t] for t in texts])


def embed_unique_pairs(embed, first, second):
    """``embed(first_ids, second_ids)`` evaluated once per distinct pair, expanded back to
    one row per input pair."""
    pairs, inverse = th.unique(th.stack([first, second], dim=1), dim=0, return_inverse=True)
    unique = embed(pairs[:, 0], pairs[:, 1])
    return unique[inverse.to(unique.device)]


def candidate_scores(score, query, candidates, chunk_size=4096):
    """``score(query, candidate_chunk)`` for chunks of ``candidates`` joined along the
    candidate axis: a ``(len(query), len(candidates))`` matrix."""
    out = [score(query, candidates[s:s + chunk_size])
           for s in range(0, len(candidates), chunk_size)]
    return th.cat(out, dim=1)


def best_on_grid(score_block, targets, grid, block_size=256):
    """The grid value with the best mean reciprocal rank (the first one on ties).

    :param score_block: ``score_block(value, start, stop)``: scores (higher is better) of the
        queries ``start:stop`` against all candidates, for the value of the parameter.
    :param targets: Position of the correct candidate of every query.
    :return: ``(best MRR, best value)``.
    """
    best = (-1.0, grid[0])
    for value in grid:
        ranks = []
        for s in range(0, len(targets), block_size):
            scores = score_block(value, s, s + block_size)
            ranks.append(ordinal_ranks_by_counting(-scores, targets[s:s + block_size]).cpu())
        mrr = float(np.mean(1.0 / th.cat(ranks).double().numpy()))
        if mrr > best[0]:
            best = (mrr, value)
    return best
