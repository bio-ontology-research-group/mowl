r"""Training data and negatives for methods that read pairs of named classes.

The order and lattice embeddings (order embeddings, box lattices, Gumbel boxes, entailment
cones, shadow cones, disk embeddings) train on the told pairs :math:`C \sqsubseteq D` between
named classes and nothing else. The functions here are the data side of that: which rows of
the normal form are kept, how a pair is coded as one integer, and how corrupted pairs that
are told pairs are drawn again. They know nothing of a method and work for any model that
trains on two-column rows.

======================================  ===============================================
:func:`named_pair_rows`                 keep the rows made of named classes only, sorted
                                        and without duplicates (optionally distinct
                                        columns, optionally without one class)
:func:`pair_codes`                      one integer per pair; the set of told pairs
:func:`redraw_known_pairs`              draw again the corruptions that are told pairs
                                        (or reflexive), without a Python loop
:func:`descendant_fractions`            the share of the classes below each class (the
                                        unary marginal of the box lattice)
======================================  ===============================================
"""

import numpy as np
import torch as th

__all__ = ["named_pair_rows", "pair_codes", "redraw_known_pairs", "descendant_fractions"]


def descendant_fractions(pairs, n):
    """For each of the ``n`` classes the number of its descendants (itself included) in the
    taxonomy given by the rows ``(sub, sup)`` of ``pairs``, divided by ``n``."""
    children = [[] for _ in range(n)]
    for sub, sup in pairs:
        if sub != sup:
            children[sup].append(sub)

    def descendants(root):
        seen, stack = {root}, [root]
        while stack:
            for c in children[stack.pop()]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen
    counts = np.array([len(descendants(i)) for i in range(n)])
    return counts / n


def named_pair_rows(rows, n_named, distinct=False, exclude_first=None, device=None):
    """The rows of ``rows`` (``(n, 2)`` integer tensor) whose entries are all below
    ``n_named`` (the named classes come first in the index), sorted lexicographically and
    without duplicates.

    :param distinct: Also drop rows whose two entries are equal (reflexive axioms).
    :param exclude_first: Drop rows whose first entry is this index (for example the index
        of the bottom class: :math:`\\bot \\sqsubseteq D` says nothing about order).
    :param device: Move the result there.
    :return: ``(m, 2)`` tensor.
    """
    keep = (rows < n_named).all(dim=1)
    if distinct:
        keep &= rows[:, 0] != rows[:, 1]
    if exclude_first is not None:
        keep &= rows[:, 0] != exclude_first
    out = th.unique(rows[keep], dim=0)
    return out if device is None else out.to(device)


def pair_codes(rows, base):
    """One integer per row of a two-column tensor, ``first * base + second``, and the sorted
    unique codes: the set of told pairs, for membership tests with :func:`redraw_known_pairs`.
    ``base`` is the number of entities."""
    return th.unique(rows[:, 0] * base + rows[:, 1])


def redraw_known_pairs(negatives, n_positives, columns, known_codes, base, pool,
                       max_redraw=10):
    """Draw again the corruptions that are told pairs or reflexive.

    ``negatives`` is a ``(k * n, 2)`` tensor laid out as ``k`` consecutive blocks of ``n``
    corruptions (:meth:`~mowl.base_models.EmbeddingELModel.generate_negatives`); block
    ``j`` corrupted column ``columns[j]``. A row whose code is in ``known_codes`` or whose
    two entries are equal gets a fresh entity (uniform from ``pool``, drawn with NumPy's
    global generator) in the column its block corrupted, up to ``max_redraw`` rounds.
    Done for all rows at once: a round costs a few tensor operations, whatever the number of
    negatives. Modifies and returns ``negatives``.

    :param known_codes: Codes of the told pairs (:func:`pair_codes`).
    :param base: Number of entities used for the codes.
    :param pool: Indices a replacement is drawn from (array-like).
    """
    if negatives is None or len(negatives) == 0:
        return negatives
    pool = np.asarray(pool)
    column_of_row = th.as_tensor(list(columns), dtype=th.long,
                                 device=negatives.device).repeat_interleave(n_positives)
    known = known_codes.to(negatives.device)
    for _ in range(max_redraw):
        codes = negatives[:, 0] * base + negatives[:, 1]
        bad = th.nonzero(th.isin(codes, known) | (negatives[:, 0] == negatives[:, 1])).flatten()
        if len(bad) == 0:
            break
        fresh = th.as_tensor(np.random.choice(pool, size=len(bad)), dtype=negatives.dtype,
                             device=negatives.device)
        negatives[bad, column_of_row[bad]] = fresh
    return negatives
