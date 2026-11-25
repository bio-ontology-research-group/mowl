"""Order entities by a score.

:func:`rank_by_score` is the ranking step of every "which individuals (classes, candidates)
fit this best" query: the names sorted by increasing cost, ties kept in the order given.
"""

import numpy as np

__all__ = ["rank_by_score"]


def rank_by_score(names, scores):
    """``names`` sorted by increasing ``scores`` (a lower cost is a better fit); equal scores
    keep their order in ``names``.

    :param names: Sequence of ``n`` names.
    :param scores: ``n`` costs.
    :return: A list of the names.
    """
    return [names[i] for i in np.argsort(scores, kind="stable")]
