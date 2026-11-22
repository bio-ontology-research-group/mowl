"""Order embedding of the CatE lattice of concept descriptions."""
from .module import CatEModule  # noqa: F401
from mowl.representation.library.hierarchy_objectives import margin_loss  # noqa: F401


def order_margin_loss(pos_scores, neg_scores, margin, negative_aggregation="per_negative",
                      neg_mask=None):
    """Loss of CatE (Zhapa-Camacho and Hoehndorf, arXiv:2305.07163v3, Eq. 7) on order
    violations: the mean positive cost plus a hinge on the corruptions; see
    :func:`mowl.representation.library.hierarchy_objectives.margin_loss`."""
    return margin_loss(pos_scores, neg_scores, margin, negative_aggregation, neg_mask)
