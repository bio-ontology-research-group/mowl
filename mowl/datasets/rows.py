"""Row tables of axioms in a canonical order.

The rows of a normal form come out of the OWL API in the order of a Java hash set, which can
differ between two builds of the same ontology. :func:`sort_rows` puts a table in
lexicographic order (first column first), so that a run seeded the same way draws the same
rows whatever that order was."""
import torch as th

__all__ = ["lexicographic_order", "sort_rows"]


def lexicographic_order(table):
    """The permutation that sorts the rows of a 2-D tensor lexicographically (stable)."""
    order = th.arange(table.shape[0], dtype=th.long, device=table.device)
    for col in range(table.shape[1] - 1, -1, -1):
        order = order[th.argsort(table[order, col], stable=True)]
    return order


def sort_rows(table):
    """Sort the rows of ``table`` (a 2-D tensor with more than one row; anything else is left
    alone) lexicographically, in place. Returns ``table``."""
    if isinstance(table, th.Tensor) and table.ndim == 2 and table.shape[0] > 1:
        table.copy_(table[lexicographic_order(table)])
    return table
