"""Row tables of axioms in a canonical order.

The rows of a normal form come out of the OWL API in the order of a Java hash set, which can
differ between two builds of the same ontology. :func:`sort_rows` puts a table in
lexicographic order (first column first), so that a run seeded the same way draws the same
rows whatever that order was."""
import numpy as np
import torch as th

__all__ = ["lexicographic_order", "sort_rows", "index_table"]


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


def index_table(rows, width, bounds, name="rows", device="cpu", fill_last=None):
    """A table of index rows ``(N, width)`` (``LongTensor`` on ``device``) from a tensor, a
    nested sequence, an array, an object with a tensor ``data`` or ``None`` (no rows).

    Every column ``c`` of ``bounds`` must hold indices in ``[0, bounds[c])``. With
    ``fill_last`` (an index), a table one column short gets that column filled with it (a
    normal form whose last class is implied, such as bottom). Raises ``ValueError`` (or
    ``TypeError``) naming ``name`` otherwise."""
    def none():
        return th.empty((0, width), dtype=th.long, device=device)
    if rows is None:
        return none()
    if isinstance(getattr(rows, "data", None), th.Tensor) and not isinstance(rows, th.Tensor):
        table = rows.data
    elif isinstance(rows, th.Tensor):
        if rows.dtype == th.bool or rows.is_floating_point() or rows.is_complex():
            raise ValueError(f"{name} tensor must have integer dtype, got {rows.dtype}")
        table = rows.to(dtype=th.long)
    elif isinstance(rows, (list, tuple, np.ndarray)):
        arr = np.asarray(rows)
        if arr.size == 0:
            return none()
        if arr.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integer indices, got dtype {arr.dtype}")
        table = th.as_tensor(arr, dtype=th.long)
    else:
        raise TypeError(f"{name} must be a Tensor, sequence, ndarray, or None, got "
                        f"{type(rows).__name__}")
    if table.numel() == 0:
        return none()
    if table.ndim != 2:
        raise ValueError(f"{name} must be a 2-D tensor of shape (N, {width}), got "
                         f"{tuple(table.shape)}")
    if fill_last is not None and table.shape[1] + 1 == width:
        table = th.cat([table, th.full((table.shape[0], 1), int(fill_last), dtype=th.long,
                                       device=table.device)], dim=1)
    if table.shape[1] != width:
        raise ValueError(f"{name} must have {width} columns, got shape {tuple(table.shape)}")
    for c, bound in enumerate(bounds):
        low, high = int(table[:, c].min()), int(table[:, c].max())
        if low < 0 or high >= bound:
            raise ValueError(f"{name} column {c} indices out of range [0, {bound}): min={low}, "
                             f"max={high}")
    return table
