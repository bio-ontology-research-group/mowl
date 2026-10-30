"""Shared utilities for multi-component box embedding modules."""
from __future__ import annotations

import numpy as np
import torch as th

__all__ = ["effective_components"]


def effective_components(
    centers: th.Tensor,
    offsets: th.Tensor,
    eps: float = 1e-3,
) -> th.Tensor:
    """Count non-degenerate, non-nested components per class on CPU using corner arithmetic.

    A component :math:`i` of class :math:`c` is counted if all of its
    half-widths satisfy :math:`|o_{c,i,m}| > \\text{eps}` and its box is not
    contained in any other valid component :math:`j \\neq i` of :math:`c`
    (checked on corners :math:`\\text{lo}_j \\le \\text{lo}_i` and
    :math:`\\text{hi}_i \\le \\text{hi}_j`; if two components have identical
    corners, only the lowest-index component is kept).

    Guarantees:
    - Returns a 1-D CPU ``torch.Tensor`` of shape ``(n_classes,)`` with integer
      counts in ``0 .. n_components``.
    - Takes ``torch.abs(offsets)`` before computing box corners and comparing
      against ``eps``.

    Non-guarantees:
    - Does not check whether a component is covered by the *union* of two or
      more other components when it is not contained in any single one.

    Parameters
    ----------
    centers : torch.Tensor
        Component centres of shape ``(n_classes, n_components, embed_dim)``.
    offsets : torch.Tensor
        Component offsets of shape ``(n_classes, n_components, embed_dim)``.
    eps : float, default 1e-3
        Minimum half-width threshold in every coordinate for a component to be
        considered non-degenerate.

    Returns
    -------
    torch.Tensor
        1-D ``int64`` tensor of length ``n_classes`` on CPU.
    """
    if centers.ndim != 3 or offsets.ndim != 3 or centers.shape != offsets.shape:
        raise ValueError(
            "centers and offsets must be 3-D tensors of identical shape "
            f"(n_classes, n_components, embed_dim); got {tuple(centers.shape)} "
            f"and {tuple(offsets.shape)}."
        )
    n, k, _ = centers.shape
    cen_np = centers.detach().cpu().numpy()
    off_np = th.abs(offsets.detach()).cpu().numpy()
    lo_np = cen_np - off_np
    hi_np = cen_np + off_np
    valid_np = np.all(off_np > eps, axis=-1)

    counts = np.zeros(n, dtype=np.int64)
    for c in range(n):
        v = valid_np[c]
        if not np.any(v):
            continue
        lo_c = lo_np[c]
        hi_c = hi_np[c]
        eff = 0
        for i in range(k):
            if not v[i]:
                continue
            contained = False
            for j in range(k):
                if i == j or not v[j]:
                    continue
                if np.all(lo_c[j] <= lo_c[i]) and np.all(hi_c[i] <= hi_c[j]):
                    equal_boxes = np.all(lo_c[i] <= lo_c[j]) and np.all(
                        hi_c[j] <= hi_c[i]
                    )
                    if not equal_boxes or j < i:
                        contained = True
                        break
            if not contained:
                eff += 1
        counts[c] = eff
    return th.from_numpy(counts)
