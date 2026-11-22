"""TransBox losses as written in the authors' code (``model/TransBox_reg.py`` of
HuiYang1997/TransBox), used by :class:`~mowl.nn.TransBoxModule` with
``formulation="reference"``.

The published numbers of Yang et al. (WWW 2025) were obtained with these formulas, which
differ from mOWL's default TransBox losses (:mod:`mowl.baselines.nn.el.transbox.losses`) in five
ways:

1. **Centre-norm term inside every axiom.** Every concept loss adds
   :math:`\\big|\\lVert c(A)\\rVert - 1\\big|` for every concept :math:`A` of the axiom
   (the paper's :math:`\\mathcal{L}^{reg}_{con}`, Eq. 13), with weight 1 (the paper's
   :math:`\\lambda = 1`, Section 6.2). The training regime squares the sum per axiom
   (Eq. 14). mOWL's default adds the centre norm once, outside, weighted by
   ``reg_factor``.
2. **C ⊑ ∃r.D (semantic enhancement)** is the inclusion of the Minkowski difference
   :math:`Box(C) - Box(D)` in :math:`Box(r)`:
   :math:`\\lVert\\max(0, |c_C - c_D - c_r| + o_C + o_D - o_r + m)\\rVert`. mOWL's default
   clamps the offset of :math:`\\exists r^{all}.D` at zero, :math:`\\max(0, o_r - o_D)`,
   which gives the same loss only while :math:`o_r \\ge o_D`.
3. **C ⊓ D ⊑ E** compares the intersection with :math:`Box(E)` using the *signed*
   half-width :math:`(\\mathrm{start} - \\mathrm{end})/2`, which is negative in every
   dimension in which the intersection is non-empty. The masked term is therefore zero
   as soon as the intersection *overlaps* :math:`Box(E)` (not: lies inside it). The
   emptiness penalty is :math:`\\lVert\\max(0, \\mathrm{start} - \\mathrm{end})\\rVert`
   (twice mOWL's). This follows the code; the paper (Section 4.4) states an inclusion.
4. **Negatives of C ⊑ ∃r.D** return the overlap
   :math:`\\lVert\\max(0, o_C + o_D + o_r - |c_C - c_D - c_r| - m)\\rVert` plus the
   centre-norm terms; the training regime pushes it to ``neg_dist`` with a squared loss
   (``(neg_dist - x)^2``), so it is pushed *up* to the target as well as down.
5. **Regularisation**: ``reg_factor`` weights only
   :math:`\\mathrm{mean}_r \\max(0, \\lVert o_r\\rVert - 1)` over the role offsets.

All losses here are unsquared per-axiom scores; ``margin`` follows the module's strict
convention (the slack margin of the code is ``margin = -γ``, which
:class:`~mowl.training.SlackMargin` sets).
"""

import torch as th
import torch.nn.functional as F


def centre_norm_penalty(centre):
    """:math:`\\big|\\lVert c\\rVert - 1\\big|` per row (``reg`` of the code)."""
    return (th.linalg.norm(centre, dim=1) - 1).abs()


def _norm(x):
    return th.linalg.norm(x, dim=1)


def gci0_loss(data, class_center, class_offset, margin):
    c, oc = class_center(data[:, 0]), class_offset(data[:, 0])
    d, od = class_center(data[:, 1]), class_offset(data[:, 1])
    dst = _norm(F.relu((c - d).abs() + oc - od + margin))
    return dst + centre_norm_penalty(c) + centre_norm_penalty(d)


def gci1_terms(data, class_center, class_offset, margin):
    """The two parts of :func:`gci1_loss`: the emptiness penalty :math:`p` and the
    masked inclusion term plus centre-norm terms :math:`q` (``gci1_loss = p + q``)."""
    c, oc = class_center(data[:, 0]), class_offset(data[:, 0])
    d, od = class_center(data[:, 1]), class_offset(data[:, 1])
    e, oe = class_center(data[:, 2]), class_offset(data[:, 2])
    start = th.maximum(c - oc, d - od)
    end = th.minimum(c + oc, d + od)
    mask = (start <= end).to(start.dtype)
    empty = _norm(F.relu(start - end))
    s = ((((start + end) / 2) - e).abs() + (start - end) / 2 - oe) * mask
    rest = _norm(F.relu(s + margin * mask))
    rest = rest + centre_norm_penalty(c) + centre_norm_penalty(d) + centre_norm_penalty(e)
    return empty, rest


def gci1_loss(data, class_center, class_offset, margin):
    p, q = gci1_terms(data, class_center, class_offset, margin)
    return p + q


def disjointness_loss(data, class_center, class_offset, margin):
    """:math:`C \\sqcap D \\sqsubseteq \\bot` (``disJointLoss`` of the code): the boxes of
    :math:`C` and :math:`D` must not overlap."""
    c, oc = class_center(data[:, 0]), class_offset(data[:, 0])
    d, od = class_center(data[:, 1]), class_offset(data[:, 1])
    dst = _norm(F.relu(-(c - d).abs() + oc + od + margin))
    return dst + centre_norm_penalty(c) + centre_norm_penalty(d)


def gci2_loss(data, class_center, class_offset, relation_center, relation_offset, margin,
              enhanced=True, neg=False):
    c, oc = class_center(data[:, 0]), class_offset(data[:, 0])
    r, orr = relation_center(data[:, 1]), relation_offset(data[:, 1])
    d, od = class_center(data[:, 2]), class_offset(data[:, 2])
    reg = centre_norm_penalty(c) + centre_norm_penalty(d)
    euc = (c - d - r).abs()
    if neg:
        return _norm(F.relu(oc + od + orr - euc - margin)) + reg
    if enhanced:
        return _norm(F.relu(euc + oc + od - orr + margin)) + reg
    # Without the enhancement: Box(C) ⊆ Box(∃r.D) (centre c_r + c_D, offset o_r + o_D).
    # The code's branch for this case drops c_C and is not used for the published runs.
    return _norm(F.relu(euc + oc - orr - od + margin)) + reg


def gci3_loss(data, class_center, class_offset, relation_center, relation_offset, margin):
    r, orr = relation_center(data[:, 0]), relation_offset(data[:, 0])
    c, oc = class_center(data[:, 1]), class_offset(data[:, 1])
    d, od = class_center(data[:, 2]), class_offset(data[:, 2])
    dst = _norm(F.relu((c + r - d).abs() + orr + oc - od + margin))
    return dst + centre_norm_penalty(c) + centre_norm_penalty(d)


def regularization_loss(relation_offset, reg_factor):
    """``reg_factor`` times the mean excess of the role offset norms over 1."""
    return reg_factor * F.relu(th.linalg.norm(relation_offset.weight, dim=1) - 1).mean()
