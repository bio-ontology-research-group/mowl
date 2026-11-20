"""Loss functions of EmEL++ [mondal2021]_ and EmEL(var) [mohapatra2021]_.

All functions work on already gathered tensors (centres of shape ``(n, d)``, radii of shape
``(n, 1)``, non-negative) and return one value per row, shape ``(n, 1)``. The equation
numbers refer to Mondal et al. (2021) unless stated otherwise.

Every loss of an axiom that involves centres or role vectors carries the normalisation terms
:math:`|\\lVert x \\rVert - 1|` of the published equations, one for each embedding vector that
occurs in the axiom.
"""

import torch as th
import torch.nn.functional as F


def unit_norm_penalty(*vectors):
    """Sum over the given vectors of :math:`|\\lVert x \\rVert - 1|`, shape ``(n, 1)``."""
    total = 0
    for v in vectors:
        total = total + th.abs(th.linalg.norm(v, dim=1, keepdim=True) - 1)
    return total


def gci0_loss(c, rc, d, rd, margin, shift=None):
    """Eq. 1: :math:`C \\sqsubseteq D`.

    :param shift: Optional vector added to the centre of :math:`C` before the distance is
        taken (the learned "SubClassOf" translation of the EmEL++ Keras code).
    """
    moved = c if shift is None else c + shift
    dist = th.linalg.norm(moved - d, dim=1, keepdim=True)
    return th.relu(dist + rc - rd - margin) + unit_norm_penalty(c, d)


def gci1_loss(c, rc, d, rd, e, re, margin, radius_term=True):
    """Eq. 2: :math:`C \\sqcap D \\sqsubseteq E`, including the term on the smaller radius.

    :param radius_term: Set to ``False`` to drop the term
        :math:`\\max(0, \\min(r_C, r_D) - r_E - \\gamma)`, which the EmEL++ baseline of the
        Box\\ :sup:`2`\\ EL code (``model/EmELpp.py``) has commented out.
    """
    dist_cd = th.linalg.norm(c - d, dim=1, keepdim=True)
    dist_ce = th.linalg.norm(c - e, dim=1, keepdim=True)
    dist_de = th.linalg.norm(d - e, dim=1, keepdim=True)
    terms = [th.relu(dist_cd - rc - rd - margin),
             th.relu(dist_ce - rc - margin),
             th.relu(dist_de - rd - margin),
             unit_norm_penalty(c, d, e)]
    if radius_term:
        terms.append(th.relu(th.minimum(rc, rd) - re - margin))
    return sum(terms)


def gci2_loss(c, rc, rel, d, rd, margin, sigma=None, sigma_reg_weight=1.0):
    """Eq. 3 (EmEL++) and Eq. 7 of [mohapatra2021]_ (EmEL(var), if ``sigma`` is given):
    :math:`C \\sqsubseteq \\exists R.D`.

    :param sigma: Non-negative variance of the role, shape ``(n, 1)``, or ``None``.
    :param sigma_reg_weight: Weight of the term :math:`+\\sigma(r)` of Eq. 7. The paper's
        loss has weight 1.
    """
    dist = th.linalg.norm(c + rel - d, dim=1, keepdim=True)
    slack = 0 if sigma is None else sigma
    loss = th.relu(dist + rc - rd - slack - margin) + unit_norm_penalty(c, d)
    if sigma is not None:
        loss = loss + sigma_reg_weight * sigma
    return loss


def gci3_loss(rel, c, rc, d, rd, margin, sigma=None, sigma_reg_weight=1.0):
    """Eq. 4 (EmEL++) and Eq. 8 of [mohapatra2021]_ (EmEL(var)):
    :math:`\\exists R.C \\sqsubseteq D`."""
    dist = th.linalg.norm(c - rel - d, dim=1, keepdim=True)
    slack = 0 if sigma is None else sigma
    loss = th.relu(dist - rc - rd - slack - margin) + unit_norm_penalty(c, d)
    if sigma is not None:
        loss = loss + sigma_reg_weight * sigma
    return loss


def gci2_neg_loss(c, rc, rel, d, rd, margin, linear=False):
    """Eq. 10: negative loss for :math:`C \\not\\sqsubseteq \\exists R.D`.

    :param linear: Drop the hinge, as the EmEL++ baseline of the Box\\ :sup:`2`\\ EL code
        does (``model/EmELpp.py``, ``neg_loss``).
    """
    dist = th.linalg.norm(c + rel - d, dim=1, keepdim=True)
    violation = rc + rd - dist + margin
    if not linear:
        violation = th.relu(violation)
    return violation + unit_norm_penalty(c, d)


def gci1_bot_loss(c, rc, d, rd, margin):
    """Eq. 6: :math:`C \\sqcap D \\sqsubseteq \\bot`."""
    dist = th.linalg.norm(c - d, dim=1, keepdim=True)
    return th.relu(rc + rd - dist + margin) + unit_norm_penalty(c, d)


def bot_loss(rc):
    """Eq. 5 and Eq. 7 of the paper (``C ⊑ ⊥`` and ``∃R.C ⊑ ⊥``): the radius of ``C``."""
    return rc


def _direction_term(a, b, direction):
    """:math:`|1 - \\cos(a, b)|` per row (``"row"``), or the term of the EmEL++ Keras
    code (``"batch"``): every coordinate is normalised over the *batch*
    (``tf.nn.l2_normalize(x, 0)``) and all products are summed, giving one number per
    batch that is added to every row."""
    if direction == "batch":
        total = (F.normalize(a, dim=0) * F.normalize(b, dim=0)).sum()
        return th.abs(1 - total).expand(len(a), 1)
    return th.abs(1 - F.cosine_similarity(a, b, dim=1).unsqueeze(1))


def role_inclusion_loss(r, s, margin, direction="row"):
    """Eq. 8: :math:`R \\sqsubseteq S`.

    The loss is symmetric in ``r`` and ``s``: the distance, the cosine and the two
    normalisation terms are all symmetric, so it also penalises a violation of
    :math:`S \\sqsubseteq R` (see the limitation recorded on the catalogue card).

    :param direction: ``"row"`` (the cosine of every pair, as in the paper) or
        ``"batch"`` (the EmEL++ code, see :func:`_direction_term`).
    """
    dist = th.linalg.norm(s - r, dim=1, keepdim=True)
    direction = _direction_term(r, s, direction)
    return th.relu(dist - margin) + direction + unit_norm_penalty(r, s)


def role_chain_loss(r1, r2, s, margin, direction="row"):
    """Eq. 9: :math:`R_1 \\circ R_2 \\sqsubseteq S`.

    :param direction: See :func:`role_inclusion_loss`.
    """
    total = r1 + r2
    dist = th.linalg.norm(s - total, dim=1, keepdim=True)
    direction = _direction_term(total, s, direction)
    return th.relu(dist - margin) + direction + unit_norm_penalty(r1, r2, s)
