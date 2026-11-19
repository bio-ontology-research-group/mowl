r"""Geometry and loss primitives of ELK-Em [singh2026]_.

Boxes are handled as pairs ``(lower, upper)`` of tensors of shape ``(*, n)``. A box is
empty when ``upper < lower`` in at least one dimension. A role :math:`r` is a tuple
``(range_lower, range_upper, error_lower, error_upper, slope, shift)`` describing the range
box :math:`R_r`, the error box :math:`E_r` and the diagonal affine map
:math:`T_r(x) = m_r x + c_r` with :math:`m_r > 0`, so that

.. math::
    r^{\mathcal{E}} = \{(x, y) \mid x \in T_r(y) \oplus E_r,\ y \in R_r\},
    \qquad (\exists r.C)^{\mathcal{E}} = T_r(C^{\mathcal{E}} \cap R_r) \oplus E_r.

The three loss primitives follow Eqs. (1)--(3) of the paper, with
:math:`d(B_x, B_y) = |c_x - c_y| - o_x - o_y` the per-dimension separation of two boxes
with centres :math:`c` and half-widths :math:`o`:

* :func:`inclusion_loss`: :math:`\ell^\gamma_\subseteq = \frac{1}{n}\lVert
  \mathrm{ReLU}(d + 2 o_x + \gamma)\rVert_1` (zero for an empty :math:`B_x`, since the
  empty box is included in every box),
* :func:`overlap_loss`: :math:`\ell^\gamma_{\cap+} = \frac{1}{n}\lVert
  \mathrm{ReLU}(d + \gamma)\rVert_1`,
* :func:`disjoint_loss`: :math:`\ell^\gamma_{\cap-} = \mathrm{ReLU}(\gamma - \max_i d_i)`.

All functions are pure and differentiable; none of them reads module parameters.
"""

import torch as th
import torch.nn.functional as F

#: Half-width used to represent the full box (:math:`\top`) and, with swapped signs, the
#: canonical empty box (:math:`\bot`) with finite numbers.
BIG = 1e6


def center_offset(lower, upper):
    """Centre and (signed) half-width of a box."""
    return (lower + upper) / 2, (upper - lower) / 2


def is_empty(lower, upper):
    """``True`` for the rows whose box is empty (``upper < lower`` in some dimension)."""
    return (lower > upper).any(dim=-1)


def separation(lx, ux, ly, uy):
    r"""Per-dimension separation :math:`d(B_x, B_y) = |c_x - c_y| - o_x - o_y`.

    :math:`d_i \leq 0` iff the projections of the two boxes on dimension :math:`i` meet.
    """
    cx, ox = center_offset(lx, ux)
    cy, oy = center_offset(ly, uy)
    return (cx - cy).abs() - ox - oy


def inclusion_loss(lx, ux, ly, uy, margin):
    r"""Inclusion primitive :math:`\ell^\gamma_\subseteq(B_x, B_y)` (Eq. 1).

    Zero iff :math:`B_x \subseteq B_y` (for ``margin = 0`` and non-empty boxes); zero by
    definition when :math:`B_x` is empty.

    :rtype: :class:`torch.Tensor` of shape ``(*,)``
    """
    cx, ox = center_offset(lx, ux)
    cy, oy = center_offset(ly, uy)
    # d + 2 o_x = |c_x - c_y| + o_x - o_y
    loss = F.relu((cx - cy).abs() + ox - oy + margin).mean(dim=-1)
    return th.where(is_empty(lx, ux), th.zeros_like(loss), loss)


def overlap_loss(lx, ux, ly, uy, margin):
    r"""Overlap primitive :math:`\ell^\gamma_{\cap+}(B_x, B_y)` (Eq. 2)."""
    return F.relu(separation(lx, ux, ly, uy) + margin).mean(dim=-1)


def disjoint_loss(lx, ux, ly, uy, margin):
    r"""Disjointness primitive :math:`\ell^\gamma_{\cap-}(B_x, B_y)` (Eq. 3).

    For ``margin > 0`` a zero loss implies :math:`B_x \cap B_y = \emptyset`.
    """
    return F.relu(margin - separation(lx, ux, ly, uy).max(dim=-1).values)


def intersection(l1, u1, l2, u2):
    """Box intersection :math:`[\\max(l_1, l_2), \\min(u_1, u_2)]`."""
    return th.maximum(l1, l2), th.minimum(u1, u2)


def minkowski_sum(l1, u1, l2, u2):
    """Box addition :math:`[l_1 + l_2, u_1 + u_2]`."""
    return l1 + l2, u1 + u2


def affine_image(lower, upper, slope, shift):
    """Image of a box under the diagonal affine map ``x -> slope * x + shift``, ``slope > 0``.

    An empty box stays empty.
    """
    return slope * lower + shift, slope * upper + shift


def existential_box(role, lower, upper):
    r"""Box of :math:`\exists r.C`: :math:`T_r(C^{\mathcal{E}} \cap R_r) \oplus E_r`.

    :param role: Role tuple ``(range_l, range_u, error_l, error_u, slope, shift)``.
    :param lower: Lower corners of :math:`C`.
    :param upper: Upper corners of :math:`C`.
    """
    rl, ru, el, eu, m, c = role
    il, iu = intersection(lower, upper, rl, ru)
    tl, tu = affine_image(il, iu, m, c)
    # The Minkowski sum of the empty set with E_r is empty, but interval arithmetic on an
    # "inverted" box would make it non-empty again whenever the gap is smaller than the
    # width of E_r (which would make a zero GCI2 loss unsound). An empty C ∩ R_r is
    # therefore returned as T_r(C ∩ R_r), which is still inverted and keeps a gradient
    # towards making C and R_r overlap.
    empty = is_empty(il, iu).unsqueeze(-1)
    return th.where(empty, tl, tl + el), th.where(empty, tu, tu + eu)


def compose_roles(r1, r2):
    r"""Over-approximation :math:`\max(r_1 \circ r_2)` of Proposition 4 of the paper.

    :math:`\langle R_{r_2},\ m_{r_1} E_{r_2} \oplus E_{r_1},\ T_{r_1} \circ T_{r_2}\rangle`.
    """
    _, _, el1, eu1, m1, c1 = r1
    rl2, ru2, el2, eu2, m2, c2 = r2
    el, eu = minkowski_sum(m1 * el2, m1 * eu2, el1, eu1)
    return rl2, ru2, el, eu, m1 * m2, m1 * c2 + c1


def role_inclusion_loss(r, s, margin):
    r"""RI0 loss of Table 2: :math:`\ell_\subseteq(R_r, R_s) + \sum_{y \in \{l^R_r, u^R_r\}}
    \ell_\subseteq(T_r(y) \oplus E_r, T_s(y) \oplus E_s)`.

    By Proposition 2 of the paper the margin-free loss is zero iff
    :math:`r^{\mathcal{E}} \subseteq s^{\mathcal{E}}`.
    """
    rl, ru, el_r, eu_r, m_r, c_r = r
    sl, su, el_s, eu_s, m_s, c_s = s
    loss = inclusion_loss(rl, ru, sl, su, margin)
    for y in (rl, ru):
        loss = loss + inclusion_loss(m_r * y + c_r + el_r, m_r * y + c_r + eu_r,
                                     m_s * y + c_s + el_s, m_s * y + c_s + eu_s, margin)
    return loss


def role_chain_loss(r1, r2, s, margin):
    r"""RI1 loss of Table 2: the RI0 loss of :math:`\max(r_1 \circ r_2)` against :math:`s`."""
    return role_inclusion_loss(compose_roles(r1, r2), s, margin)


def point_membership_loss(point, lower, upper, margin):
    r""":math:`\ell^\gamma_\subseteq([e, e], B)`: an individual as a degenerate box."""
    return inclusion_loss(point, point, lower, upper, margin)


def role_assertion_loss(subject, role, obj, margin):
    r"""Loss of :math:`r(a, b)`: :math:`\ell_\subseteq([e_a, e_a], T_r([e_b, e_b]) \oplus E_r)
    + \ell_\subseteq([e_b, e_b], R_r)` (Section 4.2 of the paper)."""
    rl, ru, el, eu, m, c = role
    image = m * obj + c
    to_subject = inclusion_loss(subject, subject, image + el, image + eu, margin)
    in_range = inclusion_loss(obj, obj, rl, ru, margin)
    return to_subject + in_range


def hinge_negative_loss(loss, threshold):
    r"""Loss of a corrupted assertion: :math:`\mathrm{ReLU}(\delta - \ell)`, with
    :math:`\delta = -\log\tau` for the target-score threshold :math:`\tau \in (0, 1)`."""
    delta = -th.log(th.tensor(threshold, dtype=loss.dtype, device=loss.device))
    return F.relu(delta - loss)


def pearson_contrastive_loss(pos_scores, neg_scores, eps=1e-8):
    r"""Pearson correlation loss of Eq. (5):
    :math:`1 - \frac{(\sigma - \bar\sigma)^\top (t - \bar t)}{\lVert\sigma - \bar\sigma\rVert
    \lVert t - \bar t\rVert + \varepsilon}`, with labels 1 for positives and 0 for negatives.

    ``pos_scores`` and ``neg_scores`` may have different lengths.
    """
    scores = th.cat([pos_scores, neg_scores])
    labels = th.cat([th.ones_like(pos_scores), th.zeros_like(neg_scores)])
    s = scores - scores.mean()
    t = labels - labels.mean()
    return 1 - (s * t).sum() / (s.norm() * t.norm() + eps)


def lexical_loss(points, features, decay):
    r"""Lexical regularisation loss of Eq. (6).

    :math:`\mathcal{L}_{lex} = \binom{m}{2}^{-1} \sum_{i<j} (\bar s^{ext}_{ij} -
    s^{geo}_{ij})^2`, with :math:`\bar s^{ext}_{ij} = (1 + \cos(z_i, z_j)) / 2` and
    :math:`s^{geo}_{ij} = \exp(-\frac{\lambda}{n}\lVert e_i - e_j\rVert_2^2)`.

    :param points: Point embeddings :math:`e_i` of the individuals, shape ``(m, n)``.
    :param features: External representations :math:`z_i`, shape ``(m, d)``.
    :param decay: :math:`\lambda > 0`.
    """
    m, n = points.shape
    if m < 2:
        return points.sum() * 0.0
    z = F.normalize(features, dim=1)
    s_ext = (1 + z @ z.t()) / 2
    sq_norms = points.pow(2).sum(dim=1)
    # Squared distances from the Gram matrix (cdist has an undefined gradient at 0).
    sq_dist = (sq_norms[:, None] + sq_norms[None, :] - 2 * points @ points.t()).clamp_min(0)
    s_geo = th.exp(-decay / n * sq_dist)
    i, j = th.triu_indices(m, m, offset=1, device=points.device)
    return (s_ext[i, j] - s_geo[i, j]).pow(2).mean()
