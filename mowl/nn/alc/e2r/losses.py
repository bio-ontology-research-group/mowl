r"""Loss terms of Embed2Reason (E2R; Garg et al., NeurIPS 2019, Sec. 4).

Representation (Sec. 4, "In E2R, we assume the underlying field of the embedding space
:math:`\mathbb{C}^d` is :math:`\mathbb{R}`"): the space :math:`\mathbb{C}^d` is simulated by
:math:`\mathbb{R}^{2d}`; coordinates :math:`0 \dots d-1` are the real parts and
:math:`d \dots 2d-1` the imaginary parts. Every predicate is an *axis-parallel* subspace,
given by a :math:`2d`-dimensional indicator vector :math:`y` (``1`` = the axis belongs to
the subspace). Entities are vectors :math:`x = (x_R, 0)`; the ordered pair
:math:`(a, b)` is :math:`x_{ab} = (x_{aR}, x_{bR})`. Concepts live in the real half,
relations use both halves.

Under this representation the quantum-logic operations of Sec. 3 become coordinatewise
operations on indicator vectors (they are used verbatim in eqs. (5)-(8)):

* :math:`S_i \wedge S_j = S_i \cap S_j`: product :math:`y_i \odot y_j`;
* :math:`S_i \vee S_j = S_i + S_j` (span, *not* set union): :math:`\max(y_i, y_j)`;
* :math:`\neg S = S^\perp`: :math:`1 - y` (on the relevant half, see
  :func:`negation_loss`);
* :math:`S_i \sqsubseteq S_j`: :math:`y_i^\top (1 - y_j) = 0`;
* membership of :math:`x` in :math:`S`: the residual :math:`(1 - y) \odot x` is zero.

Every function returns one non-negative value per row (shape ``[N]``), except the
negated membership losses, which are bounded below but may be negative (Sec. 4,
"take the negation of the corresponding membership loss").

Deviations from the printed equations (also stated in the class docstrings and the
catalogue card):

* Eq. (4), second term :math:`(1_{2d}^\top((\begin{smallmatrix}0&I\\I&0\end{smallmatrix})
  \overline{y}_j \odot x_i))^2`: the text says it is "the squared length of complex part
  of the projected vector" and that it "would be zero for :math:`L_{(O_i\in C_j)}` due to
  the assumptions about structures of :math:`x_i` and :math:`y_j`". As printed it equals
  :math:`(\sum_t \overline{y}_j[t+d]\, x_i[t])^2 = (\sum_t x_{iR}[t])^2` for a
  well-formed concept and entity, which is not zero. We could not reconstruct the
  intended term and omit it; the residual term alone is zero iff :math:`x` lies in the
  subspace.
* Eq. (4), last two terms of :math:`L_{R_i(O_p,O_q)}`: as printed the second one compares
  the imaginary half of :math:`x_{pq}` with the *real* half of :math:`x_q`, which
  contradicts the stated structure :math:`x_{pq} = (x_{pR}, x_{qR})`. We build
  :math:`x_{pq}` from the entity vectors exactly in the stated form
  (:func:`pair_vector`), so both terms are zero by construction.
* Eq. (8) for concepts: with the imaginary half of every concept fixed to ``0`` (eq. (2)),
  :math:`\overline{y}_i^\top \overline{y}_j \geq d` and the loss can never vanish. We
  evaluate the concept negation loss on the real half only, i.e. :math:`\neg C` is the
  orthogonal complement inside the concept universe :math:`\mathbb{R}^d \times \{0\}`
  that contains every entity vector. For relations eq. (8) is used on all :math:`2d`
  coordinates as printed.
"""
import torch as th


def _halves(v, dim):
    return v[..., :dim], v[..., dim:]


def complement(y):
    r"""Bit-flipped indicator :math:`\overline{y} = 1 - y` (eq. (3))."""
    return 1 - y


def meet(*ys):
    r"""Indicator of :math:`S_1 \cap \dots \cap S_k`: elementwise product (eq. (6))."""
    out = ys[0]
    for y in ys[1:]:
        out = out * y
    return out


def join(*ys):
    r"""Indicator of the span :math:`S_1 + \dots + S_k`: elementwise maximum (eq. (7))."""
    out = ys[0]
    for y in ys[1:]:
        out = th.maximum(out, y)
    return out


def pair_vector(x_p, x_q, dim):
    r"""Vector of the ordered pair :math:`(p, q)`: :math:`x_{pq} = (x_{pR}, x_{qR})`
    (Sec. 4, "Entities, Pair of Entities, and Predicates")."""
    return th.cat([x_p[..., :dim], x_q[..., :dim]], dim=-1)


# -- structural constraints ------------------------------------------------------------

def entity_structure_loss(x, dim):
    r"""Eq. (1): :math:`\|x \odot (0_d, 1_d)\|^2` (the imaginary half of an entity is 0)."""
    return (_halves(x, dim)[1] ** 2).sum(-1)


def concept_structure_loss(y, dim):
    r"""Eq. (2), :math:`L_{C}`: :math:`\|y \odot (0_d, 1_d)\|^2` (a concept has no
    imaginary axes)."""
    return (_halves(y, dim)[1] ** 2).sum(-1)


def relation_structure_loss(z, dim):
    r"""Eq. (2), :math:`L_{R}`: :math:`\min\{0, z^\top(0_d, 1_d) - 1\}^2 +
    \min\{0, z^\top(1_d, 0_d) - 1\}^2` (a relation has at least one real and at least
    one imaginary axis)."""
    real, imag = _halves(z, dim)
    too_few_imag = th.clamp(imag.sum(-1) - 1, max=0) ** 2
    too_few_real = th.clamp(real.sum(-1) - 1, max=0) ** 2
    return too_few_imag + too_few_real


def binary_loss(y):
    r"""Eq. (3): :math:`\|y \odot \overline{y}\|^2` (indicators should be 0/1)."""
    return ((y * complement(y)) ** 2).sum(-1)


# -- A-Box -----------------------------------------------------------------------------

def residual_sq(y, x):
    r"""Squared length of the residual of the orthogonal projection of :math:`x` onto
    the axis-parallel subspace :math:`y`: :math:`\|\overline{y} \odot x\|^2`."""
    return ((complement(y) * x) ** 2).sum(-1)


def unit_length_loss(x):
    r""":math:`(1 - x^\top x)^2` (last term of :math:`L_{(O_i \in C_j)}`, eq. (4))."""
    return (1 - (x * x).sum(-1)) ** 2


def membership_loss(y, x):
    r"""Eq. (4), :math:`L_{(O_i \in C_j)} = \|\overline{y}_j \odot x_i\|^2 +
    (1 - x_i^\top x_i)^2`. The second printed term is omitted (see module docstring)."""
    return residual_sq(y, x) + unit_length_loss(x)


def role_membership_loss(z, x_p, x_q, dim, unit_length=True):
    r"""Eq. (4), :math:`L_{R_i(O_p, O_q)} = \|\overline{z}_i \odot x_{pq}\|^2` with
    :math:`x_{pq}` built by :func:`pair_vector` (the structure terms are then zero).

    :param unit_length: Add :math:`(1-\|x_p\|^2)^2 + (1-\|x_q\|^2)^2`. Not printed in
        eq. (4) for role assertions, but Sec. 4 states that entities are unit vectors;
        without it an entity that occurs only in role assertions can shrink to ``0``,
        which lies in every subspace. Defaults to ``True``.
    """
    loss = residual_sq(z, pair_vector(x_p, x_q, dim))
    if unit_length:
        loss = loss + unit_length_loss(x_p) + unit_length_loss(x_q)
    return loss


def negative_membership_loss(y, x):
    r"""Negated membership loss of a corrupted assertion (Sec. 4, last paragraph):
    :math:`-\|\overline{y} \odot x\|^2 + (1 - x^\top x)^2`. Only the residual is negated;
    the unit-length term stays positive so that the loss is bounded below (by
    :math:`-5/4` when :math:`y \in [0, 1]^{2d}`)."""
    return -residual_sq(y, x) + unit_length_loss(x)


def negative_role_membership_loss(z, x_p, x_q, dim):
    r"""Negated role membership loss of a corrupted pair (Sec. 4, last paragraph):
    :math:`-\|\overline{z} \odot x_{pq}\|^2` plus the unit-length terms."""
    unit = unit_length_loss(x_p) + unit_length_loss(x_q)
    return unit - residual_sq(z, pair_vector(x_p, x_q, dim))


# -- T-Box -----------------------------------------------------------------------------

def inclusion_loss(y_sub, y_sup):
    r"""Eq. (5): :math:`L_{C_i \sqsubseteq C_j} = \|y_i \odot \overline{y}_j\|^2` (same for
    relations)."""
    return ((y_sub * complement(y_sup)) ** 2).sum(-1)


def conjunction_loss(y, *operands):
    r"""Eq. (6): :math:`L_{(C_i = C_j \sqcap C_k)} = \|y_i - (y_j \odot y_k)\|^2`, for any
    number of operands. With :math:`y_i = 0` (:math:`\bot = \{0\}`, Sec. 3) it is the
    disjointness loss :math:`\|y_j \odot y_k\|^2`."""
    return ((y - meet(*operands)) ** 2).sum(-1)


def disjunction_loss(y, *operands):
    r"""Eq. (7): :math:`L_{(C_i = C_j \sqcup C_k)} = \|y_i - \max(y_j, y_k)\|^2`."""
    return ((y - join(*operands)) ** 2).sum(-1)


def disjointness_loss(y_a, y_b):
    r""":math:`A \sqcap B = \bot`: eq. (6) with :math:`y_i = 0`, i.e.
    :math:`\|y_a \odot y_b\|^2`. The paper defines :math:`\bot = \{0\}` (Sec. 3) but has
    no separate disjointness loss."""
    return ((y_a * y_b) ** 2).sum(-1)


def negation_loss(y, y_neg, dim=None):
    r"""Eq. (8): :math:`L_{(C_i = \neg C_j)} = (y_i^\top y_j)^2 +
    (\overline{y}_i^\top \overline{y}_j)^2`.

    :param dim: If given, the loss is evaluated on the first ``dim`` coordinates only
        (concept negation inside the real half; see module docstring). ``None`` (the
        default) uses all coordinates (relations).
    """
    if dim is not None:
        y, y_neg = y[..., :dim], y_neg[..., :dim]
    overlap = (y * y_neg).sum(-1) ** 2
    uncovered = (complement(y) * complement(y_neg)).sum(-1) ** 2
    return overlap + uncovered


def range_loss(y_k, z, dim):
    r"""Eq. (9): :math:`L_{(\forall R_i \cdot C_j)}(y_k) = (y_k^\top
    (\begin{smallmatrix}0&I\\0&0\end{smallmatrix}) z_i)^2 = (y_{kR}^\top z_{iI})^2` for a
    concept :math:`C_k` that is not a descendant of :math:`C_j`: the second-argument
    (imaginary) axes of :math:`R_i` must avoid the axes of :math:`C_k`."""
    return ((y_k[..., :dim] * z[..., dim:]).sum(-1)) ** 2


# -- regularisers ----------------------------------------------------------------------

def orthogonality_loss(y_i, y_j):
    r"""Summand of eq. (11), :math:`\Omega_{\text{orth}}`: :math:`(y_i^\top y_j)^2` for
    siblings."""
    return (y_i * y_j).sum(-1) ** 2


def separation_loss(y_sub, y_sup, gap):
    r"""Summand of eq. (12), :math:`\Omega_{\text{sep}}`: :math:`(g - (y_j - y_i)^\top
    1)^2` for :math:`C_i \sqsubseteq C_j`; the paper uses :math:`g = \sqrt{d}` and allows
    any constant :math:`\geq 1`."""
    return (gap - (y_sup - y_sub).sum(-1)) ** 2
