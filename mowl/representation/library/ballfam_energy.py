r"""The two dissimilarities of RegD ([yang2025regd]_) for boxes and balls.

RegD replaces the inclusion energy between two regions by

.. math::

    E(\mathrm{reg}_u, \mathrm{reg}_v) = d_{bd}(\mathrm{reg}_u, \mathrm{reg}_v)
        + \lambda \cdot d_{dep}(\mathrm{reg}_u, \mathrm{reg}_v),

where :math:`\mathrm{reg}_u` is the *parent* (the larger region, the one that should contain
the other) and :math:`\mathrm{reg}_v` the *child*. In ontology terms, for :math:`C \sqsubseteq D`
the parent is the region of :math:`D` and the child the region of :math:`C`.

* The **boundary dissimilarity** :math:`d_{bd}` (Definition 2) is asymmetric. It is
  non-positive iff the child is contained in the parent, and then it is minus the smallest
  distance from the child to the complement of the parent. Otherwise it is the largest
  distance from a point of the child to the parent.
* The **depth dissimilarity** :math:`d_{dep}` (Definition 1) is symmetric and non-negative.
  It divides a distance between the region parameters by the product of the *sizes* of the
  regions, so it grows without bound as regions shrink. This is what gives the model
  hyperbolic-like capacity in low dimension (Theorems 1 and 2).

All functions are batched: the last axis of a box centre or offset is the embedding axis,
and all leading axes broadcast. Balls have a centre of shape ``(..., n)`` and a radius of shape
``(...)``. The radii and offsets must be strictly positive.
"""

import torch as th


def _lp_power(x, p):
    """Return :math:`\\sum_i |x_i|^p` over the last axis."""
    if p == 1:
        return th.abs(x).sum(dim=-1)
    if p == 2:
        return th.square(x).sum(dim=-1)
    return th.abs(x).pow(p).sum(dim=-1)


def ball_boundary_dissimilarity(parent_center, parent_radius, child_center, child_radius):
    r"""Boundary dissimilarity of two balls (Example 3).

    .. math:: d_{bd} = \lVert c_1 - c_2 \rVert_2 + r_2 - r_1

    where ball 1 is the parent and ball 2 the child. It is :math:`\le 0` iff the child is
    contained in the parent, and :math:`0` iff the balls are internally tangent.

    :param parent_center: Centres of the parents, shape ``(..., n)``.
    :param parent_radius: Radii of the parents, shape ``(...)``.
    :param child_center: Centres of the children, shape ``(..., n)``.
    :param child_radius: Radii of the children, shape ``(...)``.
    :return: Tensor of shape ``(...)``.
    """
    return th.linalg.norm(parent_center - child_center, dim=-1) + child_radius - parent_radius


def ball_depth_dissimilarity(parent_center, parent_radius, child_center, child_radius, p=2):
    r"""Depth dissimilarity of two balls (Example 1, with :math:`g(x) = x`, :math:`f = r`).

    .. math:: d_{dep} = \frac{\lVert c_1 - c_2 \rVert_p^p + |r_1 - r_2|^p}{r_1 r_2}

    :param p: The exponent of the :math:`L_p` norm. The paper tests ``1`` and ``2`` and finds
        ``2`` better for balls. Defaults to 2.
    :return: Tensor of shape ``(...)``.
    """
    numerator = _lp_power(parent_center - child_center, p) \
        + th.abs(parent_radius - child_radius).pow(p)
    return numerator / (parent_radius * child_radius)


def box_boundary_dissimilarity(parent_center, parent_offset, child_center, child_offset):
    r"""Boundary dissimilarity of two boxes (Example 3).

    With :math:`a = |c_1 - c_2| + o_2 - o_1` (box 1 is the parent, box 2 the child):

    .. math::
        d_{bd} = \begin{cases} \max_i a_i & \text{if } a \le 0 \text{ (child contained)},\\
        \lVert \max(a, 0) \rVert_2 & \text{otherwise.} \end{cases}

    The two cases agree at :math:`\max_i a_i = 0`, so the function is continuous. It is
    :math:`\le 0` iff the child is contained in the parent.

    :param parent_center: Centres of the parents, shape ``(..., n)``.
    :param parent_offset: Positive half side lengths of the parents, shape ``(..., n)``.
    :param child_center: Centres of the children, shape ``(..., n)``.
    :param child_offset: Positive half side lengths of the children, shape ``(..., n)``.
    :return: Tensor of shape ``(...)``.
    """
    diff = th.abs(parent_center - child_center) + child_offset - parent_offset
    contained = diff.max(dim=-1).values
    outside = th.linalg.norm(th.relu(diff), dim=-1)
    return th.where(contained < 0, contained, outside)


def box_depth_dissimilarity(parent_center, parent_offset, child_center, child_offset, p=1):
    r"""Depth dissimilarity of two boxes (Example 2: :math:`g(x) = x`,
    :math:`f = \lVert o \rVert`).

    .. math:: d_{dep} = \frac{\lVert c_1 - c_2 \rVert_p^p + \lVert o_1 - o_2 \rVert_p^p}
        {\lVert o_1 \rVert_2 \, \lVert o_2 \rVert_2}

    :param p: The exponent of the :math:`L_p` norm in the numerator. The paper tests ``1`` and
        ``2`` and finds ``1`` better for boxes. Defaults to 1. The size :math:`f` is always the
        Euclidean norm of the offset.
    :return: Tensor of shape ``(...)``.
    """
    numerator = _lp_power(parent_center - child_center, p) \
        + _lp_power(parent_offset - child_offset, p)
    denominator = th.linalg.norm(parent_offset, dim=-1) * th.linalg.norm(child_offset, dim=-1)
    return numerator / denominator


def _combine(boundary, depth, lamda, depth_on_containment):
    if not depth_on_containment:
        depth = th.where(boundary > 0, depth, th.zeros_like(depth))
    return boundary + lamda * depth


def ball_energy(parent_center, parent_radius, child_center, child_radius,
                lamda=1.0, p=2, depth_on_containment=True):
    r"""RegD energy :math:`E = d_{bd} + \lambda\, d_{dep}` of two balls (Eq. 9).

    :param lamda: Weight :math:`\lambda` of the depth dissimilarity. The paper selects it from
        ``{0, 0.5, 1}``.
    :param depth_on_containment: If ``True`` (Eq. 9 of the paper), the depth term is always
        added. If ``False``, it is added only where the child is not contained in the parent
        (:math:`d_{bd} > 0`), which is what the authors' released training code does.
    :return: Tensor of shape ``(...)``; it is :math:`\le 0` only if the depth term is absent or
        the boundary term cancels it.
    """
    boundary = ball_boundary_dissimilarity(parent_center, parent_radius, child_center,
                                           child_radius)
    depth = ball_depth_dissimilarity(parent_center, parent_radius, child_center, child_radius,
                                     p=p)
    return _combine(boundary, depth, lamda, depth_on_containment)


def box_energy(parent_center, parent_offset, child_center, child_offset,
               lamda=1.0, p=1, depth_on_containment=True):
    r"""RegD energy :math:`E = d_{bd} + \lambda\, d_{dep}` of two boxes (Eq. 9).

    See :func:`ball_energy` for the parameters.
    """
    boundary = box_boundary_dissimilarity(parent_center, parent_offset, child_center,
                                          child_offset)
    depth = box_depth_dissimilarity(parent_center, parent_offset, child_center, child_offset,
                                    p=p)
    return _combine(boundary, depth, lamda, depth_on_containment)


def negative_hinge(negative_boundary, gamma2=0.0):
    r"""Negative term :math:`\max\{\gamma_2 - d_{bd}(u, v'), 0\}` of Eq. 10.

    It is zero iff the negative child lies outside the parent by at least :math:`\gamma_2`.
    """
    return th.clamp(gamma2 - negative_boundary, min=0.0)


def contrastive_loss(positive_energy, negative_scores, gamma1=0.0):
    r"""RegD training loss of Section 3.3 (Eq. 10), per positive pair.

    .. math::
        L = \max\{E(u, v), \gamma_1\}
            + \log \sum_{(u, v') \in N} e^{\max\{\gamma_2 - d_{bd}(u, v'), 0\}}

    The first term pulls a positive pair into containment, the second pushes each sampled
    negative out of the parent by at least :math:`\gamma_2`. Following the paper, the negatives
    use the boundary dissimilarity only. The paper sums over all pairs; here the negatives are
    grouped per positive pair and the result is averaged over the positive pairs.

    :param positive_energy: Energies :math:`E(u, v)` of the positive pairs, shape ``(n,)``.
    :param negative_scores: The values :func:`negative_hinge` of the negatives, shape
        ``(n, k)``: ``k`` negatives per positive pair.
    :param gamma1: Margin :math:`\gamma_1` of the positive term. The paper uses 0 for ontologies.
    :return: Scalar tensor.
    """
    positive = th.clamp(positive_energy, min=gamma1)
    negative = th.logsumexp(negative_scores, dim=1)
    return (positive + negative).mean()


#: The boundary dissimilarity and the energy of each region shape, by shape name.
BOUNDARY = {"ball": ball_boundary_dissimilarity, "box": box_boundary_dissimilarity}
ENERGY = {"ball": ball_energy, "box": box_energy}
