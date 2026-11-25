"""Loss functions for TransBox, an EL++-closed ontology embedding method [yang2025]_.

In TransBox, atomic concepts and roles are embedded as axis-aligned boxes
defined by a center ``c`` and a non-negative offset ``o`` in R^n, and
individuals as points. A complex EL++ concept is embedded by composition:

* ``Box(C ⊓ D) = Box(C) ∩ Box(D)``
* ``Box(∃r.C)`` has center ``c(r) + c(C)`` and offset ``o(r) + o(C)``
* ``Box(∃rall.C)`` (semantic enhancement, training only) has center
  ``c(r) + c(C)`` and offset ``max(0, o(r) - o(C))``
* ``Box(r ∘ t)`` has center ``c(r) + c(t)`` and offset ``o(r) + o(t)``

A role ``r`` holds between ``a`` and ``b`` iff ``x_a - x_b ∈ Box(r)``.

All losses are returned as per-row (per-axiom) scalar tensors; the base
:class:`~mowl.base_models.elmodel.EmbeddingELModel` takes the mean.
"""

import torch as th
import torch.nn.functional as F


def inclusion_margin(c1, o1, c2, o2):
    """Elementwise subsumption margin of box 1 in box 2.

    ``s = |c1 - c2| + o1 - o2``; box 1 is included in box 2 (elementwise)
    iff ``s <= 0``.

    :param c1: Centers of the first boxes, shape ``(*, n)``.
    :type c1: :class:`torch.Tensor`
    :param o1: Non-negative offsets of the first boxes, shape ``(*, n)``.
    :type o1: :class:`torch.Tensor`
    :param c2: Centers of the second boxes, shape ``(*, n)``.
    :type c2: :class:`torch.Tensor`
    :param o2: Non-negative offsets of the second boxes, shape ``(*, n)``.
    :type o2: :class:`torch.Tensor`
    :rtype: :class:`torch.Tensor`
    """
    return (c1 - c2).abs() + o1 - o2


def inclusion_loss(margin_tensor, margin):
    """Margin inclusion loss :math:`L_{\\subseteq} = \\lVert \\max(0, s + \\gamma) \\rVert_2`.

    :param margin_tensor: Elementwise subsumption margin (see
        :func:`inclusion_margin`), shape ``(batch, n)``.
    :type margin_tensor: :class:`torch.Tensor`
    :param margin: Margin :math:`\\gamma >= 0`.
    :type margin: float
    :rtype: :class:`torch.Tensor`
    """
    return th.norm(F.relu(margin_tensor + margin), dim=1)


def intersection_box(c1, o1, c2, o2):
    """Center, signed offset and non-emptiness mask of ``Box1 ∩ Box2``.

    The intersection of two axis-aligned boxes is again a box, possibly
    empty in some dimensions. Following the paper's "extended box
    intersections", empty dimensions are tracked with a mask instead of
    collapsing to a degenerate box.

    :returns: ``(center, offset, mask)`` where ``center`` has shape
        ``(batch, n)``, ``offset`` is the signed half-width (negative where
        the intersection is empty) and ``mask`` is 1.0 where the
        intersection is non-empty.
    """
    start = th.max(c1 - o1, c2 - o2)
    end = th.min(c1 + o1, c2 + o2)
    center = (start + end) / 2
    offset = (end - start) / 2
    mask = (start <= end).to(start.dtype)
    return center, offset, mask


def existential_box(c_role, o_role, c_con, o_con, enhanced=False):
    """Center and offset of ``Box(∃r.C)`` (or ``Box(∃rall.C)`` if enhanced).

    :param c_role: Role centers, shape ``(batch, n)``.
    :type c_role: :class:`torch.Tensor`
    :param o_role: Role offsets, shape ``(batch, n)``.
    :type o_role: :class:`torch.Tensor`
    :param c_con: Concept centers, shape ``(batch, n)``.
    :type c_con: :class:`torch.Tensor`
    :param o_con: Concept offsets, shape ``(batch, n)``.
    :type o_con: :class:`torch.Tensor`
    :param enhanced: If ``True``, use the semantic enhancement ``∃rall``
        (offset ``max(0, o(r) - o(C))``).
    :type enhanced: bool
    :rtype: tuple[:class:`torch.Tensor`, :class:`torch.Tensor`]
    """
    if enhanced:
        offset = th.clamp_min(o_role - o_con, 0)
    else:
        offset = o_role + o_con
    return c_role + c_con, offset


def gci0_loss(data, class_center, class_offset, margin, neg=False):
    """Loss for GCI0: :math:`C \\sqsubseteq D` — inclusion of ``Box(C)`` in ``Box(D)``.

    :param data: Tensor of shape ``(*, 2)`` with ``C`` at ``data[:, 0]`` and
        ``D`` at ``data[:, 1]``.
    :type data: :class:`torch.Tensor`
    """
    c = class_center(data[:, 0])
    o = class_offset(data[:, 0])
    d = class_center(data[:, 1])
    od = class_offset(data[:, 1])
    return inclusion_loss(inclusion_margin(c, o, d, od), margin)


def gci0_bot_loss(*args, **kwargs):
    """Loss for GCI0 with bottom: :math:`C \\sqsubseteq \\bot`.

    The bottom concept has its own box embedding, so this is the same
    inclusion loss as :func:`gci0_loss`.
    """
    return gci0_loss(*args, **kwargs)


def gci1_loss(data, class_center, class_offset, margin, neg=False):
    """Loss for GCI1: :math:`C_1 \\sqcap C_2 \\sqsubseteq D`.

    Inclusion of the (possibly empty) intersection box in ``Box(D)``, with
    an emptiness penalty for the dimensions where the intersection is empty.
    """
    c1 = class_center(data[:, 0])
    o1 = class_offset(data[:, 0])
    c2 = class_center(data[:, 1])
    o2 = class_offset(data[:, 1])
    d = class_center(data[:, 2])
    od = class_offset(data[:, 2])

    inter_c, inter_o, mask = intersection_box(c1, o1, c2, o2)

    # Penalty for the dimensions where the intersection is empty
    penalty = th.norm(F.relu(-inter_o), dim=1)

    # Inclusion of the non-empty part of the intersection in Box(D)
    s = inclusion_margin(inter_c, inter_o, d, od) * mask
    return penalty + th.norm(F.relu(s + margin * mask), dim=1)


def gci1_bot_loss(*args, **kwargs):
    """Loss for GCI1 with bottom: :math:`C_1 \\sqcap C_2 \\sqsubseteq \\bot`."""
    return gci1_loss(*args, **kwargs)


def gci2_loss(data, class_center, class_offset, relation_center, relation_offset,
              margin, enhanced=True, neg=False):
    """Loss for GCI2: :math:`C \\sqsubseteq \\exists R.D`.

    Inclusion of ``Box(C)`` in ``Box(∃R.D)`` — or in
    ``Box(∃Rall.D)`` when ``enhanced`` (the paper's semantic enhancement).

    For ``neg=True``, the paper's negative loss
    :math:`L_{\\nsubseteq} = (1 - \\lVert \\max(0, -d - \\gamma) \\rVert)^2`
    is used, where :math:`d = |c_1 - c_2| - o_1 - o_2` is the (signed)
    elementwise box distance; it discourages the negative axiom from holding.
    """
    c = class_center(data[:, 0])
    o = class_offset(data[:, 0])
    r = relation_center(data[:, 1])
    orr = relation_offset(data[:, 1])
    d = class_center(data[:, 2])
    od = class_offset(data[:, 2])

    e_c, e_o = existential_box(r, orr, d, od, enhanced=enhanced)

    if neg:
        overlap = o + e_o - (c - e_c).abs() - margin
        return (1 - th.norm(th.clamp_min(overlap, 0), dim=1)) ** 2
    return inclusion_loss(inclusion_margin(c, o, e_c, e_o), margin)


def gci3_loss(data, class_center, class_offset, relation_center, relation_offset,
              margin, neg=False):
    """Loss for GCI3: :math:`\\exists R.C \\sqsubseteq D`.

    Inclusion of ``Box(∃R.C)`` in ``Box(D)``.
    """
    r = relation_center(data[:, 0])
    orr = relation_offset(data[:, 0])
    c = class_center(data[:, 1])
    o = class_offset(data[:, 1])
    d = class_center(data[:, 2])
    od = class_offset(data[:, 2])

    e_c, e_o = existential_box(r, orr, c, o, enhanced=False)
    return inclusion_loss(inclusion_margin(e_c, e_o, d, od), margin)


def gci3_bot_loss(*args, **kwargs):
    """Loss for GCI3 with bottom: :math:`\\exists R.C \\sqsubseteq \\bot`."""
    return gci3_loss(*args, **kwargs)


def class_assertion_loss(data, ind_embed, class_center, class_offset, margin, neg=False):
    """Loss for class assertions: :math:`C(a)`.

    The individual point ``x_a`` must lie inside ``Box(C)``.
    """
    x = ind_embed(data[:, 0])
    c = class_center(data[:, 1])
    o = class_offset(data[:, 1])
    return th.norm(F.relu((x - c).abs() - o + margin), dim=1)


def object_property_assertion_loss(data, ind_embed, relation_center, relation_offset,
                                   margin, neg=False):
    """Loss for role assertions: :math:`R(a, b)`.

    The translation ``x_a - x_b`` must lie inside ``Box(R)``.
    """
    x_a = ind_embed(data[:, 0])
    r = relation_center(data[:, 1])
    orr = relation_offset(data[:, 1])
    x_b = ind_embed(data[:, 2])
    return th.norm(F.relu((x_a - x_b - r).abs() - orr + margin), dim=1)


def role_inclusion_loss(data, relation_center, relation_offset, margin, neg=False):
    """Loss for role inclusions (EL++): :math:`r \\sqsubseteq s`.

    Inclusion of ``Box(r)`` in ``Box(s)``.
    """
    r = relation_center(data[:, 0])
    orr = relation_offset(data[:, 0])
    s = relation_center(data[:, 1])
    os_ = relation_offset(data[:, 1])
    return inclusion_loss(inclusion_margin(r, orr, s, os_), margin)


def role_chain_loss(data, relation_center, relation_offset, margin, neg=False):
    """Loss for role chain axioms (EL++): :math:`r \\circ t \\sqsubseteq s`.

    Inclusion of ``Box(r ∘ t)`` (center ``c(r) + c(t)``, offset
    ``o(r) + o(t)``) in ``Box(s)``.
    """
    r = relation_center(data[:, 0])
    orr = relation_offset(data[:, 0])
    t = relation_center(data[:, 1])
    ot = relation_offset(data[:, 1])
    s = relation_center(data[:, 2])
    os_ = relation_offset(data[:, 2])
    return inclusion_loss(inclusion_margin(r + t, orr + ot, s, os_), margin)


def regularization_loss(class_center, relation_offset, reg_factor):
    """Regularization term of the paper.

    Encourages concept box centers to have unit norm (as in TransE/ELEM) and
    role offsets to have norm at most 1 (so the set of translations of a
    role stays bounded).
    """
    center_reg = (th.linalg.norm(class_center.weight, dim=1) - 1).abs().mean()
    role_reg = F.relu(th.linalg.norm(relation_offset.weight, dim=1) - 1).mean()
    return reg_factor * (center_reg + role_reg)
