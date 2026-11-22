"""Losses and volume ratios of BoxSEL (Zhu et al., 2024).

A concept expression is a nested tuple: a class index is an atom, ``("and", e1, e2, ...)``
is a conjunction and ``("some", r, e)`` is :math:`\\exists r.e`. The region of
:math:`\\exists r.C` is :math:`(T^r)^{-1}(\\mathrm{Box}(C))`, the box
:math:`((m - b)/D, (M - b)/D)` (the same transformation as the left-existential loss of
BoxEL), because :math:`r = \\{(a, b) \\mid T^r(a) = b\\}`.
"""
import torch as th
import torch.nn.functional as F

from mowl.baselines.nn.el.boxel.losses import eps, effective_scale


def compile_expression(expr, class_index, role_index):
    """Split an expression into a *structure* (the tree without names) and its *leaves*, the
    class and role indices in left-to-right order: ``(structure, [indices])``."""
    if isinstance(expr, str):
        if expr not in class_index:
            raise KeyError(f"Unknown class {expr!r} in a conditional.")
        return ("atom",), [class_index[expr]]
    if isinstance(expr, (tuple, list)) and expr:
        head = expr[0]
        if head == "and" and len(expr) >= 3:
            parts = [compile_expression(e, class_index, role_index) for e in expr[1:]]
            return ("and",) + tuple(p[0] for p in parts), [i for p in parts for i in p[1]]
        if head == "some" and len(expr) == 3:
            if expr[1] not in role_index:
                raise KeyError(f"Unknown object property {expr[1]!r} in a conditional.")
            struct, leaves = compile_expression(expr[2], class_index, role_index)
            return ("some", struct), [role_index[expr[1]]] + leaves
    raise ValueError(f"Not a concept expression: {expr!r}. Use an IRI, ('and', e1, e2, ...) "
                     "or ('some', role_iri, e).")


def expression_box(struct, leaves, pos, min_embed, delta_embed, relation_embed, scaling_embed,
                   positive_scale=True):
    """Lower and upper corner ``(m, M, next position)`` of the box of an expression, for a
    batch of rows: ``leaves`` is a ``(B, k)`` index tensor, ``pos`` the next column.
    ``positive_scale`` as in :func:`mowl.nn.el.boxel.losses.effective_scale`."""
    kind = struct[0]
    if kind == "atom":
        m = min_embed(leaves[:, pos])
        return m, m + th.exp(delta_embed(leaves[:, pos])), pos + 1
    if kind == "some":
        rel = relation_embed(leaves[:, pos])
        scale = effective_scale(scaling_embed(leaves[:, pos]), positive_scale)
        m, big, nxt = expression_box(struct[1], leaves, pos + 1, min_embed, delta_embed,
                                     relation_embed, scaling_embed, positive_scale)
        return (m - rel) / scale, (big - rel) / scale, nxt
    m = big = None
    for sub in struct[1:]:
        m2, big2, pos = expression_box(sub, leaves, pos, min_embed, delta_embed,
                                       relation_embed, scaling_embed, positive_scale)
        m, big = (m2, big2) if m is None else (th.max(m, m2), th.min(big, big2))
    return m, big, pos


def soft_volume(m, big, temperature):
    """Softplus volume (Patel et al.; BoxSEL Eq. 21)."""
    return F.softplus(big - m, beta=temperature).prod(-1)


def hard_volume(m, big):
    """Vol of a box (Section 6): the product of ``max(0, M_i - m_i)``."""
    return th.relu(big - m).prod(-1)


def conditional_probability(c_box, d_box):
    """Proportion :math:`\\mathrm{Vol}(D \\cap C)/\\mathrm{Vol}(C)` of two boxes given as
    ``(m, M)`` pairs, with exact volumes; ``nan`` where :math:`C` has no volume."""
    vc = hard_volume(*c_box)
    inter = (th.max(c_box[0], d_box[0]), th.min(c_box[1], d_box[1]))
    return hard_volume(*inter) / vc


def conditional_loss(c_box, d_box, lower, upper, temperature):
    """Eqs. 4, 5 and 7: :math:`[l\\,\\mathrm{Vol}(C) - \\mathrm{Vol}(D \\cap C)]_+ +
    [\\mathrm{Vol}(D \\cap C) - u\\,\\mathrm{Vol}(C)]_+`, one value per row, with softplus
    volumes."""
    vc = soft_volume(*c_box, temperature)
    inter = (th.max(c_box[0], d_box[0]), th.min(c_box[1], d_box[1]))
    vdc = soft_volume(*inter, temperature)
    return F.relu(lower * vc - vdc) + F.relu(vdc - upper * vc)


def location_loss(min_embed, delta_embed, beta, epsilon=eps):
    """Eq. 9: :math:`\\sum_C\\sum_i [M_i - \\beta + \\epsilon]_+ + [-m_i - \\epsilon]_+` (mean
    over the entries here)."""
    m = min_embed.weight
    big = m + th.exp(delta_embed.weight)
    return (F.relu(big - beta + epsilon).mean() + F.relu(-m - epsilon).mean())


def volume_loss(min_embed, delta_embed, beta, epsilon=eps):
    """Eq. 10 as printed: :math:`\\sum_C [\\beta^n - \\mathrm{Vol}(C) - \\epsilon]_+`
    (mean over concepts)."""
    n = min_embed.weight.shape[1]
    vol = th.exp(delta_embed.weight).prod(-1)
    return F.relu(beta ** n - vol - epsilon).mean()
