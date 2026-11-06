"""Łukasiewicz hinge losses, minimality penalty, and crisp membership helper for GroundedEL.

Implements the element-grounded and region-soundness hinge functions from §3–§4
of the GroundedEL design:

1. **Crisp helper (:func:`crisp_membership`):** Thresholds signed box depth at
   ``0`` to produce crisp :math:`\\{0, 1\\}` membership values matching the dtype
   of the input depth tensor.
2. **Existence fact (:func:`fact`):** Enforces :math:`\\text{live}_{x_A} \\le m(p_{x_A}, A)`
   with :math:`\\text{live}_x = 1 - \\delta_x` for each class element :math:`x_A`.
3. **Normal-form GCI hinges (:func:`gci0`, :func:`gci1`, :func:`gci0_bot`,
   :func:`gci1_bot`, :func:`gci2`, :func:`gci2_dead`, :func:`gci3`,
   :func:`gci3_bot`):** Łukasiewicz t-norm hinges on element or region
   memberships and role memberships, with optional element deadness
   :math:`\\delta_x` (``dead=None`` yields the :math:`\\delta`-free region
   soundness form) and backwards dead propagation along :math:`A \\sqsubseteq \\exists r.B`.
4. **Role and ABox hinges (:func:`role_inclusion`, :func:`role_chain`,
   :func:`class_assertion`, :func:`role_assertion`):** Hinges for
   :math:`r \\sqsubseteq s`, :math:`r \\circ s \\sqsubseteq t`, :math:`A(a)`, and
   :math:`r(a, b)`.
5. **Minimality regulariser (:func:`minimality`):** Linear L1 penalty over
   random element-class memberships, random role triples, and element dead
   scores (:math:`\\varepsilon \\,\\text{mean}(m) + \\varepsilon_r \\,\\text{mean}(\\rho) + \\varepsilon_\\delta \\,\\text{mean}(\\delta)`).

Guarantees:
- **Exact zero on crisp models:** On any crisp :math:`\\{0, 1\\}` assignment of
   memberships :math:`m`, role memberships :math:`\\rho`, and dead indicators
   :math:`\\delta` that satisfies the grounded EL++ completion rules, every
   hinge returns exact ``0.0``.
- **Dead-element silencing:** For :math:`\\delta_x = 1` and valid memberships
   :math:`m, \\rho \\in [0, 1]`, all element GCI hinges (:func:`gci0`,
   :func:`gci1`, :func:`gci0_bot`, :func:`gci1_bot`, :func:`gci2`, :func:`gci3`,
   :func:`gci3_bot`) and :func:`fact` evaluate to exact ``0.0`` at :math:`x`.
- **Region reduction:** Passing ``dead=None`` removes :math:`-\\delta_x` and
   evaluates the exact region-soundness hinge on arbitrary points :math:`z \\in \\mathbb{R}^d`.

Non-guarantees:
- **Flat ReLU plateau:** When a hinge argument is strictly negative, its value
   and gradient are both ``0.0``.
- **Fuzzy zero-loss points:** At soft memberships :math:`m \\in (0, 1)`, zero
   hinge loss alone does not imply a crisp model unless paired with the
   minimality penalty or crisp thresholding.
"""
from __future__ import annotations

import math
from typing import Optional

import torch as th

__all__ = [
    "crisp_membership",
    "fact",
    "gci0",
    "gci0_region",
    "gci1",
    "gci1_region",
    "gci0_bot",
    "gci0_bot_region",
    "gci1_bot",
    "gci1_bot_region",
    "gci2_dead",
    "gci2",
    "gci2_region",
    "gci3",
    "gci3_region",
    "gci3_bot",
    "gci3_bot_region",
    "role_inclusion",
    "role_inclusion_region",
    "role_chain",
    "role_chain_region",
    "class_assertion",
    "role_assertion",
    "object_property_assertion",
    "minimality",
]


def _as_float_tensor(t: th.Tensor, ref_dtype: Optional[th.dtype] = None) -> th.Tensor:
    """Convert boolean or integer tensors to a floating-point tensor."""
    if not isinstance(t, th.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t).__name__}")
    if t.is_floating_point():
        return t
    target_dtype = ref_dtype if (ref_dtype is not None and ref_dtype.is_floating_point()) else th.float32
    return t.to(dtype=target_dtype)


def _check_same_shape(name_a: str, a: th.Tensor, name_b: str, b: th.Tensor) -> None:
    """Raise ValueError if two tensors do not have identical shapes."""
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape, got {tuple(a.shape)} vs {tuple(b.shape)}"
        )


def _subtract_dead(expr: th.Tensor, dead: Optional[th.Tensor]) -> th.Tensor:
    """Subtract ``dead`` from ``expr`` when ``dead`` is provided, validating shape compatibility."""
    if dead is None:
        return expr
    d = _as_float_tensor(dead, expr.dtype).to(device=expr.device, dtype=expr.dtype)
    if d.ndim > 0 and d.shape != expr.shape:
        raise ValueError(
            f"dead tensor shape {tuple(d.shape)} is incompatible with hinge shape {tuple(expr.shape)}"
        )
    return expr - d


def _validate_nonneg_weight(name: str, val: float) -> float:
    """Validate that a regularisation weight is a finite non-negative float."""
    if isinstance(val, th.Tensor):
        if val.dtype == th.bool or val.numel() != 1:
            raise ValueError(f"{name} must be a non-negative scalar, got {val!r}")
        v = float(val.detach().item())
    elif isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a non-negative number, got {val!r}")
    else:
        v = float(val)
    if v < 0.0 or math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be >= 0 and finite, got {val!r}")
    return v


def _validate_margin(name: str, val: float) -> float:
    """Validate that a region margin is a finite float."""
    if isinstance(val, th.Tensor):
        if val.dtype == th.bool or val.numel() != 1:
            raise ValueError(f"{name} must be a finite scalar, got {val!r}")
        v = float(val.detach().item())
    elif isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {val!r}")
    else:
        v = float(val)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be finite, got {val!r}")
    return v


def crisp_membership(depth: th.Tensor) -> th.Tensor:
    """Compute crisp box membership :math:`\\mathbb{1}[\\text{depth} > 0]` from signed depth.

    Guarantees:
    - Returns a floating-point tensor of the same shape and device as ``depth``
      with entries in :math:`\\{0.0, 1.0\\}`.
    - Entry is ``1.0`` if and only if the corresponding entry of ``depth`` is
      strictly positive (``depth > 0``).

    Non-guarantees:
    - Zero gradient everywhere (step function at ``0``).

    Parameters
    ----------
    depth : torch.Tensor
        Signed depth tensor of arbitrary shape.

    Returns
    -------
    torch.Tensor
        Crisp membership tensor of the same shape with values in ``{0.0, 1.0}``.
    """
    if not isinstance(depth, th.Tensor):
        raise TypeError(f"depth must be a torch.Tensor, got {type(depth).__name__}")
    out_dtype = depth.dtype if depth.is_floating_point() else th.float32
    return (depth > 0).to(dtype=out_dtype)


def fact(m_a: th.Tensor, dead: Optional[th.Tensor] = None) -> th.Tensor:
    """Compute the existence fact hinge :math:`\\operatorname{relu}(1 - \\delta_{x_A} - m(p_{x_A}, A))`.

    Guarantees:
    - Returns a tensor of the same shape as ``m_a``.
    - Equals ``0`` when :math:`m(p_{x_A}, A) \\ge 1 - \\delta_{x_A}` (in
      particular, when :math:`\\delta_{x_A} = 1` and :math:`m \\ge 0`, or when
      :math:`\\delta_{x_A} = 0` and :math:`m = 1`).
    - When ``dead is None``, reduces to :math:`\\operatorname{relu}(1 - m_a)`.

    Non-guarantees:
    - Does not clamp ``m_a`` or ``dead`` to ``[0, 1]``.

    Parameters
    ----------
    m_a : torch.Tensor
        Self-membership :math:`m(p_{x_A}, A)` of shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Dead score :math:`\\delta_{x_A}` of shape ``(B,)`` (or ``None`` for ``0``).

    Returns
    -------
    torch.Tensor
        Per-row existence fact hinge of shape ``(B,)``.
    """
    ma = _as_float_tensor(m_a)
    return th.relu(_subtract_dead(1.0 - ma, dead))


def gci0(
    m_a: th.Tensor,
    m_b: th.Tensor,
    dead: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the GCI0 hinge :math:`\\operatorname{relu}(m_A - m_B - \\delta_x)` for :math:`A \\sqsubseteq B`.

    Guarantees:
    - Returns a tensor of the same shape as ``m_a`` and ``m_b``.
    - When ``dead is None`` (region form), evaluates :math:`\\operatorname{relu}(m_A - m_B)`.
    - On crisp :math:`\\{0, 1\\}` inputs with :math:`\\delta_x = 0`, equals ``0``
      if and only if :math:`m_A \\le m_B`.

    Non-guarantees:
    - Vacuously ``0`` when :math:`m_A = 0` or :math:`\\delta_x = 1`.

    Parameters
    ----------
    m_a : torch.Tensor
        Sub-class membership :math:`m(x, A)` of shape ``(B,)``.
    m_b : torch.Tensor
        Super-class membership :math:`m(x, B)` of shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional deadness score :math:`\\delta_x` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row GCI0 hinge of shape ``(B,)``.
    """
    ma = _as_float_tensor(m_a)
    mb = _as_float_tensor(m_b, ma.dtype).to(dtype=ma.dtype)
    _check_same_shape("m_a", ma, "m_b", mb)
    return th.relu(_subtract_dead(ma - mb, dead))


def gci0_region(
    depth_b: th.Tensor,
    margin: float = 0.0,
) -> th.Tensor:
    """Compute the geometric region GCI0 hinge :math:`\\operatorname{relu}(\\text{margin} - \\operatorname{depth}_B(z))` for :math:`A \\sqsubseteq B` at :math:`z \\in A`.

    Guarantees:
    - Returns a tensor of the same shape as ``depth_b``.
    - Equals ``0.0`` whenever :math:`\\operatorname{depth}_B(z) \\ge \\text{margin}`
      (containing configuration).
    - Has nonzero gradient with respect to the active superclass box centres
      and half-widths whenever :math:`\\operatorname{depth}_B(z) < \\text{margin}`
      (violating point), regardless of :math:`\\tau`.

    Non-guarantees:
    - Assumes the caller sampled :math:`z \\in A`; does not depend on
      :math:`\\operatorname{depth}_A(z)`.

    Parameters
    ----------
    depth_b : torch.Tensor
        Signed depth :math:`\\operatorname{depth}(z, B)` of shape ``(B,)``.
    margin : float, default 0.0
        Containment margin ``region_margin``.

    Returns
    -------
    torch.Tensor
        Per-point geometric GCI0 hinge of shape ``(B,)``.
    """
    db = _as_float_tensor(depth_b)
    m = _validate_margin("margin", margin)
    return th.relu(m - db)


def gci1(
    m_a: th.Tensor,
    m_b: th.Tensor,
    m_c: th.Tensor,
    dead: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the GCI1 hinge :math:`\\operatorname{relu}(m_A + m_B - 1 - m_C - \\delta_x)` for :math:`A \\sqcap B \\sqsubseteq C`.

    Guarantees:
    - Returns a tensor of the same shape as ``m_a``.
    - When ``dead is None`` (region form), evaluates
      :math:`\\operatorname{relu}(m_A + m_B - 1 - m_C)`.
    - On crisp :math:`\\{0, 1\\}` inputs with :math:`\\delta_x = 0`, equals ``0``
      if and only if :math:`(m_A = 1 \\land m_B = 1) \\implies m_C = 1`.

    Non-guarantees:
    - Vacuously ``0`` whenever :math:`m_A + m_B \\le 1` or :math:`\\delta_x = 1`.

    Parameters
    ----------
    m_a : torch.Tensor
        First conjunct membership :math:`m(x, A)` of shape ``(B,)``.
    m_b : torch.Tensor
        Second conjunct membership :math:`m(x, B)` of shape ``(B,)``.
    m_c : torch.Tensor
        Super-class membership :math:`m(x, C)` of shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional deadness score :math:`\\delta_x` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row GCI1 hinge of shape ``(B,)``.
    """
    ma = _as_float_tensor(m_a)
    mb = _as_float_tensor(m_b, ma.dtype).to(dtype=ma.dtype)
    mc = _as_float_tensor(m_c, ma.dtype).to(dtype=ma.dtype)
    _check_same_shape("m_a", ma, "m_b", mb)
    _check_same_shape("m_a", ma, "m_c", mc)
    return th.relu(_subtract_dead(ma + mb - 1.0 - mc, dead))


def gci1_region(
    depth_c: th.Tensor,
    margin: float = 0.0,
) -> th.Tensor:
    """Compute the geometric region GCI1 hinge :math:`\\operatorname{relu}(\\text{margin} - \\operatorname{depth}_C(z))` for :math:`A \\sqcap B \\sqsubseteq C` at :math:`z \\in A \\cap B`.

    Guarantees:
    - Returns a tensor of the same shape as ``depth_c``.
    - Equals ``0.0`` whenever :math:`\\operatorname{depth}_C(z) \\ge \\text{margin}`
      (containing configuration).
    - Has nonzero gradient with respect to the active superclass box centres
      and half-widths whenever :math:`\\operatorname{depth}_C(z) < \\text{margin}`
      (violating intersection point), regardless of :math:`\\tau`.

    Non-guarantees:
    - Assumes the caller sampled :math:`z \\in A \\cap B`; does not depend on
      :math:`\\operatorname{depth}_A(z)` or :math:`\\operatorname{depth}_B(z)`.

    Parameters
    ----------
    depth_c : torch.Tensor
        Signed depth :math:`\\operatorname{depth}(z, C)` of shape ``(B,)``.
    margin : float, default 0.0
        Containment margin ``region_margin``.

    Returns
    -------
    torch.Tensor
        Per-point geometric GCI1 hinge of shape ``(B,)``.
    """
    dc = _as_float_tensor(depth_c)
    m = _validate_margin("margin", margin)
    return th.relu(m - dc)


def gci0_bot(
    m_a: th.Tensor,
    dead: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the GCI0-bot hinge :math:`\\operatorname{relu}(m_A - \\delta_x)` for :math:`A \\sqsubseteq \\bot`.

    Guarantees:
    - Returns a tensor of the same shape as ``m_a``.
    - When ``dead is None`` (region form), evaluates :math:`\\operatorname{relu}(m_A)`.
    - On crisp :math:`\\{0, 1\\}` inputs, equals ``0`` if and only if
      :math:`m_A = 0` or :math:`\\delta_x = 1`.

    Non-guarantees:
    - Does not directly force :math:`\\delta_{x_A} = 1` unless paired with
      :func:`fact` at :math:`x_A`.

    Parameters
    ----------
    m_a : torch.Tensor
        Sub-class membership :math:`m(x, A)` of shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional deadness score :math:`\\delta_x` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row GCI0-bot hinge of shape ``(B,)``.
    """
    ma = _as_float_tensor(m_a)
    return th.relu(_subtract_dead(ma, dead))


def gci0_bot_region(
    depth_a: th.Tensor,
    margin: float = 0.0,
) -> th.Tensor:
    """Compute the geometric region GCI0-bot hinge :math:`\\operatorname{relu}(\\operatorname{depth}_A(z) + \\text{margin})` for :math:`A \\sqsubseteq \\bot` at :math:`z \\in A`.

    Guarantees:
    - Returns a tensor of the same shape as ``depth_a``.
    - Equals ``0.0`` whenever :math:`\\operatorname{depth}_A(z) + \\text{margin} \\le 0`.
    - Has nonzero gradient with respect to the active box half-widths of :math:`A`
      whenever :math:`\\operatorname{depth}_A(z) + \\text{margin} > 0`, regardless
      of :math:`\\tau`.

    Non-guarantees:
    - Vanishes once all components of :math:`A` are inactive or have
      :math:`\\operatorname{depth}_A(z) \\le -\\text{margin}`.

    Parameters
    ----------
    depth_a : torch.Tensor
        Signed depth :math:`\\operatorname{depth}(z, A)` of shape ``(B,)``.
    margin : float, default 0.0
        Disjointness margin ``region_margin``.

    Returns
    -------
    torch.Tensor
        Per-point geometric GCI0-bot hinge of shape ``(B,)``.
    """
    da = _as_float_tensor(depth_a)
    m = _validate_margin("margin", margin)
    return th.relu(da + m)


def gci1_bot(
    m_a: th.Tensor,
    m_b: th.Tensor,
    dead: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the GCI1-bot hinge :math:`\\operatorname{relu}(m_A + m_B - 1 - \\delta_x)` for :math:`A \\sqcap B \\sqsubseteq \\bot`.

    Guarantees:
    - Returns a tensor of the same shape as ``m_a`` and ``m_b``.
    - When ``dead is None`` (region form), evaluates
      :math:`\\operatorname{relu}(m_A + m_B - 1)`.
    - On crisp :math:`\\{0, 1\\}` inputs with :math:`\\delta_x = 0`, equals ``0``
      if and only if :math:`\\neg(m_A = 1 \\land m_B = 1)`.

    Non-guarantees:
    - Vacuously ``0`` whenever :math:`m_A + m_B \\le 1` or :math:`\\delta_x = 1`.

    Parameters
    ----------
    m_a : torch.Tensor
        First conjunct membership :math:`m(x, A)` of shape ``(B,)``.
    m_b : torch.Tensor
        Second conjunct membership :math:`m(x, B)` of shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional deadness score :math:`\\delta_x` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row GCI1-bot hinge of shape ``(B,)``.
    """
    ma = _as_float_tensor(m_a)
    mb = _as_float_tensor(m_b, ma.dtype).to(dtype=ma.dtype)
    _check_same_shape("m_a", ma, "m_b", mb)
    return th.relu(_subtract_dead(ma + mb - 1.0, dead))


def gci1_bot_region(
    depth_a: th.Tensor,
    depth_b: th.Tensor,
    margin: float = 0.0,
) -> th.Tensor:
    """Compute the geometric region GCI1-bot hinge for :math:`A \\sqcap B \\sqsubseteq \\bot`.

    Evaluates
    :math:`\\mathbb{1}[\\operatorname{depth}_B(z) \\ge 0] \\operatorname{relu}(\\operatorname{depth}_A(z) + \\text{margin}) + \\mathbb{1}[\\operatorname{depth}_A(z) \\ge 0] \\operatorname{relu}(\\operatorname{depth}_B(z) + \\text{margin})`,
    penalising :math:`\\operatorname{depth}_A(z) + \\text{margin}` at :math:`z \\in B`,
    :math:`\\operatorname{depth}_B(z) + \\text{margin}` at :math:`z \\in A`, and
    both at :math:`z \\in A \\cap B`.

    Guarantees:
    - Returns a tensor of the same shape as ``depth_a`` and ``depth_b``.
    - Equals ``0.0`` whenever :math:`A` and :math:`B` are separated by at least
      ``margin``.
    - Has nonzero gradient with respect to the centres and half-widths of both
      :math:`A` and :math:`B` at any violating point :math:`z \\in A \\cap B`
      (and with respect to :math:`A` at :math:`z \\in B` with
      :math:`\\operatorname{depth}_A(z) > -\\text{margin}`), regardless of :math:`\\tau`.

    Non-guarantees:
    - Vanishes at points outside both :math:`A` and :math:`B`.

    Parameters
    ----------
    depth_a : torch.Tensor
        Signed depth :math:`\\operatorname{depth}(z, A)` of shape ``(B,)``.
    depth_b : torch.Tensor
        Signed depth :math:`\\operatorname{depth}(z, B)` of shape ``(B,)``.
    margin : float, default 0.0
        Disjointness margin ``region_margin``.

    Returns
    -------
    torch.Tensor
        Per-point geometric GCI1-bot hinge of shape ``(B,)``.
    """
    da = _as_float_tensor(depth_a)
    db = _as_float_tensor(depth_b, da.dtype).to(device=da.device, dtype=da.dtype)
    _check_same_shape("depth_a", da, "depth_b", db)
    m = _validate_margin("margin", margin)
    in_b = (db >= 0.0).to(dtype=da.dtype)
    in_a = (da >= 0.0).to(dtype=da.dtype)
    return in_b * th.relu(da + m) + in_a * th.relu(db + m)


def gci2_dead(
    m_a: th.Tensor,
    dead_filler: th.Tensor,
    dead: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the GCI2 dead-propagation hinge :math:`\\operatorname{relu}(m_A + \\delta_{x_B} - 1 - \\delta_x)` for :math:`A \\sqsubseteq \\exists r.B`.

    Propagates unsatisfiability backwards from the filler element :math:`x_B` to
    any element :math:`x` (or region point when ``dead=None``) in :math:`A`.

    Guarantees:
    - Returns a tensor of the same shape as ``m_a``.
    - On crisp :math:`\\{0, 1\\}` inputs, equals ``0`` if and only if
      :math:`(m_A = 1 \\land \\delta_{x_B} = 1) \\implies \\delta_x = 1`.

    Non-guarantees:
    - Vacuously ``0`` when :math:`\\delta_{x_B} = 0` or :math:`m_A = 0`.

    Parameters
    ----------
    m_a : torch.Tensor
        Sub-class membership :math:`m(x, A)` of shape ``(B,)``.
    dead_filler : torch.Tensor
        Deadness score :math:`\\delta_{x_B}` of the filler element :math:`x_B`,
        shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional deadness score :math:`\\delta_x` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row dead-propagation hinge of shape ``(B,)``.
    """
    ma = _as_float_tensor(m_a)
    df = _as_float_tensor(dead_filler, ma.dtype).to(device=ma.device, dtype=ma.dtype)
    _check_same_shape("m_a", ma, "dead_filler", df)
    return th.relu(_subtract_dead(ma + df - 1.0, dead))


def gci2(
    m_a: th.Tensor,
    rho: th.Tensor,
    dead: Optional[th.Tensor] = None,
    dead_filler: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the GCI2 hinge for :math:`A \\sqsubseteq \\exists r.B`, including dead propagation.

    Evaluates

    .. math::

        \\operatorname{relu}(m_A - \\rho(x, x_B, r) - \\delta_x)
        + \\operatorname{relu}(m_A + \\delta_{x_B} - 1 - \\delta_x),

    where the second term is included whenever ``dead_filler`` (:math:`\\delta_{x_B}`)
    is provided, and :math:`-\\delta_x` is omitted when ``dead is None`` (region form).

    Guarantees:
    - Returns a tensor of the same shape as ``m_a`` and ``rho``.
    - On crisp :math:`\\{0, 1\\}` inputs with ``dead_filler`` provided, equals
      ``0`` if and only if either :math:`\\delta_x = 1`, or :math:`m_A = 0`, or
      (:math:`\\rho(x, x_B, r) = 1` and :math:`\\delta_{x_B} = 0`).

    Non-guarantees:
    - When ``dead_filler is None``, only the edge hinge
      :math:`\\operatorname{relu}(m_A - \\rho - \\delta_x)` is evaluated.

    Parameters
    ----------
    m_a : torch.Tensor
        Sub-class membership :math:`m(x, A)` of shape ``(B,)``.
    rho : torch.Tensor
        Role membership :math:`\\rho(x, x_B, r)` of shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional source deadness score :math:`\\delta_x` of shape ``(B,)``.
    dead_filler : Optional[torch.Tensor], default None
        Optional filler deadness score :math:`\\delta_{x_B}` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row GCI2 hinge of shape ``(B,)``.
    """
    ma = _as_float_tensor(m_a)
    r = _as_float_tensor(rho, ma.dtype).to(dtype=ma.dtype)
    _check_same_shape("m_a", ma, "rho", r)
    edge_hinge = th.relu(_subtract_dead(ma - r, dead))
    if dead_filler is None:
        return edge_hinge
    return edge_hinge + gci2_dead(ma, dead_filler, dead=dead)


def gci2_region(
    role_depth: th.Tensor,
    margin: float = 0.0,
    agg: str = "max",
    agg_tau: float = 0.05,
    depth_a: Optional[th.Tensor] = None,
    dead_filler: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the geometric region GCI2 hinge :math:`\\operatorname{relu}(\\text{margin} - \\max_y \\operatorname{role\\_depth}(z, p_y, r))` for :math:`A \\sqsubseteq \\exists r.B` at :math:`z \\in A`.

    Supports both a 1-D ``role_depth`` of shape ``(B,)`` (evaluated at the filler
    element :math:`y = x_B`) and a 2-D ``role_depth`` of shape ``(B, M)`` over
    multiple filler elements :math:`y`, aggregating across ``dim=-1`` with ``max``
    or ``lse`` according to ``agg``. When ``depth_a`` and ``dead_filler`` are
    provided, also shrinks :math:`A`'s region when the filler :math:`x_B` is dead.

    Guarantees:
    - Returns a 1-D tensor of shape ``(B,)``.
    - Equals ``0.0`` whenever the aggregated ``role_depth`` is :math:`\\ge \\text{margin}`
      (and ``dead_filler`` is ``0`` or ``None``).
    - Has nonzero gradient with respect to the active role head/tail box centres
      and half-widths whenever the aggregated ``role_depth`` is :math:`< \\text{margin}`,
      regardless of :math:`\\tau`.

    Non-guarantees:
    - Raises :class:`ValueError` if ``agg`` is not ``"max"`` or ``"lse"``.

    Parameters
    ----------
    role_depth : torch.Tensor
        Signed product-box depth :math:`\\operatorname{role\\_depth}(z, p_y, r)`
        of shape ``(B,)`` or ``(B, M)``.
    margin : float, default 0.0
        Containment margin ``region_margin``.
    agg : str, default "max"
        Aggregation over filler candidates when ``role_depth`` is 2-D
        (``"max"`` or ``"lse"``).
    agg_tau : float, default 0.05
        Temperature for ``"lse"`` aggregation over filler candidates.
    depth_a : Optional[th.Tensor], default None
        Optional signed depth :math:`\\operatorname{depth}(z, A)` of shape ``(B,)``
        for region dead-propagation when ``dead_filler`` is provided.
    dead_filler : Optional[th.Tensor], default None
        Optional filler deadness score :math:`\\delta_{x_B}` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-point geometric GCI2 hinge of shape ``(B,)``.
    """
    rd = _as_float_tensor(role_depth)
    m = _validate_margin("margin", margin)
    if agg not in ("max", "lse"):
        raise ValueError(f"agg must be 'max' or 'lse', got {agg!r}")
    if rd.ndim == 2:
        if agg == "max":
            rd = th.amax(rd, dim=-1)
        else:
            tau_f = _validate_nonneg_weight("agg_tau", agg_tau)
            if tau_f <= 0.0:
                raise ValueError(f"agg_tau must be > 0, got {agg_tau!r}")
            rd = tau_f * th.logsumexp(rd / tau_f, dim=-1)
    elif rd.ndim != 1:
        raise ValueError(
            f"role_depth must be 1-D or 2-D, got shape {tuple(role_depth.shape)}"
        )
    edge_h = th.relu(m - rd)
    if depth_a is None or dead_filler is None:
        return edge_h
    da = _as_float_tensor(depth_a, rd.dtype).to(device=rd.device, dtype=rd.dtype)
    df = _as_float_tensor(dead_filler, rd.dtype).to(device=rd.device, dtype=rd.dtype)
    _check_same_shape("role_depth", rd, "depth_a", da)
    _check_same_shape("role_depth", rd, "dead_filler", df)
    live_gate = (df <= 0.5).to(dtype=rd.dtype)
    dead_gate = (df > 0.5).to(dtype=rd.dtype)
    return live_gate * edge_h + dead_gate * th.relu(da + m)


def gci3(
    rho: th.Tensor,
    m_b: th.Tensor,
    m_c: th.Tensor,
    dead: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the GCI3 hinge :math:`\\operatorname{relu}(\\rho_{xy} + m(p_y, B) - 1 - m_C(x) - \\delta_x)` for :math:`\\exists r.B \\sqsubseteq C`.

    Guarantees:
    - Returns a tensor of the same shape as ``rho``.
    - When ``dead is None`` (region form), evaluates
      :math:`\\operatorname{relu}(\\rho_{xy} + m(p_y, B) - 1 - m_C(x))`.
    - On crisp :math:`\\{0, 1\\}` inputs with :math:`\\delta_x = 0`, equals ``0``
      if and only if :math:`(\\rho_{xy} = 1 \\land m_B(y) = 1) \\implies m_C(x) = 1`.

    Non-guarantees:
    - Vacuously ``0`` whenever :math:`\\rho_{xy} + m_B(y) \\le 1` or :math:`\\delta_x = 1`.

    Parameters
    ----------
    rho : torch.Tensor
        Role membership :math:`\\rho(x, y, r)` of shape ``(B,)``.
    m_b : torch.Tensor
        Filler membership :math:`m(y, B)` of shape ``(B,)``.
    m_c : torch.Tensor
        Super-class membership :math:`m(x, C)` of shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional source deadness score :math:`\\delta_x` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row GCI3 hinge of shape ``(B,)``.
    """
    r = _as_float_tensor(rho)
    mb = _as_float_tensor(m_b, r.dtype).to(dtype=r.dtype)
    mc = _as_float_tensor(m_c, r.dtype).to(dtype=r.dtype)
    _check_same_shape("rho", r, "m_b", mb)
    _check_same_shape("rho", r, "m_c", mc)
    return th.relu(_subtract_dead(r + mb - 1.0 - mc, dead))


def gci3_region(
    role_depth: th.Tensor,
    depth_b: th.Tensor,
    depth_c: th.Tensor,
    margin: float = 0.0,
    smooth: bool = False,
) -> th.Tensor:
    """Compute the geometric region GCI3 hinge for :math:`\\exists r.B \\sqsubseteq C` at :math:`(z, y)`.

    Evaluates :math:`\\operatorname{relu}(\\text{margin} - \\operatorname{depth}_C(z))`
    masked by crisp :math:`(\\operatorname{role\\_depth}(z, y) > 0 \\land \\operatorname{depth}_B(y) > 0)`
    when ``smooth=False``, or gated by :math:`\\sigma(\\min(\\operatorname{role\\_depth}(z, y), \\operatorname{depth}_B(y)))`
    when ``smooth=True``.

    Guarantees:
    - Returns a tensor of the same shape as ``role_depth``.
    - Equals ``0.0`` whenever :math:`\\operatorname{depth}_C(z) \\ge \\text{margin}`
      or (when ``smooth=False``) :math:`\\min(\\operatorname{role\\_depth}(z, y), \\operatorname{depth}_B(y)) \\le 0`.
    - Has nonzero gradient with respect to the active superclass box centres and
      half-widths of :math:`C` whenever :math:`\\operatorname{role\\_depth}(z, y) > 0`,
      :math:`\\operatorname{depth}_B(y) > 0`, and :math:`\\operatorname{depth}_C(z) < \\text{margin}`,
      regardless of :math:`\\tau`.

    Non-guarantees:
    - Under ``smooth=False``, zero gradient with respect to :math:`r` and :math:`B`
      (only expands :math:`C` to cover active role-filler pairs).

    Parameters
    ----------
    role_depth : torch.Tensor
        Signed role depth :math:`\\operatorname{role\\_depth}(z, y, r)` of shape ``(B,)``.
    depth_b : torch.Tensor
        Signed filler depth :math:`\\operatorname{depth}(y, B)` of shape ``(B,)``.
    depth_c : torch.Tensor
        Signed superclass depth :math:`\\operatorname{depth}(z, C)` of shape ``(B,)``.
    margin : float, default 0.0
        Containment margin ``region_margin``.
    smooth : bool, default False
        If ``True``, replaces the crisp mask by ``sigmoid(min(role_depth, depth_b))``.

    Returns
    -------
    torch.Tensor
        Per-pair geometric GCI3 hinge of shape ``(B,)``.
    """
    rd = _as_float_tensor(role_depth)
    db = _as_float_tensor(depth_b, rd.dtype).to(device=rd.device, dtype=rd.dtype)
    dc = _as_float_tensor(depth_c, rd.dtype).to(device=rd.device, dtype=rd.dtype)
    _check_same_shape("role_depth", rd, "depth_b", db)
    _check_same_shape("role_depth", rd, "depth_c", dc)
    m = _validate_margin("margin", margin)
    rhs = th.relu(m - dc)
    if smooth:
        gate = th.sigmoid(th.minimum(rd, db))
    else:
        gate = ((rd > 0.0) & (db > 0.0)).to(dtype=dc.dtype)
    return gate * rhs


def gci3_bot(
    rho: th.Tensor,
    m_b: th.Tensor,
    dead: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the GCI3-bot hinge :math:`\\operatorname{relu}(\\rho_{xy} + m(p_y, B) - 1 - \\delta_x)` for :math:`\\exists r.B \\sqsubseteq \\bot`.

    Guarantees:
    - Returns a tensor of the same shape as ``rho`` and ``m_b``.
    - When ``dead is None`` (region form), evaluates
      :math:`\\operatorname{relu}(\\rho_{xy} + m(p_y, B) - 1)`.
    - On crisp :math:`\\{0, 1\\}` inputs with :math:`\\delta_x = 0`, equals ``0``
      if and only if :math:`\\neg(\\rho_{xy} = 1 \\land m_B(y) = 1)`.

    Non-guarantees:
    - Vacuously ``0`` whenever :math:`\\rho_{xy} + m_B(y) \\le 1` or :math:`\\delta_x = 1`.

    Parameters
    ----------
    rho : torch.Tensor
        Role membership :math:`\\rho(x, y, r)` of shape ``(B,)``.
    m_b : torch.Tensor
        Filler membership :math:`m(y, B)` of shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional source deadness score :math:`\\delta_x` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row GCI3-bot hinge of shape ``(B,)``.
    """
    r = _as_float_tensor(rho)
    mb = _as_float_tensor(m_b, r.dtype).to(dtype=r.dtype)
    _check_same_shape("rho", r, "m_b", mb)
    return th.relu(_subtract_dead(r + mb - 1.0, dead))


def gci3_bot_region(
    role_depth: th.Tensor,
    depth_b: th.Tensor,
    margin: float = 0.0,
) -> th.Tensor:
    """Compute the geometric region GCI3-bot hinge for :math:`\\exists r.B \\sqsubseteq \\bot` at :math:`(z, y)`.

    Evaluates
    :math:`\\mathbb{1}[\\operatorname{role\\_depth}(z, y) \\ge 0] \\operatorname{relu}(\\operatorname{depth}_B(y) + \\text{margin}) + \\mathbb{1}[\\operatorname{depth}_B(y) \\ge 0] \\operatorname{relu}(\\operatorname{role\\_depth}(z, y) + \\text{margin})`.

    Guarantees:
    - Returns a tensor of the same shape as ``role_depth`` and ``depth_b``.
    - Equals ``0.0`` whenever :math:`T_r` and :math:`B` are separated by at
      least ``margin``.
    - Has nonzero gradient with respect to both :math:`T_r` and :math:`B` at
      any violating pair :math:`(z, y)` with :math:`\\operatorname{role\\_depth}(z, y) > 0`
      and :math:`\\operatorname{depth}_B(y) > 0`, regardless of :math:`\\tau`.

    Non-guarantees:
    - Vanishes when neither :math:`\\operatorname{role\\_depth}(z, y) \\ge 0` nor
      :math:`\\operatorname{depth}_B(y) \\ge 0`.

    Parameters
    ----------
    role_depth : torch.Tensor
        Signed role depth :math:`\\operatorname{role\\_depth}(z, y, r)` of shape ``(B,)``.
    depth_b : torch.Tensor
        Signed filler depth :math:`\\operatorname{depth}(y, B)` of shape ``(B,)``.
    margin : float, default 0.0
        Disjointness margin ``region_margin``.

    Returns
    -------
    torch.Tensor
        Per-pair geometric GCI3-bot hinge of shape ``(B,)``.
    """
    rd = _as_float_tensor(role_depth)
    db = _as_float_tensor(depth_b, rd.dtype).to(device=rd.device, dtype=rd.dtype)
    _check_same_shape("role_depth", rd, "depth_b", db)
    m = _validate_margin("margin", margin)
    in_r = (rd >= 0.0).to(dtype=rd.dtype)
    in_b = (db >= 0.0).to(dtype=rd.dtype)
    return in_r * th.relu(db + m) + in_b * th.relu(rd + m)


def role_inclusion(
    rho_r: th.Tensor,
    rho_s: th.Tensor,
    dead: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the role-inclusion hinge :math:`\\operatorname{relu}(\\rho_r(x, y) - \\rho_s(x, y))` for :math:`r \\sqsubseteq s`.

    Guarantees:
    - Returns a tensor of the same shape as ``rho_r`` and ``rho_s``.
    - On crisp :math:`\\{0, 1\\}` inputs (with ``dead=None`` or ``dead=0``),
      equals ``0`` if and only if :math:`\\rho_r(x, y) \\le \\rho_s(x, y)`.

    Non-guarantees:
    - Vacuously ``0`` when :math:`\\rho_r(x, y) = 0`.

    Parameters
    ----------
    rho_r : torch.Tensor
        Sub-role membership :math:`\\rho(x, y, r)` of shape ``(B,)``.
    rho_s : torch.Tensor
        Super-role membership :math:`\\rho(x, y, s)` of shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional deadness score :math:`\\delta_x` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row role-inclusion hinge of shape ``(B,)``.
    """
    rr = _as_float_tensor(rho_r)
    rs = _as_float_tensor(rho_s, rr.dtype).to(dtype=rr.dtype)
    _check_same_shape("rho_r", rr, "rho_s", rs)
    return th.relu(_subtract_dead(rr - rs, dead))


def role_inclusion_region(
    role_depth_s: th.Tensor,
    margin: float = 0.0,
) -> th.Tensor:
    """Compute the geometric region role-inclusion hinge :math:`\\operatorname{relu}(\\text{margin} - \\operatorname{role\\_depth}_s(x, y))` for :math:`r \\sqsubseteq s` at :math:`(x, y) \\in r`.

    Guarantees:
    - Returns a tensor of the same shape as ``role_depth_s``.
    - Equals ``0.0`` whenever :math:`\\operatorname{role\\_depth}_s(x, y) \\ge \\text{margin}`
      (containing configuration).
    - Has nonzero gradient with respect to the active super-role head/tail box
      centres and half-widths of :math:`s` whenever
      :math:`\\operatorname{role\\_depth}_s(x, y) < \\text{margin}`, regardless of
      :math:`\\tau`.

    Non-guarantees:
    - Assumes the caller sampled :math:`(x, y) \\in H_r \\times T_r`.

    Parameters
    ----------
    role_depth_s : torch.Tensor
        Signed super-role depth :math:`\\operatorname{role\\_depth}(x, y, s)` of shape ``(B,)``.
    margin : float, default 0.0
        Containment margin ``region_margin``.

    Returns
    -------
    torch.Tensor
        Per-pair geometric role-inclusion hinge of shape ``(B,)``.
    """
    rs = _as_float_tensor(role_depth_s)
    m = _validate_margin("margin", margin)
    return th.relu(m - rs)


def role_chain(
    rho_r: th.Tensor,
    rho_s: th.Tensor,
    rho_t: th.Tensor,
    dead: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the role-chain hinge :math:`\\operatorname{relu}(\\rho_r(x, y) + \\rho_s(y, z) - 1 - \\rho_t(x, z))` for :math:`r \\circ s \\sqsubseteq t`.

    Guarantees:
    - Returns a tensor of the same shape as ``rho_r``.
    - On crisp :math:`\\{0, 1\\}` inputs (with ``dead=None`` or ``dead=0``),
      equals ``0`` if and only if
      :math:`(\\rho_r(x, y) = 1 \\land \\rho_s(y, z) = 1) \\implies \\rho_t(x, z) = 1`.

    Non-guarantees:
    - Vacuously ``0`` whenever :math:`\\rho_r(x, y) + \\rho_s(y, z) \\le 1`.

    Parameters
    ----------
    rho_r : torch.Tensor
        First chain role membership :math:`\\rho(x, y, r)` of shape ``(B,)``.
    rho_s : torch.Tensor
        Second chain role membership :math:`\\rho(y, z, s)` of shape ``(B,)``.
    rho_t : torch.Tensor
        Super-role membership :math:`\\rho(x, z, t)` of shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional deadness score :math:`\\delta_x` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row role-chain hinge of shape ``(B,)``.
    """
    rr = _as_float_tensor(rho_r)
    rs = _as_float_tensor(rho_s, rr.dtype).to(dtype=rr.dtype)
    rt = _as_float_tensor(rho_t, rr.dtype).to(dtype=rr.dtype)
    _check_same_shape("rho_r", rr, "rho_s", rs)
    _check_same_shape("rho_r", rr, "rho_t", rt)
    return th.relu(_subtract_dead(rr + rs - 1.0 - rt, dead))


def role_chain_region(
    role_depth_r: th.Tensor,
    role_depth_s: th.Tensor,
    role_depth_t: th.Tensor,
    margin: float = 0.0,
    smooth: bool = False,
) -> th.Tensor:
    """Compute the geometric region role-chain hinge for :math:`r \\circ s \\sqsubseteq t` at :math:`(x, y, z)`.

    Evaluates :math:`\\operatorname{relu}(\\text{margin} - \\operatorname{role\\_depth}_t(x, z))`
    masked by crisp :math:`(\\operatorname{role\\_depth}_r(x, y) > 0 \\land \\operatorname{role\\_depth}_s(y, z) > 0)`
    when ``smooth=False``, or gated by :math:`\\sigma(\\min(\\operatorname{role\\_depth}_r(x, y), \\operatorname{role\\_depth}_s(y, z)))`
    when ``smooth=True``.

    Guarantees:
    - Returns a tensor of the same shape as ``role_depth_r``.
    - Equals ``0.0`` whenever :math:`\\operatorname{role\\_depth}_t(x, z) \\ge \\text{margin}`
      or (when ``smooth=False``) :math:`\\min(\\operatorname{role\\_depth}_r(x, y), \\operatorname{role\\_depth}_s(y, z)) \\le 0`.
    - Has nonzero gradient with respect to the active super-role head/tail box
      centres and half-widths of :math:`t` whenever
      :math:`\\operatorname{role\\_depth}_r(x, y) > 0`,
      :math:`\\operatorname{role\\_depth}_s(y, z) > 0`, and
      :math:`\\operatorname{role\\_depth}_t(x, z) < \\text{margin}`, regardless of
      :math:`\\tau`.

    Non-guarantees:
    - Under ``smooth=False``, zero gradient with respect to :math:`r` and :math:`s`.

    Parameters
    ----------
    role_depth_r : torch.Tensor
        Signed depth :math:`\\operatorname{role\\_depth}(x, y, r)` of shape ``(B,)``.
    role_depth_s : torch.Tensor
        Signed depth :math:`\\operatorname{role\\_depth}(y, z, s)` of shape ``(B,)``.
    role_depth_t : torch.Tensor
        Signed depth :math:`\\operatorname{role\\_depth}(x, z, t)` of shape ``(B,)``.
    margin : float, default 0.0
        Containment margin ``region_margin``.
    smooth : bool, default False
        If ``True``, replaces the crisp mask by ``sigmoid(min(role_depth_r, role_depth_s))``.

    Returns
    -------
    torch.Tensor
        Per-triple geometric role-chain hinge of shape ``(B,)``.
    """
    rr = _as_float_tensor(role_depth_r)
    rs = _as_float_tensor(role_depth_s, rr.dtype).to(device=rr.device, dtype=rr.dtype)
    rt = _as_float_tensor(role_depth_t, rr.dtype).to(device=rr.device, dtype=rr.dtype)
    _check_same_shape("role_depth_r", rr, "role_depth_s", rs)
    _check_same_shape("role_depth_r", rr, "role_depth_t", rt)
    m = _validate_margin("margin", margin)
    rhs = th.relu(m - rt)
    if smooth:
        gate = th.sigmoid(th.minimum(rr, rs))
    else:
        gate = ((rr > 0.0) & (rs > 0.0)).to(dtype=rt.dtype)
    return gate * rhs


def class_assertion(
    m_a: th.Tensor,
    dead: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the ABox class-assertion hinge :math:`\\operatorname{relu}(1 - m(p_a, A))` for :math:`A(a)`.

    Guarantees:
    - Returns a tensor of the same shape as ``m_a``.
    - Equals ``0`` when :math:`m(p_a, A) \\ge 1` (or :math:`\\ge 1 - \\delta_a`
      when ``dead`` is provided).

    Non-guarantees:
    - Does not clamp ``m_a`` to ``[0, 1]``.

    Parameters
    ----------
    m_a : torch.Tensor
        Membership :math:`m(p_a, A)` of individual :math:`a` in class :math:`A`,
        shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional deadness score :math:`\\delta_a` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row class-assertion hinge of shape ``(B,)``.
    """
    ma = _as_float_tensor(m_a)
    return th.relu(_subtract_dead(1.0 - ma, dead))


def role_assertion(
    rho: th.Tensor,
    dead: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Compute the ABox role-assertion hinge :math:`\\operatorname{relu}(1 - \\rho(a, b, r))` for :math:`r(a, b)`.

    Guarantees:
    - Returns a tensor of the same shape as ``rho``.
    - Equals ``0`` when :math:`\\rho(a, b, r) \\ge 1` (or :math:`\\ge 1 - \\delta_a`
      when ``dead`` is provided).

    Non-guarantees:
    - Does not clamp ``rho`` to ``[0, 1]``.

    Parameters
    ----------
    rho : torch.Tensor
        Role membership :math:`\\rho(a, b, r)` of shape ``(B,)``.
    dead : Optional[torch.Tensor], default None
        Optional deadness score :math:`\\delta_a` of shape ``(B,)``.

    Returns
    -------
    torch.Tensor
        Per-row role-assertion hinge of shape ``(B,)``.
    """
    r = _as_float_tensor(rho)
    return th.relu(_subtract_dead(1.0 - r, dead))


object_property_assertion = role_assertion


def minimality(
    m_rand: Optional[th.Tensor] = None,
    rho_rand: Optional[th.Tensor] = None,
    dead: Optional[th.Tensor] = None,
    eps: float = 1e-2,
    eps_r: Optional[float] = None,
    eps_dead: Optional[float] = None,
) -> th.Tensor:
    """Compute the principled-negatives minimality penalty (:math:`\\varepsilon \\,\\text{mean}(m) + \\varepsilon_r \\,\\text{mean}(\\rho) + \\varepsilon_\\delta \\,\\text{mean}(\\delta)`).

    Pushes unforced class memberships, role memberships, and element dead scores
    toward ``0`` so that in the small-:math:`\\varepsilon` limit at fixed
    positions, the unique minimiser on the zero-hinge polytope is the least
    model.

    Guarantees:
    - Returns a scalar ``torch.Tensor`` of shape ``()``.
    - Returns exact ``0.0`` when ``eps == 0`` (and ``eps_r == 0``, ``eps_dead == 0``)
      or when all provided tensors are empty / ``None``.
    - Raises :class:`ValueError` if any weight is negative or non-finite.

    Non-guarantees:
    - For large :math:`\\varepsilon`, can trade off against positive hinges if not
      scaled below :math:`2^{-\\text{depth}}`.

    Parameters
    ----------
    m_rand : Optional[torch.Tensor], default None
        Sampled element-class memberships :math:`m(p_x, A)`.
    rho_rand : Optional[torch.Tensor], default None
        Sampled role memberships :math:`\\rho(x, y, r)`.
    dead : Optional[torch.Tensor], default None
        Element dead scores :math:`\\delta_x`.
    eps : float, default 1e-2
        Non-negative weight :math:`\\varepsilon` on ``m_rand.mean()`` (and default
        for ``eps_r`` and ``eps_dead`` when they are ``None``).
    eps_r : Optional[float], default None
        Non-negative weight :math:`\\varepsilon_r` on ``rho_rand.mean()``;
        defaults to ``eps`` when ``None``.
    eps_dead : Optional[float], default None
        Non-negative weight :math:`\\varepsilon_\\delta` on ``dead.mean()``;
        defaults to ``eps`` when ``None``.

    Returns
    -------
    torch.Tensor
        Scalar minimality loss tensor.
    """
    w_m = _validate_nonneg_weight("eps", eps)
    w_r = w_m if eps_r is None else _validate_nonneg_weight("eps_r", eps_r)
    w_d = w_m if eps_dead is None else _validate_nonneg_weight("eps_dead", eps_dead)

    ref_tensor: Optional[th.Tensor] = None
    for candidate in (m_rand, rho_rand, dead):
        if candidate is not None:
            ref_tensor = _as_float_tensor(candidate)
            break

    if ref_tensor is None:
        return th.zeros((), dtype=th.float32)

    total = th.zeros((), dtype=ref_tensor.dtype, device=ref_tensor.device)
    if m_rand is not None and m_rand.numel() > 0 and w_m > 0.0:
        m_f = _as_float_tensor(m_rand, ref_tensor.dtype).to(
            device=ref_tensor.device, dtype=ref_tensor.dtype
        )
        total = total + w_m * m_f.mean()
    if rho_rand is not None and rho_rand.numel() > 0 and w_r > 0.0:
        r_f = _as_float_tensor(rho_rand, ref_tensor.dtype).to(
            device=ref_tensor.device, dtype=ref_tensor.dtype
        )
        total = total + w_r * r_f.mean()
    if dead is not None and dead.numel() > 0 and w_d > 0.0:
        d_f = _as_float_tensor(dead, ref_tensor.dtype).to(
            device=ref_tensor.device, dtype=ref_tensor.dtype
        )
        total = total + w_d * d_f.mean()
    return total
