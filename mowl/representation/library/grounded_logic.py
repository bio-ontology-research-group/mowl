"""Grounded constraints from generic parts: rules read in a fuzzy logic, points against depths,
and the minimality regulariser.

A grounded method turns every axiom into constraints on the elements of a finite domain
(:mod:`~mowl.representation.library.solved_grounding`). For GroundedEL an axiom becomes a
*rule* at an element (or a pair, or a triple of elements): premises, a conclusion and an
exemption, for example :math:`A \\sqcap B \\sqsubseteq C` at :math:`x` is
:math:`A(x) \\wedge B(x) \\to C(x)` and is exempt where :math:`x` is dead (an element that
cannot exist satisfies everything). Three generic parts compute what a constraint costs:

:class:`Rules`
    The violation of a rule in a fuzzy logic (an operator family of
    :mod:`~mowl.representation.library.logic_fuzzy`): the degree of the premises together
    with the life of the element, minus the conclusion,

    .. math:: v = \\big(T(p_1, \\dots, p_n, 1 - e) - c\\big)^+ ,

    with :math:`c = 0` when the conclusion is bottom. :class:`LukasiewiczRules` writes it in
    closed form, :math:`(\\sum_i p_i - (n - 1) - c - e)^+`, which is linear inside the
    hinge: the same rule is also a **hard linear constraint**
    (:meth:`LukasiewiczRules.constraint`, the rows of a
    :class:`~mowl.representation.library.solved_solver.Program`). GroundedEL uses the
    Łukasiewicz family; Gödel and product are swaps (:data:`RULES`).
:class:`DepthHinges`
    The reading of a point against the signed depth of a region (positive inside): inside
    with a margin, outside, apart from another region at the point, and a conclusion
    guarded by premises (crisply or through :math:`\\sigma(\\min)`), aggregation over
    candidate fillers (``max`` or ``lse``). These are the *region* constraints, evaluated at
    points sampled in the regions.
:class:`Minimality`
    The regulariser :math:`\\varepsilon\\, \\overline{m} + \\varepsilon_r\\, \\overline{\\rho}
    + \\varepsilon_\\delta\\, \\overline{\\delta}` (means of memberships, relation degrees and
    deadness at random groundings) that makes the least model the minimiser among the models.

The named constraint of every axiom form is a one-line combination of these
(:mod:`mowl.nn.el.grounded.constraints`).

Interface summary (for unifying with other constructions)::

    Rules(family).violation(premises, conclusion=None, exempt=None) -> Tensor
    LukasiewiczRules().constraint(alg, premises, conclusion, exempt, name)   # alg.le(.., 0)
    DepthHinges.inside(depth, margin), outside(depth, margin), apart(da, db, margin),
        guarded(gate_a, gate_b, depth_c, margin, smooth), exists(depths, agg, agg_tau)
    Minimality(eps, eps_r, eps_dead).penalty(m, rho, dead)
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch as th

from mowl.representation.library.logic_fuzzy import Goedel, Lukasiewicz, Product

__all__ = ["Rules", "LukasiewiczRules", "GoedelRules", "ProductRules", "RULES", "get_rules",
           "DepthHinges", "Minimality", "validate_margin", "validate_nonneg_weight"]


def as_float_tensor(t, ref_dtype=None):
    """A boolean or integer tensor as a floating-point tensor."""
    if not isinstance(t, th.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t).__name__}")
    if t.is_floating_point():
        return t
    target = ref_dtype if (ref_dtype is not None and ref_dtype.is_floating_point) else th.float32
    return t.to(dtype=target)


def check_same_shape(name_a, a, name_b, b):
    if a.shape != b.shape:
        raise ValueError(f"{name_a} and {name_b} must have the same shape, got "
                         f"{tuple(a.shape)} vs {tuple(b.shape)}")


def validate_nonneg_weight(name, val):
    """A finite non-negative weight as a float."""
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


def validate_margin(name, val):
    """A finite margin as a float."""
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


# =============================================================================================
# rules
# =============================================================================================
class Rules:
    """The violation of grounded rules in the fuzzy logic ``family`` (see the module
    docstring). Arguments are degrees of the same shape (one entry per grounding); ``names``
    name them in error messages.

    :param family: A :class:`~mowl.representation.library.logic_fuzzy.Family`.
    """

    name = ""

    def __init__(self, family):
        self.family = family

    @staticmethod
    def _inputs(premises, conclusion, names):
        values = list(premises) + ([conclusion] if conclusion is not None else [])
        if not values:
            raise ValueError("A rule needs a premise or a conclusion.")
        names = list(names) if names else [f"arg{i}" for i in range(len(values))]
        ref = as_float_tensor(values[0])
        out = [ref]
        for name, v in zip(names[1:], values[1:]):
            v = as_float_tensor(v, ref.dtype).to(dtype=ref.dtype)
            check_same_shape(names[0], ref, name, v)
            out.append(v)
        n = len(premises)
        return out[:n], (out[n] if conclusion is not None else None)

    @staticmethod
    def _exempt(expr, exempt):
        if exempt is None:
            return None
        d = as_float_tensor(exempt, expr.dtype).to(device=expr.device, dtype=expr.dtype)
        if d.ndim > 0 and d.shape != expr.shape:
            raise ValueError(f"dead tensor shape {tuple(d.shape)} is incompatible with hinge "
                             f"shape {tuple(expr.shape)}")
        return d

    @staticmethod
    def total(*violations):
        """The sum of the violations of several rules at the same elements (a form that
        is read as more than one rule, e.g. an edge and the propagation of deadness)."""
        out = violations[0]
        for v in violations[1:]:
            out = out + v
        return out

    def violation(self, premises: Sequence[th.Tensor], conclusion: Optional[th.Tensor] = None,
                  exempt: Optional[th.Tensor] = None, names=()):
        """:math:`(T(p_1, \\dots, p_n, 1 - e) - c)^+`; no premise is truth 1, no conclusion
        is bottom (0), no exemption is 0."""
        premises, conclusion = self._inputs(premises, conclusion, names)
        ref = premises[0] if premises else conclusion
        truth = th.ones_like(ref) if not premises else premises[0]
        for p in premises[1:]:
            truth = self.family.and_(truth, p)
        d = self._exempt(ref, exempt)
        if d is not None:
            truth = self.family.and_(truth, 1.0 - d)
        if conclusion is not None:
            truth = truth - conclusion
        return th.relu(truth)


class LukasiewiczRules(Rules):
    """Rules in the Łukasiewicz logic, in closed form:
    :math:`(\\sum_i p_i - (n - 1) - c - e)^+`, equal to the generic reading with
    :math:`T(a, b) = (a + b - 1)^+` wherever the degrees lie in ``[0, 1]`` (the hinge
    absorbs the inner clamps). Linear inside the hinge, so each rule is also a hard linear
    constraint (:meth:`constraint`).

    The rules of GroundedEL. Part (vocabulary value): ``fuzzy operator`` (Łukasiewicz)."""

    name = "lukasiewicz"

    def __init__(self):
        super().__init__(Lukasiewicz())

    def violation(self, premises, conclusion=None, exempt=None, names=()):
        premises, conclusion = self._inputs(premises, conclusion, names)
        if premises:
            expr = premises[0]
            for p in premises[1:]:
                expr = expr + p
            if len(premises) > 1:
                expr = expr - float(len(premises) - 1)
            if conclusion is not None:
                expr = expr - conclusion
        else:
            expr = 1.0 - conclusion
        d = self._exempt(expr, exempt)
        return th.relu(expr if d is None else expr - d)

    def constraint(self, alg, premises, conclusion=None, exempt=None, name="rule"):
        """The rule as a hard constraint in the algebra ``alg`` of a
        :class:`~mowl.representation.library.solved_solver.Program`:
        :math:`\\sum_i p_i - (n - 1) - c - e \\le 0` (arrays or expressions that broadcast)."""
        expr = 1.0 if not premises else premises[0]
        for p in premises[1:]:
            expr = expr + p
        if len(premises) > 1:
            expr = expr - float(len(premises) - 1)
        if conclusion is not None:
            expr = expr - conclusion
        if exempt is not None:
            expr = expr - exempt
        alg.le(expr, 0.0 * expr, name)


class GoedelRules(Rules):
    """Rules in the Gödel logic: :math:`(\\min(p_1, \\dots, p_n, 1 - e) - c)^+`. GroundedEL
    reads the chain of two boxes of translations with it (the minimum of the two degrees).
    Not linear: it has no hard-constraint form."""

    name = "godel"

    def __init__(self):
        super().__init__(Goedel())


class ProductRules(Rules):
    """Rules in the product logic: :math:`(\\prod_i p_i (1 - e) - c)^+`."""

    name = "product"

    def __init__(self):
        super().__init__(Product())


#: Rules by the name of their logic.
RULES = {"lukasiewicz": LukasiewiczRules, "godel": GoedelRules, "product": ProductRules}


def get_rules(rules):
    """A :class:`Rules` from a name of :data:`RULES` or an instance."""
    if isinstance(rules, Rules):
        return rules
    if rules not in RULES:
        raise ValueError(f"Unknown logic {rules!r}; available: {sorted(RULES)}.")
    return RULES[rules]()


# =============================================================================================
# points against depths
# =============================================================================================
class DepthHinges:
    """Costs of a point against the signed depth of regions (positive inside). Used by
    the region constraints of GroundedEL, at points sampled in the regions; works for any
    class model with a signed depth."""

    @staticmethod
    def inside(depth, margin=0.0):
        """The point should be at least ``margin`` inside: :math:`(m - \\text{depth})^+`."""
        d = as_float_tensor(depth)
        return th.relu(validate_margin("margin", margin) - d)

    @staticmethod
    def outside(depth, margin=0.0):
        """The point should be at least ``margin`` outside: :math:`(\\text{depth} + m)^+`."""
        d = as_float_tensor(depth)
        return th.relu(d + validate_margin("margin", margin))

    @staticmethod
    def apart(depth_a, depth_b, margin=0.0, names=("depth_a", "depth_b")):
        """The point should not be in both regions: push it out of each region where it is
        in the other, :math:`[d_b \\ge 0](d_a + m)^+ + [d_a \\ge 0](d_b + m)^+`."""
        da = as_float_tensor(depth_a)
        db = as_float_tensor(depth_b, da.dtype).to(device=da.device, dtype=da.dtype)
        check_same_shape(names[0], da, names[1], db)
        m = validate_margin("margin", margin)
        in_b = (db >= 0.0).to(dtype=da.dtype)
        in_a = (da >= 0.0).to(dtype=da.dtype)
        return in_b * th.relu(da + m) + in_a * th.relu(db + m)

    @staticmethod
    def guard(depth_a, depth_b, smooth=False, dtype=None):
        """Degree to which a point satisfies two premises given by their depths: crisp
        (both positive) or :math:`\\sigma(\\min(d_a, d_b))`."""
        if smooth:
            return th.sigmoid(th.minimum(depth_a, depth_b))
        return ((depth_a > 0.0) & (depth_b > 0.0)).to(dtype=dtype)

    @classmethod
    def guarded(cls, depth_a, depth_b, depth_c, margin=0.0, smooth=False,
                names=("depth_a", "depth_b", "depth_c")):
        """A conclusion guarded by two premises: :math:`g(d_a, d_b)\\,(m - d_c)^+`."""
        da = as_float_tensor(depth_a)
        db = as_float_tensor(depth_b, da.dtype).to(device=da.device, dtype=da.dtype)
        dc = as_float_tensor(depth_c, da.dtype).to(device=da.device, dtype=da.dtype)
        check_same_shape(names[0], da, names[1], db)
        check_same_shape(names[0], da, names[2], dc)
        rhs = th.relu(validate_margin("margin", margin) - dc)
        return cls.guard(da, db, smooth=smooth, dtype=dc.dtype) * rhs

    @staticmethod
    def exists(depth, agg="max", agg_tau=0.05, name="role_depth"):
        """The depth of "some candidate": ``(B,)`` as it is, ``(B, M)`` aggregated over the
        candidates by ``max`` or :math:`\\tau \\log \\sum e^{d / \\tau}` (``lse``)."""
        rd = as_float_tensor(depth)
        if agg not in ("max", "lse"):
            raise ValueError(f"agg must be 'max' or 'lse', got {agg!r}")
        if rd.ndim == 2:
            if agg == "max":
                return th.amax(rd, dim=-1)
            tau_f = validate_nonneg_weight("agg_tau", agg_tau)
            if tau_f <= 0.0:
                raise ValueError(f"agg_tau must be > 0, got {agg_tau!r}")
            return tau_f * th.logsumexp(rd / tau_f, dim=-1)
        if rd.ndim != 1:
            raise ValueError(f"{name} must be 1-D or 2-D, got shape {tuple(depth.shape)}")
        return rd

    @staticmethod
    def select(condition, if_true, if_false):
        """``if_true`` where the crisp ``condition`` holds, else ``if_false`` (the gate of a
        region that must be empty when its filler is dead)."""
        return ((~condition).to(dtype=if_true.dtype) * if_false
                + condition.to(dtype=if_true.dtype) * if_true)


# =============================================================================================
# the minimality regulariser
# =============================================================================================
class Minimality:
    """The regulariser of the least model: :math:`\\varepsilon\\, \\overline{m} +
    \\varepsilon_r\\, \\overline{\\rho} + \\varepsilon_\\delta\\, \\overline{\\delta}`
    (``eps_r`` and ``eps_dead`` default to ``eps``). Every true membership costs a little,
    so among the models (zero violation) the least one is preferred; a positive
    ``eps_dead`` keeps an element alive unless an axiom forces it dead."""

    part = "minimality"

    def __init__(self, eps=1e-2, eps_r=None, eps_dead=None):
        self.eps = validate_nonneg_weight("eps", eps)
        self.eps_r = self.eps if eps_r is None else validate_nonneg_weight("eps_r", eps_r)
        self.eps_dead = self.eps if eps_dead is None else \
            validate_nonneg_weight("eps_dead", eps_dead)

    def penalty(self, m_rand=None, rho_rand=None, dead=None):
        ref = None
        for candidate in (m_rand, rho_rand, dead):
            if candidate is not None:
                ref = as_float_tensor(candidate)
                break
        if ref is None:
            return th.zeros((), dtype=th.float32)
        total = th.zeros((), dtype=ref.dtype, device=ref.device)
        for values, weight in ((m_rand, self.eps), (rho_rand, self.eps_r),
                               (dead, self.eps_dead)):
            if values is not None and values.numel() > 0 and weight > 0.0:
                v = as_float_tensor(values, ref.dtype).to(device=ref.device, dtype=ref.dtype)
                total = total + weight * v.mean()
        return total
