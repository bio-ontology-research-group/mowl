"""The grounded constraints of GroundedEL, one per axiom form, as combinations of generic parts.

Every axiom form is read twice (§3, §4 of the GroundedEL design):

- **at elements** (one or two elements of the finite domain): a rule in a fuzzy logic,
  :meth:`Rules.violation <mowl.representation.library.grounded_logic.Rules.violation>` of
  premises, a conclusion and an exemption (the deadness :math:`\\delta_x` of the element);
  GroundedEL uses the Łukasiewicz logic, and every function below takes ``rules=`` to read
  the same constraint in another logic;
- **at points of regions** (``*_region``): the depth of a sampled point against the
  regions of the conclusion,
  :class:`~mowl.representation.library.grounded_logic.DepthHinges`; with
  ``role_mode="translation"`` the region of :math:`\\exists r.B` is a union of boxes again,
  and the constraint is a rule on degrees of membership at the point (``*_region_trans``).

``crisp_membership`` is the crisp reading of the measure, ``minimality`` the regulariser.

=========================  ===========================================================
form                       at an element (rule: premises -> conclusion, exempt)
=========================  ===========================================================
existence of :math:`x_A`   -> A(x_A), exempt :math:`\\delta_{x_A}` (``fact``)
A ⊑ B                      A(x) -> B(x), exempt :math:`\\delta_x`
A ⊓ B ⊑ C                  A(x), B(x) -> C(x)
A ⊑ ⊥, A ⊓ B ⊑ ⊥           A(x) -> ⊥;  A(x), B(x) -> ⊥
A ⊑ ∃r.B                   A(x) -> r(x, x_B);  A(x), dead(x_B) -> ⊥ (dead propagation)
∃r.B ⊑ C, ∃r.B ⊑ ⊥         r(x, y), B(y) -> C(x);  r(x, y), B(y) -> ⊥
r ⊑ s, r ∘ s ⊑ t           r(x, y) -> s(x, y);  r(x, y), s(y, z) -> t(x, z)
A(a), r(a, b)              -> A(a);  -> r(a, b)
=========================  ===========================================================
"""
from __future__ import annotations

import torch as th

from mowl.representation.library.grounded_logic import (DepthHinges, GoedelRules,
                                                        LukasiewiczRules, Minimality,
                                                        as_float_tensor, check_same_shape)
from mowl.representation.library.grounded_parts import UnionDepth

__all__ = [
    "ConstraintSet", "ELEMENT_CONSTRAINTS", "REGION_CONSTRAINTS", "LUKASIEWICZ", "GOEDEL", "HINGES", "crisp_membership", "fact", "gci0", "gci0_region",
    "gci1", "gci1_region", "gci0_bot", "gci0_bot_region", "gci1_bot", "gci1_bot_region",
    "gci2_dead", "gci2", "gci2_region", "gci2_region_trans", "gci3", "gci3_region",
    "gci3_region_trans", "gci3_bot", "gci3_bot_region", "gci3_bot_region_trans",
    "role_inclusion", "role_inclusion_region", "role_inclusion_region_trans", "role_chain",
    "role_chain_region", "role_chain_region_trans", "class_assertion", "role_assertion",
    "object_property_assertion", "minimality",
]

#: The logic of GroundedEL's element constraints.
LUKASIEWICZ = LukasiewiczRules()
#: The logic of the chain of two boxes of translations (minimum of the two degrees).
GOEDEL = GoedelRules()
#: Points against depths.
HINGES = DepthHinges()


def crisp_membership(depth: th.Tensor) -> th.Tensor:
    """:math:`[\\text{depth} > 0]` (the crisp reading of the measure)."""
    return UnionDepth.crisp(depth)


# -- at elements ------------------------------------------------------------------------------
def fact(m_a, dead=None, rules=LUKASIEWICZ):
    """Existence of the witness :math:`x_A`: :math:`\\to A(x_A)`, exempt if dead."""
    return rules.violation([], m_a, exempt=dead, names=("m_a",))


def gci0(m_a, m_b, dead=None, rules=LUKASIEWICZ):
    """:math:`A \\sqsubseteq B` at :math:`x`: :math:`A(x) \\to B(x)`."""
    return rules.violation([m_a], m_b, exempt=dead, names=("m_a", "m_b"))


def gci1(m_a, m_b, m_c, dead=None, rules=LUKASIEWICZ):
    """:math:`A \\sqcap B \\sqsubseteq C` at :math:`x`: :math:`A(x), B(x) \\to C(x)`."""
    return rules.violation([m_a, m_b], m_c, exempt=dead, names=("m_a", "m_b", "m_c"))


def gci0_bot(m_a, dead=None, rules=LUKASIEWICZ):
    """:math:`A \\sqsubseteq \\bot` at :math:`x`: :math:`A(x) \\to \\bot`."""
    return rules.violation([m_a], None, exempt=dead, names=("m_a",))


def gci1_bot(m_a, m_b, dead=None, rules=LUKASIEWICZ):
    """:math:`A \\sqcap B \\sqsubseteq \\bot` at :math:`x`: :math:`A(x), B(x) \\to \\bot`."""
    return rules.violation([m_a, m_b], None, exempt=dead, names=("m_a", "m_b"))


def gci2_dead(m_a, dead_filler, dead=None, rules=LUKASIEWICZ):
    """Dead propagation of :math:`A \\sqsubseteq \\exists r.B`: an element of :math:`A`
    whose filler :math:`x_B` is dead is dead, :math:`A(x), \\delta_{x_B} \\to \\bot`."""
    return rules.violation([m_a, dead_filler], None, exempt=dead,
                           names=("m_a", "dead_filler"))


def gci2(m_a, rho, dead=None, dead_filler=None, rules=LUKASIEWICZ):
    """:math:`A \\sqsubseteq \\exists r.B` at :math:`x`: :math:`A(x) \\to r(x, x_B)`, plus
    the dead propagation when ``dead_filler`` is given."""
    edge = rules.violation([m_a], rho, exempt=dead, names=("m_a", "rho"))
    if dead_filler is None:
        return edge
    return rules.total(edge, gci2_dead(as_float_tensor(m_a), dead_filler, dead=dead, rules=rules))


def gci3(rho, m_b, m_c, dead=None, rules=LUKASIEWICZ):
    """:math:`\\exists r.B \\sqsubseteq C` at :math:`(x, y)`: :math:`r(x, y), B(y) \\to C(x)`."""
    return rules.violation([rho, m_b], m_c, exempt=dead, names=("rho", "m_b", "m_c"))


def gci3_bot(rho, m_b, dead=None, rules=LUKASIEWICZ):
    """:math:`\\exists r.B \\sqsubseteq \\bot` at :math:`(x, y)`: :math:`r(x, y), B(y) \\to \\bot`."""
    return rules.violation([rho, m_b], None, exempt=dead, names=("rho", "m_b"))


def role_inclusion(rho_r, rho_s, dead=None, rules=LUKASIEWICZ):
    """:math:`r \\sqsubseteq s` at :math:`(x, y)`: :math:`r(x, y) \\to s(x, y)`."""
    return rules.violation([rho_r], rho_s, exempt=dead, names=("rho_r", "rho_s"))


def role_chain(rho_r, rho_s, rho_t, dead=None, rules=LUKASIEWICZ):
    """:math:`r \\circ s \\sqsubseteq t` at :math:`(x, y, z)`:
    :math:`r(x, y), s(y, z) \\to t(x, z)`."""
    return rules.violation([rho_r, rho_s], rho_t, exempt=dead,
                           names=("rho_r", "rho_s", "rho_t"))


def class_assertion(m_a, dead=None, rules=LUKASIEWICZ):
    """:math:`A(a)`: :math:`\\to A(a)`."""
    return rules.violation([], m_a, exempt=dead, names=("m_a",))


def role_assertion(rho, dead=None, rules=LUKASIEWICZ):
    """:math:`r(a, b)`: :math:`\\to r(a, b)`."""
    return rules.violation([], rho, exempt=dead, names=("rho",))


object_property_assertion = role_assertion


def minimality(m_rand=None, rho_rand=None, dead=None, eps=1e-2, eps_r=None, eps_dead=None):
    """The minimality regulariser (:class:`~mowl.representation.library.grounded_logic.Minimality`)."""
    return Minimality(eps, eps_r, eps_dead).penalty(m_rand, rho_rand, dead)


# -- at points of regions ---------------------------------------------------------------------
def gci0_region(depth_b, margin=0.0):
    """:math:`A \\sqsubseteq B` at a point :math:`z` of :math:`A`: :math:`z` inside :math:`B`."""
    return HINGES.inside(depth_b, margin)


def gci1_region(depth_c, margin=0.0):
    """:math:`A \\sqcap B \\sqsubseteq C` at a point of :math:`A \\cap B`: inside :math:`C`."""
    return HINGES.inside(depth_c, margin)


def gci0_bot_region(depth_a, margin=0.0):
    """:math:`A \\sqsubseteq \\bot` at a point of :math:`A`: outside :math:`A`."""
    return HINGES.outside(depth_a, margin)


def gci1_bot_region(depth_a, depth_b, margin=0.0):
    """:math:`A \\sqcap B \\sqsubseteq \\bot` at a point: not in both."""
    return HINGES.apart(depth_a, depth_b, margin)


def gci2_region(role_depth, margin=0.0, agg="max", agg_tau=0.05, depth_a=None,
                dead_filler=None):
    """:math:`A \\sqsubseteq \\exists r.B` at a point :math:`z` of :math:`A`: some
    candidate filler is related (``role_depth`` ``(B,)`` or ``(B, M)`` over candidates);
    where the filler is dead, :math:`z` must leave :math:`A` instead."""
    rd = HINGES.exists(role_depth, agg=agg, agg_tau=agg_tau)
    edge = HINGES.inside(rd, margin)
    if depth_a is None or dead_filler is None:
        return edge
    da = as_float_tensor(depth_a, rd.dtype).to(device=rd.device, dtype=rd.dtype)
    df = as_float_tensor(dead_filler, rd.dtype).to(device=rd.device, dtype=rd.dtype)
    check_same_shape("role_depth", rd, "depth_a", da)
    check_same_shape("role_depth", rd, "dead_filler", df)
    return HINGES.select(df > 0.5, HINGES.outside(da, margin), edge)


def gci3_region(role_depth, depth_b, depth_c, margin=0.0, smooth=False):
    """:math:`\\exists r.B \\sqsubseteq C` at a pair :math:`(z, y)`: if related and
    :math:`y \\in B` (crisply, or :math:`\\sigma(\\min)` with ``smooth``), :math:`z \\in C`."""
    return HINGES.guarded(role_depth, depth_b, depth_c, margin, smooth,
                          names=("role_depth", "depth_b", "depth_c"))


def gci3_bot_region(role_depth, depth_b, margin=0.0):
    """:math:`\\exists r.B \\sqsubseteq \\bot` at a pair: not related with :math:`y \\in B`."""
    return HINGES.apart(depth_b, role_depth, margin, names=("depth_b", "role_depth"))


def role_inclusion_region(role_depth_s, margin=0.0):
    """:math:`r \\sqsubseteq s` at a pair of :math:`r`: inside :math:`s`."""
    return HINGES.inside(role_depth_s, margin)


def role_chain_region(role_depth_r, role_depth_s, role_depth_t, margin=0.0, smooth=False):
    """:math:`r \\circ s \\sqsubseteq t` at :math:`(x, y, z)`: guarded by :math:`r(x, y)` and
    :math:`s(y, z)`, inside :math:`t`."""
    return HINGES.guarded(role_depth_r, role_depth_s, role_depth_t, margin, smooth,
                          names=("role_depth_r", "role_depth_s", "role_depth_t"))


def gci2_region_trans(m_a_z, m_exists_z, rules=LUKASIEWICZ):
    """Translation mode, :math:`A \\sqsubseteq \\exists r.B` at :math:`z`:
    :math:`A(z) \\to (\\exists r.B)(z)`."""
    return rules.violation([m_a_z], m_exists_z, names=("m_a_z", "m_exists_z"))


def gci3_region_trans(m_exists_z, m_c_z, rules=LUKASIEWICZ):
    """Translation mode, :math:`\\exists r.B \\sqsubseteq C` at :math:`z`:
    :math:`(\\exists r.B)(z) \\to C(z)`."""
    return rules.violation([m_exists_z], m_c_z, names=("m_exists_z", "m_c_z"))


def gci3_bot_region_trans(m_exists_z, rules=LUKASIEWICZ):
    """Translation mode, :math:`\\exists r.B \\sqsubseteq \\bot` at :math:`z`."""
    return rules.violation([m_exists_z], None, names=("m_exists_z",))


def role_inclusion_region_trans(m_vr_v, m_vs_v, rules=LUKASIEWICZ):
    """Translation mode, :math:`r \\sqsubseteq s` at a translation :math:`v`:
    :math:`V_r(v) \\to V_s(v)`."""
    return rules.violation([m_vr_v], m_vs_v, names=("m_vr_v", "m_vs_v"))


def role_chain_region_trans(m_vr_v1, m_vs_v2, m_vt_sum, rules=GOEDEL):
    """Translation mode, :math:`r \\circ s \\sqsubseteq t` at translations :math:`v_1, v_2`:
    :math:`V_r(v_1), V_s(v_2) \\to V_t(v_1 + v_2)` (Gödel: the minimum of the premises)."""
    return rules.violation([m_vr_v1, m_vs_v2], m_vt_sum,
                           names=("m_vr_v1", "m_vs_v2", "m_vt_sum"))


#: The constraints read at elements (and the region constraints of translation mode that
#: are rules on degrees): they take the logic.
ELEMENT_CONSTRAINTS = ("fact", "gci0", "gci1", "gci0_bot", "gci1_bot", "gci2_dead", "gci2",
                       "gci3", "gci3_bot", "role_inclusion", "role_chain", "class_assertion",
                       "role_assertion", "gci2_region_trans", "gci3_region_trans",
                       "gci3_bot_region_trans", "role_inclusion_region_trans")
#: The constraints at points of regions (depth hinges) and the regulariser.
REGION_CONSTRAINTS = ("gci0_region", "gci1_region", "gci0_bot_region", "gci1_bot_region",
                      "gci2_region", "gci3_region", "gci3_bot_region", "role_inclusion_region",
                      "role_chain_region", "minimality", "crisp_membership")


class ConstraintSet:
    """The grounded constraints of every axiom form read in one logic: the objective part of
    GroundedEL. ``rules`` is a :class:`~mowl.representation.library.grounded_logic.Rules`
    (default Łukasiewicz, the published reading) used by every element constraint;
    ``chain_rules`` reads the chain of two boxes of translations (default Gödel); ``hinges``
    the points against depths. Attributes are the functions of this module with the logic
    bound (``constraints.gci0(m_a, m_b, dead=...)``).

    :param rules: Rules or the name of a logic (:data:`~mowl.representation.library.grounded_logic.RULES`).
    """

    def __init__(self, rules=LUKASIEWICZ, chain_rules=GOEDEL, hinges=HINGES):
        from functools import partial
        from mowl.representation.library.grounded_logic import get_rules
        self.rules = get_rules(rules)
        self.chain_rules = get_rules(chain_rules)
        self.hinges = hinges
        module = globals()
        for name in ELEMENT_CONSTRAINTS:
            setattr(self, name, partial(module[name], rules=self.rules))
        self.object_property_assertion = self.role_assertion
        self.role_chain_region_trans = partial(role_chain_region_trans, rules=self.chain_rules)
        for name in REGION_CONSTRAINTS:
            setattr(self, name, module[name])

    @property
    def parts(self):
        return {"logic": self.rules.name, "chain logic": self.chain_rules.name,
                "hinges": type(self.hinges).__name__}
