"""Temporal al-cone models of Boolean :math:`LTL_{\\mathcal{ALC}}` ontologies.

Implements the construction of Leemhuis (FCR 2024, CEUR Vol-3763, paper 3). The paper
extends the al-cone semantics of :mod:`mowl.geometric.cones` (Özçep, Leemhuis & Wolter) to
the role-free temporal description logic :math:`LTL_{\\mathcal{ALC}}` with the operators
next (:math:`\\circ`), until (:math:`\\mathcal{U}`), eventually (:math:`\\Diamond C = \\top
\\,\\mathcal{U}\\, C`) and always (:math:`\\Box C = \\neg\\Diamond\\neg C`), flexible concept
names, rigid constants and constant domains. A temporal interpretation is a sequence
:math:`\\mathcal{J}(0), \\mathcal{J}(1), \\dots` of classical interpretations over one domain.

Geometry
--------
Every time point :math:`i` gets its own al-cone interpretation, all in the same
:math:`\\mathbb{R}^n` (in the paper: on the sphere of radius :math:`i + 1`, so that an
individual is one *ray* that meets all spheres and keeps its identity over time). A concept
:math:`C` at time :math:`i` is an al-cone (:class:`~mowl.geometric.cones.ALCone`); an
individual is a ray, represented by a non-zero point (membership of a ray in an al-cone
does not depend on the point's length). The operators are

* :math:`\\sqcap`, :math:`\\neg`, :math:`\\sqcup`: intersection, polar cone and convex union
  of the al-cones of the *same* time point;
* :math:`(\\circ C)(i) = C(i + 1)`;
* :math:`(C\\,\\mathcal{U}\\,D)(i) = \\bigsqcup_{m \\ge i} \\big(D(m) \\sqcap
  \\bigsqcap_{i \\le k < m} C(k)\\big)`.

Ultimately periodic models
--------------------------
Following Theorem 1 of the paper (ultimately periodic model property of LTL) a model is
finite: :math:`L = p + m` time points, where time point :math:`L - 1` is followed by
``loop_start`` :math:`= p`. :class:`TemporalALConeInterpretation` stores exactly these
:math:`L` al-cone interpretations. The until-operator is evaluated by unrolling the
:math:`L` positions reachable from :math:`i`.

Construction (Theorem 2, direction "satisfiable implies geometric model")
------------------------------------------------------------------------
Given a (finite, ultimately periodic) classical temporal model, the *trace* of a domain
element is its sequence of concept memberships. Each distinct trace becomes one *half-axis*
of :math:`\\mathbb{R}^n` (the algebraic atoms of the Boolean algebra of al-cones,
JAIR Prop. 7, applied to the finitely many (concept, time) pairs): the atom :math:`A` at time
:math:`i` is the al-cone spanned by the half-axes of the traces with :math:`A` at :math:`i`,
and an individual is the ray on the half-axis of its trace
(:meth:`TemporalALConeInterpretation.from_lasso_model`). Every operator acts bitwise on
the half-axis sets (a Boolean algebra), so an individual lies in a concept at a time point
iff the classical model says so, and an inclusion holds geometrically iff it holds on all
traces.

Deviation from the paper (read this)
------------------------------------
The paper writes the until-semantics of Eq. (1) as a *set union* of al-cone pieces and
remarks that the resulting sets "are neither closed convex cones nor al-cones anymore".
With a set union, :math:`C \\sqsubseteq \\Diamond D` can fail geometrically although it holds
on every element (an al-cone that contains two half-axes is not covered by the union of two
al-cones, each holding one of them). mOWL therefore uses the **convex union** (the
disjunction of al-cone semantics, :math:`(X^\\circ \\cap Y^\\circ)^\\circ`) for the
disjunction inside until, so that every concept is again an al-cone at every time point and
the construction above is correct. This is the reading of "interpreted via the other
operators" for :math:`\\Diamond` and :math:`\\Box` in the paper; it is our reading for
:math:`\\mathcal{U}`.

Scope
-----
* Role-free (Boolean) :math:`LTL_{\\mathcal{ALC}}` only, as in the paper; constant domain.
* The paper gives a *model construction* and an expressivity theorem, no learning
  procedure. There is no training here.
* Satisfiability is decided only up to bounds: :func:`find_lasso_model` searches ultimately
  periodic models with a given prefix and period length; the bounds needed for completeness
  are not given in the paper.
* ABox assertions :math:`C(a)` are evaluated at time point ``0`` (the paper defines models
  of concepts at time 0 and gives no ABox semantics; this is our convention).
* One half-axis per distinct trace; the strongly faithful doubled model (JAIR Prop. 8) is
  not built, so several individuals with the same trace share one point.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mowl.geometric.cones import (
    ALCone,
    And,
    Atomic,
    BOTTOM_CONCEPT,
    Concept,
    ConceptAssertion,
    GCI,
    Not,
    Or,
    TOP_CONCEPT,
)

__all__ = [
    "Next",
    "Until",
    "Eventually",
    "Always",
    "TemporalOntology",
    "LassoModel",
    "TemporalALConeInterpretation",
    "find_lasso_model",
    "parse_concept",
    "parse_axiom",
]


# ---------------------------------------------------------------------------
# Syntax: ALC (role-free) from mowl.geometric.cones plus next and until
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Next(Concept):
    """:math:`\\circ C`: ``C`` holds at the next time point."""

    arg: Concept

    def __str__(self):
        return f"○{self.arg}"


@dataclass(frozen=True)
class Until(Concept):
    """:math:`C\\,\\mathcal{U}\\,D`: ``D`` holds now or later and ``C`` holds until then."""

    left: Concept
    right: Concept

    def __str__(self):
        return f"({self.left} U {self.right})"


def Eventually(concept: Concept) -> Concept:
    """:math:`\\Diamond C = \\top\\,\\mathcal{U}\\,C` (as in the paper)."""
    return Until(TOP_CONCEPT, concept)


def Always(concept: Concept) -> Concept:
    """:math:`\\Box C = \\neg\\Diamond\\neg C` (as in the paper)."""
    return Not(Eventually(Not(concept)))


def concept_names(concept: Concept) -> set:
    """Concept names occurring in a temporal concept."""
    if isinstance(concept, Atomic):
        return {concept.name}
    if isinstance(concept, (And, Or)):
        return set().union(*(concept_names(a) for a in concept.args)) if concept.args else set()
    if isinstance(concept, Not) or isinstance(concept, Next):
        return concept_names(concept.arg)
    if isinstance(concept, Until):
        return concept_names(concept.left) | concept_names(concept.right)
    return set()


@dataclass
class TemporalOntology:
    """A Boolean :math:`LTL_{\\mathcal{ALC}}` ontology: general concept inclusions (valid at
    every time point) and concept assertions (valid at time point 0).

    :param tbox: List of :class:`~mowl.geometric.cones.GCI`.
    :param abox: List of :class:`~mowl.geometric.cones.ConceptAssertion`.
    """

    tbox: List[GCI]
    abox: List[ConceptAssertion]

    def __init__(self, tbox: Iterable[GCI] = (), abox: Iterable[ConceptAssertion] = ()):
        self.tbox = list(tbox)
        self.abox = list(abox)

    @classmethod
    def parse(cls, tbox: Iterable[str] = (), abox: Iterable[str] = ()) -> "TemporalOntology":
        """Build from strings, see :func:`parse_axiom` (``"A & F B <= G C"``) and, for the
        ABox, ``"C(a)"`` with a concept in the same syntax."""
        return cls([parse_axiom(s) for s in tbox], [_parse_assertion(s) for s in abox])

    @property
    def axioms(self):
        return list(self.tbox) + list(self.abox)

    def concept_names(self) -> set:
        out = set()
        for g in self.tbox:
            out |= concept_names(g.sub) | concept_names(g.sup)
        for a in self.abox:
            out |= concept_names(a.concept)
        return out

    def individuals(self) -> List[str]:
        return sorted({a.individual for a in self.abox})


# -- a tiny parser ----------------------------------------------------------

_TOKEN = re.compile(r"\s*(<=|[()~&|]|[A-Za-z_][A-Za-z0-9_:#/.\-]*)")
_RESERVED = {"X", "F", "G", "U", "top", "bot"}


def _tokens(text: str) -> List[str]:
    pos, out = 0, []
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"Cannot parse {text!r} at position {pos}.")
        out.append(m.group(1))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens: Sequence[str]):
        self.t, self.i = list(tokens), 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self, expected=None):
        tok = self.peek()
        if tok is None or (expected is not None and tok != expected):
            raise ValueError(f"Expected {expected!r}, found {tok!r}.")
        self.i += 1
        return tok

    def disj(self):
        parts = [self.conj()]
        while self.peek() == "|":
            self.take()
            parts.append(self.conj())
        return parts[0] if len(parts) == 1 else Or(tuple(parts))

    def conj(self):
        parts = [self.until()]
        while self.peek() == "&":
            self.take()
            parts.append(self.until())
        return parts[0] if len(parts) == 1 else And(tuple(parts))

    def until(self):
        left = self.unary()
        if self.peek() == "U":
            self.take()
            return Until(left, self.until())
        return left

    def unary(self):
        tok = self.peek()
        if tok == "~":
            self.take()
            return Not(self.unary())
        if tok == "X":
            self.take()
            return Next(self.unary())
        if tok == "F":
            self.take()
            return Eventually(self.unary())
        if tok == "G":
            self.take()
            return Always(self.unary())
        if tok == "(":
            self.take()
            c = self.disj()
            self.take(")")
            return c
        if tok == "top":
            self.take()
            return TOP_CONCEPT
        if tok == "bot":
            self.take()
            return BOTTOM_CONCEPT
        if tok is None or tok in {"<=", ")", "&", "|", "U"}:
            raise ValueError(f"Unexpected token {tok!r}.")
        self.take()
        return Atomic(tok)


def parse_concept(text: str) -> Concept:
    """Parse a temporal concept. Syntax: names, ``top``, ``bot``, ``~C`` (negation),
    ``C & D``, ``C | D``, ``X C`` (next), ``F C`` (eventually), ``G C`` (always),
    ``C U D`` (until) and parentheses. Precedence: unary operators, ``U``, ``&``, ``|``.
    ``X``, ``F``, ``G``, ``U``, ``top`` and ``bot`` are reserved."""
    p = _Parser(_tokens(text))
    c = p.disj()
    if p.peek() is not None:
        raise ValueError(f"Unexpected trailing input {p.peek()!r} in {text!r}.")
    return c


def parse_axiom(text: str) -> GCI:
    """Parse ``"C <= D"`` into a :class:`~mowl.geometric.cones.GCI`."""
    if "<=" not in text:
        raise ValueError(f"A GCI needs '<=': {text!r}")
    left, right = text.split("<=", 1)
    return GCI(parse_concept(left), parse_concept(right))


def _parse_assertion(text: str) -> ConceptAssertion:
    m = re.fullmatch(r"\s*(.*)\(\s*([^()\s]+)\s*\)\s*", text)
    if not m:
        raise ValueError(f"A concept assertion looks like 'C(a)': {text!r}")
    return ConceptAssertion(parse_concept(m.group(1)), m.group(2))


# ---------------------------------------------------------------------------
# Classical semantics on ultimately periodic traces (independent evaluator)
# ---------------------------------------------------------------------------

def _classical(concept: Concept, trace: Sequence[FrozenSet[str]],
               loop_start: int) -> Tuple[bool, ...]:
    """Truth of ``concept`` at every time point of one element with the given ``trace``.

    Until is the least fixpoint of :math:`u_i = d_i \\lor (c_i \\land u_{succ(i)})`, which
    is independent of the unrolling used by the geometric evaluator."""
    length = len(trace)

    def succ(i):
        return i + 1 if i + 1 < length else loop_start

    def ev(c) -> Tuple[bool, ...]:
        if c is TOP_CONCEPT or c == TOP_CONCEPT:
            return (True,) * length
        if c is BOTTOM_CONCEPT or c == BOTTOM_CONCEPT:
            return (False,) * length
        if isinstance(c, Atomic):
            return tuple(c.name in trace[i] for i in range(length))
        if isinstance(c, Not):
            return tuple(not v for v in ev(c.arg))
        if isinstance(c, And):
            vals = [ev(a) for a in c.args]
            return tuple(all(v[i] for v in vals) for i in range(length))
        if isinstance(c, Or):
            vals = [ev(a) for a in c.args]
            return tuple(any(v[i] for v in vals) for i in range(length))
        if isinstance(c, Next):
            v = ev(c.arg)
            return tuple(v[succ(i)] for i in range(length))
        if isinstance(c, Until):
            cl, dr = ev(c.left), ev(c.right)
            u = [False] * length
            changed = True
            while changed:
                changed = False
                for i in range(length):
                    new = dr[i] or (cl[i] and u[succ(i)])
                    if new != u[i]:
                        u[i], changed = new, True
            return tuple(u)
        raise TypeError(f"Unsupported temporal concept {c!r} (roles are not covered).")

    return ev(concept)


@dataclass
class LassoModel:
    """A finite description of an ultimately periodic classical temporal model.

    :param traces: For every domain element its *trace*: a list of length ``L`` with the
        set of concept names that hold at each time point ``0 .. L - 1``. Time point
        ``L - 1`` is followed by ``loop_start``.
    :param loop_start: Index of the first time point of the period; ``0`` if the model is
        purely periodic.
    :param individuals: Maps individual names to a domain element (index into ``traces``).
        Constants are rigid, as in the paper.

    The domain is ``range(len(traces))``. Several elements may have the same trace.
    """

    traces: List[List[FrozenSet[str]]]
    loop_start: int = 0
    individuals: Dict[str, int] = None

    def __post_init__(self):
        self.traces = [[frozenset(s) for s in tr] for tr in self.traces]
        if not self.traces:
            raise ValueError("A model needs a non-empty domain.")
        lengths = {len(tr) for tr in self.traces}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError("All traces must have the same positive length.")
        if not 0 <= self.loop_start < len(self.traces[0]):
            raise ValueError("loop_start must be a valid time point.")
        self.individuals = dict(self.individuals or {})
        for name, d in self.individuals.items():
            if not 0 <= d < len(self.traces):
                raise ValueError(f"Individual {name!r} maps outside the domain.")

    @property
    def length(self) -> int:
        return len(self.traces[0])

    def holds(self, concept: Concept, element: int, time: int) -> bool:
        """Classical truth of ``concept`` for domain element ``element`` at ``time``."""
        values = _classical(concept, self.traces[element], self.loop_start)
        return values[self.canonical_time(time)]

    def canonical_time(self, t: int) -> int:
        L, p = self.length, self.loop_start
        if t < L:
            return t
        return p + (t - p) % (L - p)

    def satisfies(self, axiom) -> bool:
        """Classical satisfaction of a GCI (all elements, all time points) or a concept
        assertion (at time point 0)."""
        if isinstance(axiom, GCI):
            for tr in self.traces:
                s, u = (_classical(axiom.sub, tr, self.loop_start),
                        _classical(axiom.sup, tr, self.loop_start))
                if any(a and not b for a, b in zip(s, u)):
                    return False
            return True
        if isinstance(axiom, ConceptAssertion):
            return self.holds(axiom.concept, self.individuals[axiom.individual], 0)
        raise TypeError(f"Unsupported axiom {axiom!r}.")

    def is_model(self, ontology: TemporalOntology) -> bool:
        return all(self.satisfies(a) for a in ontology.axioms)


# ---------------------------------------------------------------------------
# Geometric interpretation
# ---------------------------------------------------------------------------

class TemporalALConeInterpretation:
    """A temporal al-cone interpretation: one al-cone interpretation per time point, in one
    :math:`\\mathbb{R}^n`, with rigid individuals (rays).

    :param dim: The dimension :math:`n`.
    :param loop_start: Time point that follows the last one (``L - 1``).
    :param atoms: For every concept name a list of ``L`` al-cones (one per time point).
    :param individuals: Maps individual names to non-zero points of :math:`\\mathbb{R}^n`.
    :param length: Number of time points ``L``; needed only if ``atoms`` is empty.
    """

    def __init__(self, dim: int, loop_start: int, atoms: Mapping[str, Sequence[ALCone]],
                 individuals: Optional[Mapping[str, Sequence[float]]] = None,
                 length: Optional[int] = None):
        self.dim = int(dim)
        lengths = {len(v) for v in atoms.values()}
        if len(lengths) > 1 or (length is not None and lengths - {length}):
            raise ValueError("Every concept name needs one al-cone per time point.")
        self.length = length if length is not None else (lengths.pop() if lengths else 1)
        if not 0 <= loop_start < self.length:
            raise ValueError("loop_start must be a valid time point.")
        self.loop_start = int(loop_start)
        self.atoms: Dict[str, List[ALCone]] = {}
        for name, cones in atoms.items():
            for c in cones:
                if c.dim != self.dim:
                    raise ValueError("Al-cone of the wrong dimension.")
            self.atoms[name] = list(cones)
        self.individuals: Dict[str, np.ndarray] = {}
        for name, x in (individuals or {}).items():
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != self.dim or not np.any(x):
                raise ValueError(f"Individual {name!r} must be a non-zero point of R^{self.dim}.")
            self.individuals[name] = x
        self._cache: Dict[Tuple[Concept, int], ALCone] = {}

    # -- time ---------------------------------------------------------------
    def successor(self, t: int) -> int:
        """The time point after ``t`` (time point ``L - 1`` is followed by ``loop_start``)."""
        return t + 1 if t + 1 < self.length else self.loop_start

    def canonical_time(self, t: int) -> int:
        """Time point of the stored prefix + period that represents time ``t >= 0``."""
        L, p = self.length, self.loop_start
        return t if t < L else p + (t - p) % (L - p)

    # -- concepts -----------------------------------------------------------
    def cone(self, concept: Concept, t: int = 0) -> ALCone:
        """The al-cone :math:`C^{\\mathcal{J}(t)}` of ``concept`` at time point ``t``."""
        t = self.canonical_time(t)
        key = (concept, t)
        if key not in self._cache:
            self._cache[key] = self._cone(concept, t)
        return self._cache[key]

    def _cone(self, c: Concept, t: int) -> ALCone:
        if c == TOP_CONCEPT:
            return ALCone.top(self.dim)
        if c == BOTTOM_CONCEPT:
            return ALCone.bottom(self.dim)
        if isinstance(c, Atomic):
            if c.name not in self.atoms:
                return ALCone.bottom(self.dim)
            return self.atoms[c.name][t]
        if isinstance(c, Not):
            return self.cone(c.arg, t).polar()
        if isinstance(c, And):
            out = ALCone.top(self.dim)
            for a in c.args:
                out = out.intersection(self.cone(a, t))
            return out
        if isinstance(c, Or):
            out = ALCone.bottom(self.dim)
            for a in c.args:
                out = out.join(self.cone(a, t))
            return out
        if isinstance(c, Next):
            return self.cone(c.arg, self.successor(t))
        if isinstance(c, Until):
            out = ALCone.bottom(self.dim)
            prefix = ALCone.top(self.dim)
            pos = t
            for _ in range(self.length):
                out = out.join(prefix.intersection(self.cone(c.right, pos)))
                prefix = prefix.intersection(self.cone(c.left, pos))
                pos = self.successor(pos)
            return out
        raise TypeError(f"Unsupported temporal concept {c!r} (roles are not covered).")

    # -- individuals and axioms --------------------------------------------
    def point(self, individual: str) -> np.ndarray:
        """The ray of an individual, as a point (rigid: the same at all time points)."""
        return self.individuals[individual].copy()

    def instance_of(self, individual: str, concept: Concept, t: int = 0) -> bool:
        """``True`` iff the ray of ``individual`` lies in ``concept`` at time point ``t``."""
        return self.cone(concept, t).contains_point(self.individuals[individual])

    def satisfies(self, axiom) -> bool:
        """GCI: :math:`C(t) \\subseteq D(t)` at every time point. Concept assertion
        :math:`C(a)`: :math:`a` lies in :math:`C(0)` (our convention, see module docstring)."""
        if isinstance(axiom, GCI):
            return all(self.cone(axiom.sub, t).subset_of(self.cone(axiom.sup, t))
                       for t in range(self.length))
        if isinstance(axiom, ConceptAssertion):
            return self.instance_of(axiom.individual, axiom.concept, 0)
        raise TypeError(f"Unsupported axiom {axiom!r}.")

    def violations(self, axioms: Iterable) -> list:
        return [a for a in axioms if not self.satisfies(a)]

    def is_model(self, axioms) -> bool:
        if isinstance(axioms, TemporalOntology):
            axioms = axioms.axioms
        return not self.violations(axioms)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_lasso_model(cls, model: LassoModel) -> "TemporalALConeInterpretation":
        """The construction of Theorem 2 of the paper (see the module docstring): every
        distinct trace becomes a half-axis, the atom :math:`A` at time :math:`i` is the
        al-cone of the half-axes of the traces with :math:`A` at :math:`i`, and an
        individual is the ray of its trace's half-axis.

        This is the combination of the source ``lasso model`` and the reading
        ``half-axes`` with one cell per distinct trace (``merge=True``); :attr:`module` is
        its geometry."""
        from mowl.representation.construction import RECIPES
        geometry = RECIPES["temporal_cones"].build(model)
        regions = geometry.read.classes
        bits = regions.bits
        atoms = {name: [ALCone(bits[row, t].numpy()) for t in range(model.length)]
                 for row, name in enumerate(geometry.names)}
        inds = {a: geometry.read.individuals.centre[k].numpy()
                for k, a in enumerate(geometry.individual_names)}
        out = cls(regions.dim, model.loop_start, atoms, inds, length=model.length)
        # the distinct traces, in the order of their cells
        _, cell = geometry.reading.cells(geometry.built)
        first = {}
        for element, c in enumerate(cell):
            first.setdefault(c, element)
        out.traces = [tuple(model.traces[first[c]]) for c in sorted(first)]
        out._geometry = geometry
        return out

    @property
    def module(self):
        """The geometry of the combination this interpretation was built from
        (:meth:`from_lasso_model`)."""
        geometry = getattr(self, "_geometry", None)
        if geometry is None:
            raise AttributeError("A temporal al-cone interpretation given by its cones has no "
                                 "construction; build it with from_lasso_model.")
        return geometry

    @classmethod
    def from_ontology(cls, ontology: TemporalOntology, max_prefix: int = 2, max_period: int = 3,
                      max_bits: int = 14, witnesses: bool = True
                      ) -> "TemporalALConeInterpretation":
        """Search an ultimately periodic model with :func:`find_lasso_model` and build its
        temporal al-cone model. Raises :class:`ValueError` if none exists within the
        bounds (which does not mean that the ontology is unsatisfiable)."""
        model = find_lasso_model(ontology, max_prefix, max_period, max_bits, witnesses)
        if model is None:
            raise ValueError("No ultimately periodic model found within the given bounds.")
        interp = cls.from_lasso_model(model)
        interp.lasso_model = model
        return interp


# ---------------------------------------------------------------------------
# Model search (bounded)
# ---------------------------------------------------------------------------

def find_lasso_model(ontology: TemporalOntology, max_prefix: int = 2, max_period: int = 3,
                     max_bits: int = 14, witnesses: bool = True) -> Optional[LassoModel]:
    """Search a finite ultimately periodic classical model of a role-free temporal ontology.

    Because the logic has no roles, the truth of every concept for an element depends on
    that element's trace only. A trace is *good* if it satisfies every GCI at every time
    point; the ontology is satisfiable with ``L`` time points iff every individual has a
    good trace that satisfies its assertions at time 0. All traces over the concept names
    of the ontology are enumerated, for ``L = prefix + period`` increasing, so the search
    is exponential: it is skipped for lengths with more than ``max_bits`` (names times
    ``L``) bits.

    :param witnesses: Add one good trace for every (concept name, time point) pair that
        some good trace satisfies, so that no concept name is empty in the model without
        reason (otherwise only the traces of the individuals, and one good trace, are used).
    :returns: A :class:`LassoModel`, or ``None`` if none exists within the bounds. This
        is not a decision procedure: the paper gives no bound on prefix and period.
    """
    names = sorted(ontology.concept_names())
    for length in range(1, max_prefix + max_period + 1):
        for loop_start in range(0, length):
            period = length - loop_start
            if loop_start > max_prefix or period > max_period:
                continue
            if len(names) * length > max_bits:
                continue
            good = _good_traces(ontology, names, length, loop_start)
            if not good:
                continue
            chosen: List[List[FrozenSet[str]]] = []
            inds: Dict[str, int] = {}
            ok = True
            for a in ontology.individuals():
                assertions = [x for x in ontology.abox if x.individual == a]
                found = None
                for tr in good:
                    if all(_classical(x.concept, tr, loop_start)[0] for x in assertions):
                        found = tr
                        break
                if found is None:
                    ok = False
                    break
                if found not in chosen:
                    chosen.append(found)
                inds[a] = chosen.index(found)
            if not ok:
                continue
            if witnesses or not chosen:
                for name in names:
                    for t in range(length):
                        for tr in good:
                            if name in tr[t]:
                                if tr not in chosen:
                                    chosen.append(tr)
                                break
                if not chosen:
                    chosen.append(good[0])
            return LassoModel([list(tr) for tr in chosen], loop_start, inds)
    return None


def _good_traces(ontology, names, length, loop_start):
    bits = len(names) * length
    good = []
    for mask in range(1 << bits):
        trace = [frozenset(names[j] for j in range(len(names))
                           if mask >> (t * len(names) + j) & 1)
                 for t in range(length)]
        if all(not any(a and not b for a, b in zip(_classical(g.sub, trace, loop_start),
                                                   _classical(g.sup, trace, loop_start)))
               for g in ontology.tbox):
            good.append(trace)
    return [tuple(tr) for tr in good]
