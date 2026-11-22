"""Axis-aligned cone (al-cone) models of :math:`\\mathcal{ALC}` ontologies.

Implements the al-cone semantics of Özçep, Leemhuis & Wolter (IJCAI 2020; JAIR 78,
2023): every concept is interpreted as an *axis-aligned cone*

.. math::

   X = X_1 \\times \\dots \\times X_n, \\qquad X_i \\in \\{\\mathbb{R}, \\mathbb{R}^+,
   \\mathbb{R}^-, \\{0\\}\\}

(JAIR 2023, Sec. 4). Conjunction is intersection, negation is the polar cone
:math:`X^\\circ = \\{v \\mid \\langle v, w\\rangle \\le 0 \\text{ for all } w \\in X\\}`,
disjunction is the *convex union* :math:`(X^\\circ \\cap Y^\\circ)^\\circ =
\\mathrm{conHull}(X \\cup Y)` (JAIR Def. 2, Prop. 2), :math:`\\top` is
:math:`\\mathbb{R}^n` and :math:`\\bot` is :math:`\\{0\\}`. Roles are 0/1 *incidence
matrices* over the :math:`2n` half-axes, and
:math:`(\\exists R.C)^I = \\mathrm{conHull}(\\{R^T y \\mid y \\in C^I\\})`,
:math:`(\\forall R.C)^I = (\\neg\\exists R.\\neg C)^I` (JAIR Def. 7).

Representation used here
------------------------
An al-cone is determined by the set of half-axes it contains (JAIR Sec. 4, operator
``halfAxes``). Half-axis ``2 i`` is :math:`\\mathbb{R}^+ e_i` and half-axis ``2 i + 1``
is :math:`\\mathbb{R}^- e_i`. In this encoding (JAIR Prop. 1):

* intersection is intersection of half-axis sets,
* the polar cone is the *complement* of the half-axis set
  (:math:`\\{0\\} \\leftrightarrow \\mathbb{R}`, :math:`\\mathbb{R}^+ \\leftrightarrow
  \\mathbb{R}^-` per coordinate),
* the convex union is the union of half-axis sets (Prop. 1.3, Prop. 3),
* :math:`X \\subseteq Y` iff ``halfAxes(X) ⊆ halfAxes(Y)``, and :math:`X = \\{0\\}`
  iff it contains no half-axis.

Hence the al-cones of :math:`\\mathbb{R}^n` form the Boolean algebra
:math:`O_4^n \\cong 2^{2n}` (JAIR Sec. 4 and Prop. 7), and the incidence-matrix
semantics of :math:`\\exists R` is the classical Kripke diamond on the frame whose
worlds are the half-axes. A point :math:`x` lies in an al-cone :math:`X` iff every
half-axis in its *support* :math:`\\{2i : x_i > 0\\} \\cup \\{2i+1 : x_i < 0\\}` belongs to
``halfAxes(X)``. Points whose support is not a single half-axis are *partial*: they can
lie in :math:`C \\sqcup D` without lying in :math:`C` or :math:`D`, and in neither
:math:`C` nor :math:`\\neg C` (JAIR Sec. 5.1).

Contents
--------
* A small :math:`\\mathcal{ALC}` syntax (:class:`Atomic`, :class:`Not`, :class:`And`,
  :class:`Or`, :class:`Exists`, :class:`Forall`, :data:`TOP_CONCEPT`,
  :data:`BOTTOM_CONCEPT`; axioms :class:`GCI`, :class:`ConceptAssertion`,
  :class:`RoleAssertion`) and a reader from OWLAPI (:func:`axioms_from_owl`).
* :class:`ALCone` with the cone operations.
* :class:`ALConeInterpretation` and :class:`ALConeChecker`, which decides
  satisfaction of TBox and ABox axioms in a given al-cone interpretation.
* :class:`FiniteInterpretation`, a finite classical interpretation with its own
  (independent) evaluator, readable from a mOWL
  :class:`~mowl.geometric.completion.CanonicalModel`.
* Constructions: :meth:`ALConeInterpretation.from_finite_model` (any finite classical
  model of an :math:`\\mathcal{ALC}` ontology gives an al-cone model) and
  :meth:`ALConeInterpretation.from_boolean_ontology` (JAIR Prop. 7, and the strongly
  concept-faithful doubled model of Prop. 8).

Deviations and scope (read before relying on a guarantee)
---------------------------------------------------------
* :meth:`~ALConeInterpretation.from_boolean_ontology` follows the proofs of JAIR
  Props. 7 and 8: the algebraic atoms of the Lindenbaum-Tarski algebra of a Boolean
  TBox (the TBox-consistent complete conjunctions of literals) are placed on half-axes.
  If the number of atoms is odd, one atom is placed on two half-axes (Prop. 7 only
  requires :math:`n \\ge m/2`).
* :meth:`~ALConeInterpretation.from_finite_model` is **our** construction. The
  paper's construction for full :math:`\\mathcal{ALC}` (JAIR Props. 12-15) places the
  algebraic atoms of an infinitary extension of the Lindenbaum-Tarski algebra on the
  half-axes of a possibly infinite-dimensional space and is given as a proof sketch; it
  cannot be built in general. We instead place the *elements* of a finite classical
  model on half-axes and read each role as the incidence matrix of its extension, which
  is an al-cone interpretation in the sense of JAIR Def. 7. By induction on concepts,
  using Prop. 1 and Def. 7, ``halfAxes(C^cone) = {h : f(h) ∈ C^I}`` for every
  :math:`\\mathcal{ALC}` concept, where :math:`f` maps half-axes onto domain elements;
  so the cone interpretation satisfies exactly the GCIs and assertions the classical
  model satisfies. Since :math:`\\mathcal{ALC}` has the finite model property, this
  realises the direction "satisfiable ⇒ has an al-cone model" of JAIR Prop. 12 (without
  the faithfulness part). The result is only as faithful as the finite model supplied.
* Role assertions :math:`R(a, b)` are checked as JAIR states "role-satisfiability"
  (Sec. 6, before Prop. 12): :math:`b^I \\in R\\, a^I`, i.e. the support of :math:`b^I`
  is contained in the set of :math:`R`-successor half-axes of the support of
  :math:`a^I`. For individuals on a single half-axis this coincides with the classical
  reading; for partial individuals it does not imply
  :math:`a \\in \\exists R.C` from :math:`C(b)` (the paper notes that ABox-faithfulness
  "in its original form cannot be ensured", Sec. 6).
* The IJCAI 2020 paper interprets roles differently (as relations on points, with
  :math:`\\forall R.C` the minimal al-cone containing the classical box, Def. 3). This
  module follows the JAIR definition.

References
----------
- Ö. L. Özçep, M. Leemhuis & D. Wolter, *Cone Semantics for Logics with Negation*,
  IJCAI 2020, pp. 1820-1826.
- Ö. L. Özçep, M. Leemhuis & D. Wolter, *Embedding Ontologies in the Description Logic
  ALC by Axis-Aligned Cones*, JAIR 78:287-337, 2023, Defs. 2 and 7, Props. 1-3, 5, 7,
  8, 11, 12, Examples 1, 2, 7.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Dict, FrozenSet, Iterable, List, Mapping, Optional,
                    Sequence, Set, Tuple, Union)

import numpy as np

from mowl.geometric.errors import (InconsistentOntology, InternalConsistencyError,  # noqa: F401
                                   UnsupportedAxiom)

if TYPE_CHECKING:  # pragma: no cover
    from mowl.geometric.completion import CanonicalModel

OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
OWL_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"

#: Factor names of the Boolean algebra :math:`O_4` (JAIR Fig. 5).
FACTORS = ("0", "+", "-", "R")
_FACTOR_BITS = {"0": (False, False), "+": (True, False), "-": (False, True),
                "R": (True, True), "u": (True, True)}


# ---------------------------------------------------------------------------
# ALC syntax
# ---------------------------------------------------------------------------

class Concept:
    """Base class of :math:`\\mathcal{ALC}` concept descriptions."""

    def __and__(self, other):
        return And((self, other))

    def __or__(self, other):
        return Or((self, other))

    def __invert__(self):
        return Not(self)


@dataclass(frozen=True)
class _Top(Concept):
    def __str__(self):
        return "⊤"


@dataclass(frozen=True)
class _Bottom(Concept):
    def __str__(self):
        return "⊥"


#: The top concept :math:`\\top`.
TOP_CONCEPT = _Top()
#: The bottom concept :math:`\\bot`.
BOTTOM_CONCEPT = _Bottom()


@dataclass(frozen=True)
class Atomic(Concept):
    """A concept name.

    :param name: The name (for concepts read from OWL, the IRI string).
    """

    name: str

    def __str__(self):
        return self.name.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Not(Concept):
    """Negation :math:`\\neg C`."""

    arg: Concept

    def __str__(self):
        return f"¬{self.arg}"


@dataclass(frozen=True)
class And(Concept):
    """Conjunction :math:`C_1 \\sqcap \\dots \\sqcap C_k` (``args`` is a tuple)."""

    args: Tuple[Concept, ...]

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))

    def __str__(self):
        return "(" + " ⊓ ".join(map(str, self.args)) + ")"


@dataclass(frozen=True)
class Or(Concept):
    """Disjunction :math:`C_1 \\sqcup \\dots \\sqcup C_k` (``args`` is a tuple)."""

    args: Tuple[Concept, ...]

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))

    def __str__(self):
        return "(" + " ⊔ ".join(map(str, self.args)) + ")"


@dataclass(frozen=True)
class Exists(Concept):
    """Existential restriction :math:`\\exists R.C`."""

    role: str
    filler: Concept

    def __str__(self):
        return f"∃{self.role}.{self.filler}"


@dataclass(frozen=True)
class Forall(Concept):
    """Universal restriction :math:`\\forall R.C`."""

    role: str
    filler: Concept

    def __str__(self):
        return f"∀{self.role}.{self.filler}"


@dataclass(frozen=True)
class GCI:
    """General concept inclusion :math:`C \\sqsubseteq D`."""

    sub: Concept
    sup: Concept

    def __str__(self):
        return f"{self.sub} ⊑ {self.sup}"


@dataclass(frozen=True)
class ConceptAssertion:
    """Concept assertion :math:`C(a)`."""

    concept: Concept
    individual: str

    def __str__(self):
        return f"{self.concept}({self.individual})"


@dataclass(frozen=True)
class RoleAssertion:
    """Role assertion :math:`R(a, b)` (``subject`` a, ``object`` b)."""

    role: str
    subject: str
    object: str

    def __str__(self):
        return f"{self.role}({self.subject}, {self.object})"


ALCAxiom = Union[GCI, ConceptAssertion, RoleAssertion]


def concept_names(concept: Concept) -> Set[str]:
    """Return the concept names occurring in ``concept``."""
    if isinstance(concept, Atomic):
        return {concept.name}
    if isinstance(concept, Not):
        return concept_names(concept.arg)
    if isinstance(concept, (And, Or)):
        return set().union(*(concept_names(c) for c in concept.args)) if concept.args \
            else set()
    if isinstance(concept, (Exists, Forall)):
        return concept_names(concept.filler)
    return set()


def has_roles(concept: Concept) -> bool:
    """Return ``True`` iff ``concept`` contains an existential or universal restriction."""
    if isinstance(concept, (Exists, Forall)):
        return True
    if isinstance(concept, Not):
        return has_roles(concept.arg)
    if isinstance(concept, (And, Or)):
        return any(has_roles(c) for c in concept.args)
    return False


def signature(axioms: Iterable[ALCAxiom]) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return ``(concept names, role names, individual names)`` of ``axioms``."""
    names: Set[str] = set()
    roles: Set[str] = set()
    inds: Set[str] = set()

    def walk(c):
        if isinstance(c, Atomic):
            names.add(c.name)
        elif isinstance(c, Not):
            walk(c.arg)
        elif isinstance(c, (And, Or)):
            for a in c.args:
                walk(a)
        elif isinstance(c, (Exists, Forall)):
            roles.add(c.role)
            walk(c.filler)

    for ax in axioms:
        if isinstance(ax, GCI):
            walk(ax.sub)
            walk(ax.sup)
        elif isinstance(ax, ConceptAssertion):
            walk(ax.concept)
            inds.add(ax.individual)
        elif isinstance(ax, RoleAssertion):
            roles.add(ax.role)
            inds.update((ax.subject, ax.object))
    return names, roles, inds


# ---------------------------------------------------------------------------
# Reading OWL
# ---------------------------------------------------------------------------

def concept_from_owl(expr) -> Concept:
    """Translate an OWLAPI class expression into the :math:`\\mathcal{ALC}` syntax of
    this module.

    ``owl:Thing`` and ``owl:Nothing`` become :data:`TOP_CONCEPT` and
    :data:`BOTTOM_CONCEPT`.

    :raises UnsupportedAxiom: For constructors outside :math:`\\mathcal{ALC}`.
    """
    from mowl.owlapi import ClassExpressionType

    t = expr.getClassExpressionType()
    if t == ClassExpressionType.OWL_CLASS:
        iri = str(expr.asOWLClass().toStringID())
        if iri == OWL_THING:
            return TOP_CONCEPT
        if iri == OWL_NOTHING:
            return BOTTOM_CONCEPT
        return Atomic(iri)
    if t == ClassExpressionType.OBJECT_COMPLEMENT_OF:
        return Not(concept_from_owl(expr.getOperand()))
    if t == ClassExpressionType.OBJECT_INTERSECTION_OF:
        return And(tuple(concept_from_owl(e) for e in expr.getOperandsAsList()))
    if t == ClassExpressionType.OBJECT_UNION_OF:
        return Or(tuple(concept_from_owl(e) for e in expr.getOperandsAsList()))
    if t in (ClassExpressionType.OBJECT_SOME_VALUES_FROM,
             ClassExpressionType.OBJECT_ALL_VALUES_FROM):
        prop = expr.getProperty()
        if prop.isAnonymous():
            raise UnsupportedAxiom(f"Inverse roles are not in ALC: {expr}")
        role = str(prop.asOWLObjectProperty().toStringID())
        filler = concept_from_owl(expr.getFiller())
        if t == ClassExpressionType.OBJECT_SOME_VALUES_FROM:
            return Exists(role, filler)
        return Forall(role, filler)
    raise UnsupportedAxiom(f"Class expression outside ALC: {expr}")


def axioms_from_owl(ontology, strict: bool = False
                    ) -> Tuple[List[ALCAxiom], List[str]]:
    """Read the :math:`\\mathcal{ALC}` axioms of an OWLAPI ontology.

    SubClassOf, EquivalentClasses (as pairwise GCIs in both directions),
    DisjointClasses (as pairwise :math:`C \\sqcap D \\sqsubseteq \\bot`), ClassAssertion
    and ObjectPropertyAssertion are translated. Declarations and annotations are ignored.

    :param ontology: An ``OWLOntology``.
    :param strict: If ``True``, raise on any other logical axiom; otherwise collect it.
    :returns: ``(axioms, skipped)``, where ``skipped`` holds the string form of the
        logical axioms that were not translated.
    """
    from mowl.owlapi import (Imports, OWLClassAssertionAxiom, OWLDisjointClassesAxiom,
                             OWLEquivalentClassesAxiom, OWLObjectPropertyAssertionAxiom,
                             OWLSubClassOfAxiom)

    axioms: List[ALCAxiom] = []
    skipped: List[str] = []
    for ax in ontology.getAxioms(Imports.INCLUDED):
        try:
            if isinstance(ax, OWLSubClassOfAxiom):
                axioms.append(GCI(concept_from_owl(ax.getSubClass()),
                                  concept_from_owl(ax.getSuperClass())))
            elif isinstance(ax, OWLEquivalentClassesAxiom):
                ces = [concept_from_owl(c) for c in ax.getClassExpressionsAsList()]
                for c, d in itertools.combinations(ces, 2):
                    axioms.append(GCI(c, d))
                    axioms.append(GCI(d, c))
            elif isinstance(ax, OWLDisjointClassesAxiom):
                ces = [concept_from_owl(c) for c in ax.getClassExpressionsAsList()]
                for c, d in itertools.combinations(ces, 2):
                    axioms.append(GCI(And((c, d)), BOTTOM_CONCEPT))
            elif isinstance(ax, OWLClassAssertionAxiom):
                ind = ax.getIndividual()
                if ind.isAnonymous():
                    raise UnsupportedAxiom(f"Anonymous individual: {ax}")
                axioms.append(ConceptAssertion(concept_from_owl(ax.getClassExpression()),
                                               str(ind.toStringID())))
            elif isinstance(ax, OWLObjectPropertyAssertionAxiom):
                prop = ax.getProperty()
                if prop.isAnonymous():
                    raise UnsupportedAxiom(f"Inverse role: {ax}")
                axioms.append(RoleAssertion(str(prop.asOWLObjectProperty().toStringID()),
                                            str(ax.getSubject().toStringID()),
                                            str(ax.getObject().toStringID())))
            elif ax.isLogicalAxiom():
                raise UnsupportedAxiom(f"Axiom outside the ALC reader: {ax}")
        except UnsupportedAxiom:
            if strict:
                raise
            skipped.append(str(ax))
    return axioms, skipped


# ---------------------------------------------------------------------------
# Al-cones
# ---------------------------------------------------------------------------

class ALCone:
    """An axis-aligned cone :math:`X_1 \\times \\dots \\times X_n` in :math:`\\mathbb{R}^n`.

    Stored as a boolean vector of length :math:`2n`: entry ``2 i`` says whether
    :math:`X_i \\supseteq \\mathbb{R}^+`, entry ``2 i + 1`` whether
    :math:`X_i \\supseteq \\mathbb{R}^-`.

    :param halfaxes: Boolean array of shape ``(2 n,)``.
    """

    __slots__ = ("_h",)

    def __init__(self, halfaxes):
        h = np.asarray(halfaxes, dtype=bool).reshape(-1)
        if h.size % 2 != 0 or h.size == 0:
            raise ValueError("An al-cone needs a boolean vector of even, positive length.")
        h.setflags(write=False)
        self._h = h

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_factors(cls, factors: Sequence[str]) -> "ALCone":
        """Build from per-coordinate factors ``'0'``, ``'+'``, ``'-'``, ``'R'``
        (``'u'`` is accepted for :math:`\\mathbb{R}`, as in the paper)."""
        bits = []
        for f in factors:
            if f not in _FACTOR_BITS:
                raise ValueError(f"Unknown al-cone factor {f!r}; use one of 0, +, -, R.")
            bits.extend(_FACTOR_BITS[f])
        return cls(bits)

    @classmethod
    def from_halfaxes(cls, dim: int, halfaxes: Iterable[int]) -> "ALCone":
        """Build the conic hull of the given half-axes (indices in ``range(2 * dim)``)."""
        h = np.zeros(2 * dim, dtype=bool)
        for i in halfaxes:
            h[int(i)] = True
        return cls(h)

    @classmethod
    def top(cls, dim: int) -> "ALCone":
        """The whole space :math:`\\mathbb{R}^n` (interpretation of :math:`\\top`)."""
        return cls(np.ones(2 * dim, dtype=bool))

    @classmethod
    def bottom(cls, dim: int) -> "ALCone":
        """The cone :math:`\\{0\\}` (interpretation of :math:`\\bot`)."""
        return cls(np.zeros(2 * dim, dtype=bool))

    @classmethod
    def all_cones(cls, dim: int):
        """Iterate over all :math:`4^n` al-cones of :math:`\\mathbb{R}^n`."""
        for bits in itertools.product((False, True), repeat=2 * dim):
            yield cls(bits)

    # -- accessors ----------------------------------------------------------
    @property
    def dim(self) -> int:
        """Dimension :math:`n` of the ambient space."""
        return self._h.size // 2

    @property
    def bits(self) -> np.ndarray:
        """Read-only boolean half-axis vector of length :math:`2n`."""
        return self._h

    def halfaxes(self) -> FrozenSet[int]:
        """``halfAxes(X)`` as a set of half-axis indices."""
        return frozenset(int(i) for i in np.flatnonzero(self._h))

    def factors(self) -> Tuple[str, ...]:
        """Per-coordinate factors as ``'0'``, ``'+'``, ``'-'`` or ``'R'``."""
        out = []
        for i in range(self.dim):
            p, m = bool(self._h[2 * i]), bool(self._h[2 * i + 1])
            out.append("R" if p and m else "+" if p else "-" if m else "0")
        return tuple(out)

    def generators(self) -> np.ndarray:
        """Unit vectors of the half-axes of the cone, one per row (shape ``(k, n)``).
        The cone is their conic hull."""
        rows = []
        for h in np.flatnonzero(self._h):
            v = np.zeros(self.dim)
            v[h // 2] = 1.0 if h % 2 == 0 else -1.0
            rows.append(v)
        return np.array(rows).reshape(-1, self.dim)

    # -- operations (JAIR Prop. 1) -----------------------------------------
    def _check(self, other: "ALCone"):
        if not isinstance(other, ALCone):
            raise TypeError("Expected an ALCone.")
        if other.dim != self.dim:
            raise ValueError(f"Dimension mismatch: {self.dim} vs {other.dim}.")

    def intersection(self, other: "ALCone") -> "ALCone":
        """:math:`X \\cap Y` (conjunction), componentwise in :math:`O_4` (Prop. 1.1)."""
        self._check(other)
        return ALCone(self._h & other._h)

    def polar(self) -> "ALCone":
        """Polar cone :math:`X^\\circ` (negation), componentwise orthocomplement in
        :math:`O_4` (Prop. 1.2)."""
        return ALCone(~self._h)

    def join(self, other: "ALCone") -> "ALCone":
        """Convex union :math:`(X^\\circ \\cap Y^\\circ)^\\circ = \\mathrm{conHull}(X \\cup
        Y)` (disjunction), componentwise union in :math:`O_4` (Prop. 1.3, Prop. 2)."""
        self._check(other)
        return ALCone(self._h | other._h)

    def subset_of(self, other: "ALCone") -> bool:
        """:math:`X \\subseteq Y`."""
        self._check(other)
        return bool(np.all(~self._h | other._h))

    def is_bottom(self) -> bool:
        """``True`` iff :math:`X = \\{0\\}`, the interpretation of :math:`\\bot`."""
        return not bool(self._h.any())

    def is_top(self) -> bool:
        """``True`` iff :math:`X = \\mathbb{R}^n`."""
        return bool(self._h.all())

    def contains_point(self, x, tol: float = 1e-12) -> bool:
        """``True`` iff the point ``x`` lies in the cone. Coordinates with
        :math:`|x_i| \\le` ``tol`` count as zero."""
        sup = point_support(x, tol)
        if len(np.asarray(x).reshape(-1)) != self.dim:
            raise ValueError("Point has the wrong dimension.")
        return all(self._h[h] for h in sup)

    __and__ = intersection
    __or__ = join
    __invert__ = polar
    __le__ = subset_of

    def __eq__(self, other):
        return isinstance(other, ALCone) and other.dim == self.dim and \
            bool(np.array_equal(self._h, other._h))

    def __hash__(self):
        return hash(self._h.tobytes())

    def __repr__(self):
        return "ALCone(" + " × ".join(self.factors()) + ")"


def point_support(x, tol: float = 1e-12) -> FrozenSet[int]:
    """Half-axes in the support of a point: ``2 i`` if :math:`x_i > tol`, ``2 i + 1``
    if :math:`x_i < -tol`."""
    x = np.asarray(x, dtype=float).reshape(-1)
    out = set()
    for i, v in enumerate(x):
        if v > tol:
            out.add(2 * i)
        elif v < -tol:
            out.add(2 * i + 1)
    return frozenset(out)


def halfaxis_vector(dim: int, h: int, scale: float = 1.0) -> np.ndarray:
    """The point ``scale`` times the unit vector of half-axis ``h`` in
    :math:`\\mathbb{R}^{dim}`."""
    v = np.zeros(dim)
    v[h // 2] = scale if h % 2 == 0 else -scale
    return v


def exists_cone(incidence: np.ndarray, filler: ALCone) -> ALCone:
    """:math:`(\\exists R.C)^I = \\mathrm{conHull}(\\{R^T y \\mid y \\in C^I\\})`
    (JAIR Def. 7) for a 0/1 incidence matrix ``incidence`` :math:`= R` of shape
    ``(2n, 2n)``: half-axis ``i`` belongs to the result iff ``R[j, i]`` is set for some
    half-axis ``j`` of the filler."""
    R = np.asarray(incidence, dtype=bool)
    return ALCone((R & filler.bits[:, None]).any(axis=0))


# ---------------------------------------------------------------------------
# Interpretations and the checker
# ---------------------------------------------------------------------------

class ALConeInterpretation:
    """An al-cone interpretation (JAIR Defs. 2 and 7).

    :param dim: Dimension :math:`n` of the space :math:`\\mathbb{R}^n`.
    :param concepts: Map from concept names to :class:`ALCone` objects.
    :param roles: Map from role names to incidence matrices :math:`R` (0/1 arrays of
        shape ``(2n, 2n)``). ``R[j, i] = 1`` means that half-axis ``i`` has half-axis
        ``j`` as :math:`R`-successor, so :math:`R\\, a'` is the successor incidence
        vector of :math:`a'` and :math:`R^T y'` the predecessor vector of :math:`y'`
        (JAIR Def. 7 and Example 7).
    :param individuals: Map from individual names to points of :math:`\\mathbb{R}^n`.
        By JAIR Def. 2 they must differ from :math:`\\vec{0}`.
    :param halfaxis_elements: Optional map ``half-axis -> domain element`` recorded by the
        constructions (for inspection only).
    """

    def __init__(self, dim: int, concepts: Mapping[str, ALCone],
                 roles: Optional[Mapping[str, np.ndarray]] = None,
                 individuals: Optional[Mapping[str, np.ndarray]] = None,
                 halfaxis_elements: Optional[Sequence[int]] = None):
        if dim < 1:
            raise ValueError("dim must be at least 1.")
        self.dim = int(dim)
        self.concepts: Dict[str, ALCone] = {}
        for name, cone in concepts.items():
            if not isinstance(cone, ALCone):
                cone = ALCone(cone)
            if cone.dim != self.dim:
                raise ValueError(f"Concept {name!r} has dimension {cone.dim}, not {dim}.")
            self.concepts[name] = cone
        self.roles: Dict[str, np.ndarray] = {}
        for name, mat in (roles or {}).items():
            m = np.asarray(mat, dtype=bool)
            if m.shape != (2 * self.dim, 2 * self.dim):
                raise ValueError(f"Role {name!r} needs a ({2 * dim}, {2 * dim}) matrix.")
            self.roles[name] = m
        self.individuals: Dict[str, np.ndarray] = {}
        for name, pt in (individuals or {}).items():
            p = np.asarray(pt, dtype=float).reshape(-1)
            if p.size != self.dim:
                raise ValueError(f"Individual {name!r} is not a point of R^{dim}.")
            self.individuals[name] = p
        self.halfaxis_elements = None if halfaxis_elements is None \
            else list(halfaxis_elements)

    # -- constructions ------------------------------------------------------
    @classmethod
    def from_finite_model(cls, model: "FiniteInterpretation",
                          axioms: Optional[Iterable[ALCAxiom]] = None
                          ) -> "ALConeInterpretation":
        """Build an al-cone model from a finite classical interpretation (our
        construction; see the module docstring).

        Domain element ``k`` is placed on half-axis ``k``; if the domain size is odd, the
        remaining half-axis carries a copy of element ``0`` (same concept memberships,
        same successors, same predecessors), which preserves the truth of every
        :math:`\\mathcal{ALC}` concept. Concept names become the conic hulls of the
        half-axes of their extensions, roles the incidence matrices of their extensions,
        and individual ``a`` the unit vector on the half-axis of :math:`a^I`.

        This is the combination of the source ``finite model`` and the reading
        ``half-axes`` (:mod:`mowl.representation.construction`); the interpretation's
        :attr:`module` is its geometry.

        :param model: A :class:`FiniteInterpretation`.
        :param axioms: If given, the classical model is first checked to satisfy them
            (:class:`ValueError` otherwise), and the result is verified with
            :class:`ALConeChecker` (:class:`InternalConsistencyError` otherwise).
        """
        from mowl.representation.construction import RECIPES
        axioms = None if axioms is None else list(axioms)
        if axioms is not None:
            bad = model.violations(axioms)
            if bad:
                listed = "; ".join(map(str, bad[:5]))
                raise ValueError(
                    f"The finite interpretation is not a model of the axioms: {listed}")
        if model.domain_size < 1:
            raise ValueError("The domain must not be empty.")
        geometry = RECIPES["alcone-exact"].build(
            model, source="finite model", names=list(model.concepts),
            roles=list(model.roles), individuals=list(model.individuals))
        interp = cls._from_geometry(geometry)
        if axioms is not None:
            bad = ALConeChecker(interp).violations(axioms)
            if bad:
                raise InternalConsistencyError(
                    "al-cone construction violates: " + "; ".join(map(str, bad[:5])))
        return interp

    @classmethod
    def _from_geometry(cls, geometry) -> "ALConeInterpretation":
        """The interpretation of the al-cones read by the ``half-axes`` reading."""
        regions = geometry.read.classes
        cell = regions.halfaxis_cell
        concepts = {name: ALCone(regions.bits[row, 0].numpy())
                    for row, name in enumerate(geometry.names)}
        roles = {}
        if geometry.relation_model is not None:
            for r, name in enumerate(geometry.roles):
                pairs = geometry.relation_model.pairs[r]
                # R[j, i]: half-axis i has half-axis j as successor (JAIR Def. 7)
                roles[name] = pairs[cell][:, cell].T.numpy()
        inds = {}
        if geometry.read.individuals is not None:
            inds = {a: geometry.read.individuals.centre[k].numpy()
                    for k, a in enumerate(geometry.individual_names)}
        interp = cls(regions.dim, concepts, roles, inds, halfaxis_elements=cell.tolist())
        interp._geometry = geometry
        return interp

    @classmethod
    def from_canonical_model(cls, cm: "CanonicalModel",
                             axioms: Optional[Iterable[ALCAxiom]] = None
                             ) -> "ALConeInterpretation":
        """Build an al-cone model from the finite canonical model of an
        :math:`\\mathcal{EL}^{++}` ontology (see
        :meth:`FiniteInterpretation.from_canonical_model`)."""
        return cls.from_finite_model(FiniteInterpretation.from_canonical_model(cm), axioms)

    @classmethod
    def from_ontology(cls, ontology) -> "ALConeInterpretation":
        """Build and verify an al-cone model of an :math:`\\mathcal{EL}^{++}` OWLAPI
        ontology through mOWL's :class:`~mowl.geometric.completion.CanonicalModel`.

        :raises InconsistentOntology: If the ontology is inconsistent.
        """
        from mowl.geometric.completion import CanonicalModel

        axioms, _ = axioms_from_owl(ontology)
        return cls.from_canonical_model(CanonicalModel.from_ontology(ontology), axioms)

    @classmethod
    def from_boolean_ontology(cls, axioms: Iterable[ALCAxiom], faithful: bool = False,
                              names: Optional[Iterable[str]] = None,
                              max_names: int = 16) -> "ALConeInterpretation":
        """Al-cone model of a Boolean :math:`\\mathcal{ALC}` ontology (no roles) by the
        construction of JAIR Prop. 7 (``faithful=False``) or Prop. 8 (``faithful=True``).

        The algebraic atoms of the Lindenbaum-Tarski algebra of the TBox are the
        complete conjunctions of literals over the concept names that satisfy every GCI;
        they are enumerated exhaustively and placed on half-axes (one atom is doubled if
        their number is odd). A concept name is the conic hull of the half-axes of the
        atoms below it. The result is TBox-faithful: :math:`C \\sqsubseteq D` holds iff
        it is entailed by the TBox.

        Individuals: with ``faithful=False`` each individual is put on the half-axis of
        one atom consistent with its assertions (Prop. 7, Example 1). With
        ``faithful=True`` every coordinate is doubled (Prop. 8) and the individual is put
        on the point whose support is the set of half-axes of *all* atoms consistent
        with its assertions; then :math:`a^I \\in C^I` iff :math:`\\mathcal{O} \\models
        C(a)` for every Boolean concept :math:`C` (strong concept-faithfulness).

        This is the combination of the source ``boolean atoms`` and the reading
        ``half-axes`` (doubled for ``faithful=True``).

        :param axioms: GCIs and concept assertions without roles.
        :param faithful: Use the doubled, strongly concept-faithful model of Prop. 8.
        :param names: Extra concept names to include in the signature.
        :param max_names: Refuse signatures larger than this (the atoms are enumerated).
        :raises UnsupportedAxiom: If an axiom contains a role.
        :raises InconsistentOntology: If the TBox or an individual's assertions are
            unsatisfiable.
        """
        from mowl.representation.construction import RECIPES
        axioms = list(axioms)
        geometry = RECIPES["alcone-exact"].build(
            axioms, source="boolean atoms",
            source_options={"faithful": faithful, "names": tuple(names or ()),
                            "max_names": max_names},
            reading_options={"double": bool(faithful)})
        interp = cls._from_geometry(geometry)
        bad = ALConeChecker(interp).violations(axioms)
        if bad:
            raise InternalConsistencyError(
                "Boolean al-cone construction violates: " + "; ".join(map(str, bad[:5])))
        return interp

    @property
    def module(self):
        """The geometry of this interpretation
        (:class:`~mowl.representation.construction.ConstructionGeometry`): the combination
        it was built from, or, for an interpretation given by its cones, the ``half-axes``
        reading of its own half-axes (one cell per half-axis)."""
        geometry = getattr(self, "_geometry", None)
        if geometry is None:
            from mowl.representation.library.construct_methods import cone_geometry
            geometry = self._geometry = cone_geometry(self)
        return geometry

    # -- convenience --------------------------------------------------------
    def checker(self) -> "ALConeChecker":
        """Return an :class:`ALConeChecker` for this interpretation."""
        return ALConeChecker(self)

    def cone(self, concept: Concept) -> ALCone:
        """Evaluate ``concept`` (see :meth:`ALConeChecker.cone`)."""
        return ALConeChecker(self).cone(concept)

    def __repr__(self):
        return (f"ALConeInterpretation(dim={self.dim}, concepts={len(self.concepts)}, "
                f"roles={len(self.roles)}, individuals={len(self.individuals)})")


def _double(cone: ALCone) -> ALCone:
    """Double every coordinate: factor :math:`X_i` becomes :math:`X_i \\times X_i`
    (JAIR Prop. 8)."""
    h = cone.bits.reshape(-1, 2)
    return ALCone(np.repeat(h, 2, axis=0).reshape(-1))


def _doubled_point(dim: int, support: Iterable[int]) -> np.ndarray:
    """Point of :math:`\\mathbb{R}^{2 dim}` whose membership in doubled al-cones is that
    of the half-axis set ``support`` of :math:`\\mathbb{R}^{dim}`: coordinate pair
    :math:`(2i, 2i+1)` is :math:`(1, 1)` for :math:`+`, :math:`(-1, -1)` for :math:`-`,
    :math:`(1, -1)` for both and :math:`(0, 0)` for neither."""
    s = set(support)
    x = np.zeros(2 * dim)
    for i in range(dim):
        p, m = (2 * i) in s, (2 * i + 1) in s
        if p and m:
            x[2 * i], x[2 * i + 1] = 1.0, -1.0
        elif p:
            x[2 * i] = x[2 * i + 1] = 1.0
        elif m:
            x[2 * i] = x[2 * i + 1] = -1.0
    return x


def _eval_boolean(c: Concept, val: Mapping[str, bool]) -> bool:
    if c is TOP_CONCEPT or isinstance(c, _Top):
        return True
    if isinstance(c, _Bottom):
        return False
    if isinstance(c, Atomic):
        return bool(val[c.name])
    if isinstance(c, Not):
        return not _eval_boolean(c.arg, val)
    if isinstance(c, And):
        return all(_eval_boolean(a, val) for a in c.args)
    if isinstance(c, Or):
        return any(_eval_boolean(a, val) for a in c.args)
    raise UnsupportedAxiom(f"Not a Boolean concept: {c}")


class ALConeChecker:
    """Decides satisfaction of :math:`\\mathcal{ALC}` axioms in an al-cone
    interpretation.

    * :math:`C \\sqsubseteq D` holds iff :math:`C^I \\subseteq D^I`.
    * :math:`C(a)` holds iff :math:`a^I \\neq \\vec 0` and :math:`a^I \\in C^I`.
    * :math:`R(a, b)` holds iff :math:`a^I, b^I \\neq \\vec 0` and :math:`b^I \\in
      \\mathrm{conHull}(R\\, a'^I)`, i.e. every half-axis in the support of :math:`b^I` is
      an :math:`R`-successor of some half-axis in the support of :math:`a^I` (JAIR
      Sec. 6, role-satisfiability).

    Concept names, roles or individuals missing from the interpretation raise
    :class:`KeyError`.

    :param interpretation: An :class:`ALConeInterpretation`.
    :param tol: Coordinates with absolute value at most ``tol`` count as zero.
    """

    def __init__(self, interpretation: ALConeInterpretation, tol: float = 1e-12):
        self.interpretation = interpretation
        self.tol = tol
        self._cache: Dict[Concept, ALCone] = {}

    def cone(self, concept: Concept) -> ALCone:
        """The al-cone :math:`C^I` of an :math:`\\mathcal{ALC}` concept."""
        if concept in self._cache:
            return self._cache[concept]
        interp = self.interpretation
        n = interp.dim
        if isinstance(concept, _Top):
            out = ALCone.top(n)
        elif isinstance(concept, _Bottom):
            out = ALCone.bottom(n)
        elif isinstance(concept, Atomic):
            if concept.name not in interp.concepts:
                raise KeyError(f"Concept name {concept.name!r} is not interpreted.")
            out = interp.concepts[concept.name]
        elif isinstance(concept, Not):
            out = self.cone(concept.arg).polar()
        elif isinstance(concept, And):
            out = ALCone.top(n)
            for a in concept.args:
                out = out.intersection(self.cone(a))
        elif isinstance(concept, Or):
            out = ALCone.bottom(n)
            for a in concept.args:
                out = out.join(self.cone(a))
        elif isinstance(concept, Exists):
            out = exists_cone(self._role(concept.role), self.cone(concept.filler))
        elif isinstance(concept, Forall):
            out = exists_cone(self._role(concept.role),
                              self.cone(concept.filler).polar()).polar()
        else:
            raise TypeError(f"Not an ALC concept: {concept!r}")
        self._cache[concept] = out
        return out

    def _role(self, role: str) -> np.ndarray:
        if role not in self.interpretation.roles:
            raise KeyError(f"Role {role!r} is not interpreted.")
        return self.interpretation.roles[role]

    def _point(self, a: str) -> np.ndarray:
        if a not in self.interpretation.individuals:
            raise KeyError(f"Individual {a!r} is not interpreted.")
        return self.interpretation.individuals[a]

    def instance_of(self, individual: str, concept: Concept) -> bool:
        """``True`` iff :math:`a^I \\in C^I`."""
        return self.cone(concept).contains_point(self._point(individual), self.tol)

    def successors(self, individual: str, role: str) -> ALCone:
        """The al-cone :math:`\\mathrm{conHull}(R\\, a'^I)` of the :math:`R`-successor
        half-axes of the support of :math:`a^I`."""
        R = self._role(role)
        sup = point_support(self._point(individual), self.tol)
        bits = np.zeros(2 * self.interpretation.dim, dtype=bool)
        for i in sup:
            bits |= R[:, i]
        return ALCone(bits)

    def satisfies(self, axiom: ALCAxiom) -> bool:
        """``True`` iff the interpretation satisfies ``axiom``."""
        if isinstance(axiom, GCI):
            return self.cone(axiom.sub).subset_of(self.cone(axiom.sup))
        if isinstance(axiom, ConceptAssertion):
            p = self._point(axiom.individual)
            return bool(point_support(p, self.tol)) and \
                self.cone(axiom.concept).contains_point(p, self.tol)
        if isinstance(axiom, RoleAssertion):
            pa, pb = self._point(axiom.subject), self._point(axiom.object)
            if not point_support(pa, self.tol) or not point_support(pb, self.tol):
                return False
            return self.successors(axiom.subject, axiom.role).contains_point(pb, self.tol)
        raise TypeError(f"Not an ALC axiom: {axiom!r}")

    def violations(self, axioms: Iterable[ALCAxiom]) -> List[ALCAxiom]:
        """Return the axioms that are not satisfied."""
        return [ax for ax in axioms if not self.satisfies(ax)]

    def is_model(self, axioms: Iterable[ALCAxiom]) -> bool:
        """``True`` iff every axiom is satisfied."""
        return not self.violations(axioms)


# ---------------------------------------------------------------------------
# Finite classical interpretations
# ---------------------------------------------------------------------------

class FiniteInterpretation:
    """A finite classical interpretation with domain ``{0, ..., domain_size - 1}``.

    It has its own set-based evaluator (Table 1 of JAIR), independent of the cone code,
    so it can be used to verify the al-cone constructions.

    :param domain_size: Number of domain elements.
    :param concepts: Map from concept names to sets of elements.
    :param roles: Map from role names to sets of pairs of elements.
    :param individuals: Map from individual names to elements.
    """

    def __init__(self, domain_size: int, concepts: Mapping[str, Iterable[int]],
                 roles: Optional[Mapping[str, Iterable[Tuple[int, int]]]] = None,
                 individuals: Optional[Mapping[str, int]] = None):
        self.domain_size = int(domain_size)
        dom = range(self.domain_size)
        self.concepts = {k: frozenset(int(e) for e in v) for k, v in concepts.items()}
        self.roles = {k: frozenset((int(a), int(b)) for a, b in v)
                      for k, v in (roles or {}).items()}
        self.individuals = {k: int(v) for k, v in (individuals or {}).items()}
        for k, v in self.concepts.items():
            if not v <= set(dom):
                raise ValueError(f"Extension of {k!r} leaves the domain.")
        for k, v in self.roles.items():
            if any(a not in dom or b not in dom for a, b in v):
                raise ValueError(f"Extension of role {k!r} leaves the domain.")
        for k, v in self.individuals.items():
            if v not in dom:
                raise ValueError(f"Individual {k!r} is not a domain element.")

    @classmethod
    def from_canonical_model(cls, cm: "CanonicalModel") -> "FiniteInterpretation":
        """Read a mOWL :class:`~mowl.geometric.completion.CanonicalModel` (a finite model
        of an :math:`\\mathcal{EL}^{++}` ontology).

        All concept names of the canonical model are copied, including auxiliary names
        introduced by normalisation; ``owl:Thing`` and ``owl:Nothing`` are dropped
        (they are interpreted by :data:`TOP_CONCEPT` and :data:`BOTTOM_CONCEPT`).
        """
        concepts = {}
        for name in cm.concept_names:
            if name in (OWL_THING, OWL_NOTHING):
                continue
            concepts[name] = set(cm.extension(name))
        roles = {r: set(cm.role(r)) for r in cm.role_names}
        inds = {iri: cm.element_of(iri) for iri in cm.individuals}
        return cls(len(cm.domain), concepts, roles, inds)

    def extension(self, concept: Concept) -> FrozenSet[int]:
        """Classical extension :math:`C^I`."""
        dom = frozenset(range(self.domain_size))
        if isinstance(concept, _Top):
            return dom
        if isinstance(concept, _Bottom):
            return frozenset()
        if isinstance(concept, Atomic):
            if concept.name not in self.concepts:
                raise KeyError(f"Concept name {concept.name!r} is not interpreted.")
            return self.concepts[concept.name]
        if isinstance(concept, Not):
            return dom - self.extension(concept.arg)
        if isinstance(concept, And):
            out = dom
            for a in concept.args:
                out = out & self.extension(a)
            return out
        if isinstance(concept, Or):
            out = frozenset()
            for a in concept.args:
                out = out | self.extension(a)
            return out
        if isinstance(concept, (Exists, Forall)):
            if concept.role not in self.roles:
                raise KeyError(f"Role {concept.role!r} is not interpreted.")
            pairs = self.roles[concept.role]
            ext = self.extension(concept.filler)
            if isinstance(concept, Exists):
                return frozenset(x for x, y in pairs if y in ext)
            return frozenset(x for x in dom
                             if all(y in ext for (x2, y) in pairs if x2 == x))
        raise TypeError(f"Not an ALC concept: {concept!r}")

    def satisfies(self, axiom: ALCAxiom) -> bool:
        """Classical satisfaction (JAIR Sec. 3)."""
        if isinstance(axiom, GCI):
            return self.extension(axiom.sub) <= self.extension(axiom.sup)
        if isinstance(axiom, ConceptAssertion):
            return self.individuals[axiom.individual] in self.extension(axiom.concept)
        if isinstance(axiom, RoleAssertion):
            return (self.individuals[axiom.subject], self.individuals[axiom.object]) in \
                self.roles.get(axiom.role, frozenset())
        raise TypeError(f"Not an ALC axiom: {axiom!r}")

    def violations(self, axioms: Iterable[ALCAxiom]) -> List[ALCAxiom]:
        """Return the axioms not satisfied classically."""
        return [ax for ax in axioms if not self.satisfies(ax)]


__all__ = [
    "ALCone", "ALConeChecker", "ALConeInterpretation", "And", "Atomic",
    "BOTTOM_CONCEPT", "Concept", "ConceptAssertion", "Exists", "FACTORS",
    "FiniteInterpretation", "Forall", "GCI", "Not", "Or", "RoleAssertion", "TOP_CONCEPT",
    "axioms_from_owl", "concept_from_owl", "concept_names", "exists_cone", "has_roles",
    "halfaxis_vector", "point_support", "signature",
]
