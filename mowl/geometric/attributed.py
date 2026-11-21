"""Convex geometric models of ground (temporally) attributed Horn description logic ontologies.

Implements the construction behind Theorem 6 (and, for the temporal attributes, Theorem 12) of
Bourgaux, Ozaki & Pan, *Geometric Models for (Temporally) Attributed Description Logics*
(DL 2021, arXiv:2108.12239). The paper is theoretical: it defines geometric interpretations
of attributed DLs and shows by *grounding* that every satisfiable ontology of
:math:`DL\\text{-}Lite^{\\mathcal{H},@}_{horn}` without negative role inclusions has a convex
geometric model. There is no learning procedure.

Attributed ontologies
---------------------
Axioms carry *specifiers* (:class:`Specifier`): sets of attribute-value pairs such as
``[source: s, time: 3]``. A *closed* specifier describes exactly this annotation set, an
*open* specifier ``⌊source: s⌋`` any annotation set that contains these pairs
(``S ⇒ T`` iff ``T`` is open and the pairs of ``S`` include those of ``T``). The empty open
specifier is "any annotation", so an axiom without annotations is an ordinary axiom.
How the annotations are supplied is separate from the geometry: they are plain Python data
(:class:`AttributedOntology`) or OWL axiom annotations (:func:`from_owl_ontology`).

Construction
------------
1. **Grounding** (:func:`ground`, proof of Theorem 6): every annotated name ``E@S`` becomes a
   fresh name ``E_S``, and for specifiers ``S ⇒ T`` of the same name the axiom ``E_S ⊑ E_T``
   is added. The result ``dl(O_g)`` is an ordinary Horn ontology (here normalised to the
   :math:`\\mathcal{ELH}_\\bot` normal forms of :mod:`mowl.geometric.normal_forms`).
2. A finite model is the canonical model (:class:`~mowl.geometric.completion.CanonicalModel`),
   and the convex model :math:`\\eta'` of Gutiérrez-Basulto & Schockaert (Proposition 3) is
   :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`.
3. :math:`\\eta(E@S) = \\eta'(E_S)` (and the empty region for specifiers that do not occur).

Temporal attributes
-------------------
``time`` (a time point), ``during`` (an interval ``(k, l)``, all points), ``since`` (all points
from ``x``) and ``until`` (all points up to ``x``) have the semantics of Definition 7 of the
paper. The paper states Theorem 12 (a convex model exists under restrictions) without
the grounding in the text available to us. The grounding used here is **ours** and adds to
``dl(O_g)`` the entailments of Definition 7 between the names of one concept and one
non-temporal annotation part (a conjunction on the left, so this is still Horn):

* ``E@[…, during: (k, l)]`` is equivalent to the conjunction of ``E@[…, time: j]`` for
  ``k <= j <= l``; ``until: x`` to the conjunction over ``origin <= j <= x``;
* ``since: x`` implies ``time: j`` for ``j >= x`` and ``since: y`` for ``y >= x``
  (the converse needs infinitely many premises and does not hold on an unbounded time line);
* a specifier with several temporal attributes is the conjunction of the single ones.

Only the time points that occur in the ontology (and all points of occurring intervals)
are considered. As in the paper's conditions of Theorem 12: ``before``, ``after`` and
``between`` (which express disjunction) are rejected, and specifiers on roles must be
atemporal. Temporal attributes are only allowed in closed specifiers.

Not covered: set variables and object variables (the grounding of non-ground axioms of
Section 2 is not implemented: only ground ontologies are accepted), inverse roles, negative
role inclusions (the paper excludes them, Example 11 of Gutiérrez-Basulto & Schockaert) and
role inclusions with conjunction. The ABox of an OWL ontology is read as time-less
annotated assertions; see :func:`from_owl_ontology`.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple, Union

import numpy as np

from mowl.geometric.completion import CanonicalModel
from mowl.geometric.errors import UnsupportedAxiom
from mowl.geometric.normal_forms import (
    BOTTOM,
    NF1,
    NF2,
    NF3,
    NF4,
    NF6,
    TOP,
    Axiom,
    NormalisedOntology,
    nominal_name,
)
from mowl.geometric.polyhedral import ConvexPolyhedralModel

__all__ = [
    "TIME", "DURING", "SINCE", "UNTIL", "BEFORE", "AFTER", "BETWEEN",
    "Specifier", "ANY", "Basic", "ConceptFact", "RoleFact", "ConceptInc", "RoleInc",
    "AttributedOntology", "Grounding", "ground", "AttributedConvexModel",
    "from_owl_ontology", "ATTRIBUTE_NAMESPACE",
]

#: Temporal attribute names (Section 4 of the paper).
TIME, DURING, SINCE, UNTIL, BEFORE, AFTER, BETWEEN = (
    "time", "during", "since", "until", "before", "after", "between")
_SUPPORTED_TEMPORAL = {TIME, DURING, SINCE, UNTIL}
_UNSUPPORTED_TEMPORAL = {BEFORE, AFTER, BETWEEN}

Value = Union[str, int, Tuple[int, int]]


# ---------------------------------------------------------------------------
# Syntax
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Specifier:
    """A ground specifier: attribute-value pairs, closed or open.

    :param pairs: Frozen set of ``(attribute, value)`` pairs. A value is a string (an
        annotation name), an ``int`` (a time point) or a pair ``(k, l)`` (an interval,
        value of ``during``).
    :param open: ``True`` for an open specifier ``⌊...⌋`` ("at least these pairs").
    """

    pairs: FrozenSet[Tuple[str, Value]] = frozenset()
    open: bool = False

    @classmethod
    def closed(cls, **pairs: Value) -> "Specifier":
        """``Specifier.closed(source="s", time=3)`` is ``[source: s, time: 3]``."""
        return cls(frozenset(pairs.items()), False)

    @classmethod
    def at_least(cls, **pairs: Value) -> "Specifier":
        """The open specifier ``⌊...⌋``."""
        return cls(frozenset(pairs.items()), True)

    def implies(self, other: "Specifier") -> bool:
        """The paper's ``S ⇒ T``: ``T`` is open and the pairs of ``S`` include those of ``T``."""
        return other.open and other.pairs <= self.pairs

    def temporal(self) -> List[Tuple[str, Value]]:
        names = _SUPPORTED_TEMPORAL | _UNSUPPORTED_TEMPORAL
        return sorted(((a, v) for a, v in self.pairs if a in names), key=str)

    def abstract(self) -> "Specifier":
        """This specifier without its temporal pairs."""
        return Specifier(frozenset(p for p in self.pairs
                                   if p[0] not in _SUPPORTED_TEMPORAL | _UNSUPPORTED_TEMPORAL),
                         self.open)

    def __str__(self):
        body = ", ".join(f"{a}: {v}" for a, v in sorted(self.pairs, key=str))
        return f"⌊{body}⌋" if self.open else f"[{body}]"


#: The empty open specifier: "any annotation set". ``E@ANY`` is ``E``.
ANY = Specifier(frozenset(), True)


@dataclass(frozen=True)
class Basic:
    """A basic concept ``A@S`` (``exists=False``) or ``∃R@S`` (``exists=True``)."""

    name: str
    spec: Specifier = ANY
    exists: bool = False


@dataclass(frozen=True)
class ConceptFact:
    """Concept assertion ``A(a)@S`` (``S`` closed)."""

    concept: str
    individual: str
    spec: Specifier = ANY


@dataclass(frozen=True)
class RoleFact:
    """Role assertion ``R(a, b)@S`` (``S`` closed)."""

    role: str
    subject: str
    object: str
    spec: Specifier = ANY


@dataclass(frozen=True)
class ConceptInc:
    """Ground concept inclusion ``B1 ⊓ ... ⊓ Bk ⊑ C``; ``rhs=None`` means ``⊥``."""

    lhs: Tuple[Basic, ...]
    rhs: Optional[Basic]

    def __post_init__(self):
        object.__setattr__(self, "lhs", tuple(self.lhs))
        if not self.lhs:
            raise ValueError("A concept inclusion needs at least one concept on the left.")


@dataclass(frozen=True)
class RoleInc:
    """Positive role inclusion ``P@S ⊑ Q@T`` between role names (no inverse, no negation)."""

    sub: str
    sub_spec: Specifier
    sup: str
    sup_spec: Specifier = ANY


AttributedAxiom = Union[ConceptFact, RoleFact, ConceptInc, RoleInc]


@dataclass
class AttributedOntology:
    """A ground attributed Horn ontology.

    :param axioms: Iterable of :class:`ConceptFact`, :class:`RoleFact`, :class:`ConceptInc`,
        :class:`RoleInc`.
    :param time_origin: First point of the (discrete, unbounded above) time line; the
        paper has a bounded past. Default ``0``.
    """

    axioms: List[AttributedAxiom]
    time_origin: int = 0

    def __init__(self, axioms: Iterable[AttributedAxiom] = (), time_origin: int = 0):
        self.axioms = list(axioms)
        self.time_origin = time_origin

    def add(self, axiom: AttributedAxiom) -> "AttributedOntology":
        self.axioms.append(axiom)
        return self

    def individuals(self) -> List[str]:
        out = set()
        for ax in self.axioms:
            if isinstance(ax, ConceptFact):
                out.add(ax.individual)
            elif isinstance(ax, RoleFact):
                out.update((ax.subject, ax.object))
        return sorted(out)


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

@dataclass
class Grounding:
    """Result of :func:`ground`: the ordinary ontology ``dl(O_g)`` in normal form.

    :ivar axioms: Normal-form axioms of ``dl(O_g)``.
    :ivar concept_names: All concept names (including nominals, aux names, ⊤, ⊥).
    :ivar role_names: All role names.
    :ivar per_axiom: For every original axiom (by position) the normal-form axioms that
        express it.
    :ivar specifiers: Maps each entity name to the specifiers that occur with it.
    """

    axioms: List[Axiom]
    concept_names: set
    role_names: set
    per_axiom: List[List[Axiom]]
    specifiers: Dict[Tuple[str, bool], set]

    def concept_name(self, name: str, spec: Specifier = ANY) -> str:
        """The fresh name ``A_S`` of ``A@S`` (plain ``A`` for the empty open specifier)."""
        return _name(name, spec)

    def role_name(self, name: str, spec: Specifier = ANY) -> str:
        return _name(name, spec)


def _name(entity: str, spec: Specifier) -> str:
    return entity if spec == ANY else f"{entity}@{spec}"


def _validate(spec: Specifier, is_role: bool, origin: int, where: str) -> None:
    temporal = spec.temporal()
    for attr, value in temporal:
        if attr in _UNSUPPORTED_TEMPORAL:
            raise UnsupportedAxiom(
                f"Temporal attribute {attr!r} in {where} expresses a disjunction; the "
                "construction (Theorem 12 (ii)) needs ontologies without before, after "
                "and between.")
        if spec.open:
            raise UnsupportedAxiom(
                f"Temporal attributes are only supported in closed specifiers ({where}).")
        if is_role:
            raise UnsupportedAxiom(
                f"Specifiers on roles must be atemporal (Theorem 12 (i)); got {spec} in {where}.")
        if attr == DURING:
            if not (isinstance(value, tuple) and len(value) == 2 and value[0] <= value[1]):
                raise ValueError(f"during needs an interval (k, l) with k <= l, got {value!r}.")
            lo = value[0]
        else:
            if not isinstance(value, int):
                raise ValueError(f"{attr} needs an integer time point, got {value!r}.")
            lo = value
        if lo < origin:
            raise ValueError(f"Time point {lo} lies before the time origin {origin}.")


def ground(ontology: AttributedOntology) -> Grounding:
    """Ground an attributed ontology into an ordinary one (the proof of Theorem 6, plus the
    temporal entailments of Definition 7, see the module docstring)."""
    axioms: List[Axiom] = []
    concepts: set = {TOP, BOTTOM}
    roles: set = set()
    specs: Dict[Tuple[str, bool], set] = {}
    counter = itertools.count()

    def fresh(kind: str) -> str:
        n = f"http://mowl.borg/attributed/aux#{kind}{next(counter)}"
        concepts.add(n)
        return n

    def use(entity: str, spec: Specifier, is_role: bool) -> str:
        _validate(spec, is_role, ontology.time_origin, f"{entity}@{spec}")
        specs.setdefault((entity, is_role), set()).add(spec)
        specs[(entity, is_role)].add(ANY)
        n = _name(entity, spec)
        (roles if is_role else concepts).add(n)
        return n

    exists_cache: Dict[Tuple[str, Specifier], str] = {}

    def lhs_name(b: Basic, out: List[Axiom]) -> str:
        if not b.exists:
            return use(b.name, b.spec, False)
        key = (b.name, b.spec)
        if key not in exists_cache:
            r = use(b.name, b.spec, True)
            x = fresh("ex")
            exists_cache[key] = x
            out.append(NF4(r, TOP, x))
        return exists_cache[key]

    per_axiom: List[List[Axiom]] = []
    for ax in ontology.axioms:
        out: List[Axiom] = []
        if isinstance(ax, ConceptFact):
            if ax.spec.open and ax.spec.pairs:
                raise UnsupportedAxiom("Assertions carry closed specifiers.")
            out.append(NF1(nominal_name(ax.individual), use(ax.concept, ax.spec, False)))
            concepts.add(nominal_name(ax.individual))
        elif isinstance(ax, RoleFact):
            if ax.spec.open and ax.spec.pairs:
                raise UnsupportedAxiom("Assertions carry closed specifiers.")
            out.append(NF3(nominal_name(ax.subject), use(ax.role, ax.spec, True),
                           nominal_name(ax.object)))
            concepts.update((nominal_name(ax.subject), nominal_name(ax.object)))
        elif isinstance(ax, RoleInc):
            out.append(NF6(use(ax.sub, ax.sub_spec, True), use(ax.sup, ax.sup_spec, True)))
        elif isinstance(ax, ConceptInc):
            names = [lhs_name(b, out) for b in ax.lhs]
            body = names[0]
            for nxt in names[1:]:
                z = fresh("and")
                out.append(NF2(body, nxt, z))
                body = z
            if ax.rhs is None:
                out.append(NF1(body, BOTTOM))
            elif ax.rhs.exists:
                out.append(NF3(body, use(ax.rhs.name, ax.rhs.spec, True), TOP))
            else:
                out.append(NF1(body, use(ax.rhs.name, ax.rhs.spec, False)))
        else:
            raise TypeError(f"Unsupported axiom {ax!r}.")
        axioms.extend(out)
        per_axiom.append(out)

    implied = _specifier_axioms(specs, ontology.time_origin, concepts, roles, counter)
    axioms.extend(implied)
    return Grounding(axioms, concepts, roles, per_axiom, specs)


def _specifier_axioms(specs, origin, concepts, roles, counter) -> List[Axiom]:
    """``E_S ⊑ E_T`` for ``S ⇒ T`` and the temporal entailments of Definition 7."""
    out: List[Axiom] = []
    for (entity, is_role), sset in specs.items():
        sset = set(sset)
        if not is_role:
            sset = _add_temporal_specifiers(entity, sset, origin, out, concepts, counter)
        for s in sorted(sset, key=str):
            for t in sorted(sset, key=str):
                if s != t and s.implies(t):
                    if is_role:
                        out.append(NF6(_name(entity, s), _name(entity, t)))
                    else:
                        out.append(NF1(_name(entity, s), _name(entity, t)))
    return out


def _add_temporal_specifiers(entity, sset, origin, out, concepts, counter):
    """Axioms of Definition 7 for the temporal specifiers of one concept name.

    Groups the specifiers by their atemporal part ``F``; for each group, the single temporal
    constraints (``time:j``, ``during:(k,l)``, ``since:x``, ``until:x``) get a name each
    and a specifier with several constraints is the conjunction of its single ones.
    """
    new = set(sset)
    groups: Dict[Specifier, set] = {}
    for s in sset:
        if s.temporal():
            groups.setdefault(s.abstract(), set()).add(s)
    for abstract, members in groups.items():
        def single(attr, value):
            return Specifier(abstract.pairs | {(attr, value)}, False)

        constraints = set()
        for s in members:
            constraints.update(s.temporal())
        points = set()
        for attr, v in constraints:
            if attr == DURING:
                points.update(range(v[0], v[1] + 1))
            elif attr == UNTIL:
                points.update(range(origin, v + 1))
            else:
                points.add(v)
        for attr, v in list(constraints):
            if attr == TIME:
                continue
            s = single(attr, v)
            n = _name(entity, s)
            concepts.add(n)
            if attr in (DURING, UNTIL):
                lo, hi = (v if attr == DURING else (origin, v))
                parts = []
                for j in range(lo, hi + 1):
                    p = single(TIME, j)
                    concepts.add(_name(entity, p))
                    new.add(p)
                    out.append(NF1(n, _name(entity, p)))
                    parts.append(_name(entity, p))
                body = parts[0]
                for nxt in parts[1:]:
                    z = f"http://mowl.borg/attributed/aux#t{next(counter)}"
                    concepts.add(z)
                    out.append(NF2(body, nxt, z))
                    body = z
                out.append(NF1(body, n))
            elif attr == SINCE:
                for j in sorted(points):
                    if j >= v:
                        p = single(TIME, j)
                        concepts.add(_name(entity, p))
                        new.add(p)
                        out.append(NF1(n, _name(entity, p)))
                for attr2, y in constraints:
                    if attr2 == SINCE and y > v:
                        out.append(NF1(n, _name(entity, single(SINCE, y))))
        for s in members:
            temp = s.temporal()
            if len(temp) > 1:
                n = _name(entity, s)
                parts = [_name(entity, single(a, v)) for a, v in temp]
                for p in parts:
                    concepts.add(p)
                    out.append(NF1(n, p))
                    new.add(Specifier(abstract.pairs | {temp[parts.index(p)]}, False))
                body = parts[0]
                for nxt in parts[1:]:
                    z = f"http://mowl.borg/attributed/aux#m{next(counter)}"
                    concepts.add(z)
                    out.append(NF2(body, nxt, z))
                    body = z
                out.append(NF1(body, n))
    return new


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class AttributedConvexModel:
    """A convex geometric model of a ground attributed ontology (Theorem 6 / 12).

    :param ontology: An :class:`AttributedOntology`.
    :raises mowl.geometric.errors.InconsistentOntology: if the ontology is unsatisfiable.
    :raises mowl.geometric.errors.UnsupportedAxiom: for ``before``/``after``/``between``,
        temporal specifiers on roles or in open specifiers.

    After construction: :attr:`grounding` (``dl(O_g)``), :attr:`canonical_model`
    (the finite model), :attr:`model` (the
    :class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel` :math:`\\eta'`) and
    :attr:`dimension`.
    """

    #: The model is an exact construction, not trained.
    axiom_scoring = None

    def __init__(self, ontology: AttributedOntology):
        self.ontology = ontology
        self.grounding = ground(ontology)
        normalised = NormalisedOntology(
            list(self.grounding.axioms), set(self.grounding.concept_names),
            set(self.grounding.role_names), 0)
        self.canonical_model = CanonicalModel.from_normalised(normalised)
        self.model = ConvexPolyhedralModel(self.canonical_model)

    @property
    def dimension(self) -> int:
        return self.model.dimension

    # -- regions ------------------------------------------------------------
    def concept_region(self, concept: str, spec: Specifier = ANY):
        """:math:`\\eta(A@S)`: a face of the simplex (empty for specifiers that do not
        occur in the ontology)."""
        return self.model.class_region(_name(concept, spec))

    def role_region(self, role: str, spec: Specifier = ANY):
        """:math:`\\eta(R@S)`: convex hull of concatenated vertex pairs."""
        return self.model.role_region(_name(role, spec))

    def point(self, individual: str) -> np.ndarray:
        """:math:`\\eta(a)`: the simplex vertex of the nominal of ``a``."""
        cm = self.canonical_model
        return self.model.element_points()[cm.element_of(nominal_name(individual))]

    # -- what holds ---------------------------------------------------------
    def holds_concept(self, individual: str, concept: str, spec: Specifier = ANY) -> bool:
        """``True`` iff :math:`\\eta(a) \\in \\eta(A@S)`, i.e. :math:`\\eta \\models A(a)@S`."""
        region = self.concept_region(concept, spec)
        if region.is_empty:
            return False
        try:
            return bool(region.contains(self.point(individual)))
        except KeyError:
            return False

    def holds_role(self, subject: str, role: str, obj: str, spec: Specifier = ANY) -> bool:
        """``True`` iff :math:`\\eta(a) \\oplus \\eta(b) \\in \\eta(R@S)` (concatenation as the
        linear map ``f``). The region is the hull of vertex pairs, whose vertices are exactly
        the pairs of the canonical model, so membership of a vertex pair is membership of the
        pair in the canonical role."""
        cm = self.canonical_model
        try:
            i = cm.element_of(nominal_name(subject))
            j = cm.element_of(nominal_name(obj))
        except KeyError:
            return False
        return (i, j) in cm.role(_name(role, spec))

    def included(self, sub: str, sub_spec: Specifier, sup: str, sup_spec: Specifier = ANY) -> bool:
        """``True`` iff :math:`\\eta(A@S) \\subseteq \\eta(B@T)` for two concept names."""
        a, b = self.concept_region(sub, sub_spec), self.concept_region(sup, sup_spec)
        return a.is_empty or a.is_subset(b)

    def entails_instance(self, individual: str, concept: str, spec: Specifier = ANY) -> bool:
        """Entailment of ``A(a)@S`` by the grounded ontology (canonical model). Strongly
        faithful (:math:`\\varphi(\\eta) = M`): equals :meth:`holds_concept`."""
        return self.canonical_model.instance_of(individual, _name(concept, spec))

    def is_satisfiable(self, concept: str, spec: Specifier = ANY) -> bool:
        """``True`` iff ``A@S`` is not forced to be empty."""
        return self.canonical_model.is_satisfiable(_name(concept, spec))

    def satisfies(self, index: int) -> bool:
        """Whether the geometry satisfies the original axiom number ``index`` of the
        ontology (checked on its normal-form translation with the exact model checker)."""
        return all(self.model.satisfies(nf) for nf in self.grounding.per_axiom[index])

    def violations(self) -> List[AttributedAxiom]:
        """Original axioms that are not satisfied by the geometry (empty by construction)."""
        return [ax for k, ax in enumerate(self.ontology.axioms) if not self.satisfies(k)]

    def is_model(self) -> bool:
        """``True`` iff every original axiom and every specifier axiom ``E_S ⊑ E_T`` holds
        in the geometry."""
        return not self.violations() and self.model.is_model(self.grounding.axioms)


# ---------------------------------------------------------------------------
# OWL axiom annotations
# ---------------------------------------------------------------------------

#: Annotation properties in this namespace are read as attributes: the local name is the
#: attribute, the literal value its value. ``time``, ``since`` and ``until`` take integers,
#: ``during`` an interval ``"k,l"``.
ATTRIBUTE_NAMESPACE = "http://mowl.borg/attribute#"


def from_owl_ontology(ontology, namespace: str = ATTRIBUTE_NAMESPACE,
                      time_origin: int = 0) -> Tuple[AttributedOntology, list]:
    """Read the attributed part of an OWLAPI ontology from *axiom annotations*.

    Every annotation of an axiom whose property IRI starts with ``namespace`` is an
    attribute-value pair of that axiom (local name = attribute, literal value = value;
    ``time``/``since``/``until`` integers, ``during`` as ``"k,l"``). Axioms without such
    annotations get the empty open specifier, i.e. they are ordinary axioms.

    Supported OWL axioms (named classes, named properties, named individuals):

    * ``ClassAssertion(A a)`` and ``ObjectPropertyAssertion(r a b)``: the annotations are the
      closed specifier of the assertion.
    * ``SubClassOf(B1 ⊓ ... ⊓ Bk, C)`` with ``Bi`` a named class or ``∃r.⊤`` and ``C`` a
      named class, ``∃r.⊤`` or ``⊥``: **adaptation**: the annotations of an inclusion are
      the same *closed* specifier on all concept and role names of the axiom, i.e. the
      axiom ``A@S ⊑ B@S`` ("under this annotation, A implies B"). A different specifier on
      each side needs the Python API.
    * ``SubObjectPropertyOf(r, s)`` with the same convention.

    Other axioms are skipped and returned in the second result (a list of strings).

    :returns: ``(AttributedOntology, skipped_axioms)``.
    """
    from mowl.geometric.normal_forms import _clean_iri

    axioms: List[AttributedAxiom] = []
    skipped: List[str] = []
    for ax in ontology.getAxioms():
        kind = str(ax.getAxiomType().getName())
        if kind not in {"ClassAssertion", "ObjectPropertyAssertion", "SubClassOf",
                        "SubObjectPropertyOf"}:
            if not str(ax.getAxiomType().getName()).startswith("Declaration"):
                skipped.append(str(ax))
            continue
        try:
            spec = _spec_from_annotations(ax, namespace)
            if kind == "ClassAssertion":
                c, a = ax.getClassExpression(), ax.getIndividual()
                if not (c.isClassExpressionLiteral() and not c.isAnonymous() and a.isNamed()):
                    raise ValueError("complex assertion")
                axioms.append(ConceptFact(_clean_iri(c.asOWLClass().getIRI()),
                                          str(a.asOWLNamedIndividual().getIRI().toString()),
                                          _closed(spec)))
            elif kind == "ObjectPropertyAssertion":
                p = ax.getProperty()
                if p.isAnonymous() or not (ax.getSubject().isNamed() and ax.getObject().isNamed()):
                    raise ValueError("complex assertion")
                axioms.append(RoleFact(
                    str(p.asOWLObjectProperty().getIRI().toString()),
                    str(ax.getSubject().asOWLNamedIndividual().getIRI().toString()),
                    str(ax.getObject().asOWLNamedIndividual().getIRI().toString()),
                    _closed(spec)))
            elif kind == "SubObjectPropertyOf":
                sub, sup = ax.getSubProperty(), ax.getSuperProperty()
                if sub.isAnonymous() or sup.isAnonymous():
                    raise ValueError("inverse role")
                axioms.append(RoleInc(str(sub.asOWLObjectProperty().getIRI().toString()), spec,
                                      str(sup.asOWLObjectProperty().getIRI().toString()), spec))
            else:
                axioms.append(_subclass_from_owl(ax, spec))
        except ValueError:
            skipped.append(str(ax))
    return AttributedOntology(axioms, time_origin), skipped


def _closed(spec: Specifier) -> Specifier:
    """Assertions with no annotations are plain assertions (the empty open specifier)."""
    return spec if spec.pairs else ANY


def _spec_from_annotations(ax, namespace: str) -> Specifier:
    pairs = {}
    for annotation in ax.getAnnotations():
        iri = str(annotation.getProperty().getIRI().toString())
        if not iri.startswith(namespace):
            continue
        attr = iri[len(namespace):]
        lit = annotation.getValue()
        text = str(lit.getLiteral()) if hasattr(lit, "getLiteral") else str(lit.toString())
        if attr == DURING:
            lo, hi = (int(x) for x in text.split(","))
            pairs[attr] = (lo, hi)
        elif attr in (TIME, SINCE, UNTIL, BEFORE, AFTER):
            pairs[attr] = int(text)
        else:
            pairs[attr] = text
    if not pairs:
        return ANY
    return Specifier(frozenset(pairs.items()), False)


def _subclass_from_owl(ax, spec: Specifier) -> ConceptInc:
    from mowl.geometric.normal_forms import _clean_iri

    def basic(ce, allow_bottom=False):
        t = str(ce.getClassExpressionType().getName())
        if t == "Class":
            iri = _clean_iri(ce.asOWLClass().getIRI())
            if iri == BOTTOM and allow_bottom:
                return None
            if iri == TOP:
                raise ValueError("owl:Thing")
            return Basic(iri, spec)
        if t == "ObjectSomeValuesFrom":
            filler = ce.getFiller()
            if filler.isOWLThing() and not ce.getProperty().isAnonymous():
                return Basic(str(ce.getProperty().asOWLObjectProperty().getIRI().toString()),
                             spec, exists=True)
        raise ValueError("unsupported class expression")

    sub, sup = ax.getSubClass(), ax.getSuperClass()
    if str(sub.getClassExpressionType().getName()) == "ObjectIntersectionOf":
        lhs = tuple(basic(op) for op in sub.getOperands())
    else:
        lhs = (basic(sub),)
    return ConceptInc(lhs, basic(sup, allow_bottom=True))
