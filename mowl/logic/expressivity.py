"""Expressivity of description logics, in standard notation.

An :class:`Expressivity` is a set of :class:`Construct` values (the concept and role
constructors and axiom types a logic allows) with a name in the usual description logic
notation (``EL``, ``ELHO(∘)⊥``, ``ALC``, ``SROIQ(D)``). It is used in three places:

* every method of the catalogue (:mod:`mowl.catalog`) states the expressivity it supports;
* every reasoner backend (:mod:`mowl.reasoning`) states the expressivity it is complete for;
* :func:`ontology_expressivity` and :func:`check_ontology` compute what an ontology uses and
  report the axioms that go beyond what a method supports, *before* it is embedded.

This module does not need the JVM, except for the two functions that read an ontology.
"""

import enum
import logging
import warnings
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class Construct(enum.Enum):
    """A constructor or axiom type of a description logic."""

    TOP = "⊤"
    BOTTOM = "⊥"                          #: ⊥, hence unsatisfiability and disjointness
    CONJUNCTION = "⊓"
    EXISTENTIAL = "∃r.C"
    UNIVERSAL = "∀r.C"
    DISJUNCTION = "⊔"
    NEGATION = "¬C"
    NOMINAL = "{a}"                       #: O
    ROLE_HIERARCHY = "r ⊑ s"              #: H
    ROLE_CHAIN = "r ∘ s ⊑ t"              #: complex role inclusions (R, or (∘) in EL)
    TRANSITIVITY = "Trans(r)"             #: S when added to ALC; covered by ROLE_CHAIN
    INVERSE = "r⁻"                        #: I
    FUNCTIONALITY = "Func(r)"             #: F
    CARDINALITY = "≥n r"                  #: N
    QUALIFIED_CARDINALITY = "≥n r.C"      #: Q
    SELF = "∃r.Self"
    ROLE_CHARACTERISTIC = "Sym/Asym/Refl/Irrefl/Disj(r)"   #: the remaining SROIQ role axioms
    DOMAIN_RANGE = "dom(r), ran(r)"       #: domain and range restrictions
    DATATYPE = "(D)"                      #: data properties and concrete domains
    CLASS_ASSERTION = "C(a)"              #: ABox
    ROLE_ASSERTION = "r(a,b)"             #: ABox
    INDIVIDUAL_EQUALITY = "a = b, a ≠ b"  #: ABox


C = Construct

#: Constructs that concern the ABox; they are not part of the name of a logic.
ABOX = frozenset({C.CLASS_ASSERTION, C.ROLE_ASSERTION, C.INDIVIDUAL_EQUALITY})

_EL = frozenset({C.TOP, C.CONJUNCTION, C.EXISTENTIAL})
_ALC = _EL | {C.BOTTOM, C.UNIVERSAL, C.DISJUNCTION, C.NEGATION}
_BEYOND_EL = frozenset({C.UNIVERSAL, C.DISJUNCTION, C.NEGATION, C.CARDINALITY,
                        C.QUALIFIED_CARDINALITY})


def _name(constructs):
    """The smallest standard description logic name that covers ``constructs``."""
    cs = set(constructs) - ABOX
    hierarchy = "H" if C.ROLE_HIERARCHY in cs else ""
    letters = ("O" if C.NOMINAL in cs else "") + ("I" if C.INVERSE in cs else "")
    datatype = "(D)" if C.DATATYPE in cs else ""
    if not cs & _BEYOND_EL:
        number = "F" if C.FUNCTIONALITY in cs else ""
        chain = "(∘)" if cs & {C.ROLE_CHAIN, C.TRANSITIVITY} else ""
        bottom = "⊥" if C.BOTTOM in cs else ""
        return f"EL{hierarchy}{letters}{number}{chain}{bottom}{datatype}"
    if C.QUALIFIED_CARDINALITY in cs:
        number = "Q"
    elif C.CARDINALITY in cs:
        number = "N"
    elif C.FUNCTIONALITY in cs:
        number = "F"
    else:
        number = ""
    if cs & {C.ROLE_CHAIN, C.SELF, C.ROLE_CHARACTERISTIC}:
        return f"SR{letters}{number}{datatype}"
    base = "S" if C.TRANSITIVITY in cs else "ALC"
    return f"{base}{hierarchy}{letters}{number}{datatype}"


@dataclass(frozen=True)
class Expressivity:
    """A description logic, as the set of constructs it allows.

    :param constructs: The :class:`Construct` values.
    :param label: Name to show instead of the generated one, for logics whose usual name is
        not built from letters (``"EL++"``, ``"DL-Lite_H"``, ``"taxonomy"``).
    :param note: Restrictions that the constructs do not express (for example "normal form
        only").

    Expressivities are compared by their constructs: ``a <= b`` means every construct of
    ``a`` is allowed in ``b``.
    """

    constructs: frozenset
    label: str = field(default="", compare=False)
    note: str = field(default="", compare=False)

    def __post_init__(self):
        object.__setattr__(self, "constructs", frozenset(self.constructs))

    @property
    def name(self):
        """The name in description logic notation."""
        return self.label or _name(self.constructs)

    @property
    def dl_name(self):
        """The generated standard name, also when a ``label`` is set."""
        return _name(self.constructs)

    @property
    def tbox(self):
        """The constructs without the ABox ones."""
        return Expressivity(self.constructs - ABOX, self.label, self.note)

    @property
    def has_abox(self):
        return bool(self.constructs & ABOX)

    def __le__(self, other):
        return self.constructs <= other.constructs

    def __lt__(self, other):
        return self.constructs < other.constructs

    def __or__(self, other):
        other = other.constructs if isinstance(other, Expressivity) else frozenset(other)
        return Expressivity(self.constructs | other)

    def __sub__(self, other):
        other = other.constructs if isinstance(other, Expressivity) else frozenset(other)
        return Expressivity(self.constructs - other)

    def __contains__(self, construct):
        return construct in self.constructs

    def __str__(self):
        return self.name + (" with ABox" if self.has_abox else "")

    def missing(self, other):
        """The constructs of ``other`` that this logic does not allow, sorted by name."""
        return sorted(other.constructs - self.constructs, key=lambda c: c.name)

    def with_abox(self, equality=False):
        """This logic plus class and role assertions."""
        extra = set(ABOX) if equality else {C.CLASS_ASSERTION, C.ROLE_ASSERTION}
        return Expressivity(self.constructs | extra, self.label, self.note)

    @classmethod
    def parse(cls, name):
        """Read a name in description logic notation.

        Understands the named logics of :data:`NAMED` (``"EL++"``, ``"DL-Lite_H"``, ...) and
        names built from a base (``EL``, ``ALC``, ``S``, ``SR``) and the letters ``H``,
        ``O``, ``I``, ``F``, ``N``, ``Q``, ``(∘)``, ``⊥`` / ``_bot``, ``(D)``, optionally
        followed by ``" with ABox"``.
        """
        text = name.strip()
        abox = False
        for suffix in (" with ABox", "+ABox"):
            if text.endswith(suffix):
                text, abox = text[:-len(suffix)].strip(), True
        if text in NAMED:
            found = NAMED[text]
            return found.with_abox() if abox else found
        rest = text.replace("_bot", "⊥").replace("(o)", "(∘)")
        if rest.startswith("EL"):
            cs, rest = set(_EL), rest[2:]
        elif rest.startswith("ALC"):
            cs, rest = set(_ALC), rest[3:]
        elif rest.startswith("SR"):
            cs = set(_ALC) | {C.TRANSITIVITY, C.ROLE_HIERARCHY, C.ROLE_CHAIN, C.SELF,
                              C.ROLE_CHARACTERISTIC}
            rest = rest[2:]
        elif rest.startswith("S"):
            cs, rest = set(_ALC) | {C.TRANSITIVITY}, rest[1:]
        else:
            raise ValueError(f"Cannot read {name!r} as a description logic name.")
        tokens = {"(∘)": {C.ROLE_CHAIN}, "(D)": {C.DATATYPE}, "⊥": {C.BOTTOM},
                  "H": {C.ROLE_HIERARCHY}, "O": {C.NOMINAL}, "I": {C.INVERSE},
                  "F": {C.FUNCTIONALITY}, "N": {C.CARDINALITY},
                  "Q": {C.QUALIFIED_CARDINALITY, C.CARDINALITY},
                  "+": {C.ROLE_HIERARCHY, C.ROLE_CHAIN, C.TRANSITIVITY}}
        while rest:
            for token, constructs in tokens.items():
                if rest.startswith(token):
                    cs |= constructs
                    rest = rest[len(token):]
                    break
            else:
                raise ValueError(f"Cannot read {name!r} as a description logic name "
                                 f"(at {rest!r}).")
        result = cls(cs)
        return result.with_abox() if abox else result


def _named():
    el = Expressivity(_EL)
    elpp = Expressivity(_EL | {C.BOTTOM, C.NOMINAL, C.ROLE_HIERARCHY, C.ROLE_CHAIN,
                               C.TRANSITIVITY, C.DOMAIN_RANGE, C.DATATYPE}, label="EL++",
                        note="EL with ⊥, nominals, concrete domains, role inclusions with "
                             "chains, and domain and range restrictions (Baader, Brandt "
                             "and Lutz, 2005; 2008).")
    sroiq = Expressivity(_ALC | {C.TRANSITIVITY, C.ROLE_HIERARCHY, C.ROLE_CHAIN, C.SELF,
                                 C.ROLE_CHARACTERISTIC, C.NOMINAL, C.INVERSE,
                                 C.FUNCTIONALITY, C.CARDINALITY, C.QUALIFIED_CARDINALITY,
                                 C.DOMAIN_RANGE})
    return {
        "taxonomy": Expressivity({C.TOP}, label="taxonomy",
                                 note="Subsumptions between concept names only (A ⊑ B)."),
        "propositional EL⊥": Expressivity({C.TOP, C.BOTTOM, C.CONJUNCTION},
                                          label="propositional EL⊥",
                                          note="Conjunction and ⊥ only; no roles."),
        "EL": el,
        "EL++": elpp,
        "EL+": Expressivity(_EL | {C.ROLE_HIERARCHY, C.ROLE_CHAIN, C.TRANSITIVITY},
                            label="EL+"),
        "ALC": Expressivity(_ALC),
        "DL-Lite_H": Expressivity({C.TOP, C.BOTTOM, C.EXISTENTIAL, C.INVERSE,
                                   C.ROLE_HIERARCHY, C.DOMAIN_RANGE}, label="DL-Lite_H",
                                  note="Unqualified existentials ∃r and ∃r⁻ on both "
                                       "sides, concept and role inclusions, disjointness."),
        "SROIQ": sroiq,
        "SROIQ(D)": sroiq | {C.DATATYPE},
        "OWL 2 DL": Expressivity((sroiq | {C.DATATYPE}).constructs, label="OWL 2 DL"),
    }


#: Named logics understood by :meth:`Expressivity.parse`.
NAMED = _named()


# ---- ontologies (these functions need the JVM) ------------------------------------------

def _expression_constructs(expression):
    """The constructs used by one OWL class expression node (not its sub-expressions)."""
    kind = str(expression.getClassExpressionType().getName())
    if kind == "Class":
        if expression.isOWLThing():
            return {C.TOP}
        if expression.isOWLNothing():
            return {C.BOTTOM}
        return set()
    table = {
        "ObjectIntersectionOf": {C.CONJUNCTION},
        "ObjectUnionOf": {C.DISJUNCTION},
        "ObjectComplementOf": {C.NEGATION},
        "ObjectSomeValuesFrom": {C.EXISTENTIAL},
        "ObjectAllValuesFrom": {C.UNIVERSAL},
        "ObjectHasValue": {C.EXISTENTIAL, C.NOMINAL},
        "ObjectOneOf": {C.NOMINAL},
        "ObjectHasSelf": {C.SELF},
        "ObjectMinCardinality": {C.QUALIFIED_CARDINALITY},
        "ObjectMaxCardinality": {C.QUALIFIED_CARDINALITY},
        "ObjectExactCardinality": {C.QUALIFIED_CARDINALITY},
    }
    found = set(table.get(kind, {C.DATATYPE}))
    if kind in ("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectHasValue",
                "ObjectHasSelf", "ObjectMinCardinality", "ObjectMaxCardinality",
                "ObjectExactCardinality"):
        if expression.getProperty().isAnonymous():
            found.add(C.INVERSE)
    if C.QUALIFIED_CARDINALITY in found and not expression.isQualified():
        found = (found - {C.QUALIFIED_CARDINALITY}) | {C.CARDINALITY}
    return found


_AXIOM_CONSTRUCTS = {
    "SubClassOf": set(),
    "EquivalentClasses": set(),
    "DisjointClasses": {C.BOTTOM, C.CONJUNCTION},
    "DisjointUnion": {C.BOTTOM, C.CONJUNCTION, C.DISJUNCTION},
    "SubObjectPropertyOf": {C.ROLE_HIERARCHY},
    "EquivalentObjectProperties": {C.ROLE_HIERARCHY},
    "SubPropertyChainOf": {C.ROLE_CHAIN},
    "TransitiveObjectProperty": {C.TRANSITIVITY},
    "InverseObjectProperties": {C.INVERSE},
    "FunctionalObjectProperty": {C.FUNCTIONALITY},
    "InverseFunctionalObjectProperty": {C.FUNCTIONALITY, C.INVERSE},
    "SymmetricObjectProperty": {C.INVERSE, C.ROLE_HIERARCHY},
    "AsymmetricObjectProperty": {C.ROLE_CHARACTERISTIC},
    "ReflexiveObjectProperty": {C.ROLE_CHARACTERISTIC},
    "IrreflexiveObjectProperty": {C.ROLE_CHARACTERISTIC},
    "DisjointObjectProperties": {C.ROLE_CHARACTERISTIC},
    "ObjectPropertyDomain": {C.DOMAIN_RANGE},
    "ObjectPropertyRange": {C.DOMAIN_RANGE},
    "ClassAssertion": {C.CLASS_ASSERTION},
    "ObjectPropertyAssertion": {C.ROLE_ASSERTION},
    "NegativeObjectPropertyAssertion": {C.ROLE_ASSERTION, C.NEGATION, C.NOMINAL},
    "SameIndividual": {C.INDIVIDUAL_EQUALITY},
    "DifferentIndividuals": {C.INDIVIDUAL_EQUALITY},
}


def axiom_constructs(axiom):
    """The :class:`Construct` values that one OWL API logical axiom uses."""
    kind = str(axiom.getAxiomType().getName())
    found = set(_AXIOM_CONSTRUCTS.get(kind, {C.DATATYPE}))
    for expression in axiom.getNestedClassExpressions():
        found |= _expression_constructs(expression)
    return found


def ontology_expressivity(ontology, imports=True):
    """The expressivity an ontology uses.

    :param ontology: An ``org.semanticweb.owlapi.model.OWLOntology``.
    :param imports: Include the imports closure.
    :rtype: Expressivity
    """
    from org.semanticweb.owlapi.model.parameters import Imports
    which = Imports.INCLUDED if imports else Imports.EXCLUDED
    found = set()
    for axiom in ontology.getLogicalAxioms(which):
        found |= axiom_constructs(axiom)
    return Expressivity(found)


@dataclass
class ExpressivityReport:
    """What :func:`check_ontology` found.

    :ivar supported: The expressivity the ontology was checked against.
    :ivar used: The expressivity the ontology uses.
    :ivar axioms: Number of logical axioms.
    :ivar unsupported_axioms: Number of logical axioms that use a construct outside
        ``supported``.
    :ivar by_axiom_type: ``{axiom type: number of unsupported axioms}``.
    :ivar by_construct: ``{Construct: number of axioms using it}``, for the constructs
        outside ``supported``.
    """

    supported: Expressivity
    used: Expressivity
    axioms: int = 0
    unsupported_axioms: int = 0
    by_axiom_type: dict = field(default_factory=dict)
    by_construct: dict = field(default_factory=dict)

    @property
    def ok(self):
        """``True`` if every logical axiom is within the supported expressivity."""
        return self.unsupported_axioms == 0

    def summary(self):
        if self.ok:
            return (f"All {self.axioms} logical axioms are within {self.supported} "
                    f"(the ontology uses {self.used}).")
        types = ", ".join(f"{n} {t}" for t, n in sorted(self.by_axiom_type.items()))
        constructs = ", ".join(f"{c.value} ({n})" for c, n in sorted(
            self.by_construct.items(), key=lambda kv: kv[0].name))
        return (f"{self.unsupported_axioms} of {self.axioms} logical axioms go beyond "
                f"{self.supported} (the ontology uses {self.used}) and will be ignored or "
                f"only partly used: {types}. Constructs outside the supported logic: "
                f"{constructs}.")

    def __str__(self):
        return self.summary()


def check_ontology(ontology, supported, imports=True, warn=True):
    """Compare an ontology with the expressivity a method supports.

    :param ontology: An ``org.semanticweb.owlapi.model.OWLOntology``.
    :param supported: An :class:`Expressivity`, or a name for :meth:`Expressivity.parse`.
    :param imports: Include the imports closure.
    :param warn: Emit a :class:`UserWarning` if some axioms are outside ``supported``.
    :rtype: ExpressivityReport
    """
    from org.semanticweb.owlapi.model.parameters import Imports
    if isinstance(supported, str):
        supported = Expressivity.parse(supported)
    which = Imports.INCLUDED if imports else Imports.EXCLUDED
    used, by_type, by_construct = set(), {}, {}
    total = unsupported = 0
    for axiom in ontology.getLogicalAxioms(which):
        total += 1
        constructs = axiom_constructs(axiom)
        used |= constructs
        outside = constructs - supported.constructs
        if outside:
            unsupported += 1
            kind = str(axiom.getAxiomType().getName())
            by_type[kind] = by_type.get(kind, 0) + 1
            for construct in outside:
                by_construct[construct] = by_construct.get(construct, 0) + 1
    report = ExpressivityReport(supported, Expressivity(used), total, unsupported,
                                by_type, by_construct)
    if warn and not report.ok:
        warnings.warn(report.summary(), UserWarning, stacklevel=2)
    return report
