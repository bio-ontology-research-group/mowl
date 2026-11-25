"""Strong faithfulness of geometric models of ELH ontologies (Lacerda et al., TGDK 2024).

A geometric model of an ontology is *strongly TBox faithful* if it satisfies an axiom
exactly when the ontology entails it (and *strongly IQ faithful* for instance queries
``C(a)``). The paper proves that the geometric interpretation of the canonical model has
this property, in dimension 1 with unions of intervals (Definition 12, Theorem 16) and
with convex regions in a dimension tied to the canonical model (Definition 29,
Theorem 32). Both constructions are in mOWL as
:class:`~mowl.geometric.intervals.IntervalUnionModel` and
:class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`.

This module adds what makes the property testable and usable:

- :func:`entailed`: whether the ontology entails a normal-form axiom, decided by the ELK-style
  completion (:class:`~mowl.geometric.completion.Completion`) on the ontology extended with
  fresh names, not by evaluating the canonical model, so it is independent of the model
  being checked.
- :func:`candidate_axioms` and :func:`check_strong_faithfulness`: compare, for every
  candidate axiom of the stated forms over a signature, "the geometric model satisfies it"
  with "the ontology entails it".
- :func:`paper_canonical_ontology` and :class:`StronglyFaithfulModel`: build the canonical
  model of Definition 26 (which has more elements than mOWL's
  :class:`~mowl.geometric.completion.CanonicalModel`; without them the construction is
  faithful on concept names only) and its geometric model with the chosen region family.

Only the TBox and RBox forms ``A ⊑ B``, ``A ⊓ B ⊑ C``, ``A ⊑ ∃r.B``, ``∃r.A ⊑ B`` and
``r ⊑ s`` (with ``⊤`` on the left or as filler, ``⊥`` as target) are enumerated; instance
queries are answered through nominal names of the canonical model and are not part of this
check. The FaithEL learning procedure (Lacerda, Ozaki & Guimarães, RuleML+RR 2024) is not
implemented.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from mowl.geometric.completion import CanonicalModel, Completion, CompletionResult
from mowl.geometric.normal_forms import (
    AUX_NS,
    BOTTOM,
    NF1,
    NF2,
    NF3,
    NF4,
    NF6,
    TOP,
    Axiom,
    NormalisedOntology,
    is_nominal_name,
)

DEFAULT_FORMS = ("NF1", "NF2", "NF3", "NF4", "NF6")
_FRESH_Z = AUX_NS + "Z_faith"
_FRESH_W = AUX_NS + "W_faith"


class EntailmentOracle:
    """Decides entailment of normal-form axioms from the completion of an ontology."""

    def __init__(self, cm: CanonicalModel) -> None:
        self.cm = cm
        self._cache: Dict[tuple, Tuple[CompletionResult, str]] = {}

    def _saturate(self, extra: Sequence[Axiom], key: tuple, fresh: str) -> CompletionResult:
        if key not in self._cache:
            concepts = set(self.cm.completion.concept_names) | {_FRESH_Z, _FRESH_W}
            res = Completion(list(self.cm.axioms) + list(extra), concepts,
                             set(self.cm.completion.role_names)).saturate()
            self._cache[key] = (res, fresh)
        return self._cache[key][0]

    def _unsat(self, name: str) -> bool:
        if name == BOTTOM:
            return True
        return name != TOP and not self.cm.is_satisfiable(name)

    def entails(self, ax: Axiom) -> bool:
        cm = self.cm
        if isinstance(ax, NF1):
            return self._unsat(ax.sub) or ax.sup == TOP or cm.entails(ax.sub, ax.sup)
        if isinstance(ax, NF2):
            return cm.entails_conjunction([ax.sub1, ax.sub2], ax.sup)
        if isinstance(ax, NF3):
            # O |= A <= EXISTS r.B  iff  Z in S(A) once EXISTS r.B <= Z is added (Z fresh).
            res = self._saturate([NF4(ax.role, ax.filler, _FRESH_Z)],
                                 ("nf3", ax.role, ax.filler), _FRESH_Z)
            return self._unsat(ax.sub) or _FRESH_Z in res.sub_super.get(ax.sub, set())
        if isinstance(ax, NF4):
            # O |= EXISTS r.B <= C  iff  C in S(Z) once Z <= EXISTS r.B is added (Z fresh).
            res = self._saturate([NF3(_FRESH_Z, ax.role, ax.filler)],
                                 ("nf4", ax.role, ax.filler), _FRESH_Z)
            sup = res.sub_super.get(_FRESH_Z, set())
            return ax.sup in sup or BOTTOM in sup or ax.sup == TOP
        if isinstance(ax, NF6):
            # O |= r <= s  iff  Y in S(Z) with Z <= EXISTS r.W and EXISTS s.W <= Y.
            res = self._saturate([NF3(_FRESH_Z, ax.sub, _FRESH_W),
                                  NF4(ax.sup, _FRESH_W, _FRESH_W + "_Y")],
                                 ("nf6", ax.sub, ax.sup), _FRESH_Z)
            return (_FRESH_W + "_Y") in res.sub_super.get(_FRESH_Z, set())
        raise TypeError(f"entailment of {type(ax).__name__} is not supported")


def entailed(cm: CanonicalModel, axiom: Axiom) -> bool:
    """Whether the ontology of ``cm`` entails ``axiom`` (``NF1``-``NF4``, ``NF6``)."""
    return EntailmentOracle(cm).entails(axiom)


def candidate_axioms(
    cm: CanonicalModel,
    forms: Iterable[str] = DEFAULT_FORMS,
    names: Optional[Iterable[str]] = None,
    roles: Optional[Iterable[str]] = None,
) -> Iterator[Axiom]:
    """All axioms of the given forms over ``names`` and ``roles`` (default: ``cm``'s signature)."""
    forms = set(forms)
    ns = sorted(set(cm.concept_names if names is None else names) - {TOP, BOTTOM})
    rs = sorted(cm.role_names if roles is None else roles)
    lhs = [TOP] + ns
    rhs = ns + [TOP, BOTTOM]
    if "NF1" in forms:
        for a, b in itertools.product(lhs, rhs):
            if a != b:
                yield NF1(a, b)
    if "NF2" in forms:
        for a, b in itertools.combinations(ns, 2):
            for c in rhs:
                yield NF2(a, b, c)
    if "NF3" in forms:
        for a, r, b in itertools.product(lhs, rs, [TOP] + ns):
            yield NF3(a, r, b)
    if "NF4" in forms:
        for r, b, c in itertools.product(rs, [TOP] + ns, rhs):
            yield NF4(r, b, c)
    if "NF6" in forms:
        for r, s in itertools.permutations(rs, 2):
            yield NF6(r, s)


@dataclass
class FaithfulnessReport:
    """Outcome of :func:`check_strong_faithfulness`."""

    checked: int = 0
    entailed_count: int = 0
    #: axioms that hold in the model but are not entailed (the model says too much)
    unsound: List[Axiom] = field(default_factory=list)
    #: axioms that are entailed but do not hold in the model
    incomplete: List[Axiom] = field(default_factory=list)

    @property
    def is_strongly_faithful(self) -> bool:
        return not self.unsound and not self.incomplete


def check_strong_faithfulness(
    eta,
    cm: CanonicalModel,
    forms: Iterable[str] = DEFAULT_FORMS,
    names: Optional[Iterable[str]] = None,
    roles: Optional[Iterable[str]] = None,
) -> FaithfulnessReport:
    """Compare model satisfaction with entailment for every candidate axiom.

    ``eta`` is a geometric interpretation with a ``satisfies(axiom)`` method
    (:class:`~mowl.geometric.polyhedral.ConvexPolyhedralModel`,
    :class:`~mowl.geometric.intervals.IntervalUnionModel`, ...).
    """
    oracle = EntailmentOracle(cm)
    rep = FaithfulnessReport()
    for ax in candidate_axioms(cm, forms, names, roles):
        holds = oracle.entails(ax)
        sat = eta.satisfies(ax)
        rep.checked += 1
        rep.entailed_count += int(holds)
        if sat and not holds:
            rep.unsound.append(ax)
        elif holds and not sat:
            rep.incomplete.append(ax)
    return rep


def paper_canonical_ontology(no: NormalisedOntology) -> Tuple[NormalisedOntology, Set[str]]:
    """Add the elements of Definition 26 of Lacerda et al. that mOWL's canonical model lacks.

    mOWL's :class:`~mowl.geometric.completion.CanonicalModel` has one element per concept
    name. Definition 26 of the paper also has an element ``c_{A⊓B}`` for every pair of
    concept names and an element ``c_{∃r.B}`` for every role and concept name (or ⊤);
    without them the model satisfies axioms such as ``A ⊓ B ⊑ C`` or ``∃r.A ⊑ B`` that
    are not entailed, so it is not strongly TBox faithful.

    The extra elements are added as fresh concept names: ``N_AB ⊑ A``, ``N_AB ⊑ B`` and
    ``M_rB ⊑ ∃r.B``. In the saturation, the types of ``N_AB`` are exactly the supers of
    ``A ⊓ B`` and those of ``M_rB`` exactly the supers of ``∃r.B``, as in the definition.

    Returns the extended ontology and the set of fresh names.
    """
    names = sorted(c for c in no.concept_names
                   if c not in (TOP, BOTTOM) and not is_nominal_name(c))
    fresh: Set[str] = set()
    extra: List[Axiom] = []
    for i, j in itertools.combinations(range(len(names)), 2):
        z = f"{AUX_NS}faith_conj_{i}_{j}"
        fresh.add(z)
        extra += [NF1(z, names[i]), NF1(z, names[j])]
    for ri, r in enumerate(sorted(no.role_names)):
        for j, b in enumerate([TOP] + names):
            z = f"{AUX_NS}faith_ex_{ri}_{j}"
            fresh.add(z)
            extra.append(NF3(z, r, b))
    return NormalisedOntology(
        axioms=list(no.axioms) + extra,
        concept_names=set(no.concept_names) | fresh,
        role_names=set(no.role_names),
        dropped_axioms=no.dropped_axioms), fresh


class StronglyFaithfulModel:
    """Strongly faithful geometric model of an ELH ontology (Lacerda et al., TGDK 2024).

    Builds the canonical model of Definition 26 (:func:`paper_canonical_ontology`) and its
    geometric interpretation with the chosen region family.

    :param source: a :class:`~mowl.datasets.Dataset`, an OWLAPI ontology or a
        :class:`~mowl.geometric.normal_forms.NormalisedOntology`.
    :param regions: ``"intervals"`` for the 1-dimensional non-convex model (Definition 12,
        Theorem 16) or ``"convex"`` for the convex model of dimension ``n - 1`` with ``n``
        the size of the canonical model (Definition 29, Theorem 32).

    The attribute ``signature`` is the set of concept names of the (normalised) ontology;
    axioms over it are the ones for which the model is strongly faithful.
    """

    REGIONS = ("intervals", "convex")

    def __init__(self, source, regions: str = "intervals") -> None:
        from mowl.geometric.intervals import IntervalUnionModel
        from mowl.geometric.normal_forms import normalise
        from mowl.geometric.polyhedral import ConvexPolyhedralModel

        if regions not in self.REGIONS:
            raise ValueError(f"regions must be one of {self.REGIONS}, got {regions!r}")
        no = source if isinstance(source, NormalisedOntology) else normalise(source)
        self.signature: Set[str] = {c for c in no.concept_names if not is_nominal_name(c)}
        self.role_signature: Set[str] = set(no.role_names)
        extended, self.fresh_names = paper_canonical_ontology(no)
        self.regions = regions
        self.canonical_model = CanonicalModel.from_normalised(extended)
        cls = IntervalUnionModel if regions == "intervals" else ConvexPolyhedralModel
        self.eta = cls.from_canonical_model(self.canonical_model)

    @property
    def dimension(self) -> int:
        return 1 if self.regions == "intervals" else self.eta.dimension

    def satisfies(self, axiom: Axiom) -> bool:
        return self.eta.satisfies(axiom)

    def check_strong_faithfulness(
            self, forms: Iterable[str] = DEFAULT_FORMS) -> FaithfulnessReport:
        """Compare satisfaction with entailment for all axioms of ``forms`` over the signature."""
        return check_strong_faithfulness(
            self.eta, self.canonical_model, forms=forms,
            names=self.signature, roles=self.role_signature)
