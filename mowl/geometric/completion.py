"""ELK-style completion and canonical model for normalised EL++ ontologies.

Ported from ``moose/elpp`` (Hoehndorf lab) and adapted to :mod:`mowl.geometric`
normal-form dataclasses. Implements the worklist saturation calculus of Kazakov,
Krötzsch & Simančík (*The Incredible ELK*, JAR 2014, §4, Table 2) for normalised
EL++ TBoxes and RBoxes (:class:`~mowl.geometric.normal_forms.NF1`–:class:`~mowl.geometric.normal_forms.NF4`,
:class:`~mowl.geometric.normal_forms.NF6`, :class:`~mowl.geometric.normal_forms.NF7`).

Guarantees:
- The canonical interpretation :math:`\\mathcal{I}` satisfies the normalised
  ontology :math:`\\mathcal{O}` (:math:`\\mathcal{I} \\models \\mathcal{O}`).
- For all concept names :math:`A, B`:
  :math:`A^{\\mathcal{I}} \\subseteq B^{\\mathcal{I}} \\iff \\mathcal{O} \\models A \\sqsubseteq B`.
- For every concept name :math:`A`:
  :math:`A^{\\mathcal{I}} = \\emptyset \\iff \\mathcal{O} \\models A \\sqsubseteq \\bot`.
- Constructing :class:`CanonicalModel` raises
  :class:`~mowl.geometric.errors.InconsistentOntology` if and only if
  :math:`\\bot \\in S(\\top)` (:math:`\\mathcal{O} \\models \\top \\sqsubseteq \\bot`)
  or :math:`\\bot \\in S(N_a)` (:math:`\\mathcal{O} \\models N_a \\sqsubseteq \\bot`)
  for some individual ``a`` (with the individual IRI in the exception message).

References:
- Kazakov, Krötzsch & Simančík, *The Incredible ELK: From Polynomial Procedures
  to Efficient Reasoning with EL Ontologies*, Journal of Automated Reasoning
  53(1):1–61, 2014.
- Baader, Brandt & Lutz, *Pushing the EL Envelope*, IJCAI 2005.
- Gutiérrez-Basulto & Schockaert, KR 2018 (arXiv:1805.10461), Props. 3, 4.
- Lacerda, Ozaki & Guimarães, TGDK 2(3):2, 2024, Defs. 2, 8, 12.
- Leemhuis & Kutz, NeSy 2025 (PMLR 284:303–321), Defs. 1, 3, 6, Props. 4, 5, 8.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from org.semanticweb.owlapi.model import OWLOntology

from mowl.datasets import Dataset
from mowl.geometric.errors import InconsistentOntology
from mowl.geometric.normal_forms import (
    AUX_NS,
    BOTTOM,
    NF1,
    NF2,
    NF3,
    NF4,
    NF6,
    NF7,
    TOP,
    Axiom,
    NormalisedOntology,
    individual_of_name,
    is_nominal_name,
    nominal_name,
    normalise,
)

JOINT_AUX_IRI = f"{AUX_NS}Z_joint"


@dataclass
class CompletionResult:
    """Saturated derivation produced by :class:`Completion`.

    Attributes
    ----------
    sub_super : Dict[str, Set[str]]
        Map ``C -> {D : S(C) contains D}``. Always contains ``C in sub_super[C]``
        and ``TOP in sub_super[C]`` for every concept ``C != BOTTOM`` in the
        signature, and ``BOTTOM in sub_super[BOTTOM]``.
    edges : Dict[str, Set[Tuple[str, str]]]
        Map ``C -> {(R, D) : ELK has derived an R-edge from C to D}`` closed
        under role hierarchy (:class:`NF6`) and role composition (:class:`NF7`).
    role_sub : Dict[str, Set[str]]
        Map ``R -> {S : R <=* S}`` (reflexive-transitive closure under :class:`NF6`).
    concept_names : Set[str]
        All concept names in the signature (including ``TOP`` and ``BOTTOM``).
    role_names : Set[str]
        All role names in the signature.
    """

    sub_super: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    edges: Dict[str, Set[Tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    role_sub: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    concept_names: Set[str] = field(default_factory=set)
    role_names: Set[str] = field(default_factory=set)

    def entails(self, sub: str, sup: str) -> bool:
        """Return ``True`` iff the saturation entails ``sub <= sup``."""
        sups = self.sub_super.get(sub, set())
        return sup in sups or BOTTOM in sups or sub == BOTTOM

    def is_unsatisfiable(self, name: str) -> bool:
        """Return ``True`` iff ``name <= BOTTOM`` was derived."""
        return name == BOTTOM or BOTTOM in self.sub_super.get(name, set())

    def is_inconsistent(self) -> bool:
        """Return ``True`` iff ``BOTTOM in S(TOP)``."""
        return BOTTOM in self.sub_super.get(TOP, set())

    def all_subsumptions(self) -> List[Tuple[str, str]]:
        """All non-trivial derived pairs ``(C, D)`` in ``sub_super`` with ``C != D != TOP``."""
        out: List[Tuple[str, str]] = []
        for c, sups in self.sub_super.items():
            for d in sups:
                if d != c and d != TOP:
                    out.append((c, d))
        return out


class Completion:
    """Worklist-driven ELK saturation for a normalised EL++ ontology.

    Ported from ``moose/elpp/completion.py`` (Hoehndorf lab), implementing
    rules R1–R7 of Kazakov, Krötzsch & Simančík (JAR 2014, §4, Table 2)
    including backwards ``BOTTOM`` propagation along existential edges and
    role-hierarchy / role-chain closure.
    """

    def __init__(
        self,
        axioms: Iterable[Axiom],
        concept_names: Optional[Set[str]] = None,
        role_names: Optional[Set[str]] = None,
    ) -> None:
        self.axioms: List[Axiom] = list(axioms)

        # Static rule indexes built once at __init__
        self._nf1_by_sub: Dict[str, List[NF1]] = defaultdict(list)
        self._nf2_by_sub: Dict[str, List[NF2]] = defaultdict(list)
        self._nf3_by_sub: Dict[str, List[NF3]] = defaultdict(list)
        self._nf4_by_role_filler: Dict[Tuple[str, str], List[NF4]] = defaultdict(list)
        self._nf6_by_sub: Dict[str, List[NF6]] = defaultdict(list)
        self._nf7_by_pair: Dict[Tuple[str, str], List[NF7]] = defaultdict(list)

        self._concept_names: Set[str] = {TOP, BOTTOM}
        if concept_names is not None:
            self._concept_names.update(concept_names)

        self._role_names: Set[str] = set()
        if role_names is not None:
            self._role_names.update(role_names)

        for ax in self.axioms:
            self._index(ax)

    def _index(self, ax: Axiom) -> None:
        if isinstance(ax, NF1):
            self._nf1_by_sub[ax.sub].append(ax)
            self._concept_names.update([ax.sub, ax.sup])
        elif isinstance(ax, NF2):
            self._nf2_by_sub[ax.sub1].append(ax)
            if ax.sub2 != ax.sub1:
                self._nf2_by_sub[ax.sub2].append(ax)
            self._concept_names.update([ax.sub1, ax.sub2, ax.sup])
        elif isinstance(ax, NF3):
            self._nf3_by_sub[ax.sub].append(ax)
            self._concept_names.update([ax.sub, ax.filler])
            self._role_names.add(ax.role)
        elif isinstance(ax, NF4):
            self._nf4_by_role_filler[(ax.role, ax.filler)].append(ax)
            self._concept_names.update([ax.filler, ax.sup])
            self._role_names.add(ax.role)
        elif isinstance(ax, NF6):
            self._nf6_by_sub[ax.sub].append(ax)
            self._role_names.update([ax.sub, ax.sup])
        elif isinstance(ax, NF7):
            self._nf7_by_pair[(ax.r1, ax.r2)].append(ax)
            self._role_names.update([ax.r1, ax.r2, ax.sup])
        else:  # pragma: no cover
            raise TypeError(f"Unknown axiom type: {type(ax).__name__}")

    def saturate(self) -> CompletionResult:
        """Run the worklist to fixpoint and return the :class:`CompletionResult`."""
        result = CompletionResult()
        result.concept_names = set(self._concept_names) | {TOP, BOTTOM}
        result.role_names = set(self._role_names)

        # ----- Role hierarchy: reflexive-transitive closure of NF6 -----
        for r in result.role_names:
            result.role_sub[r].add(r)
            for nf6 in self._nf6_by_sub.get(r, []):
                result.role_sub[r].add(nf6.sup)

        changed = True
        while changed:
            changed = False
            for r in list(result.role_names):
                for s in list(result.role_sub[r]):
                    for t in list(result.role_sub.get(s, set())):
                        if t not in result.role_sub[r]:
                            result.role_sub[r].add(t)
                            changed = True

        # ----- Worklist for concept-side conclusions -------------------
        worklist: deque = deque()
        edges_in: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

        def add_sub(c: str, d: str) -> None:
            if d not in result.sub_super[c]:
                result.sub_super[c].add(d)
                worklist.append(("sub", c, d))

        def add_edge(c: str, r: str, d: str) -> None:
            if (r, d) not in result.edges[c]:
                result.edges[c].add((r, d))
                edges_in[d].add((c, r))
                worklist.append(("edge", c, r, d))

        # ----- Init rule R_0 -------------------------------------------
        add_sub(BOTTOM, BOTTOM)
        for c in result.concept_names:
            if c == BOTTOM:
                continue
            add_sub(c, c)
            add_sub(c, TOP)

        # ----- Main worklist saturation loop ---------------------------
        while worklist:
            item = worklist.popleft()

            if item[0] == "sub":
                _, c, d = item

                # Top propagation: anything derived for TOP holds for all C != BOTTOM
                if c == TOP:
                    for x in result.concept_names:
                        if x != BOTTOM:
                            add_sub(x, d)

                # Rule R_sub (NF1, including NF5 / GCI0_BOT where nf1.sup == BOTTOM):
                # C <= D and D <= E  ==>  C <= E
                for nf1 in self._nf1_by_sub.get(d, []):
                    add_sub(c, nf1.sup)

                # Rule R_and (NF2, including GCI1_BOT where nf2.sup == BOTTOM):
                # C <= D and C <= D' and D and D' <= E  ==>  C <= E
                for nf2 in self._nf2_by_sub.get(d, []):
                    other = nf2.sub2 if nf2.sub1 == d else nf2.sub1
                    if other in result.sub_super.get(c, set()):
                        add_sub(c, nf2.sup)

                # Rule R_exists+ (NF3):
                # C <= D and D <= exists R.E  ==>  edge (C, R, E)
                for nf3 in self._nf3_by_sub.get(d, []):
                    add_edge(c, nf3.role, nf3.filler)

                # Rule R_bot-edge:
                # C <= BOTTOM and edge (X, S, C)  ==>  X <= BOTTOM
                if d == BOTTOM:
                    for parent, _role in list(edges_in.get(c, ())):
                        add_sub(parent, BOTTOM)

                # Rule R_exists- (NF4):
                # C <= D and edge (X, S, C) with S <=* R and exists R.D <= E  ==>  X <= E
                for parent, role in list(edges_in.get(c, ())):
                    for super_role in result.role_sub.get(role, {role}):
                        for nf4 in self._nf4_by_role_filler.get(
                            (super_role, d), []
                        ):
                            add_sub(parent, nf4.sup)

            else:  # item[0] == "edge"
                _, c, r, d = item

                if c == TOP:
                    for x in result.concept_names:
                        if x != BOTTOM:
                            add_edge(x, r, d)

                # Rule R_exists- (NF4): fire against all known super-concepts of d
                for super_role in result.role_sub.get(r, {r}):
                    for d_super in list(result.sub_super.get(d, set())):
                        for nf4 in self._nf4_by_role_filler.get(
                            (super_role, d_super), []
                        ):
                            add_sub(c, nf4.sup)

                # Rule R_bot-edge: edge to an unsatisfiable target propagates BOTTOM
                if BOTTOM in result.sub_super.get(d, set()):
                    add_sub(c, BOTTOM)

                # Rule R_chain (NF7): compose with outgoing edges from d
                for (r2, e) in list(result.edges.get(d, set())):
                    for nf7 in self._nf7_by_pair.get((r, r2), []):
                        for super_role in result.role_sub.get(nf7.sup, {nf7.sup}):
                            add_edge(c, super_role, e)

                # Symmetric R_chain (NF7): compose with incoming edges into c
                for parent, r0 in list(edges_in.get(c, ())):
                    for nf7 in self._nf7_by_pair.get((r0, r), []):
                        for super_role in result.role_sub.get(
                            nf7.sup, {nf7.sup}
                        ):
                            add_edge(parent, super_role, d)

                # Role-hierarchy lift (NF6): an R-edge is also an S-edge for every R <=* S
                for super_role in result.role_sub.get(r, set()):
                    if super_role != r:
                        add_edge(c, super_role, d)

        return result


class CanonicalModel:
    """Canonical model :math:`\\mathcal{I}` of a consistent normalised EL++ ontology.

    Domain and interpretation:
    - :math:`\\Delta^{\\mathcal{I}} = \\{d_A : A \\in N_C \\cup \\{\\top\\}, \\bot \\notin S(A)\\}`,
      ordered by sorted satisfiable concept and nominal names ``names``.
    - :math:`A^{\\mathcal{I}} = \\{d_B \\in \\Delta^{\\mathcal{I}} : A \\in S(B)\\}`
      (returned as a :class:`frozenset` of integer element positions by :meth:`extension`).
    - :math:`r^{\\mathcal{I}} = \\{(d_B, d_C) \\in \\Delta^{\\mathcal{I}} \\times \\Delta^{\\mathcal{I}} : (B, C) \\in R(r)\\}`
      (returned as a :class:`frozenset` of integer position pairs by :meth:`role`).

    Guarantees
    ----------
    - :math:`\\mathcal{I} \\models \\mathcal{O}`.
    - For concept/nominal names :math:`A, B`:
      :math:`A^{\\mathcal{I}} \\subseteq B^{\\mathcal{I}} \\iff \\mathcal{O} \\models A \\sqsubseteq B`.
    - For concept name :math:`A`:
      :math:`A^{\\mathcal{I}} = \\emptyset \\iff \\mathcal{O} \\models A \\sqsubseteq \\bot`.
    - Raises :class:`~mowl.geometric.errors.InconsistentOntology` if and only if
      :math:`\\bot \\in S(\\top)` or :math:`\\bot \\in S(N_a)` for some nominal
      concept :math:`N_a` (with the offending individual IRI in the message).
      Consequently, every individual in ``individuals`` is satisfiable on any
      constructed :class:`CanonicalModel`, and :meth:`element_of` cannot
      encounter an unsatisfiable nominal on a built model.
    - Instance checking (:meth:`instance_of`) is sound and complete for
      :math:`\\mathcal{ELH}` (with :math:`\\bot`) without nominals
      (``ObjectOneOf``), functionality (``FunctionalObjectProperty``), and range
      (``ObjectPropertyRange``) axioms.

    Non-guarantees
    --------------
    - Sound and complete for instance checking only for :math:`\\mathcal{ELH}`
      without nominals, functionality, and range axioms.
    - Passing unknown individual or class IRIs to :meth:`instance_of` or
      :meth:`assertion_contradicted` returns ``False``.
    - During :func:`~mowl.geometric.normal_forms.normalise`, complex
      ``ClassAssertion``, ``SameIndividual``, ``DifferentIndividuals``,
      ``ObjectPropertyAssertion`` with an inverse property, assertions on
      anonymous individuals, and negative or data assertions raise
      :class:`~mowl.geometric.errors.UnsupportedAxiomError` with the axiom text.
    """

    def __init__(
        self,
        axioms: Iterable[Axiom],
        completion: CompletionResult,
        concept_names: Optional[Set[str]] = None,
        role_names: Optional[Set[str]] = None,
        dropped_axioms: int = 0,
    ) -> None:
        if completion.is_inconsistent():
            raise InconsistentOntology(
                "Ontology is inconsistent: owl:Thing <= owl:Nothing derived in completion."
            )

        self.axioms: List[Axiom] = list(axioms)
        self.completion: CompletionResult = completion
        self.dropped_axioms: int = dropped_axioms

        all_concepts: Set[str] = set(completion.concept_names) | {TOP}
        if concept_names is not None:
            all_concepts.update(concept_names)
        all_concepts.discard(BOTTOM)

        self.individuals: Dict[str, str] = {
            individual_of_name(c): c
            for c in sorted(all_concepts)
            if is_nominal_name(c)
        }
        for ind_iri, nom_name in self.individuals.items():
            if completion.is_unsatisfiable(nom_name):
                raise InconsistentOntology(
                    f"Ontology is inconsistent: individual '{ind_iri}' is unsatisfiable "
                    f"({nom_name} <= owl:Nothing derived in completion)."
                )
        self._concept_names: Set[str] = {
            c for c in all_concepts if not is_nominal_name(c)
        } | {BOTTOM}
        self.role_names: Set[str] = set(completion.role_names)
        if role_names is not None:
            self.role_names.update(role_names)

        self.names: List[str] = sorted(
            c for c in all_concepts if not completion.is_unsatisfiable(c)
        )
        self.domain: List[str] = [f"d_{name}" for name in self.names]
        self.index: Dict[str, int] = {
            elem_id: idx for idx, elem_id in enumerate(self.domain)
        }
        self._name_to_pos: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.names)
        }

        # Precompute extensions for all known concept names by inverting completion.sub_super
        ext_lists: Dict[str, List[int]] = {cname: [] for cname in all_concepts}
        for idx, b_name in enumerate(self.names):
            for cname in completion.sub_super.get(b_name, ()):
                bucket = ext_lists.get(cname)
                if bucket is not None:
                    bucket.append(idx)
        self._extensions: Dict[str, FrozenSet[int]] = {
            cname: (
                frozenset()
                if completion.is_unsatisfiable(cname)
                else frozenset(idxs)
            )
            for cname, idxs in ext_lists.items()
        }
        self._extensions[BOTTOM] = frozenset()

        # Precompute role interpretations for all known roles
        role_pairs: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
        for src_name, pos_i in self._name_to_pos.items():
            for r_name, dst_name in completion.edges.get(src_name, set()):
                pos_j = self._name_to_pos.get(dst_name)
                if pos_j is not None:
                    role_pairs[r_name].add((pos_i, pos_j))
        self._roles: Dict[str, FrozenSet[Tuple[int, int]]] = {
            r_name: frozenset(pairs) for r_name, pairs in role_pairs.items()
        }

        # Cache for fresh-name conjunction saturation: frozenset(names) -> frozenset(S(Z_joint))
        self._conjunction_cache: Dict[FrozenSet[str], FrozenSet[str]] = {}
        self._joint_sat_cache: Dict[FrozenSet[str], bool] = {}

    @classmethod
    def from_normalised(cls, no: NormalisedOntology) -> CanonicalModel:
        """Build a :class:`CanonicalModel` from a :class:`~mowl.geometric.normal_forms.NormalisedOntology`."""
        comp = Completion(no.axioms, no.concept_names, no.role_names).saturate()
        return cls(
            axioms=no.axioms,
            completion=comp,
            concept_names=no.concept_names,
            role_names=no.role_names,
            dropped_axioms=no.dropped_axioms,
        )

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> CanonicalModel:
        """Normalise a mOWL :class:`~mowl.datasets.Dataset` and build its :class:`CanonicalModel`."""
        return cls.from_normalised(normalise(dataset))

    @classmethod
    def from_ontology(cls, ontology: OWLOntology) -> CanonicalModel:
        """Normalise an OWLAPI ``OWLOntology`` and build its :class:`CanonicalModel`."""
        return cls.from_normalised(normalise(ontology))

    def extension(self, name: str) -> FrozenSet[int]:
        """Return :math:`\\text{name}^{\\mathcal{I}}` as a :class:`frozenset` of domain positions."""
        if name in self._extensions:
            return self._extensions[name]
        if not self.is_satisfiable(name):
            return frozenset()
        ext = frozenset(
            idx
            for idx, b_name in enumerate(self.names)
            if name in self.completion.sub_super.get(b_name, set())
        )
        self._extensions[name] = ext
        return ext

    def role(self, role: str) -> FrozenSet[Tuple[int, int]]:
        """Return :math:`\\text{role}^{\\mathcal{I}}` as a :class:`frozenset` of ``(i, j)`` position pairs."""
        return self._roles.get(role, frozenset())

    def entails(self, a: str, b: str) -> bool:
        """Return ``True`` iff :math:`\\mathcal{O} \\models a \\sqsubseteq b`."""
        return self.completion.entails(a, b)

    def is_satisfiable(self, name: str) -> bool:
        """Return ``True`` iff :math:`\\mathcal{O} \\not\\models \\text{name} \\sqsubseteq \\bot`."""
        return not self.completion.is_unsatisfiable(name)

    def _saturate_conjunction(self, names_fset: FrozenSet[str]) -> FrozenSet[str]:
        """Saturate a fresh concept ``Z_joint <= A_i`` and return ``frozenset(S(Z_joint))``."""
        cached = self._conjunction_cache.get(names_fset)
        if cached is not None:
            return cached

        if BOTTOM in names_fset:
            res_set: FrozenSet[str] = frozenset({BOTTOM, TOP})
            self._conjunction_cache[names_fset] = res_set
            self._joint_sat_cache[names_fset] = False
            return res_set

        z_iri = JOINT_AUX_IRI
        while z_iri in self.completion.concept_names or z_iri in names_fset:
            z_iri += "_"

        extra_axioms = list(self.axioms) + [NF1(z_iri, a) for a in sorted(names_fset)]
        extra_concepts = set(self.completion.concept_names) | set(names_fset) | {z_iri}
        res = Completion(
            extra_axioms, extra_concepts, set(self.completion.role_names)
        ).saturate()

        sups = frozenset(res.sub_super.get(z_iri, set()))
        self._conjunction_cache[names_fset] = sups
        self._joint_sat_cache[names_fset] = not res.is_unsatisfiable(z_iri)
        return sups

    def joint_satisfiable(self, names: Iterable[str]) -> bool:
        """Return ``True`` iff the conjunction :math:`\\sqcap_{A \\in \\text{names}} A` is satisfiable."""
        names_fset = frozenset(names)
        if names_fset in self._joint_sat_cache:
            return self._joint_sat_cache[names_fset]
        self._saturate_conjunction(names_fset)
        return self._joint_sat_cache[names_fset]

    @property
    def concept_names(self) -> Set[str]:
        """Return concept names in the ontology signature, excluding nominal names.

        Separates genuine concept names (including auxiliary normalisation
        names, ``TOP``, and ``BOTTOM``) from nominal concept names
        (:func:`~mowl.geometric.normal_forms.is_nominal_name`) introduced for
        individuals, while ``self.names`` retains all satisfiable names
        (including nominals) so region builders treat nominals like any other
        satisfiable name.

        Guarantees
        ----------
        - Contains no nominal names (``not any(is_nominal_name(c) for c in cm.concept_names)``).
        - Includes ``TOP``, ``BOTTOM``, and all non-nominal concept names
          (satisfiable and unsatisfiable) in the normalised ontology signature.
        - Instance checking on the nominal encoding is sound and complete for
          :math:`\\mathcal{ELH}` without nominals, functionality, and range
          axioms.

        Non-guarantees
        --------------
        - Does not include nominal concept names (use ``self.names`` or
          ``self.individuals`` for nominals).
        - Sound and complete for instance checking only for :math:`\\mathcal{ELH}`
          without nominals, functionality, and range axioms; unknown IRIs return
          ``False`` in :meth:`instance_of` and :meth:`assertion_contradicted`;
          complex ``ClassAssertion``, ``SameIndividual``,
          ``DifferentIndividuals``, ``ObjectPropertyAssertion`` with an inverse
          property, assertions on anonymous individuals, and negative/data
          assertions raise :class:`~mowl.geometric.errors.UnsupportedAxiomError`
          during normalisation.

        Returns
        -------
        Set[str]
            Copy of the non-nominal concept name set.
        """
        return set(self._concept_names)

    def element_of(self, name: str) -> int:
        """Return the canonical domain element index for a concept/nominal name or individual IRI.

        Looks up ``name`` directly in ``self._name_to_pos`` (satisfiable concept
        or nominal names in ``self.names``) or via ``self.individuals[name]``
        when ``name`` is a known individual IRI.

        Guarantees
        ----------
        - For every satisfiable concept or nominal name ``c`` in ``self.names``,
          returns ``i`` such that ``self.names[i] == c`` and
          ``self.domain[i] == f"d_{c}"``.
        - For every individual IRI ``a`` in ``self.individuals``, returns
          ``element_of(self.individuals[a])``. Because an unsatisfiable nominal
          (:math:`N_a \\sqsubseteq \\bot` derived) raises
          :class:`~mowl.geometric.errors.InconsistentOntology` during
          :class:`CanonicalModel` construction, every individual ``a`` in
          ``self.individuals`` has a satisfiable nominal in ``self.names``, so
          ``element_of(a)`` cannot hit an unsatisfiable nominal on a built
          model.
        - Sound and complete for instance checking in :math:`\\mathcal{ELH}`
          without nominals, functionality, and range axioms.

        Non-guarantees
        --------------
        - Raises :class:`KeyError` for unknown IRIs and for unsatisfiable
          non-nominal concept names (including ``BOTTOM``). Unknown IRIs passed
          to :meth:`instance_of` or :meth:`assertion_contradicted` return
          ``False`` instead.
        - Sound and complete for instance checking only for :math:`\\mathcal{ELH}`
          without nominals, functionality, and range axioms; complex
          ``ClassAssertion``, ``SameIndividual``, ``DifferentIndividuals``,
          ``ObjectPropertyAssertion`` with an inverse property, assertions on
          anonymous individuals, and negative/data assertions raise
          :class:`~mowl.geometric.errors.UnsupportedAxiomError` during
          normalisation.

        Parameters
        ----------
        name : str
            Satisfiable concept name, nominal name, or individual IRI in the
            canonical model.

        Returns
        -------
        int
            Integer index :math:`0 \\le i < |\\text{self.domain}|` of the
            canonical domain element.

        Raises
        ------
        KeyError
            If ``name`` is not a satisfiable name in ``self.names`` or a known
            individual IRI in ``self.individuals``.
        """
        if name in self._name_to_pos:
            return self._name_to_pos[name]
        nom = self.individuals.get(name)
        if nom is not None and nom in self._name_to_pos:
            return self._name_to_pos[nom]
        raise KeyError(name)

    def _nominal_for(self, a: str) -> str:
        """Resolve an individual IRI (or nominal name) ``a`` to its nominal concept name ``N_a``."""
        if is_nominal_name(a):
            return a
        return self.individuals.get(a, nominal_name(a))

    def instance_of(self, a: str, C: str) -> bool:
        """Decide whether individual ``a`` is entailed to be an instance of concept ``C``.

        Evaluates :math:`\\mathcal{O} \\models N_a \\sqsubseteq C` in the
        saturated completion via :meth:`entails`.

        Guarantees
        ----------
        - Sound and complete for atomic instance checking (:math:`\\mathcal{O} \\models C(a)`)
          in :math:`\\mathcal{ELH}` (with :math:`\\bot`) without nominals
          (``ObjectOneOf``), functionality (``FunctionalObjectProperty``), and
          range (``ObjectPropertyRange``) axioms.
        - Accepts either an individual IRI ``a`` or its encoded nominal name
          ``N_a``.
        - For every known individual ``a`` in ``self.individuals``,
          ``instance_of(a, TOP)`` is ``True`` and ``instance_of(a, BOTTOM)`` is
          ``False`` (since any unsatisfiable nominal raises
          :class:`~mowl.geometric.errors.InconsistentOntology` at model
          construction).

        Non-guarantees
        --------------
        - Sound and complete for instance checking only for :math:`\\mathcal{ELH}`
          without nominals, functionality, and range axioms.
        - Returns ``False`` whenever ``a`` or ``C`` is an unknown IRI not
          present in the canonical model (in particular,
          ``instance_of(unknown, TOP)`` is ``False``).
        - Unsupported ABox assertions (complex ``ClassAssertion``,
          ``SameIndividual``, ``DifferentIndividuals``,
          ``ObjectPropertyAssertion`` with an inverse property, assertions on
          anonymous individuals, and negative/data assertions) raise
          :class:`~mowl.geometric.errors.UnsupportedAxiomError` during
          normalisation.

        Parameters
        ----------
        a : str
            Individual IRI (or nominal concept name ``N_a``).
        C : str
            Named concept IRI (or ``TOP`` / ``BOTTOM``).

        Returns
        -------
        bool
            ``True`` iff :math:`\\mathcal{O} \\models N_a \\sqsubseteq C` for
            known ``a`` and ``C``, else ``False``.
        """
        return self.entails(self._nominal_for(a), C)

    entails_assertion = instance_of

    def assertion_contradicted(self, a: str, C: str) -> bool:
        """Decide whether asserting ``C(a)`` is contradicted by the ontology.

        Evaluates :math:`\\mathcal{O} \\models N_a \\sqcap C \\sqsubseteq \\bot`
        via ``not self.joint_satisfiable({N_a, C})`` for known ``a`` and ``C``.

        Guarantees
        ----------
        - Sound and complete for deciding whether :math:`\\mathcal{O} \\cup \\{C(a)\\}`
          is inconsistent (:math:`\\mathcal{O} \\models N_a \\sqcap C \\sqsubseteq \\bot`)
          in :math:`\\mathcal{ELH}` (with :math:`\\bot`) without nominals
          (``ObjectOneOf``), functionality (``FunctionalObjectProperty``), and
          range (``ObjectPropertyRange``) axioms.
        - Accepts either an individual IRI ``a`` or its encoded nominal name
          ``N_a``.

        Non-guarantees
        --------------
        - Sound and complete for instance checking and assertion contradiction
          only for :math:`\\mathcal{ELH}` without nominals, functionality, and
          range axioms.
        - Returns ``False`` whenever ``a`` or ``C`` is an unknown IRI not
          present in the canonical model.
        - Unsupported ABox assertions (complex ``ClassAssertion``,
          ``SameIndividual``, ``DifferentIndividuals``,
          ``ObjectPropertyAssertion`` with an inverse property, assertions on
          anonymous individuals, and negative/data assertions) raise
          :class:`~mowl.geometric.errors.UnsupportedAxiomError` during
          normalisation.

        Parameters
        ----------
        a : str
            Individual IRI (or nominal concept name ``N_a``).
        C : str
            Named concept IRI (or ``TOP`` / ``BOTTOM``).

        Returns
        -------
        bool
            ``True`` iff both ``a`` and ``C`` are known to the canonical model
            and :math:`\\mathcal{O} \\models N_a \\sqcap C \\sqsubseteq \\bot`,
            else ``False``.
        """
        nom = self._nominal_for(a)
        if nom not in self._name_to_pos or (
            C not in self._concept_names and C not in self._name_to_pos
        ):
            return False
        return not self.joint_satisfiable({nom, C})

    def role_instances(self, r: str) -> Set[Tuple[str, str]]:
        """Return pairs of individual IRIs ``(a, b)`` connected by role ``r`` in the canonical model.

        Filters :meth:`role` for ``r`` to pairs ``(pos_i, pos_j)`` whose canonical
        domain elements both correspond to nominal names (:func:`is_nominal_name`),
        decoding each nominal back to its individual IRI via
        :func:`individual_of_name`.

        Guarantees
        ----------
        - Returns the exact set of individual IRI pairs ``(a, b)`` such that
          :math:`(d_{N_a}, d_{N_b}) \\in r^{\\mathcal{I}}` under the RBox closure
          (:class:`~mowl.geometric.normal_forms.NF6` role hierarchy and
          :class:`~mowl.geometric.normal_forms.NF7` role chains).
        - Non-nominal canonical domain elements :math:`d_C` are excluded from
          the returned pairs.
        - Sound and complete for instance checking in :math:`\\mathcal{ELH}`
          without nominals, functionality, and range axioms.

        Non-guarantees
        --------------
        - Returns ``set()`` if ``r`` is an unknown role IRI or has no
          nominal-to-nominal edges.
        - Sound and complete for instance checking only for :math:`\\mathcal{ELH}`
          without nominals, functionality, and range axioms; unknown IRIs return
          ``False`` in :meth:`instance_of` and :meth:`assertion_contradicted`;
          complex ``ClassAssertion``, ``SameIndividual``,
          ``DifferentIndividuals``, ``ObjectPropertyAssertion`` with an inverse
          property, assertions on anonymous individuals, and negative/data
          assertions raise :class:`~mowl.geometric.errors.UnsupportedAxiomError`
          during normalisation.

        Parameters
        ----------
        r : str
            Role (object property) IRI string.

        Returns
        -------
        Set[Tuple[str, str]]
            Set of ``(subject_individual_iri, object_individual_iri)`` pairs.
        """
        out: Set[Tuple[str, str]] = set()
        for pos_i, pos_j in self.role(r):
            src_name = self.names[pos_i]
            dst_name = self.names[pos_j]
            if is_nominal_name(src_name) and is_nominal_name(dst_name):
                out.add((individual_of_name(src_name), individual_of_name(dst_name)))
        return out

    def entails_conjunction(self, names: Iterable[str], sup: str) -> bool:
        """Return ``True`` iff :math:`\\mathcal{O} \\models (\\sqcap_{A \\in \\text{names}} A) \\sqsubseteq \\text{sup}`."""
        sups = self._saturate_conjunction(frozenset(names))
        return sup in sups or BOTTOM in sups
