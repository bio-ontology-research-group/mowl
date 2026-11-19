"""Logical-faithfulness evaluation of knowledge base completion models.

Implements the faithfulness hierarchy of Mashkova & Hoehndorf, "A hierarchy of
faithfulness criteria for knowledge base completion" (arXiv:2609.27863, 2026), referred
to below as [MH26]:

* the trichotomy of candidate axioms into **entailed**, **contradictory** and
  **undetermined** ones (Definition 1), decided per :math:`\\mathcal{EL}^{++}` normal form
  by a sound and complete saturation (:class:`LogicalStatusOracle`),
* reasoner-generated test sets :math:`T_+, T_-, T_?` (Section 6.1,
  :func:`generate_test_sets`),
* the four criteria, discrimination (L0), logical admissibility (L1), monotonic logical
  faithfulness (L2) and probabilistic logical faithfulness (L3) (Definitions 2--5), and
  every metric reported in the paper: :math:`AUC_{+/-}`, Adm@k, FVR, Pair-FVR,
  :math:`\\rho_P`, MRR and Hits@k (Section 6.1, Appendix G),
* the relative model count :math:`P(\\alpha \\mid \\mathcal{O}) = \\#(\\mathcal{O} \\cup
  \\{\\alpha\\}) / \\#(\\mathcal{O})` over a bounded Herbrand domain (Section 4,
  Appendix A, :class:`RelativeModelCounter`),
* a model-agnostic scoring interface: any :class:`~mowl.base_models.EmbeddingELModel`
  (scored with its normal-form losses) or any callable ``score_fn(axioms)``
  (:func:`as_scorer`),
* a "model-ness" report for geometric models (:func:`loss_modelness`).

The evaluation regimes :class:`FaithfulnessHierarchy` and :class:`LossModelness` package
these for use with any model and dataset.

Axioms are plain tuples of IRIs, one shape per normal form, in the column order of
:class:`mowl.datasets.el.ELDataset`:

================  ======================  ===================================
normal form       tuple                   axiom
================  ======================  ===================================
``gci0``          ``(C, D)``              :math:`C \\sqsubseteq D`
``gci1``          ``(C, D, E)``           :math:`C \\sqcap D \\sqsubseteq E`
``gci2``          ``(C, r, D)``           :math:`C \\sqsubseteq \\exists r.D`
``gci3``          ``(r, C, D)``           :math:`\\exists r.C \\sqsubseteq D`
``gci1_bot``      ``(C, D)``              :math:`C \\sqcap D \\sqsubseteq \\bot`
================  ======================  ===================================

All metric functions take *plausibilities*: higher means more plausible. Scores of
loss-based models (lower is better) are negated by the scorers.
"""

import inspect
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from mowl.evaluation.regime import EvaluationRegime
from mowl.geometric.normal_forms import (BOTTOM, NF1, NF2, NF3, NF4, NF6, NF7, TOP,
                                         is_nominal_name)

logger = logging.getLogger(__name__)

#: Normal forms evaluated by [MH26], in the order of the paper's tables.
NORMAL_FORMS = ("gci0", "gci1", "gci2", "gci3", "gci1_bot")

ENTAILED = "entailed"
CONTRADICTORY = "contradictory"
UNDETERMINED = "undetermined"
STATUSES = (ENTAILED, CONTRADICTORY, UNDETERMINED)

#: Position of the ranked ("tail") slot of each normal form (MRR, Hits@k, Adm@k).
TAIL_SLOT = {"gci0": 1, "gci1": 2, "gci2": 2, "gci3": 2, "gci1_bot": 1}

#: Metrics reported by :class:`FaithfulnessHierarchy` (the columns of [MH26], Tables 4-13,
#: plus Hits@1/100 and the four criteria as booleans).
FAITHFULNESS_METRICS = ("auc_pm", "adm@10", "fvr", "pair_fvr", "rho_p", "mrr",
                        "hits@1", "hits@10", "hits@100", "discriminative", "admissible",
                        "monotonic_faithful", "probabilistic_faithful")

WITNESS_NS = "http://mowl.borg/faithfulness/witness#"


def _check_nf(normal_form):
    if normal_form not in NORMAL_FORMS:
        raise ValueError(f"Unknown normal form {normal_form!r}. Use one of {NORMAL_FORMS}.")


# --------------------------------------------------------------------------------------
# Axiom conversion
# --------------------------------------------------------------------------------------

def to_normal_form_axiom(normal_form, axiom):
    """Convert a candidate tuple into a :mod:`mowl.geometric` normal-form axiom.

    :param normal_form: One of :data:`NORMAL_FORMS`.
    :type normal_form: str
    :param axiom: Candidate tuple (see the module docstring).
    :type axiom: tuple
    :rtype: :class:`~mowl.geometric.normal_forms.NF1` ... :class:`~mowl.geometric.normal_forms.NF4`
    """
    _check_nf(normal_form)
    if normal_form == "gci0":
        return NF1(axiom[0], axiom[1])
    if normal_form == "gci1":
        return NF2(axiom[0], axiom[1], axiom[2])
    if normal_form == "gci2":
        return NF3(axiom[0], axiom[1], axiom[2])
    if normal_form == "gci3":
        return NF4(axiom[0], axiom[1], axiom[2])
    return NF2(axiom[0], axiom[1], BOTTOM)


def from_normal_form_axiom(axiom):
    """Inverse of :func:`to_normal_form_axiom`. Returns ``(normal_form, tuple)`` or ``None``
    for axioms outside the five normal forms (role axioms, :math:`C \\sqsubseteq \\bot`,
    :math:`\\exists r.C \\sqsubseteq \\bot`)."""
    if isinstance(axiom, NF1):
        if axiom.sup == BOTTOM or axiom.sub == BOTTOM:
            return None
        return "gci0", (axiom.sub, axiom.sup)
    if isinstance(axiom, NF2):
        if axiom.sup == BOTTOM:
            return "gci1_bot", (axiom.sub1, axiom.sub2)
        return "gci1", (axiom.sub1, axiom.sub2, axiom.sup)
    if isinstance(axiom, NF3):
        return "gci2", (axiom.sub, axiom.role, axiom.filler)
    if isinstance(axiom, NF4):
        if axiom.sup == BOTTOM:
            return None
        return "gci3", (axiom.role, axiom.filler, axiom.sup)
    return None


def _iri(owl_object):
    return str(owl_object.toStringID())


def axiom_from_owl(owl_axiom):
    """Read one OWL API axiom that is already in normal form.

    Recognises ``SubClassOf`` axioms of the shapes of :data:`NORMAL_FORMS` (with
    ``owl:Nothing`` on the right for ``gci1_bot``) and binary ``DisjointClasses`` of named
    classes (``gci1_bot``).

    :return: ``(normal_form, tuple)``, or ``None`` if the axiom has another shape.
    """
    kind = str(owl_axiom.getAxiomType())
    if kind == "DisjointClasses":
        ops = list(owl_axiom.getClassExpressionsAsList())
        if len(ops) == 2 and all(str(o.getClassExpressionType()) == "Class" for o in ops):
            return "gci1_bot", (_iri(ops[0]), _iri(ops[1]))
        return None
    if kind != "SubClassOf":
        return None
    sub, sup = owl_axiom.getSubClass(), owl_axiom.getSuperClass()
    st, pt = str(sub.getClassExpressionType()), str(sup.getClassExpressionType())
    if pt != "Class":
        if st == "Class" and pt == "ObjectSomeValuesFrom":
            filler = sup.getFiller()
            if str(filler.getClassExpressionType()) == "Class":
                return "gci2", (_iri(sub), _iri(sup.getProperty()), _iri(filler))
        return None
    d = _iri(sup)
    if st == "Class":
        if d == BOTTOM:
            return None
        return "gci0", (_iri(sub), d)
    if st == "ObjectIntersectionOf":
        ops = list(sub.getOperandsAsList())
        if len(ops) != 2 or not all(str(o.getClassExpressionType()) == "Class" for o in ops):
            return None
        if d == BOTTOM:
            return "gci1_bot", (_iri(ops[0]), _iri(ops[1]))
        return "gci1", (_iri(ops[0]), _iri(ops[1]), d)
    if st == "ObjectSomeValuesFrom":
        filler = sub.getFiller()
        if str(filler.getClassExpressionType()) != "Class" or d == BOTTOM:
            return None
        return "gci3", (_iri(sub.getProperty()), _iri(filler), d)
    return None


def axioms_from_owl(ontology):
    """Collect the normal-form axioms of an OWL API ontology (imports included).

    :return: ``{normal_form: [tuple, ...]}`` in a deterministic order; axioms of other
        shapes are skipped.
    :rtype: dict
    """
    from org.semanticweb.owlapi.model.parameters import Imports
    out = defaultdict(list)
    for ax in ontology.getAxioms(Imports.INCLUDED):
        parsed = axiom_from_owl(ax)
        if parsed is not None:
            out[parsed[0]].append(parsed[1])
    return {nf: sorted(set(v)) for nf, v in out.items()}


def _signature(axioms):
    concepts, roles = set(), set()
    for ax in axioms:
        if isinstance(ax, NF1):
            concepts.update((ax.sub, ax.sup))
        elif isinstance(ax, NF2):
            concepts.update((ax.sub1, ax.sub2, ax.sup))
        elif isinstance(ax, NF3):
            concepts.update((ax.sub, ax.filler))
            roles.add(ax.role)
        elif isinstance(ax, NF4):
            concepts.update((ax.filler, ax.sup))
            roles.add(ax.role)
        elif isinstance(ax, NF6):
            roles.update((ax.sub, ax.sup))
        elif isinstance(ax, NF7):
            roles.update((ax.r1, ax.r2, ax.sup))
    return concepts, roles


def _canonical(normal_form, axiom):
    """Unordered conjunctions are stored with sorted conjuncts."""
    if normal_form == "gci1":
        c, d = sorted(axiom[:2])
        return (c, d, axiom[2])
    if normal_form == "gci1_bot":
        return tuple(sorted(axiom))
    return tuple(axiom)


# --------------------------------------------------------------------------------------
# Logical status
# --------------------------------------------------------------------------------------

class LogicalStatusOracle:
    """Decides whether a candidate normal-form axiom is entailed by, contradicts, or is
    undetermined with respect to a normalised :math:`\\mathcal{EL}^{++}` ontology
    :math:`\\mathcal{O}` ([MH26], Definition 1).

    All decisions use the worklist saturation of
    :class:`mowl.geometric.completion.Completion` (ELK calculus, sound and complete for
    the normal forms of :mod:`mowl.geometric`; mOWL's bundled ELK/HermiT through the OWL
    API are not used, so no ontology is modified on the Java side).

    **Entailed.** :math:`\\mathcal{O} \\models \\alpha`, decided by one saturation of
    :math:`\\mathcal{O}` extended with fresh query names: :math:`C \\sqsubseteq D` iff
    :math:`D \\in S(C)`; :math:`C \\sqcap D \\sqsubseteq E` iff :math:`E \\in S(Z)` for a
    fresh :math:`Z \\sqsubseteq C, Z \\sqsubseteq D`; :math:`C \\sqsubseteq \\exists r.D`
    iff :math:`Y \\in S(C)` for a fresh :math:`\\exists r.D \\sqsubseteq Y`;
    :math:`\\exists r.C \\sqsubseteq D` iff :math:`D \\in S(X)` for a fresh
    :math:`X \\sqsubseteq \\exists r.C`. These definitorial extensions are conservative,
    so the decisions are exact. An axiom whose left-hand side is unsatisfiable is
    entailed (vacuously).

    **Contradictory.** Read literally, Definition 1 calls :math:`\\alpha` contradictory iff
    :math:`\\mathcal{O} \\cup \\{\\alpha\\}` is inconsistent. For a TBox without
    individuals this never happens (every :math:`\\mathcal{EL}^{++}` TBox has the model in
    which all concepts are empty), so the paper's test sets are generated under a
    non-emptiness assumption (reference code, ``elplus_count_with_negatives``, flags
    ``--assume-nonempty``; NF2/NF3/NF4/NF7 generators: "attested" intersections and
    existentials are assumed non-empty). We make this assumption explicit with a set
    :math:`W` of *witness* concepts, each assumed non-empty unless :math:`\\mathcal{O}`
    proves it empty, and call :math:`\\alpha` **contradictory** iff
    :math:`\\mathcal{O} \\cup \\{\\alpha\\}` is inconsistent or makes some
    :math:`X \\in W` that is satisfiable w.r.t. :math:`\\mathcal{O}` unsatisfiable,
    i.e. iff :math:`\\mathcal{O} \\cup \\{\\alpha\\} \\cup \\{X(w_X) : X \\in W\\}` is
    inconsistent. This is decided exactly by saturating
    :math:`\\mathcal{O} \\cup \\{\\alpha\\}` once per candidate. ``witnesses`` selects
    :math:`W`:

    * ``"paper"`` (default): every concept name of :math:`\\mathcal{O}` (including
      auxiliary names of the normalisation), every conjunction :math:`A_1 \\sqcap A_2`
      occurring on the left of an NF2 axiom and every existential :math:`\\exists r.A`
      occurring in an NF3 or NF4 axiom of :math:`\\mathcal{O}` -- the concepts the
      reference generators assume non-empty. The reference generators are rule-based and
      may miss some contradictions that this decision procedure finds; the set it decides
      is a superset of theirs.
    * ``"names"``: only the concept names.
    * ``"none"``: no assumption; Definition 1 read literally (classical consistency of
      :math:`\\mathcal{O} \\cup \\{\\alpha\\}`, which uses the ABox, i.e. nominals).

    **Undetermined.** Neither entailed nor contradictory.

    :param axioms: Normal-form axioms of :math:`\\mathcal{O}`
        (:mod:`mowl.geometric.normal_forms` dataclasses).
    :type axioms: list
    :param concept_names: Additional concept names of the signature.
    :type concept_names: set, optional
    :param role_names: Additional role names of the signature.
    :type role_names: set, optional
    :param witnesses: ``"paper"``, ``"names"`` or ``"none"`` (see above).
    :type witnesses: str, optional
    :param classes: Named classes over which candidates are built. Defaults to all concept
        names except ``owl:Thing``, ``owl:Nothing`` and nominals.
    :type classes: iterable, optional
    :raises mowl.geometric.errors.InconsistentOntology: if :math:`\\mathcal{O}` is
        inconsistent (the trichotomy is defined for consistent ontologies only).
    """

    WITNESS_MODES = ("paper", "names", "none")

    def __init__(self, axioms, concept_names=None, role_names=None, witnesses="paper",
                 classes=None):
        from mowl.geometric.errors import InconsistentOntology
        if witnesses not in self.WITNESS_MODES:
            raise ValueError(f"witnesses must be one of {self.WITNESS_MODES}")
        self.axioms = list(axioms)
        self.witnesses = witnesses
        concepts, roles = _signature(self.axioms)
        self.concept_names = concepts | set(concept_names or ()) | {TOP, BOTTOM}
        self.role_names = roles | set(role_names or ())
        if classes is None:
            classes = (c for c in self.concept_names
                       if c not in (TOP, BOTTOM) and not is_nominal_name(c))
        self.classes = sorted(set(classes))
        self.roles = sorted(self.role_names)
        self.asserted = defaultdict(set)
        for ax in self.axioms:
            parsed = from_normal_form_axiom(ax)
            if parsed is not None:
                self.asserted[parsed[0]].add(_canonical(*parsed))

        # Witnesses: definitorial fresh names for attested conjunctions and existentials.
        self._witness_axioms = []
        checks = {TOP} | {c for c in self.concept_names if is_nominal_name(c)}
        if witnesses in ("paper", "names"):
            checks |= {c for c in self.concept_names if c not in (TOP, BOTTOM)}
        self.attested_conjunctions = sorted({
            tuple(sorted((ax.sub1, ax.sub2))) for ax in self.axioms
            if isinstance(ax, NF2) and ax.sub1 != ax.sub2})
        self.attested_existentials = sorted({
            (ax.role, ax.filler) for ax in self.axioms if isinstance(ax, (NF3, NF4))})
        if witnesses == "paper":
            for i, (a, b) in enumerate(self.attested_conjunctions):
                w = f"{WITNESS_NS}and{i}"
                self._witness_axioms += [NF1(w, a), NF1(w, b)]
                checks.add(w)
            for i, (r, a) in enumerate(self.attested_existentials):
                w = f"{WITNESS_NS}some{i}"
                self._witness_axioms.append(NF3(w, r, a))
                checks.add(w)
        self._base_axioms = self.axioms + self._witness_axioms
        self._base_concepts = self.concept_names | checks
        self._base = self._saturate(self._base_axioms, self._base_concepts, self.role_names)
        if self._base.is_inconsistent() or any(
                self._base.is_unsatisfiable(c) for c in checks if is_nominal_name(c)):
            raise InconsistentOntology("The trichotomy needs a consistent ontology.")
        self._checks = sorted(c for c in checks if not self._base.is_unsatisfiable(c))

    @staticmethod
    def _saturate(axioms, concepts, roles):
        import mowl.datasets  # noqa: F401  (import order: avoids a circular import)
        from mowl.geometric.completion import Completion
        return Completion(axioms, set(concepts), set(roles)).saturate()

    @classmethod
    def from_normalised(cls, normalised, **kwargs):
        """Build from a :class:`~mowl.geometric.normal_forms.NormalisedOntology`."""
        return cls(normalised.axioms, normalised.concept_names, normalised.role_names,
                   **kwargs)

    @classmethod
    def from_ontology(cls, ontology, **kwargs):
        """Normalise an OWL API ontology (:func:`mowl.geometric.normal_forms.normalise`)
        and build the oracle. Unless ``classes`` is given, candidates range over the class
        names of the ontology itself (auxiliary normalisation names are excluded)."""
        import mowl.datasets  # noqa: F401
        from mowl.geometric.normal_forms import normalise
        from org.semanticweb.owlapi.model.parameters import Imports
        if "classes" not in kwargs:
            kwargs["classes"] = {str(c.toStringID())
                                 for c in ontology.getClassesInSignature(Imports.INCLUDED)
                                 if str(c.toStringID()) not in (TOP, BOTTOM)}
        return cls.from_normalised(normalise(ontology), **kwargs)

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        """Build from the training ontology of a :class:`mowl.datasets.Dataset`."""
        return cls.from_ontology(dataset.ontology, **kwargs)

    # ------------------------------------------------------------------ entailment
    def _lhs_and_rhs(self, normal_form, axiom, fresh):
        """Return (extra axioms, lhs name, rhs name) such that O |= alpha iff
        rhs in S(lhs) (or lhs unsatisfiable)."""
        if normal_form == "gci0":
            return [], axiom[0], axiom[1]
        if normal_form in ("gci1", "gci1_bot"):
            key = ("and",) + tuple(sorted(axiom[:2]))
            if key not in fresh:
                z = f"{WITNESS_NS}q{len(fresh)}"
                fresh[key] = (z, [NF1(z, axiom[0]), NF1(z, axiom[1])])
            z = fresh[key][0]
            return [], z, (BOTTOM if normal_form == "gci1_bot" else axiom[2])
        if normal_form == "gci2":
            key = ("all", axiom[1], axiom[2])
            if key not in fresh:
                y = f"{WITNESS_NS}q{len(fresh)}"
                fresh[key] = (y, [NF4(axiom[1], axiom[2], y)])
            return [], axiom[0], fresh[key][0]
        key = ("some", axiom[0], axiom[1])
        if key not in fresh:
            x = f"{WITNESS_NS}q{len(fresh)}"
            fresh[key] = (x, [NF3(x, axiom[0], axiom[1])])
        return [], fresh[key][0], axiom[2]

    def entailment(self, normal_form, axioms):
        """Decide :math:`\\mathcal{O} \\models \\alpha` for a batch of candidates with one
        saturation.

        :return: Two lists of booleans: entailed, and vacuous (left-hand side
            unsatisfiable; for ``gci1_bot``: one conjunct unsatisfiable).
        """
        _check_nf(normal_form)
        axioms = [tuple(a) for a in axioms]
        fresh = {}
        pairs = [self._lhs_and_rhs(normal_form, a, fresh)[1:] for a in axioms]
        extra = [ax for _, axs in fresh.values() for ax in axs]
        role_slot = {"gci2": 1, "gci3": 0}.get(normal_form)
        names = {n for a in axioms for i, n in enumerate(a) if i != role_slot}
        roles = {a[1] for a in axioms} if normal_form == "gci2" else (
            {a[0] for a in axioms} if normal_form == "gci3" else set())
        if extra or not names <= self._base_concepts or not roles <= self.role_names:
            res = self._saturate(self._base_axioms + extra,
                                 self._base_concepts.union(
                                     names, (z for z, _ in fresh.values())),
                                 self.role_names | roles)
        else:
            res = self._base
        ent = [res.entails(lhs, rhs) for lhs, rhs in pairs]
        if normal_form == "gci1_bot":
            # The empty conjunction is what the axiom states; it is vacuous only if a
            # conjunct is empty on its own.
            unsat = [res.is_unsatisfiable(a[0]) or res.is_unsatisfiable(a[1])
                     for a in axioms]
        else:
            unsat = [res.is_unsatisfiable(lhs) for lhs, _ in pairs]
        return ent, unsat

    def is_entailed(self, normal_form, axiom):
        """:math:`\\mathcal{O} \\models \\alpha`."""
        return self.entailment(normal_form, [axiom])[0][0]

    def is_contradictory(self, normal_form, axiom):
        """:math:`\\alpha` makes :math:`\\mathcal{O}` inconsistent or empties a witness
        concept (see the class docstring)."""
        _check_nf(normal_form)
        if self.is_entailed(normal_form, axiom):
            return False
        return self._contradicts(normal_form, tuple(axiom))

    def _contradicts(self, normal_form, axiom):
        geo = to_normal_form_axiom(normal_form, axiom)
        concepts, roles = _signature([geo])
        res = self._saturate(self._base_axioms + [geo], self._base_concepts | concepts,
                             self.role_names | roles)
        if res.is_inconsistent():
            return True
        return any(res.is_unsatisfiable(c) for c in self._checks)

    def classify(self, normal_form, axioms):
        """Logical status of each candidate.

        :return: One of :data:`ENTAILED`, :data:`CONTRADICTORY`, :data:`UNDETERMINED` per
            axiom.
        :rtype: list
        """
        axioms = [tuple(a) for a in axioms]
        ent, _ = self.entailment(normal_form, axioms)
        return [ENTAILED if e else (CONTRADICTORY if self._contradicts(normal_form, a)
                                    else UNDETERMINED)
                for a, e in zip(axioms, ent)]

    def status(self, normal_form, axiom):
        """Logical status of one candidate (see :meth:`classify`)."""
        return self.classify(normal_form, [axiom])[0]


# --------------------------------------------------------------------------------------
# Test-set generation
# --------------------------------------------------------------------------------------

@dataclass
class FaithfulnessTestSet:
    """Entailed (:math:`T_+`), contradictory (:math:`T_-`) and undetermined
    (:math:`T_?`) test axioms of one normal form.

    :param normal_form: One of :data:`NORMAL_FORMS`.
    :param entailed: Axiom tuples in :math:`T_+`.
    :param contradictory: Axiom tuples in :math:`T_-` (may be empty: the GO-plus slice of
        [MH26] has none for GCI1--GCI3).
    :param undetermined: Axiom tuples in :math:`T_?`.
    :param p_values: Optional ``{axiom tuple: P(alpha | O)}`` (relative model counts) for
        :math:`T_?`, needed for :math:`\\rho_P`.
    """

    normal_form: str
    entailed: list = field(default_factory=list)
    contradictory: list = field(default_factory=list)
    undetermined: list = field(default_factory=list)
    p_values: dict = None

    def __post_init__(self):
        _check_nf(self.normal_form)
        self.entailed = [tuple(a) for a in self.entailed]
        self.contradictory = [tuple(a) for a in self.contradictory]
        self.undetermined = [tuple(a) for a in self.undetermined]
        if self.p_values is not None:
            self.p_values = {_canonical(self.normal_form, k): float(v)
                             for k, v in self.p_values.items()}

    def p_value(self, axiom):
        """:math:`P(\\alpha \\mid \\mathcal{O})` of an axiom (conjunct order ignored), or
        ``nan`` if unknown."""
        if not self.p_values:
            return float("nan")
        return self.p_values.get(_canonical(self.normal_form, axiom), float("nan"))

    @classmethod
    def from_owl(cls, normal_form, entailed=None, contradictory=None, undetermined=None,
                 p_values=None):
        """Build a test set from OWL API ontologies (or ``None``) holding the axioms of
        each class; axioms of other normal forms are ignored."""
        def read(ont):
            return [] if ont is None else axioms_from_owl(ont).get(normal_form, [])
        return cls(normal_form, read(entailed), read(contradictory), read(undetermined),
                   p_values)


def _candidate_pools(oracle, normal_form, classes, roles, attested_only):
    pairs = [(a, b) for i, a in enumerate(classes) for b in classes[i + 1:]]
    if normal_form == "gci0":
        return [classes, classes], lambda x: (x[0], x[1])
    if normal_form == "gci1_bot":
        return [pairs], lambda x: x[0]
    if normal_form == "gci1":
        lhs = ([p for p in oracle.attested_conjunctions] if attested_only else pairs)
        return [lhs, classes], lambda x: (x[0][0], x[0][1], x[1])
    exist = (list(oracle.attested_existentials) if attested_only
             else [(r, c) for r in roles for c in classes])
    exist = [e for e in exist if e[1] not in (TOP, BOTTOM)]
    if normal_form == "gci2":
        return [classes, exist], lambda x: (x[0], x[1][0], x[1][1])
    return [exist, classes], lambda x: (x[0][0], x[0][1], x[1])


def _trivial(normal_form, ax):
    if normal_form == "gci0":
        return ax[0] == ax[1]
    if normal_form == "gci1":
        return ax[2] in ax[:2] or ax[0] == ax[1]
    if normal_form == "gci1_bot":
        return ax[0] == ax[1]
    return False


def generate_test_sets(oracle, normal_forms=NORMAL_FORMS, n_entailed=1000,
                       n_contradictory=1000, n_undetermined=1000, classes=None,
                       roles=None, attested_only=True, exclude_asserted=True,
                       exclude=None, max_candidates=50000, seed=0, batch_size=512):
    """Sample :math:`T_+, T_-, T_?` per normal form by classifying random candidates with
    a :class:`LogicalStatusOracle` ([MH26], Section 6.1: "We sample test sets
    :math:`T_+ \\subset S_{\\models}`, :math:`T_? \\subset S_?`,
    :math:`T_- \\subset S_\\bot` via the reasoner").

    Candidate space, following the pools of the reference generators: ``gci0``: all pairs
    of classes; ``gci1_bot``: all unordered pairs; ``gci1``: left-hand conjunctions that
    occur in the ontology times all classes; ``gci2``: classes times existentials
    :math:`\\exists r.D` occurring in the ontology; ``gci3``: existentials
    :math:`\\exists r.C` occurring in the ontology times classes. With
    ``attested_only=False`` all conjunctions / role-class pairs are used. Trivial
    candidates (:math:`C \\sqsubseteq C`, :math:`C \\sqcap D \\sqsubseteq C`,
    :math:`C \\sqcap C \\sqsubseteq \\bot`), candidates with an unsatisfiable left-hand
    side (vacuously entailed), asserted axioms (if ``exclude_asserted``) and ``exclude``
    are skipped. Candidates are drawn uniformly without replacement until every quota is
    filled, the space is exhausted or ``max_candidates`` were examined; so each
    :math:`T` is a uniform sample of its class within the examined candidates.

    :param oracle: The oracle of the training ontology.
    :type oracle: :class:`LogicalStatusOracle`
    :param n_entailed: Size of :math:`T_+` (``None``: unlimited). Likewise
        ``n_contradictory`` and ``n_undetermined``.
    :param classes: Candidate classes (default ``oracle.classes``).
    :param roles: Candidate roles (default ``oracle.roles``; used if
        ``attested_only=False``).
    :param exclude: ``{normal_form: iterable of tuples}`` never to be used.
    :param max_candidates: Maximum number of candidates examined per normal form.
    :param seed: Random seed.
    :return: ``{normal_form: FaithfulnessTestSet}``
    :rtype: dict
    """
    rng = random.Random(seed)
    classes = sorted(oracle.classes if classes is None else classes)
    roles = sorted(oracle.roles if roles is None else roles)
    quota = {ENTAILED: n_entailed, CONTRADICTORY: n_contradictory,
             UNDETERMINED: n_undetermined}
    out = {}
    for nf in normal_forms:
        _check_nf(nf)
        skip = set()
        if exclude_asserted:
            skip |= oracle.asserted.get(nf, set())
        if exclude and nf in exclude:
            skip |= {_canonical(nf, a) for a in exclude[nf]}
        pools, build = _candidate_pools(oracle, nf, classes, roles, attested_only)
        sizes = [len(p) for p in pools]
        total = int(np.prod(sizes)) if sizes else 0
        found = {s: [] for s in STATUSES}

        def full(s):
            return quota[s] is not None and len(found[s]) >= quota[s]

        order = rng.sample(range(total), min(total, max_candidates)) if total else []
        for start in range(0, len(order), batch_size):
            if all(full(s) for s in STATUSES):
                break
            batch = []
            for idx in order[start:start + batch_size]:
                parts = []
                for size, pool in zip(reversed(sizes), reversed(pools)):
                    idx, rem = divmod(idx, size)
                    parts.append(pool[rem])
                ax = _canonical(nf, build(parts[::-1]))
                if not _trivial(nf, ax) and ax not in skip:
                    batch.append(ax)
            if not batch:
                continue
            ent, unsat = oracle.entailment(nf, batch)
            for ax, e, u in zip(batch, ent, unsat):
                if u:
                    continue
                if e:
                    if not full(ENTAILED):
                        found[ENTAILED].append(ax)
                    continue
                if full(CONTRADICTORY) and full(UNDETERMINED):
                    continue
                s = CONTRADICTORY if oracle._contradicts(nf, ax) else UNDETERMINED
                if not full(s):
                    found[s].append(ax)
        out[nf] = FaithfulnessTestSet(nf, found[ENTAILED], found[CONTRADICTORY],
                                      found[UNDETERMINED])
    return out


# --------------------------------------------------------------------------------------
# Relative model counting
# --------------------------------------------------------------------------------------

class _VarMap:
    def __init__(self):
        self.ids = {}

    def __call__(self, key):
        if key not in self.ids:
            self.ids[key] = len(self.ids) + 1
        return self.ids[key]


def herbrand_clauses(axioms, domain, var):
    """CNF encoding of normal-form axioms over a finite Herbrand domain ([MH26],
    Section 4.2 and Appendix A; the encoding of the reference implementation,
    ``elplus_count/encode.py``).

    :math:`A \\sqsubseteq B` gives :math:`\\neg A(a) \\lor B(a)`;
    :math:`A_1 \\sqcap A_2 \\sqsubseteq B` gives :math:`\\neg A_1(a) \\lor \\neg A_2(a) \\lor
    B(a)`; :math:`\\exists r.A \\sqsubseteq B` gives :math:`\\neg r(a,b) \\lor \\neg A(b)
    \\lor B(a)`; :math:`A \\sqsubseteq \\exists r.B` gives :math:`\\neg A(a) \\lor
    \\bigvee_b y_{ab}` with the Tseitin biconditional :math:`y_{ab} \\leftrightarrow
    r(a,b) \\land B(b)` (so auxiliary variables do not change the count); role inclusions
    and chains are encoded likewise. :math:`\\top` literals are dropped (true),
    :math:`\\bot` literals are dropped (false). Nominal concepts :math:`N_a` (ABox
    individuals) are interpreted as :math:`\\{a\\}`.

    :param axioms: :mod:`mowl.geometric.normal_forms` axioms.
    :param domain: Domain element names.
    :param var: Callable mapping ``("c", A, a)``, ``("r", r, a, b)`` and ``("y", ...)``
        keys to positive SAT variable ids.
    :return: List of clauses (tuples of non-zero ints), or ``None`` if a clause is empty
        (unsatisfiable).
    """
    def c(name, a):
        return var(("c", name, a))

    def r(role, a, b):
        return var(("r", role, a, b))

    clauses = []

    def add(lits):
        # lits: list of int or True (satisfied literal); False literals are omitted.
        if any(x is True for x in lits):
            return
        clauses.append(tuple(lits))

    def pos(name, a):
        if name == TOP:
            return True
        if name == BOTTOM:
            return None
        return c(name, a)

    def neg(name, a):
        if name == BOTTOM:
            return True
        if name == TOP:
            return None
        return -c(name, a)

    def lits(*xs):
        return [x for x in xs if x is not None]

    for ax in axioms:
        if isinstance(ax, NF1):
            for a in domain:
                add(lits(neg(ax.sub, a), pos(ax.sup, a)))
        elif isinstance(ax, NF2):
            for a in domain:
                add(lits(neg(ax.sub1, a), neg(ax.sub2, a), pos(ax.sup, a)))
        elif isinstance(ax, NF3):
            for a in domain:
                disj = [neg(ax.sub, a)]
                for b in domain:
                    if ax.filler == BOTTOM:
                        continue
                    if ax.filler == TOP:
                        disj.append(r(ax.role, a, b))
                        continue
                    y = var(("y", ax, a, b))
                    rab, fb = r(ax.role, a, b), c(ax.filler, b)
                    clauses.extend([(-y, rab), (-y, fb), (-rab, -fb, y)])
                    disj.append(y)
                add(lits(*disj))
        elif isinstance(ax, NF4):
            for a in domain:
                for b in domain:
                    add(lits(-r(ax.role, a, b), neg(ax.filler, b), pos(ax.sup, a)))
        elif isinstance(ax, NF6):
            for a in domain:
                for b in domain:
                    add([-r(ax.sub, a, b), r(ax.sup, a, b)])
        elif isinstance(ax, NF7):
            for a in domain:
                for b in domain:
                    for e in domain:
                        add([-r(ax.r1, a, b), -r(ax.r2, b, e), r(ax.sup, a, e)])
    if any(len(cl) == 0 for cl in clauses):
        return None
    return clauses


def _reduce(clauses, lit):
    out = []
    for cl in clauses:
        if lit in cl:
            continue
        if -lit in cl:
            cl = cl - {-lit}
            if not cl:
                return None
        out.append(cl)
    return out


class _PythonCounter:
    """Exact #SAT by DPLL with unit propagation, connected-component decomposition and
    component caching. Pure Python; adequate for small Herbrand bases (tens to a few
    hundred variables per connected component)."""

    def __init__(self):
        self.cache = {}

    def count(self, clauses, variables):
        clauses = [frozenset(cl) for cl in clauses]
        clauses = [cl for cl in clauses if not any(-x in cl for x in cl)]
        return self._count(clauses, frozenset(variables))

    def _count(self, clauses, scope):
        assigned = set()
        while True:
            unit = next((cl for cl in clauses if len(cl) == 1), None)
            if unit is None:
                break
            (lit,) = unit
            assigned.add(abs(lit))
            clauses = _reduce(clauses, lit)
            if clauses is None:
                return 0
        used = set()
        for cl in clauses:
            used.update(abs(x) for x in cl)
        result = 1 << (len(scope) - len(assigned) - len(used))
        for comp in self._components(clauses):
            result *= self._component(comp)
            if result == 0:
                return 0
        return result

    @staticmethod
    def _components(clauses):
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for cl in clauses:
            vs = [abs(x) for x in cl]
            for v in vs[1:]:
                ra, rb = find(vs[0]), find(v)
                if ra != rb:
                    parent[ra] = rb
        groups = defaultdict(list)
        for cl in clauses:
            groups[find(abs(next(iter(cl))))].append(cl)
        return list(groups.values())

    def _component(self, clauses):
        key = frozenset(clauses)
        if key in self.cache:
            return self.cache[key]
        freq = defaultdict(int)
        for cl in clauses:
            for x in cl:
                freq[abs(x)] += 1
        scope = frozenset(freq)
        v = max(freq, key=lambda x: (freq[x], -x))
        total = 0
        for lit in (v, -v):
            reduced = _reduce(clauses, lit)
            if reduced is not None:
                total += self._count(reduced, scope - {v})
        self.cache[key] = total
        return total


def count_models(clauses, variables, backend="auto"):
    """Exact number of assignments of ``variables`` satisfying ``clauses``.

    :param backend: ``"python"`` (built-in DPLL counter), ``"pyganak"`` (the Ganak exact
        counter used by [MH26] via the optional ``pyganak`` package), or ``"auto"``
        (``pyganak`` if importable, else ``"python"``). The ``pyganak`` path follows the
        reference implementation's use of its API and is not exercised by mOWL's tests.
    :rtype: int
    """
    variables = sorted(set(variables))
    if backend == "auto":
        try:
            import pyganak  # noqa: F401
            backend = "pyganak"
        except ImportError:
            backend = "python"
    if backend == "python":
        return _PythonCounter().count(clauses, variables)
    if backend == "pyganak":
        import pyganak  # noqa: F811
        remap = {v: i + 1 for i, v in enumerate(variables)}
        counter = pyganak.Counter()
        counter.new_vars(len(variables))
        counter.add_clauses([[remap[abs(x)] * (1 if x > 0 else -1) for x in cl]
                             for cl in clauses])
        return int(counter.count())
    raise ValueError(f"Unknown backend {backend!r}")


class RelativeModelCounter:
    """Relative model count :math:`P(\\alpha \\mid \\mathcal{O}) = \\#(\\mathcal{O} \\cup
    \\{\\alpha\\}) / \\#(\\mathcal{O})` over Herbrand interpretations of a bounded domain
    ([MH26], Eq. (1)).

    The domain consists of the ontology's individuals plus anonymous elements up to
    ``domain_size`` elements ([MH26] use :math:`|\\Delta| = 2`). The Herbrand base is
    built over the union of the signatures of :math:`\\mathcal{O}` and all candidates
    (atoms that occur in no clause cancel in the ratio). Connected components of the
    clause graph of :math:`\\mathcal{O}` that share no variable with :math:`\\alpha`
    also cancel exactly, so only the components touching :math:`\\alpha` are counted.

    Counting is #P-complete: the built-in counter is meant for small ontologies and
    tests; for ontologies of the size of the paper's (Pizza, GO-plus slice) install the
    optional ``pyganak`` package or use the precomputed values shipped with the
    datasets (:mod:`mowl.datasets.builtin.faithfulness`).

    :param axioms: Normal-form axioms of :math:`\\mathcal{O}`.
    :param domain_size: :math:`|\\Delta|`.
    :param backend: See :func:`count_models`.
    """

    def __init__(self, axioms, domain_size=2, backend="auto"):
        self.axioms = list(axioms)
        concepts, _ = _signature(self.axioms)
        individuals = sorted(c for c in concepts if is_nominal_name(c))
        if domain_size < len(individuals):
            raise ValueError("domain_size is smaller than the number of individuals.")
        self.domain = individuals + [f"_e{i}" for i in range(domain_size - len(individuals))]
        self.backend = backend
        self.var = _VarMap()
        base = herbrand_clauses(self.axioms, self.domain, self.var)
        # Nominals denote their own element.
        for n in individuals:
            for a in self.domain:
                base.append((self.var(("c", n, a)),) if a == n else
                            (-self.var(("c", n, a)),))
        self.clauses = base
        self._components = _PythonCounter._components([frozenset(c) for c in base])
        self._var_component = {}
        for i, comp in enumerate(self._components):
            for cl in comp:
                for x in cl:
                    self._var_component[abs(x)] = i
        self._count_cache = {}
        self._python = _PythonCounter()

    def _aux_ids(self):
        return {i for k, i in self.var.ids.items() if k[0] == "y"}

    def _count(self, clauses):
        variables = {abs(x) for cl in clauses for x in cl}
        if self.backend == "python" or (self.backend == "auto" and not _has_pyganak()):
            return self._python.count(clauses, variables)
        return count_models(clauses, variables, backend=self.backend)

    def relative_count(self, normal_form, axiom):
        """:math:`P(\\alpha \\mid \\mathcal{O})` as an exact :class:`fractions.Fraction`."""
        geo = to_normal_form_axiom(normal_form, axiom)
        extra = herbrand_clauses([geo], self.domain, self.var)
        if extra is None:
            return Fraction(0)
        touched = sorted({self._var_component[abs(x)] for cl in extra for x in cl
                          if abs(x) in self._var_component})
        base = [tuple(cl) for i in touched for cl in self._components[i]]
        key = tuple(touched)
        if key not in self._count_cache:
            self._count_cache[key] = self._count(base) if base else 1
        denom = self._count_cache[key]
        # Ground atoms of alpha that occur in no clause of O are free in O: the
        # denominator ranges over them too. Tseitin variables of alpha are not ground
        # atoms (they are functionally determined in the numerator) and are excluded.
        aux = self._aux_ids()
        base_vars = {abs(x) for cl in base for x in cl}
        new_vars = {abs(x) for cl in extra for x in cl}.difference(base_vars, aux)
        numer = self._count(base + list(extra))
        return Fraction(numer, denom * (1 << len(new_vars)))

    def __call__(self, normal_form, axioms):
        """Relative counts of several candidates, as floats."""
        return [float(self.relative_count(normal_form, a)) for a in axioms]


def _has_pyganak():
    try:
        import pyganak  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------------------

class ELModelScorer:
    """Scores candidate axioms with the normal-form losses of a trained
    :class:`~mowl.base_models.EmbeddingELModel` (``model.module(data, gci_name)``), which
    is how [MH26] score the logic-geometric models ("the evaluation score is the training
    loss itself", Appendix C). Plausibility is the negated loss. Axioms mentioning a
    class or role unknown to the model get ``nan`` and are dropped by the metrics (as the
    reference implementation drops out-of-vocabulary axioms).

    For ``gci1_bot`` the third tensor column holds the index of ``owl:Nothing`` if the
    model knows it, else the first class (the mOWL ``gci1_bot`` losses do not read it).

    :param model: Trained model.
    :param batch_size: Axioms per forward pass.
    """

    def __init__(self, model, batch_size=8192):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, normal_form, axioms):
        import torch as th
        model = self.model
        classes = model.class_index_dict
        roles = model.object_property_index_dict
        rows, keep = [], []
        for i, ax in enumerate(axioms):
            try:
                if normal_form == "gci2":
                    row = [classes[ax[0]], roles[ax[1]], classes[ax[2]]]
                elif normal_form == "gci3":
                    row = [roles[ax[0]], classes[ax[1]], classes[ax[2]]]
                elif normal_form == "gci1_bot":
                    row = [classes[ax[0]], classes[ax[1]], classes.get(BOTTOM, classes[ax[0]])]
                else:
                    row = [classes[x] for x in ax]
            except KeyError:
                continue
            rows.append(row)
            keep.append(i)
        out = np.full(len(axioms), np.nan)
        if not rows:
            return out
        module = model.module
        was_training = module.training
        module.eval()
        device = getattr(model, "device", "cpu")
        vals = []
        with th.no_grad():
            for s in range(0, len(rows), self.batch_size):
                data = th.tensor(rows[s:s + self.batch_size], dtype=th.long, device=device)
                loss = module(data, normal_form)
                vals.append(loss.reshape(len(data), -1).sum(dim=1).cpu().numpy())
        if was_training:
            module.train()
        out[keep] = -np.concatenate(vals)
        return out


class CallableScorer:
    """Wraps a user scoring function so that graph-based, language-model or any other
    models can be evaluated.

    :param score_fn: ``score_fn(axioms)`` or ``score_fn(normal_form, axioms)`` returning
        one number per axiom tuple (``nan`` for axioms it cannot score), or a dict
        ``{normal_form: score_fn(axioms)}``.
    :param higher_is_better: ``False`` for losses/distances (they are negated).
    """

    def __init__(self, score_fn, higher_is_better=True):
        self.score_fn = score_fn
        self.sign = 1.0 if higher_is_better else -1.0
        self._two_args = False
        if not isinstance(score_fn, dict):
            try:
                params = [p for p in inspect.signature(score_fn).parameters.values()
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
                self._two_args = len(params) >= 2
            except (TypeError, ValueError):
                self._two_args = False

    def __call__(self, normal_form, axioms):
        if isinstance(self.score_fn, dict):
            if normal_form not in self.score_fn:
                return np.full(len(axioms), np.nan)
            vals = self.score_fn[normal_form](list(axioms))
        elif self._two_args:
            vals = self.score_fn(normal_form, list(axioms))
        else:
            vals = self.score_fn(list(axioms))
        return self.sign * np.asarray(vals, dtype=float).reshape(-1)


def oracle_scorer(oracle, p_values=None):
    """Reference scorer from logical status: 1 for entailed, 0 for contradictory, and
    :math:`P(\\alpha\\mid\\mathcal{O})` (if ``p_values`` holds it) or 0.5 for undetermined
    axioms. It satisfies L0--L2 by construction, and L3 when ``p_values`` are given."""
    p_values = p_values or {}

    def score(normal_form, axioms):
        st = oracle.classify(normal_form, axioms)
        return np.array([1.0 if s == ENTAILED else 0.0 if s == CONTRADICTORY
                         else float(p_values.get(tuple(a), p_values.get(
                             _canonical(normal_form, a), 0.5)))
                         for a, s in zip(axioms, st)])
    return score


def as_scorer(model, higher_is_better=True):
    """Return a scorer ``(normal_form, axioms) -> plausibilities`` for ``model``: an
    :class:`ELModelScorer` for EL embedding models (anything with ``module`` and
    ``class_index_dict``), the model itself if it already has a ``score_axioms`` method,
    or a :class:`CallableScorer` for a callable or dict of callables."""
    if hasattr(model, "score_axioms"):
        return model.score_axioms
    if hasattr(model, "module") and hasattr(model, "class_index_dict"):
        return ELModelScorer(model)
    if callable(model) or isinstance(model, dict):
        return CallableScorer(model, higher_is_better=higher_is_better)
    raise TypeError("Cannot score axioms with this model; pass a score function.")


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def _clean(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[~np.isnan(x)]


def average_ranks(x):
    """Ranks 1..n of ``x`` in ascending order, ties receiving their average rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def discrimination_auc(pos, neg):
    """:math:`AUC_{+/-}` (L0): probability that a random entailed axiom is more plausible
    than a random contradictory one, ties counting 1/2 (Mann-Whitney). ``None`` if a set
    is empty."""
    pos, neg = _clean(pos), _clean(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = average_ranks(np.concatenate([pos, neg]))
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def faithfulness_violation_rate(neg, und, pos):
    """FVR (L2): :math:`\\Pr_{T_-\\times T_?\\times T_+}[\\neg(f(\\alpha^-) < f(\\alpha^?)
    < f(\\alpha^+))]`, computed exactly. ``None`` if a set is empty."""
    neg, und, pos = np.sort(_clean(neg)), _clean(und), np.sort(_clean(pos))
    if len(neg) == 0 or len(und) == 0 or len(pos) == 0:
        return None
    below = np.searchsorted(neg, und, side="left")             # #neg strictly below
    above = len(pos) - np.searchsorted(pos, und, side="right")  # #pos strictly above
    correct = float(np.sum(below.astype(float) * above))
    return 1.0 - correct / (len(neg) * len(und) * len(pos))


def pair_fvr(neg, und, strict=True):
    """Pair-FVR (L2, :math:`T_?` versus :math:`T_-`):
    :math:`\\Pr_{T_-\\times T_?}[f(\\alpha^?) \\le f(\\alpha^-)]` (with ``strict=False``:
    :math:`<`). 0.5 means :math:`T_?` is not separated from :math:`T_-`."""
    neg, und = np.sort(_clean(neg)), _clean(und)
    if len(neg) == 0 or len(und) == 0:
        return None
    side = "left" if strict else "right"
    correct = float(np.searchsorted(neg, und, side=side).sum())
    return 1.0 - correct / (len(neg) * len(und))


def spearman_rho(x, y):
    """Spearman correlation (Pearson on average ranks). ``None`` with fewer than two
    points or a constant input."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if den == 0:
        return None
    return float((rx * ry).sum() / den)


def is_discriminative(pos, neg):
    """Definition 2 on finite test sets: every entailed axiom outscores every
    contradictory one."""
    pos, neg = _clean(pos), _clean(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return bool(pos.min() > neg.max())


def is_admissible(neg, und, pos):
    """Definition 3: :math:`\\max_{T_-} f < \\min_{T_+\\cup T_?} f`."""
    neg, cons = _clean(neg), np.concatenate([_clean(pos), _clean(und)])
    if len(neg) == 0 or len(cons) == 0:
        return None
    return bool(neg.max() < cons.min())


def is_monotonic_faithful(neg, und, pos):
    """Definition 4: :math:`f(\\alpha^-) < f(\\alpha^?) < f(\\alpha^+)` for all triples."""
    neg, und, pos = _clean(neg), _clean(und), _clean(pos)
    if len(neg) == 0 or len(und) == 0 or len(pos) == 0:
        return None
    return bool(neg.max() < und.min() and und.max() < pos.min())


def is_probabilistic_faithful(scores, p):
    """Definition 5 on a finite set: :math:`P(\\alpha_1) > P(\\alpha_2) \\Rightarrow
    f(\\alpha_1) > f(\\alpha_2)` for all pairs."""
    scores, p = np.asarray(scores, dtype=float), np.asarray(p, dtype=float)
    ok = ~(np.isnan(scores) | np.isnan(p))
    scores, p = scores[ok], p[ok]
    if len(scores) < 2:
        return None
    order = np.argsort(p, kind="mergesort")
    s, q = scores[order], p[order]
    # For each distinct P level, all scores must exceed the max score of lower levels.
    best_below = -np.inf
    i = 0
    while i < len(q):
        j = i
        while j + 1 < len(q) and q[j + 1] == q[i]:
            j += 1
        if s[i:j + 1].min() <= best_below:
            return False
        best_below = max(best_below, s[i:j + 1].max())
        i = j + 1
    return True


def _group_queries(normal_form, axioms):
    slot = TAIL_SLOT[normal_form]
    groups = defaultdict(list)
    for ax in axioms:
        head = tuple(x for i, x in enumerate(ax) if i != slot)
        groups[head].append(ax[slot])
    return slot, groups


def _with_tail(head, slot, tail):
    h = list(head)
    h.insert(slot, tail)
    return tuple(h)


def ranking_metrics(scorer, normal_form, entailed, candidates, ks=(1, 10, 100)):
    """MRR and Hits@k of the entailed axioms: for each :math:`\\alpha \\in T_+` the ranked
    slot (the superclass; the second conjunct for ``gci1_bot``) is replaced by every
    candidate class, and the rank of the true one is taken among the plausibilities
    (descending; ties get the average rank; unfiltered, as in the reference code).

    :return: ``{"mrr": ..., "hits@k": ...}``; empty values are ``None``.
    """
    slot, groups = _group_queries(normal_form, entailed)
    candidates = list(candidates)
    pos = {c: i for i, c in enumerate(candidates)}
    rr, hits = [], {k: [] for k in ks}
    for head, tails in groups.items():
        scores = np.asarray(scorer(normal_form, [_with_tail(head, slot, c)
                                                 for c in candidates]), dtype=float)
        if np.all(np.isnan(scores)):
            continue
        ranks = average_ranks(np.where(np.isnan(scores), -np.inf, -scores))
        for t in tails:
            if t not in pos or np.isnan(scores[pos[t]]):
                continue
            rank = ranks[pos[t]]
            rr.append(1.0 / rank)
            for k in ks:
                hits[k].append(float(rank <= k))
    out = {"mrr": float(np.mean(rr)) if rr else None}
    for k in ks:
        out[f"hits@{k}"] = float(np.mean(hits[k])) if hits[k] else None
    return out


def admissibility_at_k(scorer, normal_form, entailed, candidates, contradictory, k=10):
    """Adm@k (L1): for each :math:`\\alpha \\in T_+` (one ranking query per axiom, as in
    the reference code), the fraction of the ``k`` most plausible completions of its
    query that are not contradictory, averaged over queries.

    :param contradictory: Either a collection of axiom tuples (membership in :math:`T_-`,
        as in the paper; then Adm@k is exact only if :math:`T_-` contains every
        contradictory candidate) or a callable ``(normal_form, axioms) -> list of bool``
        (for example from :meth:`LogicalStatusOracle.classify`) for an exact value.
    """
    slot, groups = _group_queries(normal_form, entailed)
    candidates = list(candidates)
    if callable(contradictory):
        is_contra = contradictory
    else:
        contra = {_canonical(normal_form, a) for a in contradictory}

        def is_contra(nf, axs):
            return [_canonical(nf, a) in contra for a in axs]
    fractions = []
    for head, tails in groups.items():
        axs = [_with_tail(head, slot, c) for c in candidates]
        scores = np.asarray(scorer(normal_form, axs), dtype=float)
        valid = np.flatnonzero(~np.isnan(scores))
        if len(valid) == 0:
            continue
        top = valid[np.argsort(-scores[valid], kind="mergesort")[:k]]
        flags = is_contra(normal_form, [axs[i] for i in top])
        frac = 1.0 - float(np.mean(flags))
        fractions.extend([frac] * len(tails))
    return float(np.mean(fractions)) if fractions else None


def evaluate_test_set(scorer, test_set, candidates, k=10, ks=(1, 10, 100),
                      contradictory=None):
    """All metrics of [MH26] on one :class:`FaithfulnessTestSet`.

    :param scorer: ``(normal_form, axioms) -> plausibilities`` (see :func:`as_scorer`).
    :param candidates: Candidate classes for the ranked slot (MRR, Hits@k, Adm@k).
    :param contradictory: Contradiction test for Adm@k (default: membership in
        :math:`T_-`).
    :return: ``{metric: value}`` with the keys of :data:`FAITHFULNESS_METRICS` (Adm@k under
        ``f"adm@{k}"``) and the sizes ``n_entailed``, ``n_contradictory``,
        ``n_undetermined`` of the scored sets.
    """
    nf = test_set.normal_form
    pos = np.asarray(scorer(nf, test_set.entailed), dtype=float) if test_set.entailed \
        else np.array([])
    neg = np.asarray(scorer(nf, test_set.contradictory), dtype=float) \
        if test_set.contradictory else np.array([])
    und = np.asarray(scorer(nf, test_set.undetermined), dtype=float) \
        if test_set.undetermined else np.array([])
    res = {
        "auc_pm": discrimination_auc(pos, neg),
        f"adm@{k}": (admissibility_at_k(
            scorer, nf, test_set.entailed, candidates,
            test_set.contradictory if contradictory is None else contradictory, k=k)
            if test_set.contradictory or contradictory is not None else None),
        "fvr": faithfulness_violation_rate(neg, und, pos),
        "pair_fvr": pair_fvr(neg, und),
        "rho_p": None,
        "discriminative": is_discriminative(pos, neg),
        "admissible": is_admissible(neg, und, pos),
        "monotonic_faithful": is_monotonic_faithful(neg, und, pos),
        "probabilistic_faithful": None,
    }
    if test_set.p_values:
        p = np.array([test_set.p_value(a) for a in test_set.undetermined], dtype=float)
        res["rho_p"] = spearman_rho(und, p)
        # Definition 5 over T+ (P = 1), T- (P = 0) and T? (Proposition 1).
        res["probabilistic_faithful"] = is_probabilistic_faithful(
            np.concatenate([pos, neg, und]),
            np.concatenate([np.ones(len(pos)), np.zeros(len(neg)), p]))
    res.update(ranking_metrics(scorer, nf, test_set.entailed, candidates, ks=ks))
    res["n_entailed"] = int(len(_clean(pos)))
    res["n_contradictory"] = int(len(_clean(neg)))
    res["n_undetermined"] = int(len(_clean(und)))
    return res


def evaluate_faithfulness(model, test_sets, candidates, k=10, ks=(1, 10, 100),
                          oracle=None, higher_is_better=True):
    """Evaluate a model on test sets of several normal forms.

    :param model: Anything accepted by :func:`as_scorer`.
    :param test_sets: ``{normal_form: FaithfulnessTestSet}``.
    :param candidates: Candidate classes for the ranked slot.
    :param oracle: If given, Adm@k decides contradiction exactly with
        :meth:`LogicalStatusOracle.classify` instead of membership in :math:`T_-`.
    :return: ``{normal_form: {metric: value}}``
    """
    scorer = as_scorer(model, higher_is_better=higher_is_better)

    def exact(nf, axs):
        return [s == CONTRADICTORY for s in oracle.classify(nf, axs)]
    contra = exact if oracle is not None else None
    return {nf: evaluate_test_set(scorer, ts, candidates, k=k, ks=ks,
                                  contradictory=contra)
            for nf, ts in test_sets.items()}


# --------------------------------------------------------------------------------------
# Model-ness of geometric models
# --------------------------------------------------------------------------------------

def loss_modelness(model, tol=1e-6, margin=0.0, normal_forms=None, datasets=None,
                   batch_size=8192):
    """Fraction of the training axioms of each normal form whose loss, computed with the
    margin set to ``margin`` (default 0: margin-free), is at most ``tol``.

    For a module whose loss for a normal form is zero exactly when the geometric
    interpretation satisfies the axiom (e.g. box inclusion losses), a fraction of 1.0
    means the embedding is a model of those axioms. This is not the case for every loss:
    the mOWL/reference ELEmbeddings GCI1 loss ignores the radius of the superclass
    ([MH26], Appendix C) and several modules add regularisers inside the per-axiom loss,
    so the report is a loss-level diagnostic; exact checks of exported region models are
    provided by :class:`mowl.geometric.ModelChecker`. (The multi-box modules'
    ``point_modelness`` measures point membership, a different notion.)

    :param model: An :class:`~mowl.base_models.EmbeddingELModel` (uses ``model.module``
        and, unless ``datasets`` is given, ``model.training_datasets``).
    :param tol: Numerical zero.
    :param margin: Value temporarily assigned to ``module.margin`` (if the module has
        that attribute; otherwise the losses are used as they are and
        ``"margin_overridden"`` is ``False``).
    :param normal_forms: Restrict to these GCI names.
    :param datasets: ``{gci_name: dataset with .data tensor, or tensor}``.
    :return: ``{gci_name: {"n": ..., "zero_loss": ..., "fraction": ...}, "all": {...},
        "margin_overridden": bool}``
    """
    import torch as th
    module = model.module
    datasets = model.training_datasets if datasets is None else datasets
    has_margin = hasattr(module, "margin")
    old = getattr(module, "margin", None)
    was_training = module.training
    report, n_all, z_all = {}, 0, 0
    try:
        if has_margin:
            module.margin = margin
        module.eval()
        with th.no_grad():
            for name, ds in datasets.items():
                if normal_forms is not None and name not in normal_forms:
                    continue
                data = getattr(ds, "data", ds)
                if data is None or len(data) == 0:
                    continue
                zero = 0
                for s in range(0, len(data), batch_size):
                    chunk = data[s:s + batch_size]
                    loss = module(chunk, name).reshape(len(chunk), -1).sum(dim=1)
                    zero += int((loss <= tol).sum().item())
                report[name] = {"n": int(len(data)), "zero_loss": zero,
                                "fraction": zero / len(data)}
                n_all += len(data)
                z_all += zero
    finally:
        if has_margin:
            module.margin = old
        if was_training:
            module.train()
    report["all"] = {"n": n_all, "zero_loss": z_all,
                     "fraction": (z_all / n_all) if n_all else None}
    report["margin_overridden"] = has_margin
    return report


# --------------------------------------------------------------------------------------
# Evaluation regimes
# --------------------------------------------------------------------------------------

class FaithfulnessHierarchy(EvaluationRegime):
    """The faithfulness hierarchy of [MH26] as an evaluation regime.

    Test sets are taken from the dataset if it provides them
    (``dataset.faithfulness_test_sets()``, as the datasets of
    :mod:`mowl.datasets.builtin.faithfulness` do), otherwise generated from the training
    ontology with :class:`LogicalStatusOracle` and :func:`generate_test_sets`; relative
    model counts for :math:`\\rho_P` are then computed with
    :class:`RelativeModelCounter` if ``count_models=True``.

    :param normal_forms: Normal forms (subsets) to evaluate.
    :param k: ``k`` of Adm@k.
    :param ks: ``k`` values of Hits@k.
    :param witnesses: Non-emptiness assumption of the oracle
        (:class:`LogicalStatusOracle`).
    :param n_per_class: Size of each generated test set.
    :param attested_only: See :func:`generate_test_sets`.
    :param max_candidates: See :func:`generate_test_sets`.
    :param count_models: Compute :math:`P(\\alpha\\mid\\mathcal{O})` for generated
        :math:`T_?` (exponential; small ontologies only).
    :param domain_size: Herbrand domain size for counting.
    :param exact_admissibility: Decide contradiction in Adm@k with the oracle instead of
        membership in :math:`T_-`.
    :param seed: Random seed of the test-set generation.
    """

    key = "faithfulness-hierarchy"
    metrics = FAITHFULNESS_METRICS
    subsets = NORMAL_FORMS

    def __init__(self, normal_forms=NORMAL_FORMS, k=10, ks=(1, 10, 100),
                 witnesses="paper", n_per_class=1000, attested_only=True,
                 max_candidates=50000, count_models=False, domain_size=2,
                 count_backend="auto", exact_admissibility=False, seed=0):
        self.normal_forms = tuple(normal_forms)
        self.subsets = self.normal_forms
        self.k, self.ks = k, tuple(ks)
        self.metrics = tuple(m if m != "adm@10" else f"adm@{k}" for m in FAITHFULNESS_METRICS)
        self.witnesses = witnesses
        self.n_per_class = n_per_class
        self.attested_only = attested_only
        self.max_candidates = max_candidates
        self.count_models = count_models
        self.domain_size = domain_size
        self.count_backend = count_backend
        self.exact_admissibility = exact_admissibility
        self.seed = seed

    def oracle(self, dataset):
        """The :class:`LogicalStatusOracle` of the dataset's training ontology."""
        return LogicalStatusOracle.from_dataset(dataset, witnesses=self.witnesses)

    def test_sets(self, dataset, oracle=None):
        """``{normal_form: FaithfulnessTestSet}`` for ``dataset``."""
        if hasattr(dataset, "faithfulness_test_sets"):
            sets = dataset.faithfulness_test_sets()
            return {nf: ts for nf, ts in sets.items() if nf in self.normal_forms}
        oracle = oracle or self.oracle(dataset)
        n = self.n_per_class
        sets = generate_test_sets(oracle, self.normal_forms, n, n, n,
                                  attested_only=self.attested_only,
                                  max_candidates=self.max_candidates, seed=self.seed)
        if self.count_models:
            counter = RelativeModelCounter(oracle.axioms, self.domain_size,
                                           backend=self.count_backend)
            for nf, ts in sets.items():
                ts.p_values = {a: float(counter.relative_count(nf, a))
                               for a in ts.undetermined}
        return sets

    def candidates(self, dataset, oracle=None):
        """Candidate classes of the ranked slot: the dataset's own
        ``faithfulness_candidates()`` if present, else the oracle's classes."""
        if hasattr(dataset, "faithfulness_candidates"):
            return dataset.faithfulness_candidates()
        return (oracle or self.oracle(dataset)).classes

    def evaluate(self, model, dataset):
        """Evaluate ``model`` on ``dataset``.

        :return: ``{normal_form: {metric: value}}`` for the normal forms with a test set.
        """
        oracle = None
        if not hasattr(dataset, "faithfulness_test_sets") or self.exact_admissibility:
            oracle = self.oracle(dataset)
        sets = self.test_sets(dataset, oracle)
        cands = self.candidates(dataset, oracle)
        return evaluate_faithfulness(model, sets, cands, k=self.k, ks=self.ks,
                                     oracle=oracle if self.exact_admissibility else None)


class LossModelness(EvaluationRegime):
    """:func:`loss_modelness` as an evaluation regime: per normal form, the fraction of
    training axioms with (numerically) zero margin-free loss.

    :param tol: Numerical zero.
    :param margin: Margin used for the check.
    """

    key = "modelness"
    metrics = ("fraction", "zero_loss", "n")

    def __init__(self, tol=1e-6, margin=0.0):
        self.tol, self.margin = tol, margin
        self.subsets = ("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot",
                        "gci3_bot", "all")

    def evaluate(self, model, dataset=None):
        rep = loss_modelness(model, tol=self.tol, margin=self.margin)
        rep.pop("margin_overridden")
        return rep
