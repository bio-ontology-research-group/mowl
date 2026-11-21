"""Inference tasks: what a method is asked to decide or predict.

Three vocabularies describe the same things and are related here:

* the standard reasoning tasks of description logics (subsumption checking,
  classification, satisfiability, consistency, instance checking, realization, ...),
  :class:`InferenceTask`;
* the tasks of knowledge graph completion and knowledge base completion, which rank
  candidate axioms (link prediction, subsumption prediction, ...), :class:`CompletionTask`;
* the logical status of a candidate axiom with respect to the knowledge base, which says
  what a ranking *should* do with it, :class:`LogicalStatus`, and the faithfulness
  criteria built on it, :class:`FaithfulnessCriterion`.

Almost every task reduces to one subsumption test between two class expressions
(:attr:`InferenceTask.as_subsumption`), which is why subsumption between class expressions
is the common testing interface of mOWL.

This module does not need the JVM.
"""

import enum
from dataclasses import dataclass


class Box(enum.Enum):
    """The part of a knowledge base a task is about."""

    TBOX = "TBox"      #: concept inclusions (terminology)
    RBOX = "RBox"      #: role inclusions
    ABOX = "ABox"      #: assertions about individuals
    KB = "KB"          #: the knowledge base as a whole


class LogicalStatus(enum.Enum):
    """Status of a candidate axiom α with respect to a consistent knowledge base O under
    the open world assumption (Mashkova and Hoehndorf, 2026, Definition 1). Every candidate
    axiom has exactly one status."""

    ENTAILED = "entailed"              #: O ⊨ α: every model of O satisfies α
    CONTRADICTORY = "contradictory"    #: O ∪ {α} is inconsistent: no model of O satisfies α
    UNDETERMINED = "undetermined"      #: some models of O satisfy α and some do not


class FaithfulnessCriterion(enum.Enum):
    """Criteria on a scoring function f, each implied by the next (Mashkova and Hoehndorf,
    2026, Section 4)."""

    #: every entailed axiom scores above every contradictory one
    DISCRIMINATION = "discrimination"
    #: no contradictory axiom scores above an entailed or undetermined one
    ADMISSIBILITY = "logical admissibility"
    #: contradictory < undetermined < entailed
    MONOTONIC = "monotonic logical faithfulness"
    #: the score is monotone in the relative model count P(α | O)
    PROBABILISTIC = "probabilistic logical faithfulness"


@dataclass(frozen=True)
class InferenceTask:
    """A standard reasoning task.

    :param key: Identifier, e.g. ``"subsumption"``.
    :param name: Display name.
    :param box: The :class:`Box` the task is about.
    :param question: The question, in description logic notation.
    :param answer: The form of the answer.
    :param as_subsumption: The subsumption test(s) ``C ⊑ D`` the task reduces to, or ``""``
        if it does not reduce to subsumption tests.
    :param needs: Constructs the reduction needs beyond the ones in the query (names of
        :class:`mowl.logic.Construct`), e.g. ``("NOMINAL",)`` for instance checking.
    :param owlapi: The ``OWLReasoner`` method that answers the task.
    :param summary: One or two sentences.
    """

    key: str
    name: str
    box: Box
    question: str
    answer: str
    as_subsumption: str = ""
    needs: tuple = ()
    owlapi: str = ""
    summary: str = ""


TASKS = {t.key: t for t in (
    InferenceTask(
        "subsumption", "Subsumption checking", Box.TBOX, "O ⊨ C ⊑ D ?", "yes / no",
        as_subsumption="C ⊑ D", owlapi="isEntailed(SubClassOf(C, D))",
        summary="Is every instance of C an instance of D in every model? C and D are "
                "class expressions; with concept names only this is a taxonomy query."),
    InferenceTask(
        "equivalence", "Equivalence checking", Box.TBOX, "O ⊨ C ≡ D ?", "yes / no",
        as_subsumption="C ⊑ D and D ⊑ C", owlapi="getEquivalentClasses(C)",
        summary="Two subsumption tests."),
    InferenceTask(
        "satisfiability", "Concept satisfiability", Box.TBOX, "O ⊭ C ⊑ ⊥ ?", "yes / no",
        as_subsumption="not (C ⊑ ⊥)", needs=("BOTTOM",), owlapi="isSatisfiable(C)",
        summary="Can C have an instance? Unsatisfiable means C ⊑ ⊥ is entailed."),
    InferenceTask(
        "disjointness", "Disjointness checking", Box.TBOX, "O ⊨ C ⊓ D ⊑ ⊥ ?", "yes / no",
        as_subsumption="C ⊓ D ⊑ ⊥", needs=("BOTTOM", "CONJUNCTION"),
        owlapi="isEntailed(DisjointClasses(C, D))",
        summary="Unsatisfiability of the conjunction."),
    InferenceTask(
        "classification", "Classification", Box.TBOX,
        "all pairs of concept names A, B with O ⊨ A ⊑ B", "the concept hierarchy",
        as_subsumption="A ⊑ B for all concept names A, B",
        owlapi="precomputeInferences(CLASS_HIERARCHY); getSuperClasses(A, false)",
        summary="Subsumption checking for every pair of concept names. The deductive "
                "closure used for filtering and for 'inference' benchmarks is its result."),
    InferenceTask(
        "role-subsumption", "Role subsumption", Box.RBOX, "O ⊨ r ⊑ s ?", "yes / no",
        owlapi="getSuperObjectProperties(r, false)",
        summary="Role hierarchy queries; does not reduce to concept subsumption."),
    InferenceTask(
        "consistency", "Consistency", Box.KB, "does O have a model?", "yes / no",
        as_subsumption="not (⊤ ⊑ ⊥)", needs=("BOTTOM",), owlapi="isConsistent()",
        summary="Every other task presupposes a consistent knowledge base. A candidate "
                "axiom is contradictory if adding it makes the knowledge base "
                "inconsistent."),
    InferenceTask(
        "instance-checking", "Instance checking", Box.ABOX, "O ⊨ C(a) ?", "yes / no",
        as_subsumption="{a} ⊑ C", needs=("NOMINAL",),
        owlapi="isEntailed(ClassAssertion(C, a))",
        summary="Is the individual a an instance of C in every model?"),
    InferenceTask(
        "realization", "Realization", Box.ABOX,
        "for each individual a, the most specific concept names A with O ⊨ A(a)",
        "a set of concept names per individual",
        as_subsumption="{a} ⊑ A for all concept names A, keeping the minimal ones",
        needs=("NOMINAL",), owlapi="getTypes(a, true)",
        summary="Instance checking against every concept name."),
    InferenceTask(
        "instance-retrieval", "Instance retrieval", Box.ABOX,
        "all individuals a with O ⊨ C(a)", "a set of individuals",
        as_subsumption="{a} ⊑ C for all individuals a", needs=("NOMINAL",),
        owlapi="getInstances(C, false)",
        summary="The query side of instance checking; protein function prediction is an "
                "instance retrieval query for C = ∃hasFunction.F."),
    InferenceTask(
        "role-assertion", "Role assertion checking", Box.ABOX, "O ⊨ r(a, b) ?", "yes / no",
        as_subsumption="{a} ⊑ ∃r.{b}", needs=("NOMINAL", "EXISTENTIAL"),
        owlapi="isEntailed(ObjectPropertyAssertion(r, a, b))",
        summary="The reasoning task behind link prediction."),
    InferenceTask(
        "role-filler-retrieval", "Role filler retrieval", Box.ABOX,
        "all individuals b with O ⊨ r(a, b)", "a set of individuals",
        as_subsumption="{a} ⊑ ∃r.{b} for all individuals b",
        needs=("NOMINAL", "EXISTENTIAL"), owlapi="getObjectPropertyValues(a, r)",
        summary="Tail queries (a, r, ?) of knowledge graph completion."),
    InferenceTask(
        "query-answering", "Conjunctive query answering", Box.KB,
        "all answers to a conjunctive query q over O", "a set of tuples",
        summary="Tree-shaped queries can be rolled up into instance retrieval for a class "
                "expression; cyclic queries do not reduce to subsumption tests."),
)}


def get_task(key):
    """Return the :class:`InferenceTask` with this key."""
    if key not in TASKS:
        raise KeyError(f"Unknown inference task {key!r}. Known tasks: {sorted(TASKS)}")
    return TASKS[key]


@dataclass(frozen=True)
class CompletionTask:
    """A completion task of the knowledge graph / knowledge base completion literature: a
    scoring function ranks candidate axioms of one form.

    :param key: Identifier.
    :param name: Display name.
    :param candidates: The form of the candidate axioms, with ``?`` for the ranked slot.
    :param inference_task: Key of the :class:`InferenceTask` that decides one candidate.
    :param setting: ``"knowledge graph completion"`` (ABox role assertions only) or
        ``"knowledge base completion"`` (axioms that may involve class expressions).
    :param summary: One or two sentences.
    """

    key: str
    name: str
    candidates: str
    inference_task: str
    setting: str
    summary: str = ""


KGC = "knowledge graph completion"
KBC = "knowledge base completion"

COMPLETION_TASKS = {t.key: t for t in (
    CompletionTask("link-prediction", "Link prediction", "r(a, ?) or r(?, b)",
                   "role-assertion", KGC,
                   "Rank individuals as the missing end of a role assertion. This is all "
                   "of knowledge graph completion, and the ABox part of knowledge base "
                   "completion (protein-protein interaction prediction)."),
    CompletionTask("type-prediction", "Class membership prediction", "?(a) or C(?)",
                   "instance-checking", KBC,
                   "Rank classes for an individual, or individuals for a class "
                   "expression (function prediction: C = ∃hasFunction.F)."),
    CompletionTask("subsumption-prediction", "Subsumption prediction", "C ⊑ ?",
                   "subsumption", KBC,
                   "Rank concept names as the superclass of a class expression: C a "
                   "concept name (normal form NF1), a conjunction (NF2) or an "
                   "existential restriction (NF4)."),
    CompletionTask("existential-prediction", "Existential restriction prediction",
                   "C ⊑ ∃r.?", "subsumption", KBC,
                   "Rank concept names as the filler of an existential restriction "
                   "(normal form NF3)."),
    CompletionTask("disjointness-prediction", "Disjointness prediction", "C ⊓ ? ⊑ ⊥",
                   "disjointness", KBC,
                   "Rank concept names as disjoint from C."),
    CompletionTask("axiom-ranking", "Candidate axiom ranking", "any axiom α",
                   "subsumption", KBC,
                   "Rank arbitrary candidate axioms; the setting of the faithfulness "
                   "criteria, where candidates of all three logical statuses are scored."),
)}


def get_completion_task(key):
    """Return the :class:`CompletionTask` with this key."""
    if key not in COMPLETION_TASKS:
        raise KeyError(f"Unknown completion task {key!r}. Known tasks: "
                       f"{sorted(COMPLETION_TASKS)}")
    return COMPLETION_TASKS[key]


class Target(enum.Enum):
    """What the correct answers of an evaluation are, in terms of :class:`LogicalStatus`.

    The same completion task is a different problem depending on this: ranking entailed
    axioms tests whether the embedding has learned the logic ("inference" benchmarks);
    ranking held-out, undetermined axioms tests whether it predicts plausible new knowledge
    ("prediction" benchmarks)."""

    ENTAILED = "deductive inference"       #: test axioms are entailed by the training KB
    UNDETERMINED = "inductive prediction"  #: test axioms are held out and not entailed
    STATUS = "logical status"              #: all three statuses are scored and compared
