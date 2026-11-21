"""Data model of the mOWL method catalogue.

A :class:`MethodCard` is a small, declarative description of one ontology embedding
method: what it represents concepts and roles with, which logic it covers, which
theoretical guarantees have been proven for it (and under which condition they hold),
which limitations are known, and whether and where mOWL implements it.

Cards are plain data. Creating them does not import any model code and does not need a
running JVM, so the catalogue can be queried and rendered cheaply.
"""

from dataclasses import dataclass, field
from enum import Enum
import importlib


class Family(str, Enum):
    """Top-level classification of ontology embedding methods."""

    #: The ontology is projected to a graph, which is then embedded (random walks or
    #: knowledge graph embeddings). Examples: OWL2Vec*, DL2Vec.
    GRAPH = "graph"
    #: Axioms (and annotations) are treated as sentences of a corpus. Examples: Onto2Vec,
    #: OPA2Vec.
    SYNTACTIC = "syntactic"
    #: Concepts are regions of a vector space and axioms are relations between regions,
    #: learned by minimising a loss. Examples: ELEmbeddings, ELBE, Box2EL.
    GEOMETRIC = "geometric"
    #: A geometric interpretation is constructed (not learned) from the ontology, usually
    #: from its canonical model, and is an exact model by construction.
    CONSTRUCTIVE = "constructive"
    #: Concepts are fuzzy sets over (sampled) individuals. Example: FALCON.
    FUZZY = "fuzzy"
    #: The subsumption order or lattice of concept descriptions is embedded directly.
    #: Example: CatE.
    ORDER = "order"
    #: Concept embeddings are produced by a (fine-tuned) language model from textual
    #: labels. Examples: HiT, OnT.
    LANGUAGE_MODEL = "language_model"
    #: Concepts are subspaces, distributions, or other non-region algebraic objects.
    #: Example: E2R (quantum logic).
    ALGEBRAIC = "algebraic"


class Status(str, Enum):
    """What mOWL provides for a method."""

    #: A trainable implementation exists in mOWL.
    LEARNABLE = "learnable"
    #: An exact construction (no training) is implemented in mOWL.
    CONSTRUCTIVE = "constructive"
    #: The method is documented in the catalogue but not implemented. The card's
    #: ``not_implemented_reason`` says why.
    CATALOGUED = "catalogued"


class Origin(str, Enum):
    """How the mOWL implementation relates to the publication."""

    #: Follows the published method (losses, parametrisation) as closely as possible.
    REIMPLEMENTATION = "reimplementation"
    #: The publication gives a geometry or a construction but no learning method for
    #: ontologies; the training objective in mOWL is ours. Published benchmark numbers do
    #: not exist or are not comparable.
    ADAPTATION = "adaptation"
    #: Developed in mOWL.
    ORIGINAL = "original"
    #: Not implemented.
    NONE = "none"


class RegimeKind(str, Enum):
    """Kinds of regimes."""

    #: How a model is trained: negative sampling, choice of positives, loss combination,
    #: initialisation, schedules, model selection.
    TRAINING = "training"
    #: How a model is evaluated: construction of test sets, candidate sets, filtering,
    #: metrics.
    EVALUATION = "evaluation"


class GuaranteeKind(str, Enum):
    """Kinds of theoretical statements recorded on a card."""

    #: The learned or constructed object is a model of the ontology (under ``condition``).
    SOUNDNESS = "soundness"
    #: Every satisfiable ontology of the fragment has an embedding of this kind.
    COMPLETENESS = "completeness"
    #: Weak faithfulness: only consequences (or no contradictions) of the ontology hold.
    WEAK_FAITHFULNESS = "weak_faithfulness"
    #: Strong faithfulness: an axiom holds in the embedding iff it is entailed.
    STRONG_FAITHFULNESS = "strong_faithfulness"
    #: The region family is closed under a constructor (intersection, existential, ...).
    CLOSURE = "closure"
    #: A positive expressivity result that fits none of the above.
    EXPRESSIVITY = "expressivity"
    #: A proven negative result (something the method cannot represent).
    LIMITATION = "limitation"
    #: A property of a training or evaluation regime that holds by construction or by
    #: proof (for example "sampled negatives are never entailed").
    REGIME_PROPERTY = "regime_property"


@dataclass(frozen=True)
class Reference:
    """A bibliographic reference.

    :param key: Citation key, e.g. ``"kulmanov2019"``. Use the key of
        ``docs/source/appendix/references.rst`` when the reference is listed there.
    :param citation: Full human-readable citation.
    :param url: Stable URL (DOI or arXiv), if any.
    """

    key: str
    citation: str
    url: str = ""


@dataclass(frozen=True)
class Guarantee:
    """A theoretical statement about a method, with its source.

    :param kind: The kind of statement.
    :param statement: The statement itself, in one or two sentences.
    :param source: Where it is proven, e.g. ``"singh2026, Theorem 6"``. Never leave this
        empty: a guarantee without a source does not belong on a card.
    :param condition: When it holds, e.g. ``"zero loss"``, ``"by construction"``,
        ``"always"``.
    :param verified_in_mowl: Name of a test or checker in mOWL that exercises the
        statement, if any.
    """

    kind: GuaranteeKind
    statement: str
    source: str
    condition: str = "always"
    verified_in_mowl: str = ""

    def __post_init__(self):
        if not self.source.strip():
            raise ValueError("A Guarantee needs a source (paper and theorem/section).")


@dataclass(frozen=True)
class MethodCard:
    """Catalogue entry for one ontology embedding method.

    :param key: Unique lower-case identifier, e.g. ``"elkem"``.
    :param name: Display name, e.g. ``"ELK-Em"``.
    :param family: Top-level :class:`Family`.
    :param region: Finer classification inside the family: the region shape for geometric
        methods (``"ball"``, ``"box"``, ``"union of boxes"``, ``"cone"``, ...), the
        embedding backend for graph methods, and so on.
    :param summary: Two or three sentences describing the method.
    :param logic: Description logic fragment the method is defined for.
    :param concepts: How concepts are represented.
    :param roles: How roles are represented.
    :param individuals: How individuals are represented (empty if not supported).
    :param status: What mOWL provides (:class:`Status`).
    :param origin: How the implementation relates to the paper (:class:`Origin`).
    :param implementation: Dotted path of the mOWL class implementing the method.
    :param module: Dotted path of the underlying ``torch.nn.Module``, if any.
    :param not_implemented_reason: Required when ``status`` is ``CATALOGUED``.
    :param guarantees: Proven statements (:class:`Guarantee`).
    :param limitations: Known limitations that are not formal theorems.
    :param intersection_closed: Whether the concept representation is closed under
        intersection: ``True``, ``False``, a short string when the answer is conditional
        (e.g. ``"only if the number of boxes may grow"``), or ``None`` if not applicable.
    :param related: Keys of related catalogue methods (e.g. the construction that
        realises a theoretical result, or the method this one extends).
    :param helly: What is known about the Helly number of the concept regions, e.g.
        ``"2 (axis-parallel boxes, any dimension)"``, ``"d+1 (convex sets in R^d)"``,
        ``"unbounded (non-convex)"``. Empty if not applicable.
    :param references: Publications (:class:`Reference`), primary reference first.
    :param code_url: URL of the authors' reference implementation, if any.
    :param benchmarks: Keys of :mod:`mowl.benchmarks` benchmarks for which published
        results of this method are recorded.
    :param axioms: Normal forms / axiom types the implementation trains on, e.g.
        ``("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
        "role_inclusion", "role_chain")``.
    :param optional_dependencies: Extra packages needed by the implementation.
    :param training: Keys of the training regimes (:class:`RegimeCard`) the publication
        trains the method with; the first one is the default of the mOWL implementation.
    :param evaluation: Keys of the evaluation regimes (:class:`RegimeCard`) the
        publication evaluates the method with.
    :param expressivity: The logic the method supports, in description logic notation
        readable by :meth:`mowl.logic.Expressivity.parse` (``"EL⊥ with ABox"``,
        ``"ELH(∘)⊥"``, ``"ALC"``), or a :class:`mowl.logic.Expressivity`. For an implemented
        method this is what the implementation uses of an ontology; axioms beyond it are
        ignored (see :func:`mowl.logic.check_ontology`). Empty if the method is not defined
        on a description logic (``logic`` then says what it is defined on).
    :param tasks: Keys of the inference tasks (:data:`mowl.logic.TASKS`) for which the
        method gives a score, e.g. ``("subsumption", "instance-checking")``. Empty for
        methods that only produce feature vectors.
    :param head_tasks: Keys of the inference tasks the method can be applied to only
        through a scoring head trained on its vectors
        (:class:`mowl.evaluation.model_scoring.PairClassifierHead`): the method gives
        vectors, the head gives the score.
    :param factory: Dotted path of a function ``f(dataset, **kwargs)`` that returns the
        configured model, for methods that are a configuration of a general class (a
        projector, a walker and Word2Vec). Used by the benchmark runner.
    :param assumption: The :class:`mowl.logic.WorldAssumption` (its value) of the method's
        own training objective, i.e. of how it trains when no training regime is attached:
        where its negatives come from, or ``"open world"`` if it uses none or only refuted
        ones. A training regime that provides negatives replaces it.
    :param notes: Free text (deviations from the paper, implementation remarks).
    """

    key: str
    name: str
    family: Family
    region: str
    summary: str
    logic: str
    concepts: str
    roles: str
    individuals: str = ""
    status: Status = Status.CATALOGUED
    origin: Origin = Origin.NONE
    implementation: str = ""
    module: str = ""
    not_implemented_reason: str = ""
    guarantees: tuple = ()
    limitations: tuple = ()
    intersection_closed: object = None
    helly: str = ""
    references: tuple = ()
    code_url: str = ""
    benchmarks: tuple = ()
    axioms: tuple = ()
    optional_dependencies: tuple = ()
    training: tuple = ()
    evaluation: tuple = ()
    related: tuple = ()
    expressivity: object = ""
    tasks: tuple = ()
    head_tasks: tuple = ()
    factory: str = ""
    assumption: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.key or self.key != self.key.lower() or " " in self.key:
            raise ValueError(f"Card key must be a lower-case identifier, got {self.key!r}.")
        _check_assumption(self.assumption, f"Card {self.key!r}")
        if not isinstance(self.family, Family):
            raise TypeError("family must be a mowl.catalog.Family.")
        if not isinstance(self.status, Status):
            raise TypeError("status must be a mowl.catalog.Status.")
        if not isinstance(self.origin, Origin):
            raise TypeError("origin must be a mowl.catalog.Origin.")
        if not self.references:
            raise ValueError(f"Card {self.key!r} needs at least one reference.")
        if self.status is Status.CATALOGUED:
            if not self.not_implemented_reason:
                raise ValueError(
                    f"Card {self.key!r} is catalogued only and must state "
                    f"'not_implemented_reason'.")
            if self.implementation:
                raise ValueError(
                    f"Card {self.key!r} is catalogued only but names an implementation.")
        else:
            if not self.implementation:
                raise ValueError(f"Card {self.key!r} must name its implementation.")
            if self.origin is Origin.NONE:
                raise ValueError(f"Card {self.key!r} is implemented and must set 'origin'.")
        for g in self.guarantees:
            if not isinstance(g, Guarantee):
                raise TypeError("guarantees must contain mowl.catalog.Guarantee objects.")
        for r in self.references:
            if not isinstance(r, Reference):
                raise TypeError("references must contain mowl.catalog.Reference objects.")
        from mowl.logic import TASKS
        for task in tuple(self.tasks) + tuple(self.head_tasks):
            if task not in TASKS:
                raise ValueError(f"Card {self.key!r} names the unknown inference task "
                                 f"{task!r}. Known tasks: {sorted(TASKS)}")
        self.supported_expressivity  # raises if the notation cannot be read

    @property
    def supported_expressivity(self):
        """The :class:`mowl.logic.Expressivity` of :attr:`expressivity`, or ``None`` if
        the method is not defined on a description logic."""
        from mowl.logic import Expressivity
        if not self.expressivity:
            return None
        if isinstance(self.expressivity, Expressivity):
            return self.expressivity
        return Expressivity.parse(self.expressivity)

    def check_ontology(self, ontology, warn=True):
        """Report the axioms of ``ontology`` (an OWL API ontology) that go beyond what
        this method supports. See :func:`mowl.logic.check_ontology`. Needs the JVM."""
        from mowl.logic import check_ontology
        supported = self.supported_expressivity
        if supported is None:
            raise ValueError(f"Method {self.key!r} states no description logic "
                             f"expressivity (it is defined on: {self.logic}).")
        return check_ontology(ontology, supported, warn=warn)

    @property
    def implemented(self):
        """``True`` if mOWL ships an implementation of the method."""
        return self.status is not Status.CATALOGUED

    def load(self):
        """Import and return the implementing class.

        Importing model classes requires a running JVM (``mowl.init_jvm``).

        :raises NotImplementedError: If the method is catalogued only.
        """
        if not self.implemented:
            raise NotImplementedError(
                f"{self.name} is documented in the catalogue but not implemented in mOWL: "
                f"{self.not_implemented_reason}")
        module_name, _, attr = self.implementation.rpartition(".")
        return getattr(importlib.import_module(module_name), attr)


def _check_assumption(value, owner):
    from mowl.logic import WorldAssumption
    if value and value not in {x.value for x in WorldAssumption}:
        raise ValueError(f"{owner}: assumption must be one of "
                         f"{sorted(x.value for x in WorldAssumption)}.")


@dataclass(frozen=True)
class RegimeCard:
    """Catalogue entry for a training or evaluation regime.

    A regime is a way to train or to evaluate embeddings that is independent of the
    representation. Some regimes were published on their own; many were introduced
    inside a method paper and never named. Both kinds are catalogued, so that it is
    explicit which regime a published result was obtained with, and so that any regime
    can be combined with any compatible method.

    :param key: Unique lower-case identifier, e.g. ``"deductive-closure-filtering"``.
    :param name: Display name.
    :param kind: :class:`RegimeKind`.
    :param summary: Two or three sentences describing the regime.
    :param components: What the regime determines, e.g. ``("negative sampling",
        "positives")`` for training or ``("test set", "candidate set", "filtering",
        "metrics")`` for evaluation.
    :param applies_to: Which models or logics it can be used with.
    :param references: Publications that introduce or define it, primary first.
    :param standalone: ``True`` if it was published as a contribution of its own,
        ``False`` if it is part of a method paper (or was never published separately).
    :param introduced_with: Keys of the methods whose publication introduced it.
    :param used_by: Keys of further methods whose publications use it.
    :param implementation: Dotted path of the mOWL class or function implementing it
        (a :class:`mowl.training.TrainingRegime` or
        :class:`mowl.evaluation.EvaluationRegime` subclass where possible).
    :param not_implemented_reason: Required when there is no implementation.
    :param properties: Proven or by-construction properties (:class:`Guarantee`), e.g.
        "sampled negatives are never entailed".
    :param limitations: Known limitations.
    :param task: For an evaluation regime, the key of the completion task it measures
        (:data:`mowl.logic.COMPLETION_TASKS`), or of an inference task
        (:data:`mowl.logic.TASKS`). Empty for regimes that only fix a detail (a score, a
        tie rule) and for training regimes.
    :param target: For an evaluation regime, which axioms count as correct: a value of
        :class:`mowl.logic.Target` (``"deductive inference"``, ``"inductive prediction"``
        or ``"logical status"``), or empty if the regime can be used with either.
    :param assumption: What the regime takes an axiom to be that is neither stated nor
        refuted: a value of :class:`mowl.logic.WorldAssumption` (``"open world"``,
        ``"closed world over entailed axioms"``, ``"closed world over stated axioms"``).
        For a training regime this is where its negatives come from; for an evaluation
        regime, which candidates count as wrong. Empty for regimes that do not decide it
        (losses, schedules, scoring rules, positives).
    :param notes: Free text.
    """

    key: str
    name: str
    kind: RegimeKind
    summary: str
    components: tuple
    applies_to: str
    references: tuple
    standalone: bool = False
    introduced_with: tuple = ()
    used_by: tuple = ()
    implementation: str = ""
    not_implemented_reason: str = ""
    properties: tuple = ()
    limitations: tuple = ()
    task: str = ""
    target: str = ""
    assumption: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.key or self.key != self.key.lower() or " " in self.key:
            raise ValueError(f"Regime key must be a lower-case identifier: {self.key!r}")
        _check_assumption(self.assumption, f"Regime {self.key!r}")
        if not isinstance(self.kind, RegimeKind):
            raise TypeError("kind must be a mowl.catalog.RegimeKind.")
        from mowl.logic import TASKS, COMPLETION_TASKS, Target
        if self.task and self.task not in TASKS and self.task not in COMPLETION_TASKS:
            raise ValueError(f"Regime {self.key!r} names the unknown task {self.task!r}.")
        if self.target and self.target not in {t.value for t in Target}:
            raise ValueError(f"Regime {self.key!r}: target must be one of "
                             f"{sorted(t.value for t in Target)}.")
        if not self.references:
            raise ValueError(f"Regime {self.key!r} needs at least one reference.")
        for r in self.references:
            if not isinstance(r, Reference):
                raise TypeError("references must contain mowl.catalog.Reference objects.")
        for g in self.properties:
            if not isinstance(g, Guarantee):
                raise TypeError("properties must contain mowl.catalog.Guarantee objects.")
        if not self.implementation and not self.not_implemented_reason:
            raise ValueError(
                f"Regime {self.key!r} must name its implementation or state "
                f"'not_implemented_reason'.")

    @property
    def implemented(self):
        """``True`` if mOWL ships an implementation of the regime."""
        return bool(self.implementation)

    def load(self):
        """Import and return the implementing class or function."""
        if not self.implemented:
            raise NotImplementedError(
                f"{self.name} is documented in the catalogue but not implemented in mOWL: "
                f"{self.not_implemented_reason}")
        module_name, _, attr = self.implementation.rpartition(".")
        return getattr(importlib.import_module(module_name), attr)
