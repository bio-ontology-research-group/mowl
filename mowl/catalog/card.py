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
        intersection (``None`` if not applicable).
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
    notes: str = ""

    def __post_init__(self):
        if not self.key or self.key != self.key.lower() or " " in self.key:
            raise ValueError(f"Card key must be a lower-case identifier, got {self.key!r}.")
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
