"""Syntactic (corpus-based) methods: Onto2Vec and OPA2Vec."""

from mowl.catalog.card import Family, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

_IMPL = "mowl.models.syntactic.w2v_model.SyntacticPlusW2VModel"

_LIMITATION = (
    "Axioms are treated as sentences; the embedding reflects co-occurrence of symbols, not "
    "the semantics of the axioms. The survey of Chen et al. (2025, Section II, Table I) "
    "describes sequence-based methods as only partially faithful to correlations and "
    "co-occurrences, with low interpretability.")

ONTO2VEC = MethodCard(
    key="onto2vec",
    expressivity="OWL 2 DL",
    tasks=(),
    name="Onto2Vec",
    family=Family.SYNTACTIC,
    region="Word2Vec on an axiom corpus",
    summary=(
        "Each (asserted or inferred) axiom of the ontology, written in Manchester-like "
        "syntax, is a sentence of a corpus; Word2Vec trained on the corpus gives vectors for "
        "classes, relations and annotated entities."),
    logic="OWL 2 (any axioms, treated syntactically)",
    concepts="vector",
    roles="vector",
    individuals="vector",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation=_IMPL,
    limitations=(_LIMITATION,),
    references=(R.SMAILI2018, R.MIKOLOV2013, R.CHEN2025),
    notes=(
        "Use SyntacticPlusW2VModel with generate_corpus(with_annotations=False). To "
        "include inferred axioms in the corpus, add them to the ontology first (e.g. with "
        "mowl.reasoning)."),
)

OPA2VEC = MethodCard(
    key="opa2vec",
    expressivity="OWL 2 DL",
    tasks=(),
    name="OPA2Vec",
    family=Family.SYNTACTIC,
    region="Word2Vec on axiom and annotation corpus",
    summary=(
        "Extends Onto2Vec with the ontology's annotation properties (labels, definitions, "
        "synonyms) as additional sentences, so formal and informal content are embedded "
        "together."),
    logic="OWL 2 (any axioms and annotations, treated syntactically)",
    concepts="vector",
    roles="vector",
    individuals="vector",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation=_IMPL,
    limitations=(_LIMITATION,),
    references=(R.SMAILI2019, R.MIKOLOV2013, R.CHEN2025),
    notes=(
        "Use SyntacticPlusW2VModel with generate_corpus(with_annotations=True). mOWL "
        "does not pre-train Word2Vec on an external text corpus."),
)

CARDS = (ONTO2VEC, OPA2VEC)
