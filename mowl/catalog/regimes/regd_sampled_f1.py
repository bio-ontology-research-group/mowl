"""Sampled F1 evaluation of RegD."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._benchmark_refs import YANG2025_REGD

REGIME = RegimeCard(
    key="regd-sampled-f1",
    name="Sampled subsumption classification with F1 (RegD)",
    kind=RegimeKind.EVALUATION,
    summary=(
        "Samples 1,000 test subsumptions, adds 10 negatives per positive by replacing one "
        "concept with a random class, labels a pair true iff its score is at most a "
        "threshold tuned for F1 on the validation set, and reports F1."),
    components=("test set", "negatives", "threshold", "metrics"),
    applies_to="any model scoring gci0 or gci2 axioms",
    references=(YANG2025_REGD,),
    standalone=False,
    introduced_with=("regd",),
    implementation="mowl.evaluation.sampled_f1.SampledF1Regime",
    limitations=(
        "Random negatives are not checked against the ontology and may be entailed.",
        "The F1 score depends on the sampling seed and on the 1:10 class ratio.",
    ),
    notes=(
        "arXiv:2501.17518v3, Sections 5.1-5.3, Tables 2-4, and EXP_ontology/evaluate.py of "
        "HuiYang1997/RegD. Prediction queries are A ⊑ ∃r.B (the paper writes ∃r.B ≺ ?A "
        "and reads C ⊑ D as D ≺ C). Negatives replace D in C ⊑ D and A in A ⊑ ∃r.B, as "
        "in the code (the text says 'the child'). The threshold is tuned on validation "
        "pairs sampled like the test pairs."),
)
