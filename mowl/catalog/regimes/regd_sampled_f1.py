"""Sampled subsumption classification with F1, the evaluation of RegD."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._benchmark_refs import YANG2025_REGD

REGIME = RegimeCard(
    key="regd-sampled-f1",
    name="Threshold classification of subsumptions with F1 (RegD)",
    kind=RegimeKind.EVALUATION,
    summary=(
        "Scores every test subsumption against all classes: for each candidate class X "
        "the rectified boundary dissimilarity of a fixed child region inside the region "
        "of X (the depth dissimilarity and the training loss's margins are NOT part of "
        "the score). The class of the query itself is filtered out; 10 negatives per "
        "query are drawn with replacement from the candidate scores, not checked against "
        "the ontology. A query is predicted true iff its score is at most the threshold "
        "that maximises F1 on the validation split (sampled the same way); F1 is "
        "reported. The prediction task queries A ⊑ ∃r.B rank the subclass A, with the "
        "filler's region translated by -R as the child."),
    components=("test set", "candidates", "negatives", "threshold", "metrics", "scores"),
    applies_to="EmbeddingELModel with boxes (class_offset) or balls (class_rad) and "
               "translations (rel_embed): ELBE, ELEm, RegD",
    references=(YANG2025_REGD,),
    standalone=False,
    introduced_with=("regd",),
    used_by=("regd", "regd_elem", "elbe", "elem"),
    implementation="mowl.evaluation.sampled_f1.SampledF1Regime",
    limitations=(
        "The random negatives may be entailed, which caps the achievable precision.",
        "The F1 depends on the sampling seed and on the 1:10 positive:negative ratio.",
        "The prediction-task numbers of the paper (Table 4) are NOT comparable with an "
        "evaluation that scores the query itself (positive_scores='pair'); they are "
        "reproduced only with the code's slip below.",
    ),
    notes=(
        "arXiv:2501.17518v3, Sections 5.1-5.3, Tables 2-4, and EXP_ontology/evaluate.py "
        "of HuiYang1997/RegD. What the paper says vs what the code does: the paper says "
        "1,000 subsumptions are sampled from the test set; the code evaluates ALL test "
        "subsumptions (the first 1,000 of the validation split). The paper says the "
        "threshold 'achieved the best performance on the evaluation set'; the code tunes "
        "it on the validation split at the selected checkpoint and applies it to the "
        "test split (mOWL does the same; model selection by validation F1 is the "
        "training regime 'validation-f1-selection'). The paper says negatives are made "
        "by 'replacing the child'; for C ⊑ D the code replaces the superclass D, for "
        "A ⊑ ∃r.B the subclass A. The paper describes the score only as the model's "
        "energy; the code uses relu of the boundary dissimilarity alone. Two slips of "
        "the code that the published numbers carry: (1) for the prediction task the "
        "positive score of a query is score-matrix column 1 -- the role index read as a "
        "class index, an arbitrary class instead of the query (positive_scores="
        "'reference' reproduces this; 'pair' scores the axiom itself); (2) the code "
        "trains the prediction task with the reversed inclusion (see RegDELBE's "
        "flip_gci2) but scores box(B) - R, so the signs of R differ between training "
        "and testing. For balls the code's evaluation uses absolute radii (mOWL agrees) "
        "although its training uses softplus radii, which mOWL does not."),
)
