"""Evaluation of CatE."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._benchmark_refs import ZHAPA2023

REGIME = RegimeCard(
    key="cate-filtered-ranking",
    name="Filtered ranking of memberships, subsumptions and PPIs (CatE)",
    kind=RegimeKind.EVALUATION,
    summary=(
        "Ranks the tail of every test axiom (C in C(a), D in C ⊑ D, P2 in "
        "P1 ⊑ ∃interacts.P2) among the named concepts or evaluation classes, after "
        "removing training axioms (and, for C(a), the deductive closure of the training "
        "set); filtered MR, MRR, Hits@3/10/100 and AUC only."),
    components=("test set", "candidate set", "filtering", "metrics"),
    applies_to="EmbeddingELModel",
    references=(ZHAPA2023,),
    standalone=False,
    introduced_with=("cate",),
    implementation="mowl.evaluation.filtered_ranking.cate_ranking",
    notes=(
        "arXiv:2305.07163v3, Sections 4.1-4.4. Tie handling, AUC variant and the scores "
        "of the baselines are not stated in the paper; mOWL uses its evaluator "
        "conventions (loss scores, ties by position, Riemann AUC)."),
)
