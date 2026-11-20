"""Evaluation regimes of CatE (Zhapa-Camacho and Hoehndorf, arXiv:2305.07163v3)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._benchmark_refs import ZHAPA2023

REGIMES = (
    RegimeCard(
        key="cate-filtered-ranking",
        name="Filtered ranking of memberships and subsumptions (CatE)",
        kind=RegimeKind.EVALUATION,
        summary=(
            "Ranks the tail of every test axiom (C in C(a), D in C ⊑ D) among the named "
            "concepts, filtered metrics only, with the conventions of the reference "
            "code: a query's head is removed from its own candidates, Hits@k counts "
            "ranks strictly below k, the AUC is the trapezoid over the cumulative rank "
            "curve and ties are broken by position."),
        components=("test set", "candidate set", "filtering", "metrics"),
        applies_to="EmbeddingELModel and mowl.models.cate.CatE",
        references=(ZHAPA2023,),
        standalone=False,
        introduced_with=("cate",),
        implementation="mowl.evaluation.filtered_ranking.cate_ranking",
        limitations=(
            "Hits@k with rank < k (not ≤ k) is the convention of the reference code "
            "(src/cat_completion.py: 'if rank < k'); a Hits@1 is therefore always 0, "
            "and the numbers are not comparable with regimes that use rank ≤ k.",
            "The completion queries of the published Table 3 are the released "
            "'cleaned' lists (test_cleaned.csv, 11,893 of the 14,521 GO and 3,632 of "
            "the 5,957 FoodOn test pairs); the script that produced them is not in the "
            "code release, so the lists are used as released.",
        ),
        notes=(
            "arXiv:2305.07163v3, Section 4.1 and src/cat_completion.py, "
            "src/cat_membership.py. Filtering: the deductive closure of the training "
            "ontology for subsumptions (inferred_ancestors.txt of the data release; "
            "mOWL's foodon_subsumption and go_subsumption datasets carry a "
            "deductive-closure ontology that was verified to contain exactly those "
            "pairs), the training axioms for class assertions — the paper says the "
            "closure is used for C(a) too, the code does not do it. Ranking details "
            "not stated in the paper: ties by argsort position, the trapezoid AUC of "
            "the code's compute_rank_roc, candidates = all classes including owl:Thing "
            "and owl:Nothing, and the head of a subsumption query is given the worst "
            "score (preds[head] = -10000).",
        ),
    ),
    RegimeCard(
        key="cate-ppi-ranking",
        name="Both-direction filtered PPI ranking (CatE)",
        kind=RegimeKind.EVALUATION,
        summary=(
            "For every test interaction P1 ⊑ ∃interacts.P2, rank P1 among the proteins "
            "with P2 fixed and P2 with P1 fixed, and average the metrics over the two "
            "directions; filtered metrics remove the training and validation "
            "interactions; ties by position, Hits@k with rank ≤ k, the trapezoid AUC."),
        components=("test set", "candidate set", "filtering", "metrics"),
        applies_to="EmbeddingELModel and mowl.models.cate.CatE",
        references=(ZHAPA2023,),
        standalone=False,
        introduced_with=("cate",),
        implementation="mowl.evaluation.filtered_ranking.cate_ppi_ranking",
        limitations=(
            "The two directions of one interaction are not independent, so the "
            "effective number of queries is the number of interactions, not twice it.",
        ),
        notes=(
            "The published Table 4 (left) was computed with src/evaluators.py "
            "(PPIEvaluator) — the mOWL PPI evaluation, ranking both sides with rank ≤ k "
            "and the trapezoid compute_rank_roc — which the released run script calls "
            "(use_cases/run_cat_ppi.py imports src/cat_ppi_evaluator.py). The "
            "alternative one-direction evaluation inside src/cat_ppi.py "
            "(rankdata with averaged ties, Hits@k with rank < k) was not used for the "
            "table.",
        ),
    ),
)
