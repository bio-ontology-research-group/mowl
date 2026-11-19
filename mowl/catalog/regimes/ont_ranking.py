"""Evaluation regimes of OnT (Yang, Chen, He, Gao, Horrocks, ISWC 2025)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.entries.lm_hyperbolic import YANG2025ONT, HE2024HIT

_DIFF = (
    "Differs from box2el-normal-form-ranking in three points: trivial answers (C for "
    "C ⊑ ?D; C, D for C ⊓ D ⊑ ?E) are not removed from the candidates; the candidates are "
    "all concepts of OnT's normalised ontology, including the named concepts its "
    "normaliser introduces, without ⊤/⊥; and the score is OnT's Eq. 8 with λ chosen on "
    "validation data, the complex side of NF2-NF4 embedded from its verbalisation. Ranks "
    "are raw with ties by candidate position, and the AUC is the Box2EL one (the "
    "reference code's compute_rank_roc is the same function).")

REGIMES = (
    RegimeCard(
        key="ont-normal-form-ranking",
        name="Per-normal-form axiom ranking over all concepts (OnT)",
        kind=RegimeKind.EVALUATION,
        summary=(
            "Ranks every test axiom of each normal form against the candidates obtained by "
            "replacing one concept with every concept of the normalised ontology: C ⊑ ?D, "
            "C ⊓ D ⊑ ?E, ?C ⊑ ∃r.D, ∃r.C ⊑ ?D. Raw ranks; Hits@1/10/100, median rank, "
            "MRR, mean rank and ROC AUC per normal form and combined."),
        components=("test set", "candidate set", "filtering", "metrics", "scores"),
        applies_to=("Text models with score_candidates (HiT, OnT); EL embedding models "
                    "through their scorers"),
        references=(YANG2025ONT,),
        standalone=False,
        introduced_with=("ont",),
        used_by=("hit", "elem", "elbe", "boxel", "box2el", "transbox", "opa2vec",
                 "owl2vec_star"),
        implementation="mowl.evaluation.ont_ranking.OnTRankingRegime",
        limitations=(
            "Unfiltered: other true axioms ranked above the test axiom count as errors.",
            "The test split is a random 80/10/10 split of OnT's own normalisation, so "
            "numbers are not comparable with box2el-normal-form-ranking.",
            "Trivial answers are kept: in the published test sets the answer of almost "
            "every NF4 axiom (and of a quarter to a third of the NF3 axioms) is the "
            "concept that OnT's normaliser named with the query's own verbalisation, so "
            "text encoders rank it first by construction (see the ont-* benchmark notes).",
        ),
        notes=("yang2025ont, Section 5.1 'Evaluation Metrics' and 'Experimental "
               "Protocol'; ont/evaluation/evaluator.py of HuiYang1997/OnT (commit "
               "9a2cdc3). " + _DIFF + " The baselines of Tables 2-3 were run by the OnT "
               "authors under this regime (OPA2Vec and OWL2Vec* through a classifier)."),
    ),
    RegimeCard(
        key="ont-inferred-subsumption-ranking",
        name="Ranking of inferred atomic subsumptions over all concepts (OnT)",
        kind=RegimeKind.EVALUATION,
        summary=(
            "Train on the whole normalised ontology; rank the superclass of every inferred "
            "NF1 axiom C ⊑ D among all concepts (1,000 inferred axioms for validation), "
            "as NF1 of ont-normal-form-ranking."),
        components=("test set", "candidate set", "filtering", "metrics"),
        applies_to="as ont-normal-form-ranking",
        references=(YANG2025ONT, HE2024HIT),
        standalone=False,
        introduced_with=("ont",),
        used_by=("hit", "elem", "elbe", "boxel", "box2el", "transbox", "opa2vec",
                 "owl2vec_star"),
        implementation="mowl.evaluation.ont_ranking.OnTInferenceRankingRegime",
        notes=" ".join((
            "yang2025ont, Section 5.1 'Datasets' and Section 5.3, Table 3. HiT's Multi-hop "
            "Inference (he2024hit, Section 4.1) also tests inferred subsumptions from "
            "asserted ones, but as threshold classification.", _DIFF)),
    ),
)
