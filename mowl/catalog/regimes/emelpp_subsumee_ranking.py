"""Evaluation of EmEL++."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._benchmark_refs import MONDAL2021

REGIME = RegimeCard(
    key="emelpp-subsumee-ranking",
    assumption="closed world over stated axioms",
    task="subsumption-prediction",
    target="inductive prediction",
    name="Subsumee ranking by centre distance (EmEL++)",
    kind=RegimeKind.EVALUATION,
    summary=(
        "For every test subsumption C ⊑ D, ranks all other classes by the distance of "
        "their centre from D's centre and records the rank of C; Hits@1/10/100, AUC, "
        "median and 90th percentile rank."),
    components=("test set", "candidate set", "metrics"),
    applies_to="EmbeddingELModel (centre-distance scores where available)",
    references=(MONDAL2021,),
    standalone=False,
    introduced_with=("emelpp",),
    implementation="mowl.evaluation.normal_form_ranking.SubsumeeRankingRegime",
    limitations=(
        "Centre distance is symmetric, so the direction of the subsumption is ignored.",
    ),
    notes=(
        "AAAI-MAKE 2021, Section 4.4 and Evaluating_HITS.py of kracr/EmELpp. The "
        "reference AUC drops the first distinct rank before integrating; mOWL integrates "
        "over all ranks."),
)
