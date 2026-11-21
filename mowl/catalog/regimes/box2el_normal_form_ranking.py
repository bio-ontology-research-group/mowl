"""Evaluation regimes of Box2EL (Jackermeier, Chen, Horrocks, WWW 2024)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._benchmark_refs import JACKERMEIER2024, YANG2025, KULMANOV2019

_TIES = ("Ties are broken by candidate position (argsort of argsort), as in the reference "
         "code; there is no tie averaging.")

REGIMES = (
    RegimeCard(
        key="box2el-normal-form-ranking",
        task="subsumption-prediction",
        name="Per-normal-form subsumption ranking (Box2EL)",
        kind=RegimeKind.EVALUATION,
        summary=(
            "Ranks every test axiom of each EL++ normal form against the candidates "
            "obtained by replacing one atomic concept with every class of the ontology: "
            "C ⊑ ?D (C removed), C ⊓ D ⊑ ?E (C, D removed), ?C ⊑ ∃r.D, ∃r.C ⊑ ?D. Raw "
            "ranks; Hits@1/10/100, median rank, MRR, mean rank and a ROC AUC, per normal "
            "form and combined."),
        components=("test set", "candidate set", "filtering", "metrics", "scores"),
        applies_to=("EmbeddingELModel; centre-distance scores (Appendix F) for Box2EL, "
                    "ELBE and ELEmbeddings, the model's loss otherwise"),
        references=(JACKERMEIER2024, YANG2025),
        standalone=False,
        introduced_with=("box2el",),
        used_by=("transbox",),
        implementation="mowl.evaluation.normal_form_ranking.NormalFormRankingRegime",
        limitations=(
            "Unfiltered: other true axioms ranked above the test axiom count as errors.",
            "Scores are model-specific centre distances, so numbers of different models "
            "are comparable only under the paper's scoring functions.",
        ),
        notes=(
            "Section 4.2 'Evaluation protocol', Appendix F (scores) and G (metrics); the "
            "ranked side per normal form and the AUC (trapezoid of TPR over FPR with "
            "num_classes - 1 negatives per query) are taken from evaluate.py of "
            "KRR-Oxford/BoxSquaredEL. " + _TIES + " TransBox (WWW 2025, Appendix E) "
            "re-implements it with a modified C ⊓ D ⊑ E score for empty intersections."),
    ),
    RegimeCard(
        key="inferred-subsumption-ranking",
        task="subsumption-prediction",
        target="deductive inference",
        name="Ranking of reasoner-inferred subsumptions (Box2EL)",
        kind=RegimeKind.EVALUATION,
        summary=(
            "'Approximating deductive reasoning': train on the full ontology, then rank "
            "the atomic subsumptions C ⊑ D that an EL++ reasoner infers but that are not "
            "asserted, as in the NF1 ranking of box2el-normal-form-ranking."),
        components=("test set", "candidate set", "filtering", "metrics"),
        applies_to="EmbeddingELModel",
        references=(JACKERMEIER2024,),
        standalone=False,
        introduced_with=("box2el",),
        implementation=("mowl.evaluation.normal_form_ranking."
                        "InferredSubsumptionRankingRegime"),
        notes=(
            "Section 4.4, Table 6. The test set is part of the data (90% of the inferred, "
            "non-asserted subsumptions; 10% validation). EmEL++ evaluated on inferences "
            "earlier, with test/training overlap (footnote 5). " + _TIES),
    ),
    RegimeCard(
        key="ppi-ranking",
        task="link-prediction",
        target="inductive prediction",
        name="Protein-protein interaction ranking (Box2EL code)",
        kind=RegimeKind.EVALUATION,
        summary=(
            "For every test axiom {P1} ⊑ ∃interacts.{P2}, ranks P1 among all proteins; "
            "raw and filtered (training interactions removed) Hits@10/100, mean rank and "
            "AUC of the cumulative rank curve."),
        components=("test set", "candidate set", "filtering", "metrics"),
        applies_to="EmbeddingELModel scoring gci2 axioms",
        references=(JACKERMEIER2024, KULMANOV2019),
        standalone=False,
        introduced_with=("elem",),
        used_by=("box2el",),
        implementation="mowl.evaluation.filtered_ranking.ppi_ranking",
        notes=(
            "The task and metrics come from ELEmbeddings (Kulmanov et al. 2019); this "
            "regime follows evaluate_ppi.py of the Box2EL code, which ranks the head "
            "protein P1 and computes AUC as the trapezoid of the cumulative rank curve "
            "divided by the number of proteins. " + _TIES),
    ),
)
