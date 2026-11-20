"""Regime cards: training and evaluation designs that the Box2EL and TransBox numbers
depend on, found by comparing the reference code with the papers."""

from mowl.catalog.card import RegimeCard, RegimeKind, Guarantee, GuaranteeKind
from mowl.catalog.regimes._regime_refs import JACKERMEIER2024, YANG2025

_ZERO_LOSS = Guarantee(
    GuaranteeKind.REGIME_PROPERTY,
    "With a slack margin γ > 0 the inclusion loss is zero whenever the first box exceeds "
    "the second by at most γ in every dimension. Zero loss therefore does not imply that "
    "the axiom holds, and a zero-loss embedding need not be a model of the ontology.",
    "by inspection of the loss ‖max(0, |c1 − c2| + o1 − o2 − γ)‖ (Jackermeier et al. "
    "2024, Section 3.2)",
    "margin > 0")

REGIMES = (
    RegimeCard(
        key="slack-inclusion-margin",
        name="Slack margin on inclusions",
        kind=RegimeKind.TRAINING,
        summary=(
            "The margin γ loosens every inclusion loss: a box may stick out of the box "
            "that should contain it by up to γ. Training then pulls centres together "
            "without having to nest the regions, which suits evaluations that rank by "
            "centre distance, but the result is in general not a model."),
        components=("loss", "margin"),
        applies_to="EL modules that declare margin_semantics (ELEm, EmEL++, ELBE, "
                   "Box2EL, TransBox)",
        references=(JACKERMEIER2024, YANG2025),
        standalone=False,
        introduced_with=("elem",),
        used_by=("box2el", "elbe", "emelpp", "transbox"),
        implementation="mowl.training.margin.SlackMargin",
        properties=(_ZERO_LOSS,),
        limitations=(
            "The soundness statements of the methods ('zero loss implies a model') hold "
            "for margin ≤ 0 only.",),
        notes=(
            "Sign conventions differ between code bases. Slack: ELEmbeddings, Box2EL "
            "(paper and code), the ELBE baseline in the Box2EL code (model/Elbe.py, "
            "'euc + cr - dr - margin'), the TransBox code (model/TransBox_reg.py). "
            "Strict: the ELBE code of Peng et al. ('+ margin'), the loss as written in "
            "the TransBox paper (Section 4.3), and mOWL's ELBE and TransBox modules. The "
            "regime sets the margin by meaning, whatever the module's convention. The "
            "published Box2EL-benchmark numbers of ELBE and TransBox were obtained with "
            "a slack margin."),
    ),
    RegimeCard(
        key="strict-inclusion-margin",
        name="Strict margin on inclusions",
        kind=RegimeKind.TRAINING,
        summary=(
            "The margin γ tightens every inclusion loss: a box must lie inside the box "
            "that should contain it with a gap of at least γ on every side."),
        components=("loss", "margin"),
        applies_to="EL modules that declare margin_semantics",
        references=(YANG2025,),
        standalone=False,
        introduced_with=("elbe",),
        used_by=("transbox",),
        implementation="mowl.training.margin.StrictMargin",
        properties=(
            Guarantee(
                GuaranteeKind.REGIME_PROPERTY,
                "Zero loss implies that the inclusion holds (with a gap of γ), so the "
                "soundness statements of the methods apply.",
                "by inspection of the loss ‖max(0, |c1 − c2| + o1 − o2 + γ)‖",
                "margin ≥ 0"),),
        notes=("See slack-inclusion-margin for which code base uses which form."),
    ),
    RegimeCard(
        key="box2el-code-training",
        name="Training design of the Box2EL code",
        kind=RegimeKind.TRAINING,
        summary=(
            "One step per epoch on 512 random axioms of every normal form (equal weight "
            "per normal form), negatives for C ⊑ ∃r.D with each side corrupted, a slack "
            "margin, optionally a per-axiom squared loss to a target (ELBE baseline) and "
            "a learning-rate step, and selection of the parameters with the best "
            "validation median rank of C ⊑ D every 100 epochs."),
        components=("batching", "negative sampling", "margin", "loss", "schedule",
                    "initialisation", "model selection"),
        applies_to="EmbeddingELModel",
        references=(JACKERMEIER2024,),
        standalone=False,
        introduced_with=("box2el",),
        used_by=("elem", "emelpp", "elbe", "boxel"),
        implementation="mowl.training.reference.Box2ELCodeTraining",
        properties=(_ZERO_LOSS,),
        limitations=(
            "Model selection looks at C ⊑ D only; the selected parameters can be poor on "
            "the other normal forms.",
            "The design is tuned to centre-distance ranking and does not aim at a model "
            "of the ontology.",
        ),
        notes=(
            "From train.py, model/*.py and configs.json of KRR-Oxford/BoxSquaredEL; the "
            "paper states the optimiser, the number of epochs and the validation "
            "interval (Section 4.2) and refers to Appendix H for hyperparameters. A "
            "composition of per-normal-form-minibatch, per-column-corruption, "
            "slack-inclusion-margin, mse-to-target, step-lr-decay and "
            "validation-rank-selection. The baselines of the code (ELEm, EmEL++, ELBE, "
            "BoxEL) are trained with three more parts, switched on by baseline_code=True: "
            "fixed-negatives, joint-normalised-initialisation and "
            "box2el-baseline-losses. Without them ELEm and EmEL++ fall far below the "
            "paper on C ⊓ D ⊑ E and ∃r.C ⊑ D, and ELEm, EmEL++ and ELBE score far above "
            "it on GO."),
    ),
    RegimeCard(
        key="transbox-code-training",
        name="Training design of the TransBox code",
        kind=RegimeKind.TRAINING,
        summary=(
            "The design of box2el-code-training with 1024 axioms per normal form and "
            "step; the margin is a slack margin although the paper writes a strict one."),
        components=("batching", "negative sampling", "margin", "model selection"),
        applies_to="EmbeddingELModel",
        references=(YANG2025, JACKERMEIER2024),
        standalone=False,
        introduced_with=("transbox",),
        implementation="mowl.training.reference.TransBoxCodeTraining",
        properties=(_ZERO_LOSS,),
        notes=("From model/TransBox_reg.py and configs_our.json of HuiYang1997/TransBox, "
               "which is derived from the Box2EL code."),
    ),
    RegimeCard(
        key="transbox-normal-form-ranking",
        name="Per-normal-form subsumption ranking (TransBox variant)",
        kind=RegimeKind.EVALUATION,
        summary=(
            "The ranking of box2el-normal-form-ranking with one changed rule: for "
            "C ⊓ D ⊑ ?E every candidate gets score 0 when the boxes of C and D do not "
            "intersect, so that such a query gets an uninformative rank."),
        components=("test set", "candidate set", "filtering", "metrics", "scores"),
        applies_to="EmbeddingELModel with a centre-distance scorer",
        references=(YANG2025, JACKERMEIER2024),
        standalone=False,
        introduced_with=("transbox",),
        used_by=("box2el", "elbe", "boxel"),
        implementation="mowl.evaluation.normal_form_ranking."
                       "TransBoxNormalFormRankingRegime",
        limitations=(
            "Not comparable with box2el-normal-form-ranking for C ⊓ D ⊑ E and for the "
            "combined ranks: the TransBox paper reports Hits@100 of 0.00 for Box2EL and "
            "ELBE on GALEN where the Box2EL paper reports 0.28 and 0.11.",),
        notes=("Yang et al. 2025, Section 6.2 and Appendix E; evaluate.py of the TransBox "
               "code (compute_nf2_ranks). TransBox itself is scored with the empty "
               "dimensions masked (compute_nf2_ranks_transbox)."),
    ),
    RegimeCard(
        key="empty-intersection-zero-scoring",
        name="Zero score for empty intersections",
        kind=RegimeKind.EVALUATION,
        summary=(
            "Scoring rule for C ⊓ D ⊑ E: if the boxes of C and D are disjoint, every "
            "candidate E gets score 0 instead of the distance to the centre of the "
            "degenerate intersection box."),
        components=("scores",),
        applies_to="Box and ball models with a box reader (Box2EL, ELBE, ELEm, EmEL++)",
        references=(YANG2025,),
        standalone=False,
        introduced_with=("transbox",),
        implementation="mowl.evaluation.el_scoring.centre_distance_zero_empty_scores",
        notes=("With all scores equal, the rank of the true answer is its position in "
               "the candidate list (ordinal ties)."),
    ),
    RegimeCard(
        key="overlap-volume-scoring",
        name="Ranking by overlap volume (BoxEL baseline)",
        kind=RegimeKind.EVALUATION,
        summary=(
            "Candidate axioms are scored by one minus the soft volume of the "
            "intersection of the two sides (relative to the transformed left-hand side "
            "for C ⊑ ∃r.D). The score the BoxEL baseline of the Box2EL benchmarks was "
            "evaluated with."),
        components=("scores",),
        applies_to="BoxEL",
        references=(JACKERMEIER2024,),
        standalone=False,
        introduced_with=("boxel",),
        implementation="mowl.evaluation.el_scoring.boxel_overlap_scores",
        limitations=(
            "The reference function for ∃r.C ⊑ D (compute_nf4_ranks_boxel) ranks the "
            "class whose index equals the role index instead of D; the published BoxEL "
            "numbers for that normal form (e.g. Hits@100 0.69 on GALEN) and the combined "
            "numbers that include them cannot be reproduced by a correct evaluation.",),
        notes=("evaluate.py of KRR-Oxford/BoxSquaredEL, compute_nf*_ranks_boxel. For "
               "C ⊓ D ⊑ E the score is one minus the absolute volume of a box in 25-50 "
               "dimensions, which is about 1 for every candidate; the published 0.00 "
               "for that normal form reflects these ties. Reproducible with the recipe "
               "(box2el-code-training with baseline_code): the BoxEL numbers for C ⊑ D, "
               "C ⊓ D ⊑ E, C ⊑ ∃r.D and the deductive-reasoning numbers."),
    ),
)
