"""Regime cards: how the baselines of the Box2EL code (ELEm, EmEL++, ELBE, BoxEL) are
trained differently from Box2EL itself, found by comparing ``train.py`` and
``model/{Elem,EmELpp,Elbe,BoxEL}.py`` of KRR-Oxford/BoxSquaredEL with mOWL."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import JACKERMEIER2024, KULMANOV2019, MONDAL2021

REGIMES = (
    RegimeCard(
        key="fixed-negatives",
        assumption="closed world over stated axioms",
        name="Negatives drawn once before training",
        kind=RegimeKind.TRAINING,
        summary=(
            "Every training axiom is corrupted once before training; every step then "
            "takes a random sample of these fixed negatives. The model sees the same "
            "negatives throughout training instead of fresh ones at every step."),
        components=("negative sampling",),
        applies_to="EmbeddingELModel",
        references=(JACKERMEIER2024,),
        standalone=False,
        introduced_with=(),
        used_by=("elem", "emelpp", "elbe", "boxel"),
        implementation="mowl.training.negatives.FixedNegatives",
        limitations=(
            "With few negatives per axiom the model can learn to separate exactly these "
            "pairs; the ranking of other candidates is then not trained.",),
        notes=(
            "train.py of KRR-Oxford/BoxSquaredEL: sample_negatives(train_data, 1) is "
            "called once before training for every model with negative_sampling = False "
            "(all baselines); each step takes 512 random rows of the resulting "
            "'nf3_neg0' table (tail- and head-corrupted copies of every C ⊑ ∃r.D), "
            "independently of the positives of the step. Box2EL itself redraws its "
            "negatives at every step. The papers do not mention this."),
    ),
    RegimeCard(
        key="joint-normalised-initialisation",
        name="Joint unit-norm initialisation of centre and size",
        kind=RegimeKind.TRAINING,
        summary=(
            "The centre and the radius (or offset) of a class are drawn uniformly from "
            "[-1, 1] as one vector, which is then scaled to unit norm. Radii and offsets "
            "start small (about 0.06 for ELEm in 200 dimensions) instead of at norm 1."),
        components=("initialisation",),
        applies_to="ELEm, EmEL++, ELBE (modules registered in "
                   "mowl.training.initialisation.JOINT_GROUPS)",
        references=(KULMANOV2019, JACKERMEIER2024),
        standalone=False,
        introduced_with=("elem",),
        used_by=("emelpp", "elbe"),
        implementation="mowl.training.initialisation.JointNormalisedInitialisation",
        notes=(
            "model/Elem.py, model/EmELpp.py (nn.Embedding(classes, dim + 1)) and "
            "model/Elbe.py (nn.Embedding(classes, 2 * dim)) of the Box2EL code, and the "
            "ELEmbeddings code of Kulmanov et al., normalise the whole row. mOWL's "
            "modules keep centre and radius (offset) in separate tables and normalise "
            "each to norm 1, so an ELEm radius starts at |r| = 1."),
    ),
    RegimeCard(
        key="unbounded-negative-loss",
        name="Negative loss without hinge",
        kind=RegimeKind.TRAINING,
        summary=(
            "The negative loss of C ⊑ ∃R.D is r_C + r_D − ‖c + R − d‖ + γ without the "
            "max(0, ·): it never reaches zero, keeps pushing the two balls apart and "
            "shrinks their radii for the whole of training."),
        components=("loss",),
        applies_to="ELEm, EmEL++",
        references=(JACKERMEIER2024,),
        standalone=False,
        introduced_with=(),
        used_by=("elem", "emelpp"),
        implementation="mowl.training.loss_forms.UnboundedNegativeLoss",
        limitations=(
            "The loss is unbounded below; only the unit-norm penalty keeps the centres "
            "from moving apart without limit.",),
        notes=(
            "neg_loss in model/Elem.py and model/EmELpp.py of the Box2EL code. The "
            "ELEmbeddings paper and EmEL++ (Eq. 10) write the loss with a hinge, as "
            "mOWL's modules do by default."),
    ),
    RegimeCard(
        key="per-axiom-norm-penalty",
        name="Unit-norm penalty in every axiom loss",
        kind=RegimeKind.TRAINING,
        summary=(
            "The penalty |‖c‖ − 1| of every class centre occurring in an axiom is added "
            "to the loss of that axiom, for positives and negatives, instead of being "
            "applied once per step to the mean over all classes."),
        components=("loss", "regularisation"),
        applies_to="ELEm (EmEL++ always does this)",
        references=(KULMANOV2019, MONDAL2021, JACKERMEIER2024),
        standalone=False,
        introduced_with=("elem",),
        used_by=("emelpp",),
        implementation="mowl.training.loss_forms.PerAxiomNormPenalty",
        notes=(
            "reg(x) in every nf*Loss of model/Elem.py of the Box2EL code and of the "
            "original ELEmbeddings code; EmEL++, Eq. 1-10. mOWL's ELEm module applies the "
            "penalty once per step (regularization_loss)."),
    ),
    RegimeCard(
        key="box2el-baseline-losses",
        name="Loss details of the Box2EL code's baselines",
        kind=RegimeKind.TRAINING,
        summary=(
            "The loss terms of ELEm, EmEL++ and BoxEL as the Box2EL code implements "
            "them: no hinge on the negative loss and a per-axiom unit-norm penalty (ELEm, "
            "EmEL++), no radius term in C ⊓ D ⊑ E (EmEL++), summed batch losses and the "
            "code's side regulariser (BoxEL)."),
        components=("loss", "regularisation"),
        applies_to="ELEm, EmEL++, BoxEL (other modules unchanged)",
        references=(JACKERMEIER2024,),
        standalone=False,
        introduced_with=(),
        used_by=("elem", "emelpp", "boxel"),
        implementation="mowl.training.loss_forms.Box2ELBaselineLosses",
        limitations=(
            "BoxEL's code regulariser takes the Frobenius norm of all lower corners of a "
            "batch (one number per batch), not a per-box norm.",),
        notes=(
            "model/Elem.py, model/EmELpp.py, model/BoxEL.py of KRR-Oxford/BoxSquaredEL. "
            "Part of box2el-code-training (option baseline_losses / baseline_code)."),
    ),
)
