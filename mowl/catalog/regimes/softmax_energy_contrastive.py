"""The training loss the RegD code uses on the normal form of the task."""

from mowl.catalog.card import Guarantee, GuaranteeKind as K, Reference, RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import YANG2025_REGD

YU2024 = Reference(
    "yu2024shadowcones",
    "Yu, T., Liu, T. J. B., Tseng, A., Sa, C. D. (2024). Shadow Cones: A Generalized "
    "Framework for Partial Order Embeddings. ICLR 2024.",
    "https://openreview.net/forum?id=O2jsRQ2UB9")

REGIME = RegimeCard(
    key="softmax-energy-contrastive",
    name="Softmax over positive and negative energies (RegD code)",
    kind=RegimeKind.TRAINING,
    summary=(
        "For every positive, the cross-entropy of e^-E against its K sampled negatives: "
        "a positive is satisfied when its energy reaches gamma_1 (contained, for "
        "gamma_1 = 0) and a negative when its energy reaches gamma_2 (no longer "
        "contained). Positives and negatives are compared on the same scale, and the "
        "loss stops pushing a negative once it is not contained."),
    components=("loss",),
    applies_to="RegD modules (mowl.nn.el.regd; the raw energies of the negatives "
               "require their scoring context)",
    references=(YANG2025_REGD, YU2024),
    standalone=False,
    introduced_with=("regd",),
    used_by=("regd", "regd_elem"),
    implementation="mowl.training.losses.SoftmaxEnergyContrastiveLoss",
    properties=(
        Guarantee(
            K.REGIME_PROPERTY,
            "With gamma_1 = gamma_2 = 0 the loss of a positive pair can only reach zero "
            "together with all its negatives: the positive must be contained and every "
            "negative not contained, and the loss keeps pushing whichever side is not "
            "yet satisfied, so a zero-loss embedding is a model of the sampled axioms "
            "(the negatives are random and may be entailed; see the limitations).",
            "the loss is -log(exp(-max(E+,0)) / sum exp(-min(E-gamma_2,0)))",
            condition="gamma_1 = gamma_2 = 0 and no entailed negative is drawn"),
    ),
    limitations=(
        "The negatives are random classes and are not checked against the ontology, so "
        "a 'satisfied' negative can be a true subsumption.",
    ),
    notes=(
        "PluginRegD.loss_cross of HuiYang1997/RegD (EXP_ontology/model/pluginRegD.py), "
        "which the paper inherits from the ShadowCones code. It is NOT Eq. 10 of the "
        "paper: Eq. 10 clamps the positive energy only and takes a max-margin over the "
        "negatives' boundary dissimilarity; the code takes a softmax over the full "
        "energies (boundary + depth) of one positive and its 10 negatives, and only on "
        "the normal form of the task (C ⊑ D for the inference task, C ⊑ ∃r.D for the "
        "prediction task; the other normal forms keep the host model's squared-error "
        "loss). The published Tables 3 and 4 were obtained with this loss."),
)
