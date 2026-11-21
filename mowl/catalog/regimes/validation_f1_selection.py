"""Model selection by validation F1, as the RegD code does."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import YANG2025_REGD

REGIME = RegimeCard(
    key="validation-f1-selection",
    name="Selection by validation F1 (RegD code)",
    kind=RegimeKind.TRAINING,
    summary=(
        "Every N epochs, classify the first 1,000 validation axioms of the task's normal "
        "form exactly as at test time -- 10 random candidate classes per axiom, the "
        "threshold tuned for F1 on that split itself -- and keep the parameters with the "
        "best F1 (ties: the later epoch)."),
    components=("model selection", "validation"),
    applies_to="EmbeddingELModel with a region-based module (boxes or balls)",
    references=(YANG2025_REGD,),
    standalone=False,
    introduced_with=("regd",),
    used_by=("regd", "regd_elem", "elbe", "elem"),
    implementation="mowl.training.selection.ValidationF1Selection",
    limitations=(
        "Model selection optimises the reported metric, including its threshold, on the "
        "validation split; the quantity is noisy (random negatives, 1,000 axioms).",
        "For the prediction task the positive score carries the reference evaluation's "
        "slip (see regd-sampled-f1), so the selected checkpoint is the best one under "
        "that slip, as in the reference code.",
    ),
    notes=(
        "train() of EXP_ontology/baseline_sweep.py (HuiYang1997/RegD): every 100 epochs "
        "compute_ranks(val_data[nf][:1000], threshold=None) and keep the best F1, "
        "independently of the training loss. The threshold kept with the best "
        "checkpoint is the one the code applies to the test split."),
)
