"""Exact-model check of trained region embeddings."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._benchmark_refs import LEEMHUIS2025

REGIME = RegimeCard(
    key="exact-model-check",
    name="Exact-model and Helly check of trained regions",
    kind=RegimeKind.EVALUATION,
    summary=(
        "Reads the class regions off a trained box or multi-box embedding and decides "
        "exactly which propositional normal-form axioms they satisfy, whether every "
        "satisfiable concept name is non-empty, and how many Helly certificates "
        "(pairwise satisfiable, jointly unsatisfiable triples) the regions violate."),
    components=("model check", "degeneracy", "metrics"),
    applies_to=("models with single-box regions (ELBE, Box2EL, TransBox) or with "
                "all_class_regions() (MultiBoxEL); propositional EL_bot ontologies"),
    references=(LEEMHUIS2025,),
    standalone=False,
    implementation="mowl.evaluation.exact_model.ExactModelRegime",
    limitations=(
        "Role axioms are not checked.",
        "For unions of boxes, an inclusion into a union of two or more boxes is not "
        "decided and is reported as unchecked.",
    ),
    notes=(
        "Defined in mOWL for the helly-tk benchmark; it composes the exact checks of "
        "mowl.geometric (ModelChecker, CanonicalModel, helly_check). The Helly "
        "certificates follow Leemhuis and Kutz (NeSy 2025)."),
)
