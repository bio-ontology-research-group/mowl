"""Evaluation regime: loss-level model-ness of a geometric EL++ embedding."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind

BOURGAUX2024 = Reference(
    "bourgaux2024",
    "Bourgaux, C., Guimarães, R., Koudijs, R., Lacerda, V., Ozaki, A. (2024). Knowledge "
    "Base Embeddings: Semantics and Theoretical Properties. In Proceedings of the 21st "
    "International Conference on Principles of Knowledge Representation and Reasoning "
    "(KR 2024), 823-833.",
    "https://doi.org/10.24963/kr.2024/77")

REGIME = RegimeCard(
    key="modelness",
    name="Model-ness (fraction of training axioms with zero margin-free loss)",
    kind=RegimeKind.EVALUATION,
    summary=(
        "For each normal form, the fraction of training axioms whose loss, recomputed "
        "with the margin set to 0, is numerically zero. For losses that vanish exactly "
        "when the regions satisfy the axiom, a fraction of 1 means the embedding is a "
        "model of the training ontology ('soundness w.r.t. the loss', Bourgaux et al. "
        "2024, Property 3)."),
    components=("test set", "metrics"),
    applies_to=("EL embedding models whose module exposes per-normal-form losses and a "
                "'margin' attribute."),
    references=(BOURGAUX2024,),
    standalone=False,
    implementation="mowl.evaluation.faithfulness.LossModelness",
    limitations=(
        "Loss-level only: zero loss implies satisfaction only for losses that are exact "
        "geometric characterisations. The ELEmbeddings GCI1 loss ignores the superclass "
        "radius (Mashkova & Hoehndorf 2026, Appendix C), and some modules include "
        "regularisers in per-axiom losses. Exact checks of exported region models: "
        "mowl.geometric.ModelChecker.",
    ),
    notes="Not a published regime; a diagnostic used in mOWL alongside the Helly benchmarks.",
)
