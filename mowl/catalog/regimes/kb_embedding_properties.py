"""Evaluation regime: semantic properties of KB embeddings (Bourgaux et al., KR 2024)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes.modelness import BOURGAUX2024

REGIME = RegimeCard(
    key="kb-embedding-properties",
    assumption="open world",
    task="axiom-ranking",
    target="deductive inference",
    name="KB embedding properties: model, entailment closure, weak/strong faithfulness",
    kind=RegimeKind.EVALUATION,
    summary=(
        "Given a satisfaction oracle for an embedding (an exact geometric model checker "
        "or zero margin-free loss), checks whether it is a model of the ontology, whether "
        "it satisfies every entailed candidate (TBox entailment closure, Definition 4), "
        "whether every satisfied candidate is consistent with the ontology (weak TBox "
        "faithfulness, Definition 5) and whether every satisfied candidate is entailed "
        "(strong TBox faithfulness, Definition 6)."),
    components=("test set", "metrics"),
    applies_to="Region-based EL++ embeddings with a satisfaction oracle.",
    references=(BOURGAUX2024,),
    standalone=True,
    implementation="mowl.evaluation.properties.KBEmbeddingProperties",
    limitations=(
        "The definitions quantify over all TBox axioms; mOWL checks a finite sample of "
        "concept-name candidates per normal form (generated as for the faithfulness "
        "hierarchy), so True only covers the sample, while False is a counterexample.",
        "Weak faithfulness uses classical consistency by default (witnesses='none'), as "
        "in Definition 5.",
    ),
)
