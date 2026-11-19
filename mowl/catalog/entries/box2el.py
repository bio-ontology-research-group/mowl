"""Box²EL (Jackermeier et al., 2024)."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="box2el",
    name="Box²EL",
    family=Family.GEOMETRIC,
    region="box",
    summary=(
        "Concepts are axis-parallel boxes with an additional bump vector; each role is a "
        "pair of boxes (head and tail). An axiom C ⊑ ∃r.D holds when the box of C, "
        "translated by the bump of D, lies in the head box of r. This lets roles be "
        "many-to-many."),
    logic="EL++ in normal form (ELHO(∘)⊥ in the analysis of Bourgaux et al., 2024)",
    concepts="axis-parallel box plus bump vector",
    roles="two boxes (head, tail)",
    individuals="point (as a degenerate box) with a bump",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.boxsquaredel.model.BoxSquaredEL",
    module="mowl.nn.el.boxsquaredel.module.BoxSquaredELModule",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion"),
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "If γ ≤ 0 and the Box²EL loss of O is zero, the embeddings (concepts as their "
            "boxes, roles as the product of head and tail boxes) are a model of O.",
            "jackermeier2024, Theorem 1 (proof: Appendix C)",
            condition="zero loss, γ ≤ 0"),
        Guarantee(
            K.EXPRESSIVITY,
            "Fully ABox-expressive.",
            "bourgaux2024, Table 4"),
        Guarantee(
            K.LIMITATION,
            "Not sound under the embedding semantics (Property 1): a Box²EL-model with "
            "non-zero bumps exists for an unsatisfiable KB. At zero loss all bumps are 0, "
            "so A ⊑ ∃r.B and C ⊑ ∃r.D imply A ⊑ ∃r.D and C ⊑ ∃r.B. Not complete.",
            "bourgaux2024, Example 6 and Table 4"),
        Guarantee(
            K.LIMITATION,
            "Some Box²EL-models are not ABox-entailed, and for {r1 ∘ r2 ⊑ r3, ∃r3.C ⊑ D} "
            "every Box²EL-model satisfies ∃r1.C ⊑ D, which is not entailed (role "
            "composition is not represented faithfully).",
            "bourgaux2024, Examples 7 and 9"),
        Guarantee(
            K.LIMITATION,
            "Boxes have the Helly property in every dimension, so some satisfiable "
            "ontologies have no box model in any R^n and no zero-loss Box²EL embedding.",
            "leemhuis2025, Propositions 4 and 5"),
    ),
    limitations=(
        "Evaluated on the Pizza ontology, Box²EL separates entailed from contradictory "
        "C ⊓ D ⊑ E axioms only weakly (AUC+/− 0.39) and handles disjointness well "
        "(Mashkova & Hoehndorf 2026, Section 6.2; empirical).",
        "The authors' implementation defines the 'intersection' of disjoint intervals as "
        "the gap between them (yang2025, Section 4.4).",
    ),
    intersection_closed=True,
    helly="2 (axis-parallel boxes, any dimension; eckhoff1988, as stated in leemhuis2025, "
          "Proposition 4)",
    references=(R.JACKERMEIER2024, R.BOURGAUX2024, R.LEEMHUIS2025, R.MASHKOVA2026),
    code_url="https://github.com/KRR-Oxford/BoxSquaredEL",
    notes=(
        "The mOWL class is ``BoxSquaredEL``. mOWL adds optional exact-model box "
        "initialisation (``init_boxes``) and anchoring (``anchor_weight``); these are "
        "training aids, not part of the published method. MultiBoxEL with one component "
        "reproduces Box²EL's losses exactly."),
)
