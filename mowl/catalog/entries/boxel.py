"""BoxEL (Xiong et al., 2022)."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="boxel",
    expressivity="EL⊥ with ABox",
    tasks=("subsumption", "satisfiability", "disjointness",
           "instance-checking", "role-assertion",),
    name="BoxEL",
    family=Family.GEOMETRIC,
    region="box",
    summary=(
        "Concepts are axis-parallel boxes given by a lower corner and a size, individuals "
        "are points, and roles are affine transformations (diagonal scaling plus "
        "translation). Losses are based on a soft-volume disjointness measure."),
    logic="EL++ in normal form (ELO⊥ in the analysis of Bourgaux et al., 2024)",
    concepts="axis-parallel box (min corner, delta)",
    roles="affine transformation (diagonal scaling and translation)",
    individuals="point",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.boxel.model.BoxEL",
    module="mowl.nn.el.boxel.module.BoxELModule",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion"),
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "Per normal form: if the loss term of C ⊑ D, C ⊓ D ⊑ E, C ⊑ ∃r.D or ∃r.C ⊑ D is "
            "zero, the geometric interpretation satisfies the axiom (excluding the "
            "inconsistent nominal cases {a} ⊑ ⊥ and {a} ⊓ {a} ⊑ ⊥).",
            "xiong2022, Propositions 2-5 (proofs: Propositions 7-10)",
            condition="zero loss"),
        Guarantee(
            K.SOUNDNESS,
            "Sound under the embedding semantics: the existence of a BoxEL-model implies "
            "that the KB is satisfiable (individuals are points, so they cannot be "
            "embedded as empty regions as in ELEm).",
            "bourgaux2024, Table 4 and Example 1",
            condition="always"),
        Guarantee(
            K.LIMITATION,
            "Not complete, and has none of the entailment-closure, weak/strong faithfulness "
            "or full-expressiveness properties of the framework.",
            "bourgaux2024, Table 4"),
        Guarantee(
            K.LIMITATION,
            "Boxes have the Helly property in every dimension, so some satisfiable "
            "ontologies have no box model in any R^n. Under its box semantics a BoxEL "
            "embedding is a box interpretation, so such ontologies admit no zero-loss "
            "BoxEL embedding.",
            "leemhuis2025, Propositions 4 and 5"),
        Guarantee(
            K.LIMITATION,
            "The box space is incomplete with respect to EL++: some satisfiable EL++ KBs "
            "have no box interpretation.",
            "aharoni2024, Theorem 4.37"),
    ),
    limitations=(
        "Inverse role transformations assume non-zero scaling entries (xiong2022, NF4 "
        "discussion).",
    ),
    intersection_closed=True,
    helly="2 (axis-parallel boxes, any dimension; eckhoff1988, as stated in leemhuis2025, "
          "Proposition 4)",
    references=(R.XIONG2022, R.BOURGAUX2024, R.LEEMHUIS2025, R.AHARONI2024),
    code_url="https://github.com/Box-EL/BoxEL",
)
