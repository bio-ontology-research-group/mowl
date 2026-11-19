"""ELBE / ELBoxEmbeddings (Peng et al., 2022)."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="elbe",
    name="ELBE",
    family=Family.GEOMETRIC,
    region="box",
    summary=(
        "Concepts are axis-parallel boxes (centre and offset) and roles are translation "
        "vectors. Replacing ELEmbeddings' balls by boxes makes the intersection of two "
        "concepts again a region of the same kind."),
    logic="EL++ in normal form (ELO⊥ in the analysis of Bourgaux et al., 2024)",
    concepts="axis-parallel box (centre, offset)",
    roles="translation vector",
    individuals="point (mOWL ABox losses)",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.elbe.model.ELBE",
    module="mowl.nn.el.elbe.module.ELBEModule",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion"),
    guarantees=(
        Guarantee(
            K.CLOSURE,
            "The intersection of two axis-parallel boxes is an axis-parallel box "
            "(intersectional closure), which balls lack.",
            "peng2020, abstract and Section 1"),
        Guarantee(
            K.LIMITATION,
            "Not sound and not complete for ELO⊥ in normal form under the embedding "
            "semantics, and has none of the entailment-closure, weak/strong faithfulness "
            "or full-expressiveness properties of the framework.",
            "bourgaux2024, Table 4"),
        Guarantee(
            K.LIMITATION,
            "Axioms ∃r.C ⊑ ⊥ are approximated by C ⊑ ⊥, which implies but is not "
            "equivalent to them.",
            "bourgaux2024, Section 5"),
        Guarantee(
            K.LIMITATION,
            "Axis-parallel boxes have the Helly property in every dimension, so there is a "
            "satisfiable ontology (e.g. A ⊓ B ⊓ C ⊑ ⊥ with individuals in each pairwise "
            "conjunction, or T_3) with no box model in any R^n. ELBE cannot reach zero loss "
            "on it.",
            "leemhuis2025, Propositions 4 and 5; falcon2022, Appendix A.4, Lemma 1(2)"),
    ),
    limitations=(
        "ELBE has no soundness theorem of its own; mOWL's ELBE trains with an MSE loss "
        "that pushes positive scores to 0 and negative scores to 1.",
    ),
    intersection_closed=True,
    helly="2 (axis-parallel boxes, any dimension; eckhoff1988, as stated in leemhuis2025, "
          "Proposition 4)",
    references=(R.PENG2020, R.BOURGAUX2024, R.LEEMHUIS2025, R.ECKHOFF1988),
    notes=(
        "mOWL adds optional exact-model box initialisation and anchoring "
        "(``init_boxes=mowl.geometric.ExactBoxInit``, ``anchor_weight``); this is a "
        "training aid, not part of the published method. The ``ELBE`` wrapper defaults to "
        "``margin=0`` whereas ``ELBEModule`` defaults to ``margin=0.1``."),
)
