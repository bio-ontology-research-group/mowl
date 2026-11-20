"""ELEmbeddings (Kulmanov et al., 2019)."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="elem",
    name="ELEmbeddings",
    family=Family.GEOMETRIC,
    region="ball",
    summary=(
        "Concepts are open n-balls and roles are translation vectors. Each EL++ normal form "
        "has a hinge loss whose zero set is meant to encode the axiom; individuals are "
        "treated as nominal concepts (balls)."),
    logic="EL++ in normal form (ELO⊥ in the analysis of Bourgaux et al., 2024)",
    concepts="n-ball (centre, radius)",
    roles="translation vector",
    individuals="nominal concept embedded as a ball (in the paper); point in mOWL's ABox losses",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.elembeddings.model.ELEmbeddings",
    module="mowl.nn.el.elem.module.ELEmModule",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion"),
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "If the loss of the normalised TBox is zero, the TBox has a model (balls as "
            "concepts, translations as roles). Bourgaux et al. show the theorem only holds "
            "with a strictly negative margin and with the corrected NF2 loss (radius of A2 "
            "instead of A1 in one term); a strictly negative margin in turn prevents "
            "equivalent concepts from reaching zero loss.",
            "kulmanov2019, Theorem 1 (stated for γ ≤ 0); corrected in bourgaux2024, "
            "Example 5 and footnote 1",
            condition="zero loss, margin γ < 0, corrected NF2 loss"),
        Guarantee(
            K.LIMITATION,
            "Not sound and not complete for ELO⊥ in normal form under the embedding "
            "semantics, and has none of the entailment-closure, weak/strong faithfulness "
            "or full-expressiveness properties of the framework. An ELEm-model exists for "
            "the unsatisfiable KB {A ⊑ ⊥, A(a)} (both balls of radius 0).",
            "bourgaux2024, Table 4 and Example 1"),
        Guarantee(
            K.LIMITATION,
            "For T = {B ⊓ C ⊑ ⊥} and A = {r(a,b), r(a,c), B(b), C(c)} every ELEm-model "
            "makes the ball of a empty, so it satisfies every assertion about a (not weakly "
            "ABox-faithful); for T = {∃r.C ⊑ A, ∃r.D ⊑ B, A ⊓ B ⊑ ⊥} every ELEm-model "
            "satisfies C ⊓ D ⊑ ⊥, which is not entailed (not strongly TBox-faithful).",
            "bourgaux2024, Examples 8 and 10"),
        Guarantee(
            K.LIMITATION,
            "Axioms ∃r.C ⊑ ⊥ are approximated by C ⊑ ⊥, which implies but is not "
            "equivalent to them.",
            "bourgaux2024, Section 5"),
        Guarantee(
            K.LIMITATION,
            "The intersection of two balls is not a ball, so A ⊓ B ≡ C can only be "
            "represented when A ≡ B ≡ C; this breaks with D ⊑ B, D ⊓ C ⊑ ⊥ and D "
            "non-empty.",
            "falcon2022, Appendix A.4, Lemma 1(1)"),
    ),
    limitations=(
        "The authors' released code, and the reference codes derived from it, leave out the "
        "fourth term of the gci1 loss of Eq. (2) of Kulmanov et al., which relates the "
        "radius of the superclass E to the radii of C and D; Theorem 1 does not carry over "
        "to that loss (Mashkova & Hoehndorf 2026, Appendix C). mOWL includes the term by "
        "default (gci1_radius_term=True); gci1_radius_term=False gives the released loss, "
        "which the recipes use where a publication was run with such code. Even with the "
        "term, the intersection of two balls is not a ball, so zero loss is a sufficient "
        "condition only in the sense of the published loss.",
        "Evaluated with the released loss on the Pizza ontology, ELEmbeddings ranks "
        "contradictory C ⊓ D ⊑ E axioms above entailed ones (AUC+/− near 0.02-0.05); "
        "disjointness is separated well (Mashkova & Hoehndorf 2026, Section 6.2, Table 3; "
        "empirical, not a theorem).",
    ),
    intersection_closed=False,
    helly=("d+1 (balls are convex sets in R^d; Helly's theorem as used in bourgaux2024, "
           "Remark 2). Balls are not limited by Helly number 2; their problem is that "
           "intersections of balls are not balls."),
    references=(R.KULMANOV2019, R.BOURGAUX2024, R.MASHKOVA2026, R.FALCON2022),
    notes=(
        "The two-term margin conventions differ between the paper (γ) and mOWL "
        "(``margin``, default 0). ``ExactBoxInit`` does not apply to balls."),
)
