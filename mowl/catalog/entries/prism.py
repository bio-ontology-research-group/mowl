"""PrismEL: prism embeddings (Federico, Leemhuis & Kutz, DL 2026)."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="prism",
    expressivity="EL⊥ with ABox",
    tasks=("subsumption", "satisfiability", "disjointness",
           "instance-checking", "role-assertion",),
    name="PrismEL",
    family=Family.GEOMETRIC,
    region="prism",
    summary=(
        "Every concept is a prism: an axis-aligned, upward-pointing equilateral triangle "
        "in two support coordinates times an axis-parallel box in the remaining d − 2 "
        "coordinates. Prisms are closed under intersection but, unlike boxes, three "
        "prisms can intersect pairwise without a common point."),
    logic="EL++ concept normal forms and ABox assertions (paper: ELHO(∘))",
    concepts="equilateral triangle × axis-parallel box (2d − 1 parameters)",
    roles="translation vector (federico2026b, Section 6)",
    individuals="point",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.prism.model.PrismEL",
    module="mowl.nn.el.prism.module.PrismModule",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion"),
    guarantees=(
        Guarantee(
            K.CLOSURE,
            "The intersection of two prisms is a prism; under a prism interpretation every "
            "ELHO(∘) concept is interpreted as a prism.",
            "federico2026b, Propositions 5 and 7"),
        Guarantee(
            K.EXPRESSIVITY,
            "There is an ontology with a prism model (already in R^2) but no box model in "
            "any dimension (A ⊓ B ⊓ C ⊑ ⊥ with individuals in each pairwise conjunction).",
            "federico2026b, Corollary 8"),
        Guarantee(
            K.EXPRESSIVITY,
            "For every box interpretation in R^n there is a prism interpretation in "
            "R^(n+2) satisfying exactly the same ELHO(∘) GCIs, assertions and role "
            "inclusions.",
            "federico2026b, Theorem 9"),
        Guarantee(
            K.LIMITATION,
            "For each n there is an EL ontology with a box model in R^n but no prism model "
            "in R^n, so the increase in dimension cannot always be avoided.",
            "federico2026b, Theorem 14"),
        Guarantee(
            K.LIMITATION,
            "A non-empty intersection of finitely many prisms in R^(2+m) equals the "
            "intersection of at most 2n − 1 of them (n = 2 + m).",
            "federico2026b, Lemma 13"),
        Guarantee(
            K.SOUNDNESS,
            "With objective='structural' and margin 0, zero structural box loss plus zero "
            "existence term implies that the materialised prisms satisfy the concept "
            "normal forms with every satisfiable class and listed conjunction non-empty.",
            "mOWL (original): docs/source/geometric/index.rst, 'Structural objective'",
            condition="zero structural loss, margin 0",
            verified_in_mowl="tests/baselines/test_prism_structural_losses.py::"
                             "test_prism_structural_zero_at_exact_t3"),
    ),
    limitations=(
        "Federico et al. give no loss and no experiments ('experimental evaluation left "
        "for future work'); the training objective is mOWL's (ELBE-style MSE on support "
        "coordinate losses, or the structural objective).",
        "Under the structural objective, a defined class that is empty only in its box "
        "part is read as empty (gated by non-emptiness, as in MultiBoxEL; "
        "empty_box_reading='signed', the default). empty_box_reading='absolute' is the "
        "earlier reading that sees a phantom non-empty box in gci2, gci3, gci3_bot and "
        "class_assertion. gci3_bot still shrinks the triangle of C, which conflicts with "
        "the existence term (mOWL geometric documentation, 'Prism non-guarantees').",
        "The objective is non-convex; reaching a zero-loss model is not guaranteed.",
        "The negative loss of C ⊑ ∃r.D uses the exact triangle overlap test (zero iff the "
        "translated triangle and that of D overlap); gci2_negative_triangle='inradius' is "
        "the earlier per-direction test on the inscribed discs. gci3 (∃r.C ⊑ D) only asks "
        "for overlap, as in ELBE, not inclusion.",
    ),
    intersection_closed=True,
    helly=("3. Derived: the triangle factor consists of convex sets in R^2 (Helly number "
           "≤ 3 by Helly's theorem) and federico2026b, Section 5.1 gives three pairwise "
           "intersecting triangles with empty common intersection; the box factor has "
           "Helly number 2 (eckhoff1988), and a product of families intersects iff each "
           "factor does."),
    references=(R.FEDERICO2026_PRISM, R.LEEMHUIS2025, R.ECKHOFF1988),
    notes=(
        "Origin is an adaptation: the geometry follows Definition 4 and Proposition 5 of "
        "Federico et al.; roles as translations follow their Section 6, which notes that prisms "
        "are closed under translation, so translation-based box models carry over "
        "directly (only role-as-region models such as TransBox would need an "
        "approximation, because equilateral triangles are not closed under pointwise "
        "addition). ExactBoxInit can seed a prism containing each initialisation box."),
)
