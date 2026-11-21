"""Exact constructions in :mod:`mowl.geometric` (no training).

``ExactBoxInit`` has no card: it is an initialiser for learnable box models (ELBE, Box²EL,
MultiBoxEL, MultiELBE, PrismEL), not an embedding method, and is mentioned on their cards.
``helly_check`` and ``ModelChecker`` are diagnostics used by the constructions below.
"""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

_DOCS = "mOWL docs, docs/source/geometric/index.rst, 'Guarantees by model and procedure'"

CANONICAL_MODEL = MethodCard(
    key="canonical_model",
    expressivity="ELH(∘)⊥ with ABox",
    tasks=("subsumption", "satisfiability", "disjointness", "instance-checking",
           "role-assertion", "classification", "consistency",),
    name="Canonical model (EL++ completion)",
    family=Family.CONSTRUCTIVE,
    region="finite canonical interpretation (one element per satisfiable name)",
    summary=(
        "Saturates a normalised EL++ TBox, RBox and ABox with an ELK-style completion "
        "calculus and reads off the canonical model: one domain element per satisfiable "
        "concept or nominal name. It is the basis of every construction in "
        "mowl.geometric."),
    logic="Normalised EL++ (NF1-NF4, NF6, NF7, ⊥ forms) with atomic ABox assertions",
    concepts="set of canonical elements",
    roles="relation between canonical elements",
    individuals="fresh nominal concept name N_a",
    status=Status.CONSTRUCTIVE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.geometric.completion.CanonicalModel",
    guarantees=(
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "Sound and complete on concept-name subsumption and unsatisfiability with "
            "respect to the normalised axiom set; raises InconsistentOntology iff O ⊨ ⊤ ⊑ ⊥ "
            "or some individual is unsatisfiable.",
            _DOCS + ", citing kazakov2014",
            condition="by construction",
            verified_in_mowl="tests/geometric/test_completion.py::TestCompletion::"
                             "test_matches_elk_on_normalised"),
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "Instance checking and assertion contradiction are sound and complete for ELH⊥ "
            "without nominals, functionality or range axioms, via the nominal encoding N_a.",
            _DOCS + ", 'Limitations and caveats' item 1",
            condition="by construction, ELH⊥ fragment"),
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "mOWL's canonical model (one element per concept name) satisfies every "
            "entailed axiom, and A ⊑ B and A ⊑ ∃r.B exactly when they are entailed (tested "
            "on small ontologies). It also satisfies some non-entailed A ⊓ B ⊑ C, ∃r.A ⊑ B "
            "and r ⊑ s, so the full statement of Theorem 27 does not hold for it.",
            "lacerda2024, Theorem 27 (restricted: mOWL's model is smaller than Definition 26)",
            condition="by construction for A ⊑ B; tested for A ⊑ ∃r.B",
            verified_in_mowl="tests/geometric/test_faithfulness.py::TestStrongFaithfulness::"
                             "test_name_only_canonical_model_is_faithful_only_on_some_forms"),
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "The canonical model of Definition 26 (extra elements for every conjunction of two "
            "names and every existential restriction), built by "
            "mowl.geometric.faithfulness.StronglyFaithfulModel, satisfies A ⊑ B, "
            "A ⊓ B ⊑ C, A ⊑ ∃r.B, ∃r.A ⊑ B and r ⊑ s over the signature iff they are "
            "entailed (tested on small ELH⊥ ontologies).",
            "lacerda2024, Theorem 27 and Theorems 28, 32",
            condition="by construction, ELH",
            verified_in_mowl="tests/geometric/test_faithfulness.py::TestStrongFaithfulness::"
                             "test_both_constructions_are_strongly_tbox_faithful"),
    ),
    limitations=(
        "Unsupported ABox axioms (complex class assertions, same/different individuals, "
        "inverse-property assertions, negative or data assertions) raise "
        "UnsupportedAxiomError in normalise().",
    ),
    references=(R.KAZAKOV2014, R.BAADER2005, R.LACERDA2024),
    notes=(
        "The soundness and completeness statement is taken from the mOWL geometric "
        "documentation, which cites Kazakov et al. (2014); it is checked in mOWL by a "
        "differential test against the ELK reasoner. mOWL's canonical model is not "
        "literally Definition 26 of Lacerda et al. (see the guarantees above); "
        "StronglyFaithfulModel builds Definition 26."),
)

CONVEX_POLYHEDRAL = MethodCard(
    key="convex_polyhedral",
    expressivity="ELH(∘)⊥",
    tasks=("subsumption", "satisfiability", "disjointness",),
    name="Convex polyhedral model",
    family=Family.CONSTRUCTIVE,
    region="convex polytope (simplex face)",
    summary=(
        "Maps every canonical element to a vertex of the standard simplex, every concept "
        "name to the face spanned by its elements, and every role to the convex hull of "
        "concatenated vertex pairs. The result is an exact convex model."),
    logic="Full normalised EL++ (propositional only when thicken=True)",
    concepts="face of the standard simplex in R^(n-1) (or a thickened full-dimensional face)",
    roles="convex hull of concatenated vertex pairs in R^(2(n-1))",
    individuals="simplex vertex of the nominal N_a",
    status=Status.CONSTRUCTIVE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.geometric.polyhedral.ConvexPolyhedralModel",
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "The constructed interpretation is an exact, non-degenerate model of the "
            "normalised ontology; every model is verified by ModelChecker before it is "
            "returned.",
            "gutierrez2018, Proposition 3; " + _DOCS,
            condition="by construction",
            verified_in_mowl="tests/geometric/test_polyhedral.py::TestConvexPolyhedralModel::"
                             "test_fixtures_and_family_are_exact_nondegenerate_strongly_"
                             "faithful_models"),
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "Strongly faithful on concept names: η(A) ⊆ η(B) iff O ⊨ A ⊑ B (the "
            "construction realises the canonical model, φ(η) = M).",
            "gutierrez2018, Proposition 3 (Theorem 32 of lacerda2024 needs the larger Definition 26 "
            "model: see card faithel)",
            condition="by construction"),
        Guarantee(
            K.COMPLETENESS,
            "For quasi-chained rule ontologies with the finite model property (hence "
            "ELHI⊥ in normal form), a KB is satisfiable iff it has a convex geometric "
            "model.",
            "gutierrez2018, Corollary 1; bourgaux2024, Example 3 and Table 4",
            condition="always"),
        Guarantee(
            K.LIMITATION,
            "For each n there is a datalog KB over n concept names whose convex models need "
            "dimension at least n − 1, so a dimension linear in the number of names is "
            "unavoidable in the worst case.",
            "gutierrez2018, Proposition 4"),
        Guarantee(
            K.LIMITATION,
            "For a language with conjunction interpreted as intersection of convex regions "
            "in R^d, the TBox {C1 ⊓ ... ⊓ Ck ⊑ ⊥} with k > d + 1 has no strongly "
            "TBox-faithful model.",
            "bourgaux2024, Remark 2"),
    ),
    limitations=(
        "The ambient dimension is n − 1 with n the number of canonical elements (including "
        "auxiliary names and ⊤); this is an upper bound, not minimal.",
        "Without thicken=True concept regions are lower-dimensional faces with zero volume; "
        "thicken=True is only allowed for role-free ontologies.",
        "Bounding-box export (to_bounding_boxes) is lossy and is not a model.",
    ),
    intersection_closed=True,
    helly="d+1 with d = n − 1 (convex sets in R^d; Helly 1923 as used in bourgaux2024, "
          "Remark 2); the dimension grows with the ontology, so this is not a fixed bound.",
    references=(R.GUTIERREZ2018, R.LACERDA2024, R.BOURGAUX2024),
    notes=(
        "Realises the construction of Gutiérrez-Basulto & Schockaert (card "
        "``convex_regions``) for EL++, and corresponds to the convex construction of "
        "Lacerda et al. (card ``faithel``)."),
)

INTERVAL_UNION = MethodCard(
    key="interval_union",
    expressivity="ELH(∘)⊥",
    tasks=("subsumption", "satisfiability", "disjointness",),
    name="Interval-union model",
    family=Family.CONSTRUCTIVE,
    region="union of intervals (1-D)",
    summary=(
        "Assigns each canonical element its own interval on the real line; a concept is the "
        "union of the intervals of its elements and a role is a finite union of "
        "rectangles in R^2. The result is an exact, non-convex model in one dimension."),
    logic="Full normalised EL++",
    concepts="finite union of closed intervals in R",
    roles="finite union of axis-aligned rectangles in R^2",
    individuals="point in the interval of N_a",
    status=Status.CONSTRUCTIVE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.geometric.intervals.IntervalUnionModel",
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "For all ELH axioms α, a classical interpretation satisfies α iff its geometric "
            "interpretation (Definition 12) does; mOWL verifies every constructed model "
            "with ModelChecker.",
            "lacerda2024, Definition 12 and Theorem 13; " + _DOCS,
            condition="by construction",
            verified_in_mowl="tests/geometric/test_intervals.py::TestIntervalUnionModel::"
                             "test_fixtures_and_family_are_exact_nondegenerate_strongly_"
                             "faithful_models"),
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "The geometric interpretation of the canonical model of Definition 26 is strongly "
            "IQ- and TBox-faithful; ELH has the strong faithfulness property over "
            "m-dimensional geometric interpretations for every m ≥ 1. This card's class "
            "uses mOWL's smaller canonical model, for which only concept-name subsumption "
            "is exact; the faithful model is mowl.geometric.faithfulness.StronglyFaithfulModel "
            "(regions=\"intervals\", m = 1).",
            "lacerda2024, Theorem 16 and Corollary 17",
            condition="by construction (Definition 26 model)",
            verified_in_mowl="tests/geometric/test_faithfulness.py::TestStrongFaithfulness::"
                             "test_both_constructions_are_strongly_tbox_faithful"),
        Guarantee(
            K.COMPLETENESS,
            "Every satisfiable ELHO(∘) ontology has a multibox model in R^n for every n; "
            "the interval-union model is the n = 1 case.",
            "federico2026a, Theorem 3",
            condition="always"),
        Guarantee(
            K.LIMITATION,
            "ELH does not have the strong faithfulness property over convex 1-dimensional "
            "geometric models, so non-convexity is needed in one dimension.",
            "lacerda2024, Theorem 18"),
    ),
    intersection_closed=True,
    helly="None (non-convex regions; no Helly bound applies)",
    references=(R.LACERDA2024, R.FEDERICO2026_MULTIBOX),
    notes=(
        "Lacerda et al. prove the results for ELH; mOWL applies the construction to full "
        "normalised EL++ (⊥, role chains, nominals encoded as names) and relies on "
        "ModelChecker to verify each model. Unions of intervals are closed under "
        "intersection (with more intervals). Realises the non-convex construction "
        "behind FaithEL (card ``faithel``)."),
)

BOX_MILP = MethodCard(
    key="box_milp",
    expressivity="propositional EL⊥",
    tasks=("subsumption", "satisfiability", "disjointness",),
    name="Box model builder (MILP)",
    family=Family.CONSTRUCTIVE,
    region="box",
    summary=(
        "Searches for an exact axis-parallel box model of a propositional EL⊥ ontology with "
        "a mixed-integer linear programme, after a Helly check that can prove that no "
        "non-degenerate box model exists in any dimension."),
    logic="Propositional EL⊥ (NF1, NF2 and ⊥ forms; no roles)",
    concepts="axis-parallel box",
    roles="not supported",
    status=Status.CONSTRUCTIVE,
    origin=Origin.ORIGINAL,
    implementation="mowl.geometric.boxes.BoxModelBuilder",
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "Any returned BoxModel (including a time-limited incumbent) has been verified "
            "by ModelChecker; otherwise BoxModelNotFound is raised.",
            _DOCS,
            condition="by construction (verified output)",
            verified_in_mowl="tests/geometric/test_boxes.py::TestBoxes::"
                             "test_time_limited_incumbent_is_post_checked"),
        Guarantee(
            K.LIMITATION,
            "NotBoxRepresentable is raised iff a non-degenerate Helly certificate is found, "
            "which proves that no non-degenerate box model exists in any dimension "
            "(axis-parallel boxes have the Helly property).",
            "leemhuis2025, Propositions 4 and 5; eckhoff1988",
            verified_in_mowl="tests/geometric/test_boxes.py::TestBoxes::"
                             "test_t3_not_representable"),
    ),
    limitations=(
        "A passing Helly check does not imply that a box model exists (no sufficiency "
        "theorem is known); the MILP may be inconclusive and is exponential in the worst "
        "case.",
        "Searches dimensions 1, 2, 4, ... up to d_max (default 8); returned boxes are often "
        "zero-width.",
        "With role axioms, Helly-ABox-closure diagnostics are incomplete: a satisfiable, "
        "ABox-closed ontology can fail to be Helly-satisfiable (leemhuis2026, preprint v1, "
        "Proposition 38).",
    ),
    intersection_closed=True,
    helly="2 (axis-parallel boxes, any dimension; eckhoff1988, as stated in leemhuis2025, "
          "Proposition 4)",
    references=(R.LEEMHUIS2025, R.LEEMHUIS2026, R.ECKHOFF1988),
    notes=(
        "Original to mOWL. The Helly diagnostics (mowl.geometric.helly.helly_check, "
        "certificate_triangles) implement the Helly-satisfiability ideas of Leemhuis & "
        "Kutz. ExactBoxInit(method='milp') uses this builder to seed learnable box models."),
)

CARDS = (CANONICAL_MODEL, CONVEX_POLYHEDRAL, INTERVAL_UNION, BOX_MILP)
