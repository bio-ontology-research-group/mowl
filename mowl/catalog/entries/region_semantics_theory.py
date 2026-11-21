"""Region-based semantics from the theory literature.

These works give geometric semantics and constructions with proven guarantees but no
learning procedure for ontologies. Where mOWL realises the construction, the card is
CONSTRUCTIVE and names the :mod:`mowl.geometric` class; the others stay catalogued.
"""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind as K, MethodCard, Origin,
                               Reference, Status)
from mowl.catalog.entries import _refs as R

CONVEX_REGIONS = MethodCard(
    key="convex_regions",
    name="Convex geometric models of existential rules",
    family=Family.GEOMETRIC,
    region="convex region (relations: convex regions over concatenated vectors)",
    summary=(
        "Gutiérrez-Basulto & Schockaert interpret every concept as a convex region of R^n "
        "and every k-ary relation as a convex region of R^(kn) containing the "
        "concatenations of related entity vectors. They characterise which rule languages "
        "such models capture exactly."),
    logic="Quasi-chained existential rules (includes ELHI⊥ in normal form)",
    concepts="convex region in R^n",
    roles="convex region in R^(2n) over concatenated vectors",
    individuals="point",
    expressivity="ELH(∘)⊥",
    status=Status.CONSTRUCTIVE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.geometric.convex_regions.ConvexRegionModel",
    related=("convex_polyhedral", "faithel"),
    limitations=(
        "Constructive only: the paper gives no learning procedure, and mOWL does not add one.",
        "Implemented for concepts (unary) and roles (binary) with the normal forms of "
        "mowl.geometric (NF1-NF4, NF6, NF7, bottom forms). Existential rules of higher "
        "arity, and rules outside the quasi-chained fragment, are not represented.",
        "The input is a finite model, which must be supplied (for instance the canonical "
        "model); the construction does not search for one.",
        "Coordinates are the standard simplex of R^(m-1), an affine image of the paper's "
        "m unit vectors of R^m (the paper notes m - 1 suffices for datalog).",
    ),
    guarantees=(
        Guarantee(
            K.COMPLETENESS,
            "For a quasi-chained ontology and a finite model M of a KB K, K has a convex "
            "geometric model η with φ(η) = M (m-dimensional, m = number of objects of M).",
            "gutierrez2018, Proposition 3",
            condition="by construction",
            verified_in_mowl="tests/geometric/test_faithfulness.py::TestConvexRegionModel::"
                             "test_phi_of_eta_is_m_for_arbitrary_finite_models"),
        Guarantee(
            K.SOUNDNESS,
            "For quasi-chained rules with the finite model property, a KB is satisfiable "
            "iff it has a convex geometric model; such methods are sound, complete, "
            "TBox/ABox-entailed and able to be strongly faithful (ELH).",
            "gutierrez2018, Corollary 1; bourgaux2024, Example 3 and Table 4 (column "
            "'Convex')",
            verified_in_mowl="tests/geometric/test_faithfulness.py::TestConvexRegionModel::"
                             "test_model_of_ontology_gives_convex_model_of_ontology"),
        Guarantee(
            K.LIMITATION,
            "For each n there is a datalog KB over n concept names whose convex geometric "
            "models need at least n − 1 dimensions.",
            "gutierrez2018, Proposition 4"),
        Guarantee(
            K.LIMITATION,
            "Rules outside the quasi-chained fragment can make convex models impossible: "
            "R1(X, Y) ∧ R2(X, Y) → ⊥ with D = {R1(a1,a1), R1(a2,a2), R2(a1,a2), R2(a2,a1)} "
            "is satisfiable but has no convex geometric model (role disjointness).",
            "gutierrez2018, Example 3; bourgaux2024, Example 2",
            verified_in_mowl="tests/geometric/test_faithfulness.py::TestConvexRegionModel::"
                             "test_role_disjointness_example_has_no_convex_model_but_"
                             "construction_fails_there"),
        Guarantee(
            K.LIMITATION,
            "With concepts as convex regions in R^d, ⊥ as ∅ and conjunction as "
            "intersection, the TBox {C1 ⊓ ... ⊓ Ck ⊑ ⊥} with k > d + 1 has no strongly "
            "TBox-faithful model (Helly's theorem).",
            "bourgaux2024, Remark 2"),
        Guarantee(
            K.LIMITATION,
            "Not guaranteed (∀) to be strongly ABox- or TBox-faithful.",
            "bourgaux2024, Table 4 (column 'Convex')"),
    ),
    intersection_closed=True,
    helly="d+1 (convex sets in R^d; Helly 1923, footnote 3 of gutierrez2018)",
    references=(R.GUTIERREZ2018, R.BOURGAUX2024),
    notes=(
        "mOWL applies the construction of Proposition 3 to an arbitrary finite "
        "interpretation (mowl.geometric.convex_regions.FiniteInterpretation) and checks "
        "φ(η) = M by convex-hull membership. Additionally tested, beyond the paper's "
        "statement: η satisfies a normal-form axiom iff M does. The canonical-model "
        "instance is the 'convex_polyhedral' card."),
)

FAITHEL = MethodCard(
    key="faithel",
    name="Strongly faithful ELH embeddings (FaithEL)",
    family=Family.GEOMETRIC,
    region="union of intervals (non-convex, 1-D) / convex hull of one-hot vectors",
    summary=(
        "Lacerda, Ozaki & Guimarães construct region-based models of ELH from the "
        "canonical model that are strongly faithful: an axiom holds in the embedding iff "
        "it is entailed. First with non-convex regions in any dimension (even 1), then "
        "with convex regions in a dimension tied to the canonical model. FaithEL is the "
        "learning method built on these constructions."),
    logic="Normalised ELH (TBox, ABox, role inclusions)",
    concepts="union of intervals (non-convex) or convex hull of element vectors (convex)",
    roles="union of regions in R^2 / convex hull of concatenated vectors",
    individuals="point",
    expressivity="ELH⊥",
    tasks=("subsumption", "satisfiability", "disjointness",),
    status=Status.CONSTRUCTIVE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.geometric.faithfulness.StronglyFaithfulModel",
    related=("interval_union", "convex_polyhedral", "convex_regions"),
    limitations=(
        "Only the constructions of the TGDK article are implemented. The FaithEL learning "
        "procedure (RuleML+RR 2024) is not: that paper could not be obtained and no code "
        "was located, so nothing is trained and no FaithEL numbers are reproduced.",
        "The non-convex construction is the 1-dimensional one (the article proves it for "
        "every dimension m >= 1; mOWL builds m = 1).",
        "Strong faithfulness is tested for TBox and RBox axioms of the forms A <= B, "
        "A and B <= C, A <= exists r.B, exists r.A <= B and r <= s over the concept "
        "names of the normalised ontology (with top on the left or as filler and bottom "
        "as target). Strong IQ faithfulness (instance queries) is not tested here.",
        "mOWL applies the constructions to normalised ELH with bottom; role chains and "
        "nominals are outside the tested forms.",
    ),
    guarantees=(
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "The m-dimensional geometric interpretation of the canonical model "
            "(Definition 12) is a strongly IQ- and TBox-faithful model of a satisfiable "
            "ELH ontology, for every m ≥ 1.",
            "lacerda2024, Theorem 16 and Corollary 17",
            condition="by construction",
            verified_in_mowl="tests/geometric/test_faithfulness.py::TestStrongFaithfulness::"
                             "test_both_constructions_are_strongly_tbox_faithful"),
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "The d-dimensional convex geometric interpretation of the canonical model "
            "(convex hull, Definition 29) is a strongly IQ- and TBox-faithful model of a "
            "normalised ELH ontology.",
            "lacerda2024, Theorems 28, 31 and 32",
            condition="by construction",
            verified_in_mowl="tests/geometric/test_faithfulness.py::TestStrongFaithfulness::"
                             "test_both_constructions_are_strongly_tbox_faithful"),
        Guarantee(
            K.LIMITATION,
            "ELH does not have the strong faithfulness property over convex 1-dimensional "
            "geometric models.",
            "lacerda2024, Theorem 18"),
        Guarantee(
            K.EXPRESSIVITY,
            "Model checking of normal-form CIs, IQs and role inclusions in a finite convex "
            "geometric interpretation takes polynomial time.",
            "lacerda2024, Theorems 35-37"),
    ),
    intersection_closed=True,
    helly="None for the non-convex construction; d+1 for the convex one (d grows with the "
          "canonical model)",
    references=(
        R.LACERDA2024,
        Reference(
            "lacerda2024faithel",
            "Lacerda, V., Ozaki, A., Guimarães, R. (2024). FaithEL: Strongly TBox Faithful "
            "Knowledge Base Embeddings for EL. In Rules and Reasoning (RuleML+RR 2024), "
            "191-199."),
    ),
    notes=(
        "The FaithEL paper itself was not read; only the TGDK article's constructions are "
        "implemented. Strong TBox faithfulness needs the canonical model of Definition 26, "
        "which has an element for every conjunction of two names and every existential "
        "restriction; mOWL's CanonicalModel (one element per name) is faithful on concept "
        "names only, so StronglyFaithfulModel adds the missing elements as fresh names "
        "(mowl.geometric.faithfulness.paper_canonical_ontology). The cards 'interval_union' "
        "and 'convex_polyhedral' use the smaller model."),
)

_AT = "tests/models/test_attributed_dl_model.py::"
_TC = "tests/models/test_temporal_cones_model.py::"
_AT_TEMPORAL = "; ".join((_AT + "test_temporal_during_is_the_conjunction_of_time_points",
                          _AT + "test_conditions_of_theorem_12_are_enforced"))

ATTRIBUTED_DL = MethodCard(
    key="attributed_dl",
    name="Geometric models for (temporally) attributed DLs",
    family=Family.CONSTRUCTIVE,
    region="convex region (simplex face)",
    summary=(
        "Bourgaux, Ozaki & Pan extend convex geometric models to attributed description "
        "logics, whose axioms carry annotations such as validity time, by grounding the "
        "attributed ontology into an equisatisfiable quasi-chained one. mOWL implements "
        "this grounding for ground ontologies and builds the convex model of the grounded "
        "ontology with the polyhedral construction."),
    logic="DL-Lite^H_horn with (temporal) attributes, without negative role inclusions",
    concepts="convex region (face of the standard simplex) per annotated concept name A@S",
    roles="convex hull of concatenated vertex pairs per annotated role name R@S",
    individuals="point (simplex vertex of the nominal)",
    status=Status.CONSTRUCTIVE,
    origin=Origin.ADAPTATION,
    implementation="mowl.geometric.attributed.AttributedConvexModel",
    expressivity="ELH⊥ with ABox",
    tasks=("subsumption", "satisfiability", "instance-checking"),
    guarantees=(
        Guarantee(
            K.COMPLETENESS,
            "Every satisfiable DL-Lite^H,@_horn ontology without negative role inclusions "
            "has a convex geometric model. mOWL constructs it for ground ontologies without "
            "inverse roles (the grounding of set and object variables and inverse roles are "
            "not implemented); satisfiability is the satisfiability of the grounded ontology, "
            "an InconsistentOntology error otherwise.",
            "bourgaux2021, Theorem 6",
            condition="by construction (ground ontologies, no inverse roles)",
            verified_in_mowl=_AT + "test_geometry_is_strongly_faithful_to_the_grounded_"
                                   "ontology_and_convex"),
        Guarantee(
            K.COMPLETENESS,
            "The same holds for the temporal variant DL-Lite^H,T,@_horn when all specifiers "
            "on roles are atemporal and before, after and between do not occur. mOWL "
            "enforces these two conditions (UnsupportedAxiom otherwise) and, because the "
            "grounding for temporal attributes is not in the text of the paper we read, "
            "uses its own (see notes): time, during, since and until, in closed "
            "specifiers.",
            "bourgaux2021, Theorem 12",
            condition="by construction (our grounding of the temporal attributes)",
            verified_in_mowl=_AT_TEMPORAL),
        Guarantee(
            K.LIMITATION,
            "A satisfiable DL-Lite^H,T,@_horn ontology may have no convex geometric model "
            "when role specifiers are temporal (Example 10: R(a,a)@[time:1], R(b,b)@[time:1], "
            "R(a,b)@[time:2], R(b,a)@[time:2], ∃R@[during:[1,2]] ⊑ A, "
            "∃R@[time:1] ⊓ A ⊑ ⊥).",
            "bourgaux2021, Example 10",
            verified_in_mowl=_AT + "test_conditions_of_theorem_12_are_enforced"),
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "The convex model realises the canonical model of the grounded ontology: an "
            "annotated assertion holds in the geometry iff it is entailed by the grounded "
            "ontology (φ(η) = M, Proposition 3 of Gutiérrez-Basulto & Schockaert, used in "
            "the proof of Theorem 6).",
            "bourgaux2021, proof of Theorem 6 (appendix C); gutierrez2018, Proposition 3",
            condition="by construction",
            verified_in_mowl=_AT + "test_geometry_is_strongly_faithful_to_the_grounded_"
                                   "ontology_and_convex"),
    ),
    limitations=(
        "Only ground ontologies: annotations are specifiers over constants. Set variables "
        "and object variables (X : S (...)) of Section 2 are not supported, so the "
        "grounding of Appendix C is only partly implemented.",
        "Inverse roles, negative role inclusions and role conjunction are not supported "
        "(the paper excludes negative role inclusions; inverse roles are outside the EL "
        "canonical model of mowl.geometric).",
        "Only the time points that occur in the ontology are considered; before, after and "
        "between are rejected; temporal attributes need closed specifiers and are rejected "
        "on roles.",
        "No learning procedure: the paper has none (Section 6: 'we still require an "
        "embedding technique that would construct such a model'). The dimension is the "
        "number of canonical elements minus one.",
    ),
    intersection_closed=True,
    helly="d+1 (convex sets in R^d)",
    references=(
        Reference(
            "bourgaux2021",
            "Bourgaux, C., Ozaki, A., Pan, J. Z. (2021). Geometric Models for (Temporally) "
            "Attributed Description Logics. In Proceedings of DL 2021.",
            "https://arxiv.org/abs/2108.12239"),
        R.GUTIERREZ2018,
    ),
    related=("convex_polyhedral", "canonical_model", "convex_regions"),
    notes=(
        "Input is separate from the geometry: an AttributedOntology of Python objects "
        "(Specifier, Basic, ConceptFact, RoleFact, ConceptInc, RoleInc), or OWL axiom "
        "annotations with properties in the namespace http://mowl.borg/attribute# "
        "(from_owl_ontology; for a SubClassOf axiom the annotation is the same closed "
        "specifier on all names, an adaptation). Implemented: grounding E@S -> E_S with "
        "E_S ⊑ E_T for S ⇒ T (proof of Theorem 6), then the polyhedral construction. "
        "Our addition (the text of Theorem 12's proof is not in the paper we read): the "
        "entailments of Definition 7 between time:j, during:(k,l), since:x and until:x "
        "of one concept name as Horn axioms (during and until as a conjunction of time "
        "points; since only implies later time points and since:y for y >= x). The "
        "paper's linear map f is vector concatenation. Region of a specifier that does not "
        "occur in the ontology is empty, as in the paper."),
)

TEMPORAL_CONES = MethodCard(
    key="temporal_cones",
    name="Temporal al-cone models",
    family=Family.CONSTRUCTIVE,
    region="cone (axis-aligned, closed convex)",
    summary=(
        "Leemhuis extends axis-aligned cone semantics for ALC to a Boolean temporal "
        "description logic (LTL_ALC with operators such as next and eventually), "
        "interpreting concepts at each time point as closed convex cones. mOWL builds the "
        "model of the existence proof from an ultimately periodic classical model and "
        "searches such a model for small ontologies."),
    logic="Boolean LTL_ALC (role-free ALC) with constant domains",
    concepts="axis-aligned closed convex cone per time point",
    roles="not covered (the ALC part is role-free)",
    individuals="ray (non-zero point), rigid over time",
    status=Status.CONSTRUCTIVE,
    origin=Origin.ADAPTATION,
    implementation="mowl.geometric.temporal_cones.TemporalALConeInterpretation",
    expressivity="ALC",
    tasks=("satisfiability",),
    guarantees=(
        Guarantee(
            K.COMPLETENESS,
            "A Boolean LTL_ALC ontology with constant domains is satisfiable iff it has a "
            "geometric model on some finite R^n based on a temporal al-cone "
            "interpretation. mOWL implements the direction 'satisfiable implies "
            "geometric model': from a finite ultimately periodic classical model "
            "(from_lasso_model; found by a bounded search in find_lasso_model, which is not "
            "a decision procedure) every distinct element trace becomes a half-axis. "
            "The geometry agrees with the classical semantics for all concepts (membership "
            "of individuals and inclusion) on randomly generated models.",
            "leemhuis2024temporal, Theorem 2",
            condition="by construction, with convex union for until (see notes)",
            verified_in_mowl=_TC + "test_geometry_agrees_with_classical_semantics_on_random_"
                                   "models; " + _TC + "test_example_3_of_the_paper_has_a_"
                                   "geometric_model"),
    ),
    limitations=(
        "No roles (as in the paper), no learning procedure (the paper calls the model 'a first "
        "step towards a learning approach').",
        "Satisfiability is only searched up to a given prefix and period length and an "
        "exponential trace enumeration; the paper gives no bound for completeness.",
        "One half-axis per distinct trace; individuals with equal traces share a point "
        "(the strongly faithful doubled model of Prop. 8 of the ALC paper is not built).",
    ),
    related=("alcone", "alcone-exact"),
    references=(
        Reference(
            "leemhuis2024temporal",
            "Leemhuis, M. (2024). Embedding Temporal Description Logic Ontologies by "
            "Cone-based Geometric Models. In Proceedings of FCR 2024, CEUR-WS Vol-3763.",
            "https://ceur-ws.org/Vol-3763/paper3.pdf"),
    ),
    notes=(
        "Builds on the ALC cone semantics of Özçep, Leemhuis & Wolter (cards 'alcone', "
        "'alcone-exact'; the al-cone class is mowl.geometric.cones.ALCone). Input is "
        "separate from the geometry: temporal concepts and axioms are Python objects "
        "(Next, Until, Eventually, Always plus the ALC classes) or strings (parse_axiom: "
        "'~M & F M <= F (C U M)'); ABox assertions are evaluated at time 0 (our "
        "convention). Deviation: the paper writes until as a set union of al-cone pieces "
        "and notes that the result is no al-cone; with a set union C ⊑ ◇D can fail "
        "geometrically on a classical model, so mOWL uses the convex union (the "
        "disjunction of al-cone semantics) inside until, keeping every concept an al-cone "
        "at every time point (test_convex_union_inside_until_is_needed_for_correctness). "
        "The paper's spheres of radius i+1 are not represented: one al-cone interpretation "
        "per time point in one R^n is equivalent for membership of rays."),
)

CARDS = (CONVEX_REGIONS, FAITHEL, ATTRIBUTED_DL, TEMPORAL_CONES)
