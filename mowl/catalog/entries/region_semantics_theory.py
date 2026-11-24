"""Region-based semantics from the theory literature (catalogued only).

These works give geometric semantics and constructions with proven guarantees but no
learning procedure for ontologies. Where mOWL realises the construction, the card says
which :mod:`mowl.geometric` class does.
"""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Reference
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
    not_implemented_reason=(
        "The paper gives a construction from a finite model, not a learning method. mOWL "
        "implements the EL++ instance of the construction as "
        "mowl.geometric.polyhedral.ConvexPolyhedralModel (card 'convex_polyhedral'); "
        "general quasi-chained rules of higher arity are outside mOWL's OWL-based data "
        "model, and no trainable convex-region learner is implemented."),
    guarantees=(
        Guarantee(
            K.COMPLETENESS,
            "For a quasi-chained ontology and a finite model M of a KB K, K has a convex "
            "geometric model η with φ(η) = M (m-dimensional, m = number of objects of M).",
            "gutierrez2018, Proposition 3"),
        Guarantee(
            K.SOUNDNESS,
            "For quasi-chained rules with the finite model property, a KB is satisfiable "
            "iff it has a convex geometric model; such methods are sound, complete, "
            "TBox/ABox-entailed and able to be strongly faithful (ELH).",
            "gutierrez2018, Corollary 1; bourgaux2024, Example 3 and Table 4 (column "
            "'Convex')"),
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
            "gutierrez2018, Example 3; bourgaux2024, Example 2"),
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
    not_implemented_reason=(
        "The FaithEL training procedure is published in a RuleML+RR 2024 paper that we "
        "could not obtain, and no code was located. The constructions themselves are in "
        "mOWL: the 1-D non-convex one (Definition 12, Theorem 16) as "
        "mowl.geometric.intervals.IntervalUnionModel (card 'interval_union') and the "
        "convex one (Theorem 32) corresponds to "
        "mowl.geometric.polyhedral.ConvexPolyhedralModel (card 'convex_polyhedral')."),
    guarantees=(
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "The m-dimensional geometric interpretation of the canonical model "
            "(Definition 12) is a strongly IQ- and TBox-faithful model of a satisfiable "
            "ELH ontology, for every m ≥ 1.",
            "lacerda2024, Theorem 16 and Corollary 17",
            condition="by construction"),
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "The d-dimensional convex geometric interpretation of the canonical model "
            "(convex hull, Definition 29) is a strongly IQ- and TBox-faithful model of a "
            "normalised ELH ontology.",
            "lacerda2024, Theorems 28, 31 and 32",
            condition="by construction"),
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
    notes="The FaithEL paper itself was not read; only the TGDK article's constructions are "
          "recorded.",
)

ATTRIBUTED_DL = MethodCard(
    key="attributed_dl",
    name="Geometric models for (temporally) attributed DLs",
    family=Family.GEOMETRIC,
    region="convex region",
    summary=(
        "Bourgaux, Ozaki & Pan extend convex geometric models to attributed description "
        "logics, whose axioms carry annotations such as validity time, by grounding the "
        "attributed ontology into an equisatisfiable quasi-chained one."),
    logic="DL-Lite^H_horn with (temporal) attributes, without negative role inclusions",
    concepts="convex region",
    roles="convex region over concatenated vectors",
    individuals="point",
    not_implemented_reason=(
        "Theory only (no learning procedure), and attributed/temporal annotations are not "
        "represented in mOWL's OWL-based datasets."),
    guarantees=(
        Guarantee(
            K.COMPLETENESS,
            "Every satisfiable DL-Lite^H,@_horn ontology without negative role inclusions "
            "has a convex geometric model.",
            "bourgaux2021, Theorem 6"),
        Guarantee(
            K.COMPLETENESS,
            "The same holds for the temporal variant DL-Lite^H,T,@_horn when all specifiers "
            "on roles are atemporal and before, after and between do not occur.",
            "bourgaux2021, Theorem 12"),
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
)

TEMPORAL_CONES = MethodCard(
    key="temporal_cones",
    name="Temporal al-cone models",
    family=Family.GEOMETRIC,
    region="cone (axis-aligned, closed convex)",
    summary=(
        "Leemhuis extends axis-aligned cone semantics for ALC to a Boolean temporal "
        "description logic (LTL_ALC with operators such as next and eventually), "
        "interpreting concepts at each time point as closed convex cones."),
    logic="Boolean LTL_ALC (role-free ALC) with constant domains",
    concepts="axis-aligned closed convex cone per time point",
    roles="not covered (the ALC part is role-free)",
    individuals="point",
    not_implemented_reason=(
        "Theory only (no learning procedure); temporal DLs are not supported by mOWL's "
        "datasets or normalisers."),
    guarantees=(
        Guarantee(
            K.COMPLETENESS,
            "A Boolean LTL_ALC ontology with constant domains is satisfiable iff it has a "
            "geometric model on some finite R^n based on a temporal al-cone "
            "interpretation.",
            "leemhuis2024temporal, Theorem 2"),
    ),
    references=(
        Reference(
            "leemhuis2024temporal",
            "Leemhuis, M. (2024). Embedding Temporal Description Logic Ontologies by "
            "Cone-based Geometric Models. In Proceedings of FCR 2024, CEUR-WS Vol-3763.",
            "https://ceur-ws.org/Vol-3763/paper3.pdf"),
    ),
    notes="Builds on the ALC cone semantics of Özçep, Leemhuis & Wolter (card 'alcone').",
)

CARDS = (CONVEX_REGIONS, FAITHEL, ATTRIBUTED_DL, TEMPORAL_CONES)
