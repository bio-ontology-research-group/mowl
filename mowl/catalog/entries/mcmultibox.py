"""MultiBoxEL, volume-based (Aharoni, 2024)."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="mcmultibox",
    name="MultiBoxEL (Aharoni, volume-based)",
    family=Family.GEOMETRIC,
    region="union of boxes",
    summary=(
        "Concepts are unions of n axis-parallel boxes in R^d and roles are unions of n boxes "
        "in R^{2d} (domain x codomain). Every normal form is trained with the loss "
        "1 - Vol(A ∩ B)/Vol(A), with volumes of unions of boxes estimated on a fixed "
        "quasi-Monte-Carlo point set through a soft membership function; existentials use the "
        "projection of the role onto its domain."),
    logic=("EL++ in theory (finite-model completeness); trained on NF1-NF4, role inclusions "
           "and role chains; no bottom forms, assertions or negatives"),
    concepts="union of n axis-parallel boxes in R^d (centre, offset)",
    roles="union of n axis-parallel boxes in R^{2d} (domain x codomain)",
    individuals="single box (weights exist, no loss uses them)",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.mcmultibox.model.VolumeMultiBoxEL",
    module="mowl.nn.el.mcmultibox.module.VolumeMultiBoxELModule",
    axioms=("gci0", "gci1", "gci2", "gci3", "role_inclusion", "role_chain"),
    guarantees=(
        Guarantee(
            K.COMPLETENESS,
            "Every satisfiable EL++ knowledge base has a multibox interpretation: given a "
            "finite model, one box per domain element and unions of these boxes interpret "
            "concepts and roles (Theorem 5.20); with the finite model property of EL++ "
            "(Theorem 4.29) this gives completeness of multibox spaces for EL++ (stated as "
            "Theorem 5.21).",
            "aharoni2024, Theorem 5.20 (with Theorems 4.29 and 5.21)",
            condition=("the number of boxes per multibox is not bounded: the construction "
                       "uses one box per domain element of a finite model"),
            verified_in_mowl="tests/nn/test_mcmultibox.py::test_t3_admits_an_exact_two_box_"
                             "configuration"),
        Guarantee(
            K.EXPRESSIVITY,
            "ALC is complete with respect to quasi-multibox sets (QuaMultiBox), a modified "
            "multibox space with a Boolean-algebra structure that supports complement and "
            "universal restriction. Representation result only: mOWL does not implement "
            "QuaMultiBox or an ALC loss, and the thesis does not test it.",
            "aharoni2024, Theorem 6.20 (with Theorems 6.18 and 6.19)",
            condition="unbounded number of boxes; not implemented in mOWL"),
    ),
    limitations=(
        "No soundness result: the thesis proves that multibox models exist but not that a "
        "zero loss of the sampled-volume objective is a model (volumes are estimated on a "
        "finite point set with a soft membership function).",
        "Prone to weak local minima (aharoni2024, Section 7.2.1): the loss can be lowered "
        "by shrinking boxes instead of satisfying subsumptions; the thesis calls for "
        "regularisation, which is not part of the model.",
        "Intersections of multiboxes are mostly empty in higher dimensions, so nothing is "
        "learned on NF2; the thesis therefore uses d = 5 (Sections 7.1.3 and 7.3).",
        "The Monte-Carlo estimate needs at least 10,000 points (fewer gave zero evaluation "
        "scores in the thesis, Section 7.1.3); memory limits the number of points and boxes.",
        "Expensive: about 8 hours per training run on a GPU in the thesis (Section 7.1.3).",
        "The role chain loss is the thesis' simplified version, not fully semantic: it "
        "compares dom(R1) x cod(R2) with S and ignores the shared middle element "
        "(Section 7.1.1).",
        "The thesis defines no loss for bottom forms, class or role assertions, and no "
        "negative sampling; mOWL skips these axioms, so disjointness axioms are ignored "
        "and neg_capable_gcis is empty.",
        "The thesis' claim that the inner term of the soft membership exceeds 0.5 iff the "
        "point lies in the multibox holds for a minimum over coordinates but not for the "
        "mean it writes (the mean exceeds 0.5 for every point); mOWL defaults to the "
        "minimum.",
    ),
    intersection_closed=True,
    helly=("not applicable: unions of boxes are not convex, so the Helly number is not "
           "bounded by 2 (our remark; the thesis does not discuss Helly numbers)"),
    references=(R.AHARONI2024, R.FEDERICO2026_MULTIBOX, R.LEEMHUIS2025),
    benchmarks=(),
    notes=(
        "Reimplementation written from the thesis text (Chapter 7); the thesis' own code "
        "was not used. This is the earliest multibox-style EL++ embedding we know of "
        "(2024), predating Federico, Leemhuis and Kutz (2026). It is NOT mOWL's "
        "``MultiBoxEL`` (key ``multiboxel``), which also represents classes as unions of k "
        "boxes but uses Box^2EL inclusion, gated pairwise intersection and role semantics "
        "(distance-based losses on the components); here roles are unions of boxes in R^{2d} "
        "and all losses are sampled volume ratios.\n"
        "Choices of ours, not in the thesis: scrambled Sobol sequence for the QMC points "
        "(deterministic given the seed); aggregate over coordinates 'min' by default "
        "('mean' available); intersection as pointwise minimum of memberships (equal to the "
        "union of pairwise box intersections); the projection is computed exactly on boxes "
        "(a role box contributes its domain box when its codomain box meets the filler) with "
        "a soft overlap indicator instead of sampling the existential witness; bound b_o = 1; "
        "initialisation; Adam with learning rate 0.01; epsilon in the volume ratio; "
        "mini-batches are splits of each normal form into 5 batches with one step each. "
        "The evaluation distance of Section 7.1.2 is implemented for NF1 and NF2 (fraction "
        "of points sampled from the left concept outside the right concept; thesis counts "
        "points inside); for NF3 and NF4 the thesis gives none and the training loss is "
        "used.\n"
        "Thesis hyper-parameters: d = 5, n = 5 boxes, 20,000 points, 5 epochs, 5 batches "
        "(Section 7.1.3).\n"
        "Context: Theorem 4.37 shows that single-box spaces are incomplete for EL++; the "
        "witness is a knowledge base with a:A, b:B, c:C, A, B, C pairwise disjoint and "
        "AB, BC, AC with AB ⊓ BC ⊑ B etc., the same pattern as the T3 counterexample of "
        "Leemhuis and Kutz. mOWL's tests build an exact multibox model of T3 with n = 2 "
        "boxes in d = 1 and d = 2.\n"
        "Benchmark status (checked against Chapter 7 of the thesis): Tables 7.2-7.4 report "
        "mean/median rank and Top10/Top100 per normal form on the Box2EL prediction splits "
        "of GALEN and GO (public data). The experiment is not added as a benchmark: (i) "
        "the ranking score of a multibox (a count of sampled points of one box inside "
        "the candidate, Section 7.1.2) leaves the number of points, the box whose points "
        "are used and the aggregation over boxes unspecified; (ii) the Top-k values move "
        "in steps of 0.002, i.e. about 500 test queries per normal form, but the "
        "sub-sampling is not described; (iii) Top-k is defined with a 0-based rank "
        "(R(i) < k); (iv) the thesis' code is not available. With every Top10 at or "
        "below 0.06 the published numbers would not discriminate a reproduction from a "
        "failure anyway."),
)
