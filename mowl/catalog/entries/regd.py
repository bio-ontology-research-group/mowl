"""RegD (Yang and Chen, 2026)."""

import dataclasses

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind as K, MethodCard, Origin,
                               Reference, Status)
from mowl.catalog.entries import _refs as R

YANG2025_REGD = Reference(
    "yang2025regd",
    "Yang, H., Chen, J. (2025). RegD: Hierarchical Embeddings via Dissimilarity between "
    "Arbitrary Euclidean Regions. arXiv:2501.17518 (v3, 2026; KR 2026).",
    "https://arxiv.org/abs/2501.17518")

CARD = MethodCard(
    key="regd",
    expressivity="EL⊥ with ABox",
    tasks=("subsumption", "satisfiability", "disjointness", "instance-checking",
           "role-assertion"),
    name="RegD",
    family=Family.GEOMETRIC,
    region="box or ball (dissimilarity-based)",
    summary=(
        "A replacement for the inclusion energy between two regions: a boundary "
        "dissimilarity (zero or negative iff the child region is inside the parent) plus a "
        "depth dissimilarity that divides a parameter distance by the product of the region "
        "sizes, which gives hyperbolic-like capacity in low dimension. In ontology "
        "embedding it is plugged into ELBE (boxes) and ELEm (balls) in place of their "
        "inclusion energy, with a contrastive loss with sampled negatives."),
    logic="hierarchies (partial orders); inside EL++ normal forms through the host model",
    concepts="axis-parallel box (RegDELBE) or ball (RegDELEm)",
    roles="translation vector (host model)",
    individuals="point (host model's ABox losses)",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.regd.model.RegDELBE",
    module="mowl.nn.el.regd.module.RegDELBEModule",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion"),
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Depth dissimilarity of balls (and, with offsets, of boxes): (1) for two regions "
            "and any gap Δ, every sufficiently small region inside the second one is more "
            "than Δ further from the first than the second is; (2) any ball contains n "
            "sub-balls whose pairwise depth dissimilarity exceeds any M. This is the "
            "hyperbolic-like capacity (unbounded separation of layers and of siblings). "
            "Theorem 2 states the same for general parameterised regions.",
            "yang2025regd, Theorem 1 (balls; the same for boxes) and Theorem 2",
            condition="always (a property of the dissimilarity, not of a trained model)",
            verified_in_mowl="tests/nn/test_regd.py::TestDepthDissimilarityTheorems"),
        Guarantee(
            K.EXPRESSIVITY,
            "With p = 2, g(x) = arcosh(x + 1) and f(r) = sqrt(2) r, the depth dissimilarity "
            "of balls is the hyperbolic distance between the points (c, r) of the upper "
            "half-space (Prop. 1); with linear g it preserves the order of hyperbolic "
            "distances (Prop. 2). mOWL uses g(x) = x.",
            "yang2025regd, Propositions 1 and 2",
            condition="always",
            verified_in_mowl="tests/nn/test_regd.py::TestDepthDissimilarityTheorems::"
                             "test_proposition_1_isometry_with_hyperbolic_space"),
        Guarantee(
            K.EXPRESSIVITY,
            "The boundary dissimilarity is exact for inclusion: d_bd(reg1, reg2) <= 0 iff "
            "reg2 is a subset of reg1, and it is 0 iff reg2 touches the boundary of reg1 from "
            "inside (the printed statement says 'reg1 is internally tangent to reg2', which "
            "contradicts its own proof). It is monotone along inclusion chains.",
            "yang2025regd, Proposition 3 and the remark after Definition 2",
            condition="always",
            verified_in_mowl="tests/nn/test_regd.py::TestBoundaryDissimilarityProperties"),
        Guarantee(
            K.EXPRESSIVITY,
            "Faithful embedding of any partial order in R^2: for every partially ordered set "
            "(V, <) there are regions f(v) in R^2 with v < v' iff f(v) is a subset of f(v') "
            "for all distinct v, v' (Definition 4.1). The proof uses convex hulls of points on "
            "a regular polygon, not boxes or balls. It concerns hierarchies only: it says "
            "nothing about conjunction, existential restrictions or disjointness, and is an "
            "existence statement, not a property of a trained RegD model.",
            "yang2025regd, Definition 4.1 and Theorem 4.2",
            condition="existence, with arbitrary (convex polygon) regions",
            verified_in_mowl="tests/nn/test_regd.py::TestFaithfulnessConstruction"),
    ),
    limitations=(
        "Hierarchies only. RegD replaces the inclusion energy; conjunction, existential "
        "restrictions, bottom and assertions are handled by the host model (ELBE or ELEm), "
        "and inherit its guarantees and limitations. The paper proves no soundness or "
        "completeness theorem for the trained ontology embeddings.",
        "The paper reports that RegD does not always help on the prediction task (Table 4: "
        "GALEN with ELBE, Anatomy with ELEM).",
    ),
    helly="",
    references=(YANG2025_REGD, R.PENG2020, R.KULMANOV2019),
    code_url="https://github.com/HuiYang1997/RegD",
    benchmarks=tuple(f"regd-{o}-{t}" for o in ("galen", "go", "anatomy")
                     for t in ("inference", "prediction")),
    notes=(
        "Two models: RegDELBE (boxes, implementation above) and RegDELEm "
        "(mowl.models.regd.model.RegDELEm, balls). Replaced inclusion energies follow the "
        "authors' code: gci0, gci1 (ELBE only, plus ELBE's non-emptiness term), gci2 (child "
        "translated by the role) and gci3. The built-in loss of these classes is Eq. 10 "
        "(mowl.training.LogSumExpContrastiveLoss applied to gci0 and gci2 with 10 "
        "negatives); the recipes reproduce the paper's numbers with the code's loss "
        "instead (training regime 'regd-code-training': the softmax cross-entropy of "
        "mowl.training.SoftmaxEnergyContrastiveLoss on the task's normal form only, the "
        "host model's squared-error loss elsewhere). Deviations and ambiguities: Eq. 9 "
        "adds the depth term to every positive pair but the authors' training code only "
        "to non-contained pairs in training (flag depth_on_containment: true is Eq. 9, false is "
        "the code's training reading and the recipes' choice; the code's scoring adds it "
        "always); the "
        "loss of Eq. 10 uses d_bd for the negatives, the code the full energy; the "
        "negatives replace the superclass or filler (the code), while the text says the "
        "child; for the prediction task the code's nf3Loss_ours constrains the reversed "
        "inclusion box(D) + R inside box(C) (flag flip_gci2) and its evaluation scores "
        "box(D) - R, so the signs of R differ between training and testing; the code "
        "clips the parent offsets at exp(-1e-4) and uses softplus radii for the ball "
        "host, mOWL does neither; the paper's Section 5.2 swaps boxes and balls for "
        "ELBE/ELEM. Margins are 0 for ontologies (appendix C.1); the DAG experiments "
        "(mammal, noun, MCG, Hearst) are not implemented. With radius='softplus' every "
        "form uses the softplus radius (consistent_radius=False restores the old mix with "
        "|radius|). Eq. 10's negative term pushes the negative out of the parent only "
        "(non-inclusion, not disjointness): with gamma2 = 0 it is almost always zero."),
)

#: The same method with ELEm (balls) as the host model. A separate key, so the ball
#: variant can be selected by name (for example by the benchmark runner).
CARDS = (
    dataclasses.replace(
        CARD, key="regd_elem", name="RegD (ELEm host)", region="ball (dissimilarity-based)",
        concepts="ball (RegDELEm)",
        implementation="mowl.models.regd.model.RegDELEm",
        module="mowl.nn.el.regd.module.RegDELEmModule",
        related=("regd", "elem")),
)
