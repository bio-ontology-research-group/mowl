"""Multibox models: MultiBoxEL and MultiELBE (concepts as unions of k boxes)."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

_AXIOMS = ("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
           "class_assertion", "object_property_assertion")

_REPRESENTABILITY = Guarantee(
    K.COMPLETENESS,
    "For every n, an ELHO(∘) ontology is satisfiable iff it has a multibox model with "
    "domain R^n (each element of a finite model becomes a small box). The number of boxes "
    "per concept is not bounded by the theorem; it grows with the size of the finite "
    "model used.",
    "federico2026a, Theorem 3",
    condition="enough boxes per concept (not bounded by a fixed k)")

_SEMANTIC_FAITHFULNESS = Guarantee(
    K.STRONG_FAITHFULNESS,
    "Semantic faithfulness: O ⊨ φ iff every multibox model of O satisfies φ.",
    "federico2026a, Proposition 4",
    condition="over the class of all multibox models (not a single learned model)")

_AHARONI = Guarantee(
    K.COMPLETENESS,
    "EL++ is complete with respect to multibox spaces: every satisfiable EL++ KB has a "
    "multibox interpretation (built from a finite model), whereas single boxes are "
    "incomplete.",
    "aharoni2024, Theorems 5.20, 5.21 and 4.37",
    condition="enough boxes per concept")

_HELLY = ("No Helly-2 bound: unions of k ≥ 2 boxes are not convex, and by federico2026a, "
          "Theorem 3 every satisfiable ELHO(∘) ontology (including T_3) has a multibox model. "
          "With a fixed k mOWL can still fail to represent ontologies that need more than k "
          "boxes per concept.")

_INTERSECTION_NOTE = (
    "Intersection is closed for finite unions of boxes but multiplies the number of boxes "
    "(k·k pairwise intersections; federico2026a, Section 4), so the family of unions of at "
    "most k boxes used by mOWL is not closed under intersection.")

MULTIBOXEL = MethodCard(
    key="multiboxel",
    name="MultiBoxEL",
    family=Family.GEOMETRIC,
    region="union of boxes",
    summary=(
        "Every concept is a union of k axis-parallel boxes with Box²EL role semantics "
        "(head/tail boxes and bumps). With k ≥ 2 concepts can be non-convex, which escapes "
        "the Helly obstruction of single boxes."),
    logic="EL++ concept normal forms and ABox assertions (no role inclusions or chains)",
    concepts="union of k axis-parallel boxes, shared bump per concept",
    roles="two boxes (head, tail), as in Box²EL",
    individuals="point with bump, as in Box²EL",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.multiboxel.model.MultiBoxEL",
    module="mowl.nn.el.multiboxel.module.MultiBoxELModule",
    axioms=_AXIOMS,
    guarantees=(
        _REPRESENTABILITY,
        _SEMANTIC_FAITHFULNESS,
        _AHARONI,
        Guarantee(
            K.SOUNDNESS,
            "With objective='structural', structural_definitions=True and margin 0, zero "
            "sum of the structural box losses (gci0, gci0_bot, gci1, gci1_bot) and the "
            "existence term implies that the materialised regions are a geometric model of "
            "the box normal forms in which every satisfiable class and listed conjunction "
            "is non-empty (up to boundary contact). Role normal forms inherit Box²EL's "
            "guarantee.",
            "mOWL (original): docs/source/geometric/index.rst, 'Structural objective'",
            condition="zero structural loss, margin 0",
            verified_in_mowl="tests/nn/test_multiboxel_structural.py::"
                             "test_structural_losses_zero_at_exact_t3"),
        Guarantee(
            K.EXPRESSIVITY,
            "With n_components=1, hard aggregation and zero component/diversity "
            "regularisation, every GCI loss and the regularisation loss equal Box²EL's to "
            "within 1e-6.",
            "mOWL (original): MultiBoxEL docstring",
            condition="k = 1, hard aggregation",
            verified_in_mowl="tests/nn/test_multiboxel_losses.py"),
    ),
    limitations=(
        "Neither Federico et al. nor Aharoni give a bound on the number of boxes or a loss; "
        "mOWL fixes k (``n_components``) in advance.",
        "Under the default objective='axiom', zero gci1_bot loss implies that all k² "
        "component pairs are disjoint, but not conversely (separation is measured "
        "coordinatewise).",
        "The objective is non-convex; reaching a zero-loss model from random "
        "initialisation is not guaranteed (see the structural objective and witness-point "
        "recipe in the geometric documentation).",
    ),
    intersection_closed=False,
    helly=_HELLY,
    references=(R.FEDERICO2026_MULTIBOX, R.AHARONI2024, R.JACKERMEIER2024,
                R.LEEMHUIS2025),
    notes=(
        "Origin is an adaptation: Federico et al. (2026) define multibox semantics and a "
        "construction from a finite model but leave the loss open; Aharoni (2024) trains "
        "a BoxEL-based multibox model. mOWL's MultiBoxEL uses a Box²EL backbone with gated "
        "pairwise intersections, optional soft aggregation, point supervision "
        "(objective='point'|'mixed'|'hybrid') and a structural objective "
        "(objective='structural'), all mOWL additions. Exact-model initialisation "
        "(ExactBoxInit) can seed and anchor the boxes. " + _INTERSECTION_NOTE),
)

MULTIELBE = MethodCard(
    key="multielbe",
    name="MultiELBE",
    family=Family.GEOMETRIC,
    region="union of boxes",
    summary=(
        "Every concept is a union of k axis-parallel boxes with ELBE inclusion losses and "
        "translation roles. The ELBE counterpart of MultiBoxEL."),
    logic="EL++ concept normal forms and ABox assertions (no role inclusions or chains)",
    concepts="union of k axis-parallel boxes",
    roles="translation vector",
    individuals="box",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.multielbe.model.MultiELBE",
    module="mowl.nn.el.multielbe.module.MultiELBEModule",
    axioms=_AXIOMS,
    guarantees=(
        _REPRESENTABILITY,
        _SEMANTIC_FAITHFULNESS,
        Guarantee(
            K.EXPRESSIVITY,
            "With n_components=1, hard aggregation and zero component/diversity "
            "regularisation, every GCI loss and the MSE training loss equal ELBE's at equal "
            "margin to within 1e-6 (gci3 keeps ELBE's overlap formula).",
            "mOWL (original): MultiELBE docstring",
            condition="k = 1, hard aggregation, equal margin",
            verified_in_mowl="tests/nn/test_multielbe_losses.py"),
    ),
    limitations=(
        "gci3 keeps ELBE's coordinatewise overlap formula rather than full box inclusion, "
        "so that k = 1 matches ELBE.",
        "The structural objective is not available for MultiELBE.",
        "The objective is non-convex; reaching a zero-loss model is not guaranteed.",
    ),
    intersection_closed=False,
    helly=_HELLY,
    references=(R.FEDERICO2026_MULTIBOX, R.PENG2020, R.LEEMHUIS2025),
    notes=(
        "Origin is an adaptation of the multibox semantics of Federico et al. (2026) to "
        "ELBE's losses. MultiELBE defaults to margin=0.1 whereas the ELBE wrapper "
        "defaults to margin=0. Supports init_boxes/anchor_weight (ExactBoxInit), soft "
        "aggregation and point supervision. " + _INTERSECTION_NOTE),
)

CARDS = (MULTIBOXEL, MULTIELBE)
