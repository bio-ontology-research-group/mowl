"""Octagon embeddings (Charpenay & Schockaert, IJCAI 2024), adapted to EL++ concepts."""

from mowl.catalog.card import (
    Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Reference, Status)

_PAPER = "charpenay2024"
_OURS = "mOWL analysis, docs/source/methods/octagon.rst"

CARD = MethodCard(
    key="octagon",
    expressivity="EL⊥ with ABox",
    tasks=("subsumption", "satisfiability", "disjointness", "instance-checking",
           "role-assertion"),
    name="Octagon embeddings (OctagonEL)",
    family=Family.GEOMETRIC,
    region="octagon (product of 2-D octagons)",
    summary=(
        "Concepts are products of axis-aligned 2-D octagons (bounds on x, y, y - x and "
        "x + y) over consecutive coordinate pairs; roles are translations. Octagons were "
        "introduced as relation regions for knowledge graph embedding; mOWL adapts them "
        "to EL++ concepts. Unlike boxes, three octagons can intersect pairwise without a "
        "common point, so T3 has an exact model in two dimensions."),
    logic="EL++ (without role inclusions and role chains)",
    concepts="Product of 2-D octagons over the coordinate pairs (x_{2i}, x_{2i+1}); "
             "centre plus eight non-negative radii per pair (5d numbers).",
    roles="Translation vector t_R (R = {(x, x + t_R)}).",
    individuals="Points.",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.octagon.model.OctagonEL",
    module="mowl.nn.el.octagon.module.OctagonELModule",
    guarantees=(
        # --- From the paper; all about octagons as knowledge graph RELATION regions ---
        Guarantee(
            K.CLOSURE,
            "The intersection of two 2-D octagons is an octagon, given by the componentwise "
            "max/min of the eight bounds (stated for the octagons of relation embeddings).",
            f"{_PAPER}, Section 3 (intersection formula)"),
        Guarantee(
            K.CLOSURE,
            "The relational composition of two non-empty normalised octagons is an octagon "
            "with closed-form bounds (relation embeddings; not used by the mOWL concept "
            "model, whose roles are translations).",
            f"{_PAPER}, Proposition 2"),
        Guarantee(
            K.EXPRESSIVITY,
            "Octagon parameters can be normalised (tightened) in closed form, and the "
            "normalised bounds decide emptiness.",
            f"{_PAPER}, Proposition 1",
            verified_in_mowl="tests/nn/test_octagon_geometry.py::"
                             "test_tightening_equals_paper_proposition_1"),
        Guarantee(
            K.EXPRESSIVITY,
            "Every knowledge graph is captured exactly by some coordinate-wise octagon "
            "relation embedding.",
            f"{_PAPER}, Proposition 6 (Proposition 16 in the appendix)"),
        Guarantee(
            K.EXPRESSIVITY,
            "Every set of symmetry, inversion, hierarchy and intersection rules has an "
            "octagon relation embedding that captures exactly its entailed rules of these "
            "kinds and no asymmetry, mutual exclusion or composition rule.",
            f"{_PAPER}, Proposition 8 (Proposition 19 in the appendix)"),
        Guarantee(
            K.EXPRESSIVITY,
            "Sets of regular composition rules (each relation at most once per rule) whose "
            "entailed extended composition rules are regular or trivial have an octagon "
            "relation embedding capturing exactly the entailed ones.",
            f"{_PAPER}, Proposition 10"),
        Guarantee(
            K.LIMITATION,
            "Some consistent set of hierarchy, intersection and mutual exclusion rules forces "
            "every convex coordinate-wise relation embedding (octagons included) to satisfy "
            "a non-entailed hierarchy rule; the proof uses Helly's theorem in the plane.",
            f"{_PAPER}, Proposition 7 (Proposition 17 in the appendix)"),
        Guarantee(
            K.LIMITATION,
            "Some sets of composition rules in which a relation occurs twice cannot be "
            "captured by octagon relation embeddings without unintended composition rules.",
            f"{_PAPER}, Proposition 9 (Proposition 18 in the appendix)"),
        # --- Ours: octagon products as EL++ concept regions ---
        Guarantee(
            K.CLOSURE,
            "Products of 2-D octagons are closed under intersection (factorwise) and "
            "translation, which is what EL conjunction and translation existentials need.",
            f"{_OURS} (follows from {_PAPER}, Section 3)"),
        Guarantee(
            K.LIMITATION,
            "Products of 2-D octagons (in any dimension) have Helly number exactly 3: if every "
            "three of finitely many such products intersect, all of them do (Helly's theorem "
            "in each factor plane). Hence T4 = {A_i = conjunction of the X_j, j != i (i = "
            "1..4), X1 and X2 and X3 and X4 below bottom} has no octagon-product model with "
            "non-empty concept names.",
            f"{_OURS}, Section 'Helly analysis'; Helly (1923)",
            verified_in_mowl="tests/baselines/test_octagon_module.py::"
                             "test_t4_definitions_force_a_common_point"),
        Guarantee(
            K.EXPRESSIVITY,
            "T3 = {A = X1 and X2, B = X2 and X3, C = X1 and X3, A, B, C pairwise disjoint}, "
            "which has no box model (Leemhuis & Kutz 2025), has an exact octagon model in "
            "d = 2 with zero loss.",
            _OURS,
            condition="by construction",
            verified_in_mowl="tests/baselines/test_octagon_module.py::"
                             "test_t3_exact_configuration_is_a_model_by_sampling"),
        Guarantee(
            K.EXPRESSIVITY,
            "Boxes are octagons with slack diagonal bounds, so every box model with "
            "translation roles in R^n is an octagon-product model in R^n (n even) or "
            "R^(n+1) (n odd).",
            _OURS),
        Guarantee(
            K.SOUNDNESS,
            "If the gci0, gci1, gci2, gci3, class and role assertion losses are zero (any "
            "margin >= 0) and the gci1_bot loss is zero with disjoint_margin > 0, then "
            "interpreting concept names as their octagon products, roles as translations "
            "and individuals as points is a model of these normal forms.",
            _OURS,
            condition="zero loss",
            verified_in_mowl="tests/baselines/test_octagon_module.py::"
                             "test_t3_exact_configuration_is_a_model_by_sampling"),
    ),
    limitations=(
        "Concept names are non-empty by construction; C below bottom and exists R.C below "
        "bottom are approximated by flattening C (zero loss iff empty interior).",
        "With margin 0 the disjointness loss also accepts touching regions; use "
        "disjoint_margin > 0 (the model's default is 0.1).",
        "Role inclusions and role chains are not supported (translations would force "
        "equal role vectors).",
        "The gci1 loss also requires the intersection to be non-empty (stronger than the "
        "axiom, as in ELBE and the prism model).",
        "Optimisation gap: from random initialisation, training on T3 (d = 2) gets close "
        "to but does not reach the exact model (residual about 1e-2 with a width floor, "
        "or about 1e-5 through degenerate near-point regions without it); see "
        "tests/models/test_octagon_model.py::test_t3_from_random_initialisation.",
        "No published results on ontology benchmarks exist.",
    ),
    intersection_closed=True,
    helly="3 (products of 2-D octagons, any dimension; mOWL analysis)",
    references=(
        Reference(
            _PAPER,
            "Charpenay, V., & Schockaert, S. (2024). Capturing Knowledge Graphs and Rules "
            "with Octagon Embeddings. In Proceedings of IJCAI 2024.",
            "https://arxiv.org/abs/2401.16270"),
        Reference(
            "leemhuis2025",
            "Leemhuis, M., & Kutz, O. (2025). Understanding the Expressive Capabilities of "
            "Knowledge Base Embeddings under Box Semantics. NeSy 2025, PMLR 284.",
            "https://proceedings.mlr.press/v284/leemhuis25a.html"),
        Reference(
            "federico2026prisms",
            "Federico, G., Leemhuis, M., & Kutz, O. (2026). Introducing Prism Embeddings: A "
            "New Family of Geometric Ontology Embeddings. DL 2026, CEUR-WS Vol-4230.",
            "https://ceur-ws.org/Vol-4230/dl-paper32.pdf"),
        Reference(
            "helly1923",
            "Helly, E. (1923). Über Mengen konvexer Körper mit gemeinschaftlichen Punkten. "
            "Jahresbericht der Deutschen Mathematiker-Vereinigung 32, 175-176."),
    ),
    code_url="https://github.com/vcharpenay/uvxy",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion"),
    notes=(
        "From the paper (Charpenay & Schockaert, IJCAI 2024): the 2-D octagon region (Eq. "
        "6), its closed-form normalisation and emptiness test (Proposition 1, implemented "
        "in mowl.nn.el.octagon.geometry.tighten and checked against a transcription of "
        "the proposition), and closure under intersection (Section 3). In the paper "
        "octagons are knowledge graph RELATION regions over (head, tail) coordinate pairs; "
        "all guarantees taken from the paper are statements about such relation "
        "embeddings and do not transfer to this concept model as such. Ours (mOWL): using "
        "products of 2-D octagons over consecutive coordinate pairs as EL++ CONCEPT "
        "regions; roles as translations (as in ELBE and the prism model); the "
        "parametrisation (centre plus non-negative radii, non-empty by construction); all "
        "losses, which are exact (zero iff the geometric condition, see "
        "mowl.representation.library.shapes_octagon_math), including gci3 as the exact inclusion C - t_R in D "
        "instead of ELBE's overlap; negative sampling for gci2 with an exact non-inclusion "
        "loss; the Helly analysis (Helly number 3 for products of planar convex sets, the "
        "T3 model in d = 2 and the T4 non-representability). The paper's learning "
        "objective (sigmoid band distances, self-adversarial negatives) is for link "
        "prediction and is not used. On boxes the gci0/gci2 losses have the same zero set "
        "as ELBE's and lie between 1 and sqrt(5) times ELBE's value. Role chains are not "
        "supported because roles are translations; using the paper's octagon relations "
        "with their composition (Proposition 2) would be a different design. No published "
        "ontology benchmark results exist.\n"
        "Benchmark status (checked against the paper, Section 6 and Appendix C): the only "
        "published experiments are knowledge-graph link prediction on FB15k-237 and WN18RR "
        "(Tables 1, 3, 4) with octagons as RELATION regions over (head, tail) pairs, "
        "self-adversarial negatives and up to 1,000 dimensions. The data are public, but "
        "the experiment evaluates a different model from mOWL's octagon CONCEPT model, so "
        "running mOWL's octagon on it would not reproduce anything; a benchmark entry is "
        "therefore not added. Reproducing Table 1 would require implementing the uvxy "
        "relation model itself (a new method, code at github.com/vcharpenay/uvxy)."),
)
