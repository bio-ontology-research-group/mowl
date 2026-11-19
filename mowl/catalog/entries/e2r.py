"""Catalogue card of Embed2Reason (E2R), quantum-logic subspace embeddings."""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind, MethodCard, Origin,
                               Reference, Status)

_GARG = Reference(
    "garg2019",
    "Garg, D., Ikbal, S., Srivastava, S. K., Vishwakarma, H., Karanam, H., Subramaniam, "
    "L. V. (2019). Quantum Embedding of Knowledge for Reasoning. In Advances in Neural "
    "Information Processing Systems 32 (NeurIPS 2019). Supplementary "
    "material: Quantum_Embedding_Supplementary_Material_V1.pdf on the proceedings page.",
    "https://papers.nips.cc/paper_files/paper/2019/hash/"
    "cb12d7f933e7d102c52231bf62b8a678-Abstract.html")
_BVN = Reference(
    "birkhoff1936",
    "Birkhoff, G., von Neumann, J. (1936). The Logic of Quantum Mechanics. Annals of "
    "Mathematics 37(4):823-843.",
    "https://doi.org/10.2307/1968621")

_N = "tests/nn/test_e2r_module.py::"
_M = "tests/models/test_e2r_model.py::"

_LATTICE = Guarantee(
    GuaranteeKind.CLOSURE,
    "Every predicate (unary or binary, atomic or compound) is a linear subspace; with "
    "⊥ = {0} and ⊤ = Σ the subspaces ordered by inclusion form a bounded lattice with "
    "conjunction = intersection, disjunction = vector sum (span) and negation = "
    "orthogonal complement (the quantum logic of Birkhoff and von Neumann). The family "
    "of (axis-parallel) subspaces is closed under all three operations.",
    "garg2019, Sec. 3 (bullets defining ∧, ∨, ¬); Sec. 4, eqs. (6)-(8) for the "
    "axis-parallel indicator form",
    condition="always",
    verified_in_mowl=f"{_N}TestGeneralSubspaces::test_meet_join_complement_against_ranks; "
                     f"{_N}TestAxisParallel::test_indicator_operations_match_projectors")
_ZERO_LOSS = Guarantee(
    GuaranteeKind.SOUNDNESS,
    "For 0/1 indicator vectors the losses are zero iff the axiom holds in the subspace "
    "semantics: inclusion (eq. 5) iff subspace inclusion, conjunction (eq. 6) iff "
    "intersection, disjunction (eq. 7) iff span, negation (eq. 8) iff orthogonal "
    "complement, membership (eq. 4) iff the (unit) entity or pair vector lies in the "
    "subspace. Hence an embedding with zero loss and binary indicators is a model of the "
    "trained axioms in this semantics.",
    "garg2019, Sec. 4, eqs. (4)-(8) (stated there as 'necessary and sufficient "
    "conditions')",
    condition="zero loss and binary indicators (eq. (3) zero); concept negation in mOWL "
              "is evaluated on the real half (see notes)",
    verified_in_mowl=f"{_N}TestLossSemantics; "
                     f"{_M}TestExactModels::test_zero_loss_embedding_is_a_model")
_DISTRIBUTIVE = Guarantee(
    GuaranteeKind.EXPRESSIVITY,
    "Projectors onto axis-parallel subspaces commute, and for subspaces with commuting "
    "projectors the distributive law holds; hence the lattice of axis-parallel subspaces "
    "used by E2R is distributive (indeed Boolean: it is the power set of the axes).",
    "garg2019 supplementary material, Theorems 8, 10 and 11 (Boolean: mOWL analysis)",
    condition="by construction (axis-parallel embedding)",
    verified_in_mowl=f"{_N}TestAxisParallel::test_distributive_law_holds")
_NON_DISTRIBUTIVE = Guarantee(
    GuaranteeKind.LIMITATION,
    "For general (oblique) subspaces the distributive law S_i ∧ (S_j ∨ S_k) = "
    "(S_i ∧ S_j) ∨ (S_i ∧ S_k) fails, so quantum logic is not classical (Boolean) "
    "logic. E2R avoids this only by restricting predicates to axis-parallel subspaces. "
    "Example (ours): three distinct lines a, b, c in R^2 give a ∧ (b ∨ c) = a but "
    "(a ∧ b) ∨ (a ∧ c) = {0}.",
    "garg2019, Sec. 3 (second bullet) and Sec. 4 ('Choosing the embedding dimension'); "
    "supplementary material, Sec. 2",
    condition="always (for oblique subspaces)",
    verified_in_mowl=f"{_N}TestNonDistributivity::test_three_lines_in_the_plane")
_SPAN = Guarantee(
    GuaranteeKind.LIMITATION,
    "Disjunction is the span of the subspaces, not their set union. Consequence (mOWL "
    "analysis): membership is not classical even for axis-parallel subspaces; an entity "
    "vector can lie in C ⊔ D but in neither C nor D, and in neither C nor ¬C, so "
    "x ∈ C ⊔ D does not entail x ∈ C or x ∈ D as ALC semantics requires.",
    "garg2019, Sec. 3, second bullet ('+ sign means the vector sum and not the set "
    "theoretic union'); the consequence for individuals is mOWL analysis",
    condition="always",
    verified_in_mowl=f"{_N}TestNonDistributivity::test_disjunction_is_span_not_union")
_COLLAPSE = Guarantee(
    GuaranteeKind.LIMITATION,
    "Subspace collapse: without negation and universal type restrictions the objective "
    "has a degenerate zero-loss solution in which all predicate subspaces are equal; "
    "the regularisers of eqs. (11)-(12) or negated memberships of corrupted assertions "
    "are needed.",
    "garg2019, Sec. 4, 'The Problem of Subspace Collapse'",
    condition="no negation or universal restriction axioms")
_ORTH_CONFLICT = Guarantee(
    GuaranteeKind.LIMITATION,
    "The sibling-orthogonality regulariser of eq. (11) is positive on every model in "
    "which two siblings overlap; e.g. in the T3 pattern X1, X2, X3 are siblings but "
    "X1 ⊓ X2 must be non-trivial, so no model of T3 has zero regularised loss "
    "(mOWL analysis).",
    "mOWL analysis of garg2019, eq. (11)",
    condition="regulariser eq. (11) with positive weight",
    verified_in_mowl=f"{_M}TestExactModels::test_t3_orthogonality_regulariser_conflicts")

_HELLY = (
    "Not applicable in the usual sense: every subspace contains the origin, so "
    "intersections are never empty; ⊥ = {0} (garg2019, Sec. 3) and 'disjoint' means "
    "intersecting only in {0}. The paper defines no disjointness axiom; mOWL encodes "
    "A ⊓ B ⊑ ⊥ as eq. (6) with the left side {0}. For axis-parallel subspaces this is the "
    "same as orthogonality (supplementary Corollary 4: P1 P2 = 0); for oblique subspaces "
    "orthogonality is strictly stronger. mOWL analysis (proof in docs/source/methods/"
    "e2r.rst): for non-trivial intersection (≠ {0}) the Helly number of the linear "
    "subspaces of R^n, and of the axis-parallel ones, is n (the intersection of any "
    "family is the intersection of at most n members; tight for the n coordinate "
    "hyperplanes). The T3 pattern (A ≡ X1 ⊓ X2, B ≡ X2 ⊓ X3, C ≡ X1 ⊓ X3, A, B, C "
    "pairwise disjoint and non-trivial) has a model iff the concept space has dimension "
    ">= 3 (d = 3 in E2R's C^d): X1 = span(e1, e3), X2 = span(e1, e2), X3 = span(e2, e3), "
    "A, B, C the three axes.")

CARD = MethodCard(
    key="e2r",
    name="E2R (Embed2Reason)",
    family=Family.ALGEBRAIC,
    region="linear subspace (quantum logic)",
    summary="Concepts and relations are axis-parallel linear subspaces of C^d (simulated "
            "as R^2d over the reals), entities are unit vectors in the real half and an "
            "entity pair is the complex vector (x_a + i x_b). Membership is lying in the "
            "subspace, subsumption is subspace inclusion, conjunction intersection, "
            "disjunction span and negation the orthogonal complement (quantum logic). "
            "Subspaces are learned as relaxed 0/1 indicator vectors with one loss per "
            "axiom type.",
    logic="ALC as stated (garg2019, Sec. 2); the losses cover named-concept inclusion, "
          "⊓/⊔/¬ definitions of named concepts, the universal type restriction ∀R.C "
          "used as a range constraint (⊤ ⊑ ∀R.C), role inclusion and role ⊓/⊔/¬, "
          "concept and role assertions; no loss for ∃R.C",
    concepts="axis-parallel subspace of the real half of R^2d (0/1 indicator vector)",
    roles="axis-parallel subspace of R^2d containing pair vectors (x_aR, x_bR)",
    individuals="unit vector (x_R, 0)",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.e2r.model.E2R",
    module="mowl.nn.alc.e2r.module.E2RModule",
    guarantees=(_LATTICE, _ZERO_LOSS, _DISTRIBUTIVE, _NON_DISTRIBUTIVE, _SPAN, _COLLAPSE,
                _ORTH_CONFLICT),
    limitations=(
        "No loss for existential restrictions ∃R.C (defined in Sec. 2, no loss given); "
        "these axioms are skipped.",
        "∀R.C is only trained as a range restriction (eq. 9) and only against the "
        "non-descendants of C chosen from the told hierarchy; the loss assumes that these "
        "concepts are disjoint from C (a tree-shaped hierarchy).",
        "Role conjunction, disjunction and negation (eqs. 6-8 for relations) have no OWL 2 "
        "syntax; the module implements them, the model never extracts them.",
        "Relations are subspaces of pair vectors, so composition, transitivity and "
        "inverses are not represented (the paper's explanation of the WN18 results, "
        "Sec. 5).",
        "Several printed equations are inconsistent with the text (second term of "
        "eq. (4), pair-structure terms of eq. (4), eq. (8) for concepts); see notes.",
    ),
    intersection_closed=True,
    helly=_HELLY,
    references=(_GARG, _BVN),
    code_url="",
    benchmarks=(),
    axioms=("class_assertion", "object_property_assertion", "subclass_of",
            "equivalent_classes", "disjoint_classes", "sub_object_property_of",
            "object_property_range"),
    evaluation=("kg-filtered-link-prediction",),
    related=("alcone",),
    notes="REIMPLEMENTATION of the losses of Sec. 4 (eqs. 1-12) with these deviations: "
          "(1) the second term of eq. (4), printed as (1^T((0 I; I 0) ȳ_j ⊙ x_i))^2, "
          "equals (Σ x_iR)^2 ≠ 0 for well-formed concepts although the text says it "
          "vanishes; it is omitted. (2) Pair vectors are built as (x_pR, x_qR) from the "
          "entity vectors, so the pair-structure terms of eq. (4) (whose second term as "
          "printed compares the imaginary half of x_pq with the real half of x_q) are "
          "zero by construction. (3) Role assertions also carry unit-length terms for "
          "both entities. (4) Eq. (8) for concepts is evaluated on the real half, "
          "otherwise it cannot vanish together with eq. (2); ¬C is the complement inside "
          "the concept universe R^d x {0}. (5) Disjointness A ⊓ B ⊑ ⊥ (and A ⊑ ¬B) uses "
          "eq. (6) with ⊥ = {0}; the paper has no disjointness loss. (6) Negatives: only "
          "the residual of the membership loss is negated (the unit-length term stays), "
          "so the objective is bounded below; corruptions asserted in the training data "
          "are dropped. (7) Indicators are clipped to [0, 1] after each step; "
          "initialisation, learning rate, term weights and epochs are not stated in the "
          "paper. Sec. 4 offers negatives as an alternative to eqs. (11)-(12), Sec. 5 "
          "uses both; mOWL enables both by default (3 negatives per positive, Sec. 5). "
          "The paper's Table 1 results (FB15K, WN18, LUBM1U) are not recorded: FB15K and "
          "WN18 are knowledge-graph benchmarks outside mOWL's scope, and the LUBM1U "
          "reasoning protocol cannot be reproduced (69,628 training triples of unstated "
          "selection, eight custom test queries whose answer sets, candidate sets and "
          "ranking of binary queries are not specified). No authors' code was found.",
)
