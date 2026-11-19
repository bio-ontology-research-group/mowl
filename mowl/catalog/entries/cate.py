"""CatE (Zhapa-Camacho & Hoehndorf): order embedding of the lattice of ALC concept
descriptions, with partial saturation."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, \
    Reference, Status

ZHAPA2023 = Reference(
    "zhapa2023",
    "Zhapa-Camacho, F., Hoehndorf, R. Lattice-based ALC ontology embeddings with "
    "saturation. arXiv:2305.07163 (v3, 2024). Earlier version: Lattice-preserving ALC "
    "ontology embeddings, NeSy 2024, LNCS, pp. 355-369.",
    "https://arxiv.org/abs/2305.07163")
DUC2021 = Reference(
    "duc2021",
    "Le Duc, C. (2021). Category-theoretical semantics of the description logic ALC. In "
    "Proceedings of DL 2021, CEUR-WS vol. 2954.")
BRIEULLE2022 = Reference(
    "brieulle2022",
    "Brieulle, L., Le Duc, C., Vaillant, P. (2022). Reasoning in the description logic "
    "ALC under category semantics (extended abstract). In Proceedings of DL 2022, "
    "CEUR-WS vol. 3263.")
VENDROV2016 = Reference(
    "vendrov2016",
    "Vendrov, I., Kiros, R., Fidler, S., Urtasun, R. (2016). Order-Embeddings of Images "
    "and Language. In Proceedings of ICLR 2016.",
    "https://arxiv.org/abs/1511.06361")

CARD = MethodCard(
    key="cate",
    name="CatE",
    family=Family.ORDER,
    region="order embedding of the concept-description lattice",
    summary=(
        "The concept descriptions of an ALC ontology and their sub-descriptions are "
        "arranged in a preorder (the 'lattice') following the category-theoretical "
        "semantics of ALC: each constructor and axiom adds a fixed set of morphisms, and "
        "partial saturation rules can add more. Every element gets a vector in R^n with "
        "the product order, trained with the order-violation score of Vendrov et al. and "
        "a margin loss with corrupted morphisms."),
    logic="ALC",
    concepts=("vector in R^n; the element's order is the coordinatewise order (one vector "
              "per concept description in the lattice, not a region)"),
    roles=("elements r_{∃r.C} ⊑ r of an auxiliary role preorder, and elements "
           "domain(r_{∃r.C}), codomain(r_{∃r.C}), all embedded as vectors in the same "
           "ordered space"),
    individuals="vector of the nominal element {a} (C(a) is the morphism {a} ⊑ C)",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.cate.model.CatE",
    module="mowl.nn.graph.cate.module.CatEModule",
    axioms=("SubClassOf", "EquivalentClasses", "DisjointClasses", "ClassAssertion",
            "ObjectPropertyAssertion"),
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "The product order on R^n (a ⪯ b iff a_i ≤ b_i for all i) is a partial order.",
            "zhapa2023, Theorem 1"),
        Guarantee(
            K.EXPRESSIVITY,
            "If the loss L of Eq. 7 is zero, every morphism A ⊑ B of the lattice satisfies "
            "f(A) ⪯ f(B) ('lattice-preserving' in the paper's sense: the map preserves the "
            "order, hence reflexivity and transitivity, and A ⊑ B, B ⊑ A gives f(A) = "
            "f(B)). The theorem does not state that meets or joins are preserved: f(A ⊓ B) "
            "is only below f(A) and f(B), not their coordinatewise minimum.",
            "zhapa2023, Theorem 2",
            condition="zero loss (over all non-morphisms, which requires a transitively "
                      "closed lattice when γ > 0)",
            verified_in_mowl="tests/models/test_cate_model.py::"
                             "test_zero_loss_embedding_preserves_every_morphism"),
        Guarantee(
            K.LIMITATION,
            "CatE cannot make inferences over concepts that are not explicitly stated in "
            "the lattice; for the PPI task the elements ∃interacts.p had to be added a "
            "priori.",
            "zhapa2023, Section 5 (and Section 4.4)",
            verified_in_mowl="tests/models/test_cate_model.py::"
                             "test_unseen_complex_concept_raises"),
        Guarantee(
            K.LIMITATION,
            "The lattice construction is not complete: only the concept descriptions of "
            "the knowledge base and their sub-expressions are elements, and saturation is "
            "partial (a fixed point may not be reached).",
            "zhapa2023, Sections 3 and 3.1"),
    ),
    limitations=(
        "No interpretation (model) of the ontology is produced: elements are vectors, not "
        "sets, and nothing makes ⊥ empty or ¬C the complement of C in the vector space. "
        "The paper places CatE between graph-projection methods and model-constructing "
        "methods (Section 5).",
        "Lattice size: every operator adds a constant number of elements and morphisms "
        "(O(n) for a formula with n operators, Section 3), but the training lattices are "
        "large (Table 1: 364,849 to 4,479,085 morphisms) and S1 plus the transitive "
        "closure was too costly for GO and FoodOn (Section 4.5).",
        "Theorem 2's premise L = 0 is attainable with margin γ > 0 only if the set of "
        "morphisms is transitively closed (including the morphisms through ⊥ and ⊤): for "
        "A ⊑ B ⊑ C with A ⊑ C missing, f(A) ⪯ f(C) makes the hinge of the negative A ⊑ C "
        "equal to γ (mOWL observation from Eq. 7). Without the closure the two terms of "
        "Eq. 7 compete; on a 10-axiom test ontology Cat ⊑ Animal kept a violation of about "
        "0.5 (margin 0.5) because {tom} ⊑ Animal is a negative. The reference code also "
        "keeps corruptions that are morphisms (e.g. X ⊑ ⊤) as negatives, so its loss cannot "
        "reach zero either.",
        "Scores of different concept descriptions tie at 0 whenever the order holds, so "
        "ranking metrics depend on the tie-breaking rule.",
    ),
    intersection_closed=("not applicable: A ⊓ B is an element with its own learned vector, "
                         "not computed from f(A) and f(B)"),
    helly="not applicable (concepts are points of an ordered vector space, not regions)",
    references=(ZHAPA2023, DUC2021, BRIEULLE2022, VENDROV2016),
    code_url="https://github.com/bio-ontology-research-group/catE",
    benchmarks=("cate-ore1-membership", "cate-go-completion", "cate-foodon-completion",
                "cate-yeast-ppi"),
    training=("fixed-column-corruption", "cyclic-lr"),
    evaluation=("cate-filtered-ranking", "loss-scoring"),
    related=("graph_kge", "order_embeddings", "falcon"),
    notes=(
        "Construction (Section 3) per constructor: A ⊓ B ⊑ A, B; A, B ⊑ A ⊔ B; for ¬C the "
        "elements C ⊓ ¬C ⊑ ⊥ and ⊤ ⊑ C ⊔ ¬C (Fig. 2c); for ∃r.C the role element "
        "r_{∃r.C} ⊑ r, codomain(r_{∃r.C}) ⊑ C and domain(r_{∃r.C}) ≡ ∃r.C; ∀r.C ≡ ¬∃r.¬C; "
        "an axiom C ⊑ D adds C ⊑ D and ⊤ ⊑ ¬C ⊔ D; C(a) adds {a} ⊑ C; r(a, b) adds "
        "{a} ⊑ ∃r.{b}; every element X has ⊥ ⊑ X ⊑ ⊤. mOWL builds it with the existing "
        "CategoricalProjector (the reference code's projector), which deviates from the "
        "paper's text as follows: it adds the negation diagram for every concept name "
        "(not only negated ones) and only C ⊓ ¬C ⊑ ⊥, ⊤ ⊑ C ⊔ ¬C (no projections and "
        "injections) for complex C, which mOWL completes (option complete_negation); it "
        "does not apply the conditional parts of the ⊓, ⊔ and ¬ rules ('if X ⊑ A ⊓ B is "
        "found...'): the ⊓/⊔ ones follow from transitivity, the ¬ ones are saturation "
        "rules (2), (3); it ignores role inclusion axioms, so the role-dependent rules for "
        "∃ and ∀ never apply and are not implemented (the ∀ rule as printed, 'codomain(r) "
        "⊑ C', is sound only for s = r; 'codomain(s) ⊑ C' is presumably meant); it adds "
        "Not(Domain(r_{∃r.¬C})) ≡ ∀r.C and, for r(a, b), {a} ⊑ domain(r), "
        "{b} ⊑ codomain(r). "
        "Saturation (Section 3.1): S1 = rules (1) C ⊑ ¬D ⇒ D ⊑ ¬C, (2) C ⊓ D ⊑ ⊥ ⇒ "
        "C ⊑ ¬D, (3) ⊤ ⊑ C ⊔ D ⇒ ¬C ⊑ D (printed with ⊥ ⊑ C ⊔ D, a typo), Tr = transitive "
        "closure; rule (5) C ⊑ D ⇒ ∃r.C ⊑ ∃r.D was not used in the paper and is offered "
        "only restricted to existing elements. mOWL implements the rules in "
        "mowl.projection.categorical.saturation rather than the projector's own "
        "saturate(), whose contraposition step adds only the last morphism of a loop. "
        "Training: Eq. 6 score with p = 2, Eq. 7 margin loss, tail corruption uniform over "
        "all elements; the reference code averages the k negative scores before the "
        "hinge (option negative_aggregation='mean'), uses Adam with weight decay 1e-4 and "
        "CyclicLR(lr/10, lr, 20 steps), validates every 100 epochs by raw MRR with "
        "patience 10. Paper hyperparameters: Appendix A, Table 6. "
        "Why Helly-type limits do not apply: the Helly obstruction of box methods "
        "(e.g. the T3 pattern A ≡ X1 ⊓ X2, B ≡ X2 ⊓ X3, C ≡ X1 ⊓ X3 with A, B, C pairwise "
        "disjoint) says that convex regions that intersect pairwise must intersect "
        "jointly. CatE has no regions and no emptiness test: X1 ⊓ X2 is an element with a "
        "free vector that only has to lie below f(X1), f(X2) and above f(⊥), and "
        "disjointness is the morphism A ⊓ B ⊑ ⊥ (or A ⊑ ¬B), i.e. another order "
        "constraint. Any finite preorder has an exact order embedding into (R^n, ≤) "
        "(map x to the indicator vector of its down-set, n = number of elements), so "
        "every lattice CatE builds, including T3's, has a zero-violation embedding in "
        "high enough dimension. The corresponding question for CatE is which patterns "
        "of meets, joins and disjointness the order can represent as lattice "
        "operations (f(A ⊓ B) = min(f(A), f(B)), f(A ⊔ B) = max(...)): (R^n, min, max) is "
        "a distributive lattice without complements, so only patterns realisable in a "
        "distributive lattice can be represented that way (T3 is: X1 = (1,0,1), "
        "X2 = (1,1,0), X3 = (0,1,1) in {0,1}^3 gives pairwise meets A, B, C with pairwise "
        "meets (0,0,0) = f(⊥)); CatE's loss does not ask for this, it only enforces the "
        "order on the morphisms of the lattice. This analysis is mOWL's, not the "
        "paper's."),
)
