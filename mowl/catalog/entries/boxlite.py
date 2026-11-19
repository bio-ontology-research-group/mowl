"""BoxLitE (Lourenço, Morgan, Ozaki, Pavlović & Sallinger, KR 2026)."""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind as K, MethodCard, Origin,
                               Reference, Status)
from mowl.catalog.entries import _refs as R

LOURENCO2026 = Reference(
    "lourenco2026",
    "Lourenço, B. F., Morgan, H., Ozaki, A., Pavlović, A., Sallinger, E. (2026). BoxLitE: A "
    "Faithful Knowledge Base Embedding Based on Convex Optimization. In Proceedings of KR "
    "2026. arXiv:2605.23937.",
    "https://arxiv.org/abs/2605.23937")

MORGAN2026 = Reference(
    "morgan2026",
    "Morgan, H., Pavlović, A., Ozaki, A., Lourenço, B. F., Sallinger, E. (2026). Convexity "
    "for Trustworthy Knowledge Graph Embeddings: From Interpretability to Reliability. In "
    "Proceedings of the 19th Forschungsforum der österreichischen Fachhochschulen (FHK "
    "2026).")

ARTALE2009 = Reference(
    "artale2009",
    "Artale, A., Calvanese, D., Kontchakov, R., Zakharyaschev, M. (2009). The DL-Lite Family "
    "and Relations. Journal of Artificial Intelligence Research 36:1-69.",
    "https://doi.org/10.1613/jair.2820")

ABBOUD2020 = Reference(
    "boxe2020",
    "Abboud, R., Ceylan, İ. İ., Lukasiewicz, T., Salvatori, T. (2020). BoxE: A Box Embedding "
    "Model for Knowledge Base Completion. In Advances in Neural Information Processing "
    "Systems 33 (NeurIPS 2020).",
    "https://arxiv.org/abs/2007.06267")

OZCEP2020 = Reference(
    "ozcep2020",
    "Özçep, Ö. L., Leemhuis, M., Wolter, D. (2020). Cone Semantics for Logics with "
    "Negation. In Proceedings of IJCAI 2020, 1820-1826.",
    "https://doi.org/10.24963/ijcai.2020/252")

_T = "tests/models/test_boxlite_model.py::"

CARD = MethodCard(
    key="boxlite",
    name="BoxLitE",
    family=Family.GEOMETRIC,
    region="box (convex programme)",
    summary=(
        "Box embedding of DL-Lite_H knowledge bases (concept and role inclusions with "
        "inverse roles, negation on the right, ABox) whose parameters are the solution of a "
        "convex second-order cone programme: the TBox, box consistency and box widths are "
        "linear constraints, the ABox enters the objective through signed distances, "
        "negation is a convex complement box, and negatives are a convex regulariser over "
        "existential (and named) concept assertions. Every satisfiable DL-Lite_H KB has a "
        "BoxLitE embedding that is a box-consistent, weakly faithful model."),
    logic="DL-Lite_H",
    concepts=("box per concept name; ∃R = Head(R) enlarged by Bump(R) "
              "([L^H - U^B, U^H - L^B]); ∃R⁻ likewise from Tail(R); ¬C = complement box "
              "[-s_Ω - L_C, s_Ω - U_C] (not the set complement)"),
    roles=("three boxes per role name: head, tail and bump; R⁻ swaps head and tail; "
           "R(a, b) iff pos(a) + bump(b) ∈ Head(R), pos(b) + bump(a) ∈ Tail(R) and "
           "bump(a), bump(b) ∈ Bump(R)"),
    individuals="position and bump vector in the universe box Ω = [-s_Ω, s_Ω]^d",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.boxlite.model.BoxLitE",
    axioms=("concept_inclusion", "negative_inclusion", "role_inclusion",
            "concept_assertion", "role_assertion"),
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "For nonnegative λ the training problem min f_λ(z) s.t. z ∈ C_K is convex: f_λ "
            "is convex and C_K is polyhedral (Theorem 4); it can be reformulated as an "
            "equivalent second-order cone programme (Theorem 5), so every local optimum is "
            "global.",
            "lourenco2026, Theorems 4 and 5 (proofs: Appendix E)",
            condition="always (λ ≥ 0)",
            verified_in_mowl=_T + "test_programme_is_dcp_and_dpp (CVXPY DCP/DPP check)"),
        Guarantee(
            K.SOUNDNESS,
            "Every feasible point of the programme (optimal or not) is a box-consistent "
            "interpretation that satisfies the TBox and is (weakly) TBox faithful.",
            "lourenco2026, Theorem 4 (ii) with Proposition 1",
            condition="by construction (linear constraints), any λ",
            verified_in_mowl=_T + "test_tbox_holds_for_any_lambda"),
        Guarantee(
            K.SOUNDNESS,
            "If λ_1 = 0 and the optimal value is ≤ 0, the solution is a box-consistent model "
            "of the KB and hence weakly KB faithful.",
            "lourenco2026, Theorem 4 (iii) with Proposition 2",
            condition="λ_1 = 0 and optimal value ≤ 0",
            verified_in_mowl=(_T + "test_solution_is_verified_model (independent checker "
                              "mowl.models.boxlite.checker.check_model)")),
        Guarantee(
            K.COMPLETENESS,
            "For a satisfiable DL-Lite_H KB, λ = 0, d ≥ |N_C| + |N_R|(2 + |N_I| + 2|N_R|) "
            "(Corollary 2) and s_Ω as in Theorem 3, the programme has an optimal solution "
            "with value ≤ 0. With (iii) this decides satisfiability: optimum ≤ 0 iff the KB "
            "is satisfiable.",
            "lourenco2026, Theorem 4 (iv) and Corollary 2",
            condition="λ = 0, d ≥ d_min of Corollary 2, ε ≤ s_Ω/8",
            verified_in_mowl=(f"{_T}test_satisfiable_kbs_are_recognised, "
                              f"{_T}test_unsatisfiable_kbs_have_positive_optimum")),
        Guarantee(
            K.COMPLETENESS,
            "A KB with empty ABox has a TBox-faithful box model in every d ≥ |N_C| + 3|N_R|, "
            "and C_K is nonempty for such d.",
            "lourenco2026, Corollary 1 and Theorem 4 (i)",
            condition="existence; empty ABox"),
        Guarantee(
            K.STRONG_FAITHFULNESS,
            "Every satisfiable DL-Lite_H KB K has a box-consistent box interpretation η_{I_K}, "
            "built from the canonical model I_K (Definitions 1 and 6, s_Ω = 4, d = |N^∃_C| + "
            "|N_R|·|Δ^{I_K}|), with K ⊨ α iff η_{I_K} ⊨ α for every DL-Lite_H axiom α "
            "(strong KB faithfulness in the sense of Özçep et al. 2020 and Bourgaux et al. "
            "2024, Definition 6). The theorem statement claims KB faithfulness; the proof "
            "gives the strong form. This is an existence result: the optimiser is not "
            "guaranteed to return this embedding.",
            "lourenco2026, Theorem 3 (proof: Appendix D, Theorem 1, Lemmas 1-4, Theorem 6)",
            condition="existence (construction from the canonical model; not implemented)"),
        Guarantee(
            K.WEAK_FAITHFULNESS,
            "A box-consistent box interpretation that is a model of K is weakly KB faithful: "
            "every DL-Lite_H axiom it satisfies is consistent with K (Definition 5, adapted "
            "from Özçep et al. 2020 and Bourgaux et al. 2024, Definition 5). Without an "
            "ABox, every model of the TBox is weakly TBox faithful because DL-Lite_H TBoxes "
            "are always satisfiable.",
            "lourenco2026, Propositions 1 and 2 (proofs: Appendix D)",
            condition="model of K and box consistent",
            verified_in_mowl="mowl.models.boxlite.checker.ModelCheck.weakly_kb_faithful"),
        Guarantee(
            K.LIMITATION,
            "With conjunction interpreted as box intersection, C ⊓ D ⊑ E becomes "
            "L_E ≤ max(L_C, L_D) and min(U_C, U_D) ≤ U_E; the set {(a, b, c) | min(a, b) ≤ c} "
            "is not convex ((1,0,0) and (0,1,0) are in it, their midpoint is not), so no "
            "nonnegative convex loss in the box bounds vanishes exactly when C ⊓ D ⊑ E "
            "holds. This is why the approach is restricted to DL-Lite_H and does not extend "
            "to EL or ALC (conjunction on the left).",
            "lourenco2026, Section 7 ('The logic fragment') and footnote 8",
            condition="box bounds are optimisation variables"),
    ),
    limitations=(
        "DL-Lite_H only: basic concepts A, ∃R, ∃R⁻; no conjunction or qualified existentials "
        "on either side, no role disjointness or functionality, no negative assertions. The "
        "mOWL extractor (mowl.datasets.dllite) reports every axiom outside the fragment.",
        "Box consistency reserves one dimension per basic concept, so d ≥ |N_C| + 2|N_R| "
        "(Section 5.1). The completeness guarantee needs d ≥ |N_C| + |N_R|(2 + |N_I| + "
        "2|N_R|), linear in the number of individuals (1,197 for F_v1), while the "
        "experiments use d = 32.",
        "Scale: the programme has O((|N_I| + |N_C| + |N_R| + |T|)d) linear constraints plus "
        "the cones of the objective; for F_v4 (3,014 assertions, 895 individuals) the "
        "compiled SOCP has about 3.5M variables and 2.3M constraints (Table 8: 3,584k and "
        "2,367k); MOSEK solved it in 26 s (Table 3).",
        "The experiments use λ_1 > 0 (Table 9), for which Theorem 4 (iii) does not apply: "
        "the embeddings behind Table 2 are not guaranteed to satisfy the ABox.",
        "The negative regulariser takes the maximum over all |N^∃_C|·|N_I| potential "
        "concept assertions not in the ABox; consequences of the KB are not excluded. "
        "Scores exist for assertions only (s = -loss, Section 5.2).",
        "Hyperparameters are tuned by grid search; each configuration is a full solve "
        "(Section 6, Q3).",
    ),
    intersection_closed=("boxes are closed under intersection, but DL-Lite_H has no "
                         "conjunction, so it is never used"),
    helly=("2 (axis-parallel boxes: leemhuis2025, Proposition 4). This is no obstruction in "
           "DL-Lite_H: negative inclusions are binary (B ⊑ ¬B') and there is no conjunction "
           "on the left, so the T3-type ontology of leemhuis2025 (Proposition 5, a ternary "
           "A ⊓ B ⊓ C ⊑ ⊥ with pairwise witnessed intersections) cannot be written; "
           "moreover, since every derivation in DL-Lite_H starts from a single basic "
           "concept, a DL-Lite_H KB entails a disjointness of several concepts only if it "
           "entails the disjointness of two of them (mOWL's argument, consistent with "
           "Theorem 3 of lourenco2026)."),
    references=(LOURENCO2026, MORGAN2026, ARTALE2009, ABBOUD2020, R.BOURGAUX2024, OZCEP2020,
                R.LEEMHUIS2025),
    code_url="https://github.com/AleksVap/BoxLitE",
    benchmarks=("boxlite-family-v1", "boxlite-family-v2", "boxlite-family-v3",
                "boxlite-family-v4"),
    optional_dependencies=("cvxpy",),
    training=("existential-concept-negatives",),
    evaluation=("kg-filtered-link-prediction",),
    related=("boxe",),
    notes=(
        "Status LEARNABLE, not CONSTRUCTIVE: CONSTRUCTIVE is for interpretations constructed "
        "from the ontology (e.g. its canonical model) without training. BoxLitE's embedding "
        "is fitted to the data by optimisation, depends on the regularisation weights, the "
        "data-dependent negatives and the dimension, and is selected on validation data; it "
        "is guaranteed to be a model only when λ_1 = 0 and the optimum is ≤ 0 (Theorem 4 "
        "iii-iv). The constructive embedding of Theorem 3 (Definition 6) is not "
        "implemented. Not trained by gradient descent: the programme is written with CVXPY "
        "(optional dependency) and solved by a conic solver; the paper used MOSEK, mOWL "
        "tries Clarabel, then ECOS, then SCS (Clarabel stalls on F_v1, ECOS solves it); "
        "solver='MOSEK' works if installed. The "
        "optimal value is solver independent; optimal embeddings agree up to solver "
        "tolerance when the optimum is unique. Two variants: 'paper' (Sections 3 and 5 as "
        "printed: boxes {L+ε ≤ x ≤ U-ε}, individuals in [-s_Ω+ε, s_Ω-ε], Eq. 1, maximum of "
        "per-assertion signed distances, Σ‖U-L‖₂ per box, negatives D(a) ∉ A for all "
        "D ∈ N^∃_C, where R(a, b) witnesses ∃R(a) and ∃R⁻(b)) and 'reference' (the authors' "
        "code at commit 7d53063, which produced Table 2: closed boxes [L, U], individuals in "
        "[-s_Ω, s_Ω], L+U ≤ -s_Ω-2ε in the reserved dimension, one signed distance of the "
        "stacked lower (resp. upper) residuals of all assertions, assertion term "
        "max(concept term, role term) with a missing term counted as 0, width regulariser "
        "= sum of 2-norms of the row-major reshape of the stacked width vector to d rows "
        "(not per box), negatives without ∃R⁻ for roles that occur in the ABox). The "
        "reference loader's TBox for the Family data differs from Figure 2 (see the "
        "boxlite-family benchmarks). Signed distances are modelled with the "
        "partial-minimisation formula of Proposition 4 (the reference code uses CVXPY's "
        "support-function transform of the same set). TBoxes need not be in named form: "
        "the constraints of Section 5.1 do not use it."),
)
