"""ELK-Em (Singh, Jiménez-Ruiz & Weyde, 2026)."""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind as K, MethodCard, Origin,
                               Reference, Status)
from mowl.catalog.entries import _refs as R

SINGH2026 = Reference(
    "singh2026",
    "Singh, N., Jiménez-Ruiz, E., Weyde, T. (2026). ELK-Em: Closure-Aware Embeddings for "
    "EL+⊥ Ontologies. In Third Workshop on Knowledge Graphs and Neurosymbolic AI "
    "(KG-NeSy 2026), ISWC 2026 Workshops Joint Proceedings, CEUR-WS Vol-4313, paper 14.",
    "https://ceur-ws.org/Vol-4313/")

SINGH2026_TR = Reference(
    "singh2026tr",
    "Singh, N., Jiménez-Ruiz, E., Weyde, T. (2026). ELK-Em: Closure-Aware Embeddings for "
    "EL+⊥ Ontologies (Technical Report), CR-ELKEm-TR.pdf in the authors' repository.",
    "https://github.com/city-artificial-intelligence/elk-em")

_TESTS = "tests/models/test_elkem_model.py::TestSoundness::"
_NN_TESTS = "tests/nn/test_elkem_module.py::"

CARD = MethodCard(
    key="elkem",
    expressivity="ELH(∘)⊥ with ABox",
    tasks=("subsumption", "satisfiability", "disjointness",
           "instance-checking", "role-assertion",),
    name="ELK-Em",
    family=Family.GEOMETRIC,
    region="box",
    summary=(
        "Concepts are axis-parallel boxes and individuals points. A role is a range box, an "
        "error box and a diagonal affine map with positive slope, so ∃r.C = T_r(C ∩ R_r) ⊕ "
        "E_r is again a box and every normal form of EL+⊥, including role inclusions and "
        "role chains, has a loss whose zero set implies the axiom. Trained with a Pearson "
        "contrastive term; individuals can be placed by a lexical regulariser that aligns "
        "point distances with the similarity of external representations."),
    logic="EL+⊥ (EL++ without nominals and concrete domains; role inclusions, binary role "
          "chains, concept and role assertions)",
    concepts="axis-parallel box [l, u]; ⊤ the full box, ⊥ the empty box",
    roles="triple ⟨R_r, E_r, T_r⟩: range box, error box, diagonal affine map T_r(x) = "
          "m_r x + c_r with m_r > 0; r = {(x, y) | x ∈ T_r(y) ⊕ E_r, y ∈ R_r}",
    individuals="point (degenerate box)",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.elkem.model.ELKEm",
    module="mowl.nn.el.elkem.module.ELKEmModule",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion", "role_inclusion", "role_chain"),
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "If every TBox axiom and every positive ABox assertion of a normalised ontology "
            "has zero geometric loss, the embedding is a model of the ontology.",
            "singh2026, Theorem 6 (proof: singh2026tr, Theorem 8, Appendix A.4)",
            condition="zero loss",
            verified_in_mowl=_TESTS + "test_theorem_6_every_axiom_holds"),
        Guarantee(
            K.SOUNDNESS,
            "Under the assumptions of Theorem 6, the embedding satisfies every axiom "
            "derivable from the ontology by ELK's inference rules (R0, R⊤, R⊥, R⊓−, R⊓+, "
            "R∃, R⊑, R_H, R∘).",
            "singh2026, Corollary 7 (proof: singh2026tr, Corollary 9)",
            condition="zero loss",
            verified_in_mowl=_TESTS + "test_corollary_7_derived_axioms_hold"),
        Guarantee(
            K.CLOSURE,
            "Every EL+⊥ concept description is interpreted as an axis-parallel box: "
            "intersection, ⊕ and the positive-slope map T_r preserve boxes.",
            "singh2026, Section 4.1 (singh2026tr, Lemma 1)",
            condition="by construction",
            verified_in_mowl=(_NN_TESTS + "TestExistentialBox::"
                              "test_box_matches_relational_definition")),
        Guarantee(
            K.EXPRESSIVITY,
            "Rule R∃: if C ⊆ D then (∃r.C) ⊆ (∃r.D).",
            "singh2026, Proposition 1",
            condition="by construction"),
        Guarantee(
            K.EXPRESSIVITY,
            "r ⊆ s iff R_r ⊆ R_s and T_r(y) ⊕ E_r ⊆ T_s(y) ⊕ E_s for all y ∈ R_r, iff the "
            "same holds at the two corners l^R_r, u^R_r; this is what the RI0 loss checks.",
            "singh2026, Proposition 2",
            condition="always",
            verified_in_mowl=_NN_TESTS + "TestRoleInclusion::test_proposition_2_by_sampling"),
        Guarantee(
            K.EXPRESSIVITY,
            "If r ⊆ s then role assertions transfer from r to s and (rule R_H) "
            "(∃r.C) ⊆ (∃s.C) for every concept C.",
            "singh2026, Proposition 3",
            condition="always"),
        Guarantee(
            K.EXPRESSIVITY,
            "max(r1∘r2) = ⟨R_r2, m_r1 E_r2 ⊕ E_r1, T_r1 ∘ T_r2⟩ over-approximates r1 ∘ r2, "
            "with equality iff T_r2(R_r2) ⊕ E_r2 ⊆ R_r1; otherwise r1 ∘ r2 is not of the "
            "form ⟨R, E, T⟩.",
            "singh2026, Proposition 4",
            condition="always",
            verified_in_mowl=_NN_TESTS + "TestRoleComposition::"
            "test_proposition_4_over_approximation_and_equality"),
        Guarantee(
            K.EXPRESSIVITY,
            "If max(r1∘r2) ⊆ s then r1 ∘ r2 ⊑ s holds and rule R∘ holds; the RI1 loss "
            "enforces this sufficient (not necessary) condition.",
            "singh2026, Proposition 5",
            condition="always",
            verified_in_mowl=_NN_TESTS + "TestRoleComposition::"
            "test_proposition_5_chain_condition_is_sufficient"),
        Guarantee(
            K.LIMITATION,
            "Completeness is open: the RI1 condition is sufficient but not necessary, so "
            "not every consistent ontology need admit a zero-loss embedding.",
            "singh2026, Section 4.3"),
        Guarantee(
            K.LIMITATION,
            "Concepts are boxes and ∃r.C is a box for every box C, so an ELK-Em embedding is "
            "a box interpretation in the sense of Leemhuis & Kutz and has the Helly property "
            "(Helly number 2): there are classically satisfiable ontologies with no box "
            "model in any dimension, hence no zero-loss ELK-Em embedding.",
            "leemhuis2025, Definition 1, Propositions 4 and 5"),
    ),
    limitations=(
        "Disjointness axioms with a conjunct fixed to ⊥ (⊥ ⊓ ⊤ ⊑ ⊥, ∃r.⊥ ⊑ ⊥) hold trivially "
        "and cost 0; the primitive of Eq. 3 would charge the margin, and the module's "
        "variant='reference' keeps that value.",
        "The score exp(−ℓ⁰) saturates at 1 for every satisfied axiom, so it ranks a large "
        "candidate set poorly (singh2026, Section 5.2).",
        "In mOWL, as in the authors' code, the box of every named class that is not fixed "
        "to ⊥ is non-empty (half-widths are |·| + ε). With the Helly property this rules "
        "out zero loss for some TBoxes without assertions, e.g. X ⊑ A, X ⊑ B, Y ⊑ B, "
        "Y ⊑ C, Z ⊑ C, Z ⊑ A, A ⊓ B ⊑ W, W ⊓ C ⊑ ⊥ (an observation of mOWL, not of the "
        "paper).",
    ),
    intersection_closed=True,
    helly="2 (axis-parallel boxes, any dimension; eckhoff1988, as stated in leemhuis2025, "
          "Proposition 4)",
    references=(SINGH2026, SINGH2026_TR, R.LEEMHUIS2025, R.ECKHOFF1988),
    code_url="https://github.com/city-artificial-intelligence/elk-em",
    benchmarks=("box2el-galen-inference", "box2el-go-inference", "box2el-anatomy-inference"),
    notes=(
        "Follows Table 2 and Eqs. (1)-(6) of the paper; initialisation, the parametrisation "
        "(half-widths |·|, slope |·| + ε, error box centred at 0 — no loss of generality, "
        "its centre is absorbed by the shift), the box-size penalty (reg_factor), the "
        "contrastive weight 3.0, one mini-batch per normal form and epoch, and "
        "checkpoint selection by validation median rank follow the authors' code. "
        "Differences: per-axiom losses are not squared (the paper sums them; the code "
        "squares them); the gci2 loss has no overlap term (the code adds one, Table 2 "
        "does not); the empty set ⊕ E_r is kept empty when computing ∃r.C (interval "
        "arithmetic would make it non-empty, which would break soundness of the gci2 "
        "loss without an overlap term); overlap terms are dropped for rows with a box "
        "fixed to ⊥; unsatisfiable classes are propagated through gci0 and gci2 axioms. "
        "The cosine learning-rate schedule of the protein experiment (Table 4) is not "
        "implemented. The lexical regulariser takes a user-supplied matrix of external "
        "representations of individuals (no encoder is bundled)."),
)
