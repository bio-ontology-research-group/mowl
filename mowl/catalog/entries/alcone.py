"""Catalogue cards of axis-aligned cone (al-cone) embeddings for ALC."""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind, MethodCard, Origin,
                               Reference, Status)

_JAIR = Reference(
    "ozcep2023",
    "Özçep, Ö. L., Leemhuis, M., Wolter, D. (2023). Embedding Ontologies in the "
    "Description Logic ALC by Axis-Aligned Cones. Journal of Artificial Intelligence "
    "Research 78:287-337.",
    "https://doi.org/10.1613/jair.1.13939")
_IJCAI = Reference(
    "ozcep2020",
    "Özçep, Ö. L., Leemhuis, M., Wolter, D. (2020). Cone Semantics for Logics with "
    "Negation. IJCAI 2020, pp. 1820-1826.",
    "https://doi.org/10.24963/ijcai.2020/252")
_ORTHO = Reference(
    "leemhuis2020",
    "Leemhuis, M., Özçep, Ö. L., Wolter, D. (2020). Orthologics for Cones. "
    "arXiv:2008.03172. Journal version: Learning with cone-based geometric models and "
    "orthologics, Annals of Mathematics and Artificial Intelligence 90:1159-1195, 2022.",
    "https://arxiv.org/abs/2008.03172")
_AMAI = Reference(
    "leemhuis2022",
    "Leemhuis, M., Özçep, Ö. L., Wolter, D. (2022). Learning with cone-based geometric "
    "models and orthologics. Annals of Mathematics and Artificial Intelligence "
    "90:1159-1195.",
    "https://doi.org/10.1007/s10472-022-09806-1")
_TEMPORAL = Reference(
    "leemhuis2024",
    "Leemhuis, M. (2024). Embedding Temporal Description Logic Ontologies by Cone-based "
    "Geometric Models. FCR 2024, CEUR Workshop Proceedings 3763, paper 3.",
    "https://ceur-ws.org/Vol-3763/paper3.pdf")

_T = "tests/geometric/test_cones.py::"
_N = "tests/nn/test_alcone_module.py::"
_M = "tests/models/test_alcone_model.py::"

_CLOSURE = Guarantee(
    GuaranteeKind.CLOSURE,
    "Al-cones are closed under intersection (conjunction), polarity (negation) and convex "
    "union (X° ∩ Y°)° = conHull(X ∪ Y) (disjunction); all three act componentwise on the "
    "factors {0}, R+, R-, R, and (X°)° = X.",
    "ozcep2023, Proposition 1, Corollary 1, Proposition 2; ozcep2020, Proposition 1",
    condition="always",
    verified_in_mowl=_T + "TestConeOperations")
_EXISTS = Guarantee(
    GuaranteeKind.CLOSURE,
    "With roles interpreted as 0/1 incidence matrices over half-axes, (∃R.C)^I = "
    "conHull({R^T y | y ∈ C^I}) is again an al-cone, (∃R.⊥)^I = ⊥^I and ∃R distributes over "
    "the convex union (a normal, additive operator).",
    "ozcep2023, Definition 7 and Proposition 11",
    condition="always",
    verified_in_mowl=_T + "TestPaperExamples::test_prop11_exists_is_normal_and_additive")
_BOOL_COMPLETE = Guarantee(
    GuaranteeKind.COMPLETENESS,
    "A Boolean ALC ontology is classically satisfiable iff it has an al-cone model in some "
    "finite R^n (algebraic atoms of the Lindenbaum-Tarski algebra on half-axes, "
    "n >= m/2 for m atoms).",
    "ozcep2023, Proposition 7; ozcep2020, Proposition 2",
    condition="by construction",
    verified_in_mowl=_T + "TestBooleanConstruction::test_is_model")
_BOOL_FAITHFUL = Guarantee(
    GuaranteeKind.STRONG_FAITHFULNESS,
    "A satisfiable Boolean ALC ontology has a strongly concept-faithful and TBox-faithful "
    "al-cone model in R^{2n} (n atoms; every coordinate doubled): an individual lies in C "
    "iff O |= C(a), and C ⊑ D holds iff it is entailed.",
    "ozcep2023, Proposition 8; ozcep2020, Proposition 3",
    condition="by construction",
    verified_in_mowl=_T + "TestBooleanConstruction::test_strong_concept_faithfulness_prop8")
_HALFSPACE = Guarantee(
    GuaranteeKind.STRONG_FAITHFULNESS,
    "The same holds with the sub-family of products of {0} and R only (orthogonal "
    "subspaces), in R^n for n atoms. Not implemented in mOWL.",
    "ozcep2023, Proposition 9; ozcep2020, Proposition 4",
    condition="by construction")
_ALC_COMPLETE = Guarantee(
    GuaranteeKind.COMPLETENESS,
    "An ALC ontology is classically satisfiable iff it is satisfied by a concept-faithful "
    "al-cone model on some, possibly infinite-dimensional, R^n with incidence matrices for "
    "roles. mOWL implements the direction 'satisfiable => al-cone model' (without "
    "faithfulness) by its own construction from a finite classical model, which exists by "
    "the finite model property of ALC; see notes.",
    "ozcep2023, Proposition 12 (proof via Propositions 13-15, Lemmas 3-6, partly as proof "
    "sketches); IJCAI version for bounded quantifier rank: ozcep2020, Proposition 5",
    condition="by construction (from a finite model supplied by the user or by "
              "mowl.geometric.completion.CanonicalModel for EL++)",
    verified_in_mowl=_T + "TestFiniteModelConstruction")
_DISTRIBUTIVE = Guarantee(
    GuaranteeKind.EXPRESSIVITY,
    "A class of closed convex cones is distributive iff any two cones either intersect "
    "non-trivially or are polar to each other (<x, y> <= 0); every distributive cone model "
    "can be represented by an al-cone model of the same dimension.",
    "ozcep2023, Propositions 5 and 6",
    condition="always",
    verified_in_mowl=_T + "TestConeOperations::test_prop5_condition_holds_for_al_cones")
_PARTIAL = Guarantee(
    GuaranteeKind.EXPRESSIVITY,
    "Al-cone models are partial: a point can lie in C ⊔ D without lying in C or D, and in "
    "neither C nor ¬C, so an embedding can leave the class of an individual open.",
    "ozcep2023, Sec. 5.1 (Example 1, Fig. 10)",
    condition="always",
    verified_in_mowl=_T + "TestPaperExamples::test_example1_figure10")
_ORTHOLOGIC = Guarantee(
    GuaranteeKind.LIMITATION,
    "For arbitrary closed convex cones, intersection, polarity and conic hull form an "
    "ortholattice that is not distributive (also not orthomodular): conHull(X ∪ Y) ∩ U = U "
    "while X ∩ U = Y ∩ U = {0}. Classical (Boolean) ALC semantics requires the "
    "restriction to al-cones (or another distributive class, Prop. 5); the cone logic "
    "itself is an orthologic.",
    "ozcep2023, Proposition 4, Sec. 4.2 and Figs. 3(c), 7(b); leemhuis2020, Sec. 3.1, "
    "Table 1 and Fig. 3",
    condition="always",
    verified_in_mowl=_T + "TestConeOperations::test_distributivity_fails_for_paper_non_al_cone")
_INFINITE = Guarantee(
    GuaranteeKind.LIMITATION,
    "A concept-faithful al-cone model of a full ALC ontology is in general "
    "infinite-dimensional (infinitely many algebraic atoms, e.g. cyclic ABoxes such as "
    "loves(narcis, narcis)); role assertions are only represented on the conceptual level, "
    "and ABox-faithfulness for roles 'in its original form cannot be ensured'.",
    "ozcep2023, Sec. 6 (opening paragraph), Sec. 6.1 and Example 3",
    condition="always")
_LEARN_SOUND = Guarantee(
    GuaranteeKind.SOUNDNESS,
    "Learned model (ours): with margin > 0, if the loss of an axiom is zero then the "
    "discretised al-cone interpretation (logit > 0 => half-axis in the cone) satisfies it; "
    "conversely, if the discretised interpretation satisfies it and all logits have "
    "absolute value >= margin, the loss is zero.",
    "mOWL (our lemma, mowl/nn/alc/alcone/losses.py module docstring; no published learner)",
    condition="zero loss, exact operators (temperature 0), margin > 0",
    verified_in_mowl=f"{_N}TestLossSemantics::test_zero_loss_iff_satisfied; "
                     f"{_M}TestALConeModel::test_trained_model_checked_by_exact_checker")

_HELLY = ("Not applicable in the usual sense: every al-cone contains 0, so any family "
          "has a common point. Disjointness in cone semantics is 'intersection = {0}'. "
          "For non-trivial intersection (≠ {0}) the Helly number of the al-cones of R^n is "
          "2n (ours: they are the subsets of the 2n half-axes; tight for the 2n cones "
          "missing one half-axis each), so it grows with the dimension, unlike boxes.")

_NOTES_SHARED = (
    "Representation: an al-cone of R^n is the set of half-axes it contains (JAIR Prop. 1); "
    "the al-cones form the Boolean algebra O_4^n ≅ 2^{2n}, and the incidence-matrix "
    "semantics of ∃R (JAIR Def. 7) is the Kripke diamond on the frame whose worlds are "
    "half-axes. Disjointness C ⊓ D ⊑ ⊥ means C^I ∩ D^I = {0}: Helly-type arguments about "
    "empty intersections of convex sets (as for boxes, where pairwise intersecting boxes "
    "have a common point) do not transfer, because all cones share the origin; for "
    "non-trivial intersections the bound is 2n (our analysis, verified exhaustively for "
    "n <= 2 in tests/geometric/test_cones.py::TestHelly). On the T3 pattern of Leemhuis & "
    "Kutz (A ≡ X1 ⊓ X2, B ≡ X2 ⊓ X3, C ≡ X1 ⊓ X3, A, B, C pairwise disjoint) the Prop. 7 "
    "construction gives a model in R^4 with A, B, C ≠ {0}, pairwise non-trivial X_i ∩ X_j "
    "and X1 ∩ X2 ∩ X3 = {0}. The JAIR role semantics (incidence matrices) differs from the "
    "IJCAI 2020 one (relations on points, ∀R.C as the minimal al-cone containing the "
    "classical box); mOWL follows JAIR. Role assertions R(a, b) are checked as JAIR's "
    "role-satisfiability b^I ∈ R a^I. The learning algorithms of the AMAI 2022 journal "
    "version (weakly supervised multi-label learning) could not be obtained (publisher and "
    "repository PDFs were not downloadable) and are not reflected here.")

CARDS = (
    MethodCard(
        key="alcone",
        name="Al-cone embedding (learned)",
        family=Family.GEOMETRIC,
        region="axis-aligned cone",
        summary="Concepts are axis-aligned cones (products of {0}, R+, R-, R); negation is "
                "the polar cone, disjunction the convex union, roles are incidence "
                "matrices over half-axes. mOWL learns one logit per half-axis for every "
                "concept, role entry and individual and trains with Gödel operators and "
                "margin losses whose zero set is exactly satisfaction of the discretised "
                "model.",
        logic="ALC (TBox GCIs, EquivalentClasses, DisjointClasses; class and object "
              "property assertions)",
        concepts="al-cone: per coordinate one of {0}, R+, R-, R (two half-axis logits)",
        roles="0/1 incidence matrix over the 2n half-axes (logits)",
        individuals="point with coordinates in {-1, 0, 1} (support logits)",
        status=Status.LEARNABLE,
        origin=Origin.ADAPTATION,
        implementation="mowl.models.alcone.model.ALConeModel",
        module="mowl.nn.alc.alcone.module.ALConeModule",
        guarantees=(_LEARN_SOUND, _CLOSURE, _EXISTS, _BOOL_COMPLETE, _ALC_COMPLETE,
                    _DISTRIBUTIVE, _PARTIAL, _ORTHOLOGIC, _INFINITE),
        intersection_closed=True,
        helly=_HELLY,
        references=(_JAIR, _IJCAI, _ORTHO, _AMAI, _TEMPORAL),
        benchmarks=(),
        axioms=("subclass_of", "equivalent_classes", "disjoint_classes", "class_assertion",
                "object_property_assertion"),
        notes="ADAPTATION: the papers give constructions and proofs only; JAIR Sec. 7 "
              "states that a practical learning algorithm and its evaluation are out of "
              "scope. Added by mOWL: the half-axis logit parametrisation, Gödel (min/max/"
              "negation) evaluation of complex concepts, hinge losses per axiom with the "
              "zero-set property, a point-validity loss for individuals (non-zero, not "
              "both signs on one axis), a non-degeneracy regulariser for named classes, "
              "and a smooth (log-sum-exp) relaxation annealed to the exact operators "
              "during training, with parameter noise scaled by the temperature. No "
              "negative sampling and no anonymous entities. Training can stop in a local "
              "minimum with a few violated axioms (on a 4-class family ontology 6 of 8 "
              "seeds reached zero loss, on a 3-class ontology 8 of 8); check "
              "ALConeModel.violations(). No published benchmark numbers exist: JAIR "
              "Sec. 7 leaves learning and evaluation to future work and the IJCAI 2020 "
              "and orthologic papers are theoretical, so no experiment can be added to "
              "the benchmark set (the Ann. Math. AI 2022 paper could not be read here and "
              "is not checked). "
              f"{_NOTES_SHARED}"),
    MethodCard(
        key="alcone-exact",
        name="Al-cone model construction",
        family=Family.CONSTRUCTIVE,
        region="axis-aligned cone",
        summary="Exact al-cone models: the Boolean construction of JAIR Props. 7 and 8 "
                "(algebraic atoms on half-axes, optional doubled strongly "
                "concept-faithful model) and, for full ALC, a model built from any finite "
                "classical model (each element on a half-axis, roles as incidence "
                "matrices), with an exact checker for TBox and ABox axioms.",
        logic="ALC (Boolean ALC for the Prop. 7/8 construction)",
        concepts="al-cone: product of {0}, R+, R-, R",
        roles="0/1 incidence matrix over the 2n half-axes",
        individuals="non-zero point",
        status=Status.CONSTRUCTIVE,
        origin=Origin.REIMPLEMENTATION,
        implementation="mowl.geometric.cones.ALConeInterpretation",
        guarantees=(_BOOL_COMPLETE, _BOOL_FAITHFUL, _HALFSPACE, _ALC_COMPLETE, _CLOSURE,
                    _EXISTS, _DISTRIBUTIVE, _PARTIAL, _ORTHOLOGIC, _INFINITE),
        intersection_closed=True,
        helly=_HELLY,
        references=(_JAIR, _IJCAI, _ORTHO, _TEMPORAL),
        benchmarks=(),
        notes="from_boolean_ontology follows the proofs of JAIR Props. 7 and 8 (atoms are "
              "enumerated, so at most 16 concept names by default; with an odd number of "
              "atoms one atom occupies two half-axes). from_finite_model is ours: the "
              "paper's full-ALC construction (Props. 12-15, srank-indexed atoms of an "
              "infinitary algebra) is infinite-dimensional and given as a proof sketch, "
              "so mOWL places the elements of a finite model on half-axes instead; by "
              "induction on concepts the cone model satisfies exactly the axioms the "
              "finite model satisfies, so it is only as faithful as that model. Finite "
              "models can be given explicitly (FiniteInterpretation) or taken from "
              "mowl.geometric.completion.CanonicalModel for EL++ ontologies. "
              f"{_NOTES_SHARED}"),
)
