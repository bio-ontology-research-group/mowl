"""Catalogue cards of EmEL++ and EmEL(var)."""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind, MethodCard, Origin,
                               Reference, Status)
from mowl.catalog.entries import _refs as R

_MONDAL = Reference(
    "mondal2021",
    "Mondal, S., Bhatia, S., Mutharaju, R. (2021). EmEL++: Embeddings for EL++ Description "
    "Logic. AAAI Spring Symposium on Combining Machine Learning and Knowledge Engineering "
    "(AAAI-MAKE 2021), CEUR Workshop Proceedings 2846, paper 19.",
    "https://ceur-ws.org/Vol-2846/paper19.pdf")
_MOHAPATRA = Reference(
    "mohapatra2021",
    "Mohapatra, B., Bhatia, S., Mutharaju, R., Srinivasaraghavan, G. (2021). Why Settle for "
    "Just One? Extending EL++ Ontology Embeddings with Many-to-Many Relationships. "
    "arXiv:2110.10555 (SemRec workshop at ISWC 2021).",
    "https://arxiv.org/abs/2110.10555")
_MOHAPATRA_SUMMARY = Reference(
    "mohapatra2021a",
    "Mohapatra, B., Bhatia, S., Mutharaju, R., Srinivasaraghavan, G. (2021). EmELvar: A "
    "NeuroSymbolic Reasoner for the EL++ Description Logic. SemREC'21 workshop summary, "
    "ISWC 2021.")
_KULMANOV = Reference(
    "kulmanov2019",
    "Kulmanov, M., Liu-Wei, W., Yan, Y., Hoehndorf, R. (2019). EL Embeddings: Geometric "
    "construction of models for the Description Logic EL++. IJCAI 2019.",
    "https://arxiv.org/abs/1902.10499")

_AXIOMS = ("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
           "class_assertion", "object_property_assertion", "role_inclusion", "role_chain")

_SYMMETRY = Guarantee(
    GuaranteeKind.LIMITATION,
    "The role inclusion loss is symmetric in r and s, so r ⊑ s necessarily also enforces "
    "s ⊑ r: the method cannot faithfully capture the role hierarchy although a loss for it "
    "exists. In the region semantics, an embedding satisfies r ⊑ s iff the role vectors are "
    "equal.",
    "chen2025, Sec. III.B (Embedding Simple Ontology), paragraph on theoretical analysis; "
    "bourgaux2024, Sec. 3 (semantics of EmEL++: E |= r ⊑ s iff E(r) = E(s))",
    condition="zero loss (margin 0)",
    verified_in_mowl="tests/nn/test_emelpp_module.py::TestRoleAxioms::"
                     "test_role_inclusion_loss_is_symmetric")
_CHAIN = Guarantee(
    GuaranteeKind.LIMITATION,
    "The role chain loss depends on r1 and r2 only through r1 + r2, so r1 ∘ r2 ⊑ s "
    "necessarily implies r2 ∘ r1 ⊑ s; composition cannot be represented faithfully.",
    "chen2025, Sec. III.B (Embedding Simple Ontology), paragraph on theoretical analysis; "
    "bourgaux2024, Sec. 3 (E |= r1 ∘ r2 ⊑ s iff E(r1) + E(r2) = E(s)) and Sec. 5",
    condition="zero loss (margin 0)",
    verified_in_mowl="tests/nn/test_emelpp_module.py::TestRoleAxioms::"
                     "test_role_chain_loss_is_commutative")
_UNSOUND = Guarantee(
    GuaranteeKind.LIMITATION,
    "EmEL++ is not sound: an embedding model can exist for an unsatisfiable knowledge base "
    "(for example {A ⊑ ⊥, A(a)}), because a nominal can be mapped to an empty ball.",
    "bourgaux2024, Proposition 18 (proof: Example 1)",
    condition="zero loss")
_TBOX = Guarantee(
    GuaranteeKind.LIMITATION,
    "EmEL++ is not fully TBox-expressive: ∃r.C ⊑ ⊥ and C ⊑ ⊥ have the same truth conditions, "
    "so ∃r.C ⊑ ⊥ is approximated by C ⊑ ⊥. Models can also satisfy consequences that are not "
    "entailed, e.g. T = {∃r.C ⊑ A, ∃r.D ⊑ B, A ⊓ B ⊑ ⊥} forces C ⊓ D ⊑ ⊥.",
    "bourgaux2024, Proposition 21 and Example 10; Example 8 (not weakly ABox-faithful)",
    condition="zero loss")
_ONE_TO_MANY = Guarantee(
    GuaranteeKind.LIMITATION,
    "Roles embedded as transformations (here translations) of R^d are functions and cannot "
    "represent one-to-many relations faithfully.",
    "bourgaux2024, Sec. 3 (paragraph on BoxE); mohapatra2021, Sec. 1 and 4 (the motivation of "
    "EmEL(var))",
    condition="always",
    verified_in_mowl="tests/nn/test_emelpp_module.py::TestOneToMany::"
                     "test_translation_cannot_satisfy_one_to_many")

_NOTES_COMMON = (
    "Differences to the paper: radii are used through their absolute value (the paper "
    "requires non-negative radii and mentions a penalty for negative ones without giving it); "
    "the normalisation terms |‖x‖ - 1| of Eq. 1-10 are part of each axiom loss and there is "
    "no separate regularisation loss; individuals are nominal classes; the garbled "
    "normalisation terms of Eq. 2 are read as |‖e_v(x)‖ - 1| like in Eq. 1. Differences to "
    "ELEmbeddings (mowl.nn.ELEmModule): normalisation inside each loss, role axiom "
    "losses. The balls of EmEL++ are not closed under "
    "intersection (chen2025, Sec. III.B); the intersection loss only constrains the centre of "
    "E, as in ELEmbeddings. Balls are convex sets, so their Helly number is d + 1 in R^d; no "
    "Helly-number result is recorded for this method. No soundness or faithfulness theorem "
    "for EmEL++ is stated in the papers.")

_NOTES_VAR = (
    "EmEL(var): the role inclusion and role chain losses and Eq. 10 are those of "
    "EmEL++ (mohapatra2021: the other normal forms 'remain the same'), so the two "
    "role limitations above carry over; the other limitations of EmEL++ were analysed "
    "by bourgaux2024 for EmEL++ only and are not claimed here. The paper does not say "
    "how sigma is initialised or kept non-negative: mOWL initialises |sigma| = 0.1 and "
    "uses its absolute value. As published, the term +sigma(r) of Eq. 7 and 8 has "
    "weight 1 in every axiom, so max(0, x - sigma) + sigma >= max(0, x): the total loss "
    "is never lower than that of EmEL++ and the gradient with respect to sigma is zero "
    "whenever the hinge is active (tests/nn/test_emelpp_module.py::TestOneToMany). "
    "The constructor option sigma_reg_weight (default 1 = the paper) rescales this term "
    "and is NOT from the paper. The improvements reported in the paper (Tables 1-3) are "
    "therefore not explained by the loss as printed; the authors' EmEL(var) code was "
    "not available to check.")

CARDS = (
    MethodCard(
        key="emelpp",
        expressivity="ELH(∘)⊥ with ABox",
        tasks=("subsumption", "satisfiability", "disjointness",
               "instance-checking", "role-assertion",),
        name="EmEL++",
        family=Family.GEOMETRIC,
        region="ball",
        summary="ELEmbeddings extended with losses for role inclusions R ⊑ S and role "
                "chains R1 ∘ R2 ⊑ S. Classes are n-balls, roles are translations, the loss "
                "of every normal form carries unit-norm terms on the embedding vectors.",
        logic="EL++ (normal form: GCIs, ⊥, role inclusions and chains; ABox via nominals)",
        concepts="n-ball (centre, non-negative radius)",
        roles="translation vector",
        individuals="nominal classes (balls)",
        status=Status.LEARNABLE,
        origin=Origin.REIMPLEMENTATION,
        implementation="mowl.models.emelpp.model.EmELPP",
        module="mowl.nn.el.emelpp.module.EmELPPModule",
        guarantees=(_SYMMETRY, _CHAIN, _UNSOUND, _TBOX, _ONE_TO_MANY),
        intersection_closed=False,
        helly="d+1 (balls are convex sets in R^d)",
        references=(_MONDAL, _KULMANOV, R.BOURGAUX2024, R.CHEN2025),
        code_url="https://github.com/kracr/EmELpp",
        benchmarks=(),
        axioms=_AXIOMS,
        notes=_NOTES_COMMON),
    MethodCard(
        key="emelvar",
        expressivity="ELH(∘)⊥ with ABox",
        tasks=("subsumption", "satisfiability", "disjointness",
               "instance-checking", "role-assertion",),
        name="EmEL(var)",
        family=Family.GEOMETRIC,
        region="ball",
        summary="EmEL++ with a learned variance sigma(r) per role that relaxes the "
                "translation in C ⊑ ∃R.D and ∃R.C ⊑ D, intended to let one role relate a "
                "class to many fillers (one-to-many and many-to-many roles).",
        logic="EL++ (normal form: GCIs, ⊥, role inclusions and chains; ABox via nominals)",
        concepts="n-ball (centre, non-negative radius)",
        roles="translation vector and variance sigma(r) >= 0",
        individuals="nominal classes (balls)",
        status=Status.LEARNABLE,
        origin=Origin.REIMPLEMENTATION,
        implementation="mowl.models.emelpp.model.EmELVar",
        module="mowl.nn.el.emelpp.module.EmELVarModule",
        guarantees=(_SYMMETRY, _CHAIN),
        intersection_closed=False,
        helly="d+1 (balls are convex sets in R^d)",
        references=(_MOHAPATRA, _MOHAPATRA_SUMMARY, _MONDAL, _KULMANOV, R.BOURGAUX2024,
                    R.CHEN2025),
        benchmarks=(),
        axioms=_AXIOMS,
        notes=" ".join((_NOTES_COMMON, _NOTES_VAR))),
)
