"""FALCON (Tang, Hinnerichs et al., 2022)."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="falcon",
    name="FALCON",
    family=Family.FUZZY,
    region="fuzzy set over named and sampled entities",
    summary=(
        "Entities are embedded as vectors and a neural network maps (concept, entity) pairs "
        "to degrees of membership. Complex ALC concepts are evaluated with fuzzy operators "
        "(t-norm, negation, existential over sampled entities), so each training step "
        "builds an approximate fuzzy model of the ontology; entailment is approximated over "
        "several such models."),
    logic="ALC (TBox and ABox)",
    concepts="membership function f_I(f_e(·)) given by an MLP",
    roles="membership function over entity pairs",
    individuals="vector",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.falcon.model.FALCONModel",
    module="mowl.nn.alc.falcon.module.FALCONModule",
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "If the loss is 0 and the t-norm satisfies θ(x, y) = 0 iff x = 0 or y = 0, the "
            "interpretation given by the membership functions (domain R^n) is a classical "
            "model of the TBox and ABox. The Gödel and product t-norms satisfy the "
            "condition; the Łukasiewicz t-norm does not.",
            "falcon2022, Theorem 1 (called 'Faithfulness' in the paper; proof in Appendix "
            "A.4)",
            condition="zero loss, strict t-norm (product or Gödel/min)"),
        Guarantee(
            K.COMPLETENESS,
            "Representation completeness: for every finite model of T ∪ A there are an "
            "entity embedding and a membership network with loss ≤ ε for any ε > 0. The "
            "theorem shows existence only, not how to find them.",
            "falcon2022, Theorem 2",
            condition="existence, loss arbitrarily close to 0"),
        Guarantee(
            K.EXPRESSIVITY,
            "Entailed statements can be made true to any degree α ≤ 1 in generated "
            "approximate models, and non-entailed ones false to some degree; approximate "
            "entailment coincides with classical entailment when all models are considered "
            "and α = 1.",
            "falcon2022, Corollary 1",
            condition="in the limit of all models"),
    ),
    limitations=(
        "The proof of Theorem 1 argues over the finite set E of named and sampled entities "
        "on which the loss is computed (falcon2022, Appendix A.4).",
        "mOWL offers t_norm='Łukasiewicz', for which Theorem 1 does not apply "
        "(falcon2022, paragraph after Corollary 1).",
    ),
    intersection_closed=None,
    references=(R.FALCON2022,),
    notes=(
        "FALCON's paper also states (Appendix A.4, Lemma 1) that balls and axis-parallel "
        "boxes are not representation complete; that statement is recorded on the ELEm and "
        "ELBE cards."),
)
