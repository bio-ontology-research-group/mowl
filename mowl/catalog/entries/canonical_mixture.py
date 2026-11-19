"""CanonicalMixture: individuals as distributions over canonical-model elements (mOWL)."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

_TESTS = "tests/nn/test_canonical_mixture.py::"

CARD = MethodCard(
    key="canonical_mixture",
    name="CanonicalMixture",
    family=Family.CONSTRUCTIVE,
    region="distribution over canonical-model elements",
    summary=(
        "The canonical model of the EL TBox is constructed once; concept extensions are "
        "fixed sets of canonical elements. Only individuals are learned: each individual is "
        "a softmax distribution over the canonical elements compatible with its "
        "training-entailed types, with low-rank logits. Used for ABox (class assertion) "
        "completion."),
    logic="Normalised EL++ knowledge bases with individuals (instance checking is exact "
          "for ELH⊥ without nominals, functionality or range axioms)",
    concepts="set of canonical-model elements (constructed, not learned)",
    roles="relation between canonical-model elements (constructed)",
    individuals="masked softmax distribution over admissible canonical elements (learned)",
    status=Status.LEARNABLE,
    origin=Origin.ORIGINAL,
    implementation="mowl.models.canonical_mixture.model.CanonicalMixture",
    module="mowl.nn.el.canonical_mixture.module.CanonicalMixtureModule",
    axioms=("class_assertion", "object_property_assertion"),
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Every class assertion entailed by the training KB (including assertions "
            "entailed through roles) has probability 1 (score exactly 0.0), for every "
            "parameter value.",
            "mOWL (original): CanonicalMixtureModule docstring, 'Faithful by construction'",
            condition="by construction (all parameter values)",
            verified_in_mowl=_TESTS + "test_role_entailed_type_scores_exact_zero"),
        Guarantee(
            K.WEAK_FAITHFULNESS,
            "Every class assertion contradicted by the training KB has probability 0 "
            "(score +inf), for every parameter value.",
            "mOWL (original): CanonicalMixtureModule docstring, 'Faithful by construction'",
            condition="by construction (all parameter values)",
            verified_in_mowl=_TESTS + "test_entailed_and_contradicted_scores_at_init"),
        Guarantee(
            K.EXPRESSIVITY,
            "Hierarchy violation rate is 0: if O ⊨ C ⊑ D then score(a, D) ≤ score(a, C) for "
            "every individual and every parameter value.",
            "mOWL (original): CanonicalMixtureModule docstring, 'Zero hierarchy violation "
            "rate'",
            condition="by construction (all parameter values)",
            verified_in_mowl=_TESTS + "test_hierarchy_violation_rate_zero_on_500_random_triples"),
    ),
    limitations=(
        "Concept regions are not learned, so the model cannot place a concept differently "
        "from the canonical model; it only ranks undetermined class assertions.",
        "Requires individuals; raises ValueError on TBox-only datasets.",
    ),
    references=(R.BAADER2005, R.KAZAKOV2014, R.LACERDA2024),
    notes=(
        "Developed in mOWL; there is no publication yet. Family: classified as "
        "constructive because the interpretation of every concept and role is the "
        "constructed canonical model (mowl.geometric.completion.CanonicalModel via "
        "mowl.geometric.membership.MembershipStructure) and the guarantees follow from "
        "that construction, not from training. Only the placement of individuals inside "
        "the canonical model is learned, which is why its status is learnable. It is not a "
        "region embedding (no Euclidean regions) and not a fuzzy method (no fuzzy "
        "operators). The references give the canonical model it builds on."),
)
