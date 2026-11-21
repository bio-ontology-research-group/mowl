"""Language-model based hyperbolic ontology embeddings: HiT (He et al., NeurIPS 2024) and
OnT (Yang et al., ISWC 2025)."""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind as K, MethodCard, Origin,
                               Reference, Status)

HE2024HIT = Reference(
    "he2024hit",
    "He, Y., Yuan, Z., Chen, J., Horrocks, I. (2024). Language Models as Hierarchy "
    "Encoders. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024). "
    "arXiv:2401.11374.",
    "https://arxiv.org/abs/2401.11374")

YANG2025ONT = Reference(
    "yang2025ont",
    "Yang, H., Chen, J., He, Y., Gao, Y., Horrocks, I. (2025). Language Models as Ontology "
    "Encoders. In The Semantic Web - ISWC 2025. arXiv:2507.14334.",
    "https://arxiv.org/abs/2507.14334")

_OPTIONAL = ("sentence-transformers", "transformers")

_REGION = "point in the Poincaré ball (text-encoded)"

_NO_GUARANTEE = (
    "No model-theoretic guarantee is claimed: the paper proves no soundness, completeness "
    "or faithfulness result, and the embedding is not an interpretation (concepts are "
    "points, not sets).")

_HELLY = (
    "Helly number: not applicable. Concepts are points scored by a distance-and-norm "
    "heuristic, not regions with set-theoretic semantics, so there is no intersection of "
    "concept regions whose emptiness a Helly-type argument could constrain.")

_INDUCTIVE = Guarantee(
    K.EXPRESSIVITY,
    "Inductive: any concept with a textual label or verbalisation, including concepts "
    "unseen in training and concepts of another ontology, is embedded by encoding its "
    "text with the re-trained encoder.",
    "he2024hit, Section 4.1 (Mixed-hop Prediction on entities not seen in training, "
    "transfer setting, Table 3); yang2025ont, Section 1 and Section 5.4 (transfer, Fig. 3)",
    condition="by construction (empirical quality only)",
    verified_in_mowl="tests/models/test_lm_hyperbolic_models.py::test_inductive_unseen_class")

HIT = MethodCard(
    key="hit",
    expressivity="taxonomy",
    tasks=("subsumption",),
    name="HiT (Hierarchy Transformer)",
    family=Family.LANGUAGE_MODEL,
    region=_REGION,
    summary=(
        "Re-trains a sentence-transformer encoder (mean pooling, no normalisation) so that "
        "the embeddings of entity labels, read as points of the Poincaré ball of radius "
        "√d that circumscribes the encoder's output hypercube, are clustered by a "
        "hyperbolic triplet loss (child closer to its parent than to a negative) and "
        "ordered by a centripetal loss (parents closer to the origin). Subsumption is "
        "scored by −(d(x_C, x_D) + λ(‖x_D‖ − ‖x_C‖))."),
    logic="atomic subsumption hierarchies (taxonomies; no roles)",
    concepts="point of the Poincaré ball: encoding of the class label",
    roles="not represented",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.hit.model.HiT",
    module="mowl.nn.lm.hit.HiTModule",
    axioms=("gci0",),
    guarantees=(_INDUCTIVE,),
    limitations=(
        "Needs textual labels (rdfs:label, or an informative IRI fragment); classes "
        "with opaque identifiers and no labels cannot be embedded meaningfully.",
        "Taxonomies only: complex concepts, roles and conjunctions are not represented "
        "(yang2025ont, Section 1).",
        "Not a region model: no set-theoretic semantics for conjunction or "
        "disjointness, and no transitivity guarantee; the subsumption score is a "
        "heuristic whose weight λ and decision threshold are tuned on validation data "
        "(he2024hit, Section 3).",
        "Pre-trained language understanding may be lost during re-training, and entity "
        "names can be ambiguous (he2024hit, Section 7).",
    ),
    intersection_closed=None,
    references=(HE2024HIT, YANG2025ONT),
    code_url="https://github.com/KRR-Oxford/HierarchyTransformers",
    benchmarks=("ont-galen-prediction", "ont-go-prediction", "ont-anatomy-prediction",
                "ont-galen-inference", "ont-go-inference", "ont-anatomy-inference"),
    optional_dependencies=_OPTIONAL,
    training=("pairwise-margin-ranking", "transitive-reduction-positives",
              "sibling-hard-negatives", "hit-code-training", "ont-code-training"),
    evaluation=("hierarchy-threshold-f1", "ont-normal-form-ranking",
                "ont-inferred-subsumption-ranking"),
    related=("ont",),
    notes=(
        _NO_GUARANTEE + " " + _HELLY + " Defaults follow he2024hit, Appendix B: "
        "α = 5.0, β = 0.1, AdamW, learning rate 1e-5, 500 warm-up steps, batch 256, 10 "
        "negatives per positive (random under the closed-world assumption, or siblings "
        "first: hard negatives). Differences: the paper selects the epoch and λ on "
        "validation F1 with a tuned threshold (its 'hierarchy-threshold-f1' regime); "
        "mOWL selects them by validation ranking MRR over λ ∈ {0, 0.1, ..., 1}, which is "
        "how the OnT paper (Section 5.1) evaluates HiT. Published numbers on mOWL "
        "benchmarks: the HiT rows of Tables 2-3 of yang2025ont, obtained by the OnT "
        "authors (1 epoch, α = 3.0, β = 0.5). HiT's own WordNet, SNOMED CT and transfer "
        "results (Tables 2-3, 8) use datasets mOWL does not ship."),
)

ONT = MethodCard(
    key="ont",
    expressivity="EL",
    tasks=("subsumption",),
    name="OnT (Ontology Transformer)",
    family=Family.LANGUAGE_MODEL,
    region=_REGION,
    summary=(
        "Extends HiT to EL: every concept, atomic or complex, is the encoding of its "
        "verbalisation (\"V(C) and V(D)\", \"something that V(r) some V(C)\") as a point "
        "of the Poincaré ball; every role is a rotation of the ball (optionally followed "
        "by a Möbius scaling) computed from the encoding of the role label. Axioms C ⊑ D "
        "are trained with HiT's losses, and extra losses align ∃r.D with the rotated "
        "x_D and place C ⊓ D below C and D."),
    logic="EL (normal forms NF1-NF4; trained on any EL subsumption whose sides can be "
          "verbalised)",
    concepts="point of the Poincaré ball: encoding of the concept's verbalisation",
    roles="rotation R(Θ_r) (and scaling k_r) of the ball, linear in the role-label "
          "encoding",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.ont.model.OnT",
    module="mowl.nn.lm.ont.OnTModule",
    axioms=("gci0", "gci1", "gci2", "gci3"),
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Rotations preserve hyperbolic distances and norms, so with k_r = 1 the "
            "score of role images equals the score of the fillers: s(f_r(x_C) ≺ f_r(x_D)) "
            "= s(x_C ≺ x_D). This is how OnT relates A ⊑ B to ∃r.A ⊑ ∃r.B; it holds for "
            "the role images f_r(x), not for the text encodings of ∃r.A and ∃r.B that "
            "are scored in the evaluation.",
            "yang2025ont, Proposition 1",
            condition="by construction (k_r = 1)",
            verified_in_mowl="tests/nn/test_lm_hyperbolic.py::"
                             "test_proposition_1_rotation_invariance"),
        _INDUCTIVE,
    ),
    limitations=(
        "Needs textual labels for classes and roles; classes with opaque identifiers and "
        "no labels cannot be embedded meaningfully.",
        "Not a region model: no set-theoretic semantics. Conjunction C ⊓ D is the "
        "encoding of the text \"V(C) and V(D)\", a point pushed below the points of C and "
        "D by the hierarchy loss (Eq. 7); it is not an intersection. ⊥ and disjointness "
        "are not represented (the normal forms of Section 3.1 have no ⊥).",
        "The deductive pattern A ⊑ B ⇒ ∃r.A ⊑ ∃r.B is only encouraged: the text "
        "encoding of ∃r.A is pulled towards the role image f_r(x_A) by the loss of Eq. 6.",
        "Role inclusions and role chains are not used (yang2025ont, Section 6, future "
        "work).",
        "Relies on OnT's own name-preserving normaliser for complex ontologies; mOWL "
        "does not reimplement it and uses the ontology's axioms as they are.",
    ),
    intersection_closed=None,
    references=(YANG2025ONT, HE2024HIT),
    code_url="https://github.com/HuiYang1997/OnT",
    benchmarks=("ont-galen-prediction", "ont-go-prediction", "ont-anatomy-prediction",
                "ont-galen-inference", "ont-go-inference", "ont-anatomy-inference"),
    optional_dependencies=_OPTIONAL,
    training=("pairwise-margin-ranking", "ont-code-training"),
    evaluation=("ont-normal-form-ranking", "ont-inferred-subsumption-ranking"),
    related=("hit",),
    notes=(
        _NO_GUARANTEE + " " + _HELLY + " Baselines: the paper excludes CatE and FALCON "
        "because 'catE cannot handle unseen complex concepts that appear in our "
        "experimental settings, and FALCON's implementation is not publicly available' "
        "(yang2025ont, Section 5.1, 'Baselines'). Implementation choices: as in the "
        "reference code, the scaling k_r of Eq. 4 is computed but not applied (option "
        "role_scaling=True applies it), rotations pair coordinates i and i + d/2 (the "
        "paper's matrix pairs adjacent ones; option role_pairing='adjacent'), losses are "
        "averaged over batches rather than summed, the negative of an axiom C ⊑ ∃r.D is "
        "∃r.X for a random X, every training step adds L_r and L_⊓ on random samples of "
        "the existential restrictions and conjunctions of the ontology, and λ is chosen "
        "on validation NF1 axioms by MRR over {0, 0.1, ..., 1} (paper; the current "
        "reference code searches 0 ... 1.9). Conjunctions with n > 2 operands use the "
        "mean of the n hierarchy losses (Eq. 7 for n = 2). The verbaliser follows "
        "Section 4.1 by default (style 'paper'); the published data use 'V(r) some V(C)' "
        "without 'something that' (style 'release')."),
)

CARDS = (HIT, ONT)
