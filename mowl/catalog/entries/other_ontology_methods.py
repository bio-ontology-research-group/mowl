"""Further ontology embedding methods that mOWL catalogues but does not implement."""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind as K, MethodCard, Origin,
                               Reference, Status)

EIKE = MethodCard(
    key="eike",
    assumption="closed world over stated axioms",
    training=("pairwise-margin-ranking", "either-side-corruption",),
    evaluation=("kg-filtered-link-prediction", "common-filtered-ranking",
                "triple-classification-thresholds", "assertion-link-prediction"),
    expressivity="taxonomy with ABox",
    tasks=("subsumption", "instance-checking", "role-assertion"),
    name="EIKE",
    family=Family.GEOMETRIC,
    region="axis-aligned ellipsoid (plus language-model vectors)",
    summary=(
        "Embeds an ontology in two spaces: in the extensional space concepts are "
        "axis-aligned ellipsoids and instances are points; in the intensional space "
        "concepts are vectors, optionally initialised from a pre-trained language model "
        "applied to their names and descriptions. Trained with margin ranking losses on "
        "SubClassOf, InstanceOf and instance relation triples."),
    logic="Ontology as triples: SubClassOf, InstanceOf and instance relations, plus text",
    concepts="axis-aligned ellipsoid (extensional) and vector (intensional)",
    roles="relation vector between instances (TransE)",
    individuals="point (extensional); virtual vector M i (intensional)",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.eike.model.EIKE",
    module="mowl.nn.el.eike.module.EIKEModule",
    axioms=("gci0", "class_assertion", "object_property_assertion"),
    optional_dependencies=("a sentence encoder of your choice (e.g. sentence-transformers), "
                           "only to compute the optional intensional vectors yourself",),
    intersection_closed=False,
    helly="d+1 (ellipsoids are convex sets in R^d)",
    references=(
        Reference("wang2024eike",
                  "Wang, K., Qi, G., Chen, J., Huang, Y., Wu, T. (2024). Embedding "
                  "Ontologies via Incorporating Extensional and Intensional Knowledge. "
                  "arXiv:2402.01677 (IJCAI 2024).",
                  "https://arxiv.org/abs/2402.01677"),),
    notes=(
        "ADAPTATION of a triple-based method to OWL: SubClassOf triples are the "
        "atomic subclass axioms (gci0 of the EL normalisation; the gci0 rows that "
        "normalisation of complex axioms introduces, with auxiliary classes, are also "
        "used), InstanceOf triples are class assertions with a named class, relational "
        "triples are object property assertions. Axioms with existential or conjunctive "
        "parts (gci1-gci3), bottom forms and role axioms are not used; no complex concept "
        "is embedded, so only subsumption between named classes, instance checking and "
        "role assertion are scored. Implemented from the paper's Section 4, checked "
        "against the LaTeX source: all scores are exactly as printed there. The "
        "ellipsoid definition has no scale but the printed scores contain a per-concept "
        "quantity r_c that the paper never defines (it also uses r_c for the SubClassOf "
        "relation), which mOWL implements as a learnable positive scalar per concept "
        "(initialised to 1); the printed f_sub^ex (squared normalised centre distance "
        "plus r_i^2 - r_j^2) is weaker than ellipsoid containment, so a zero extensional "
        "subclass score does not imply containment of the regions and no soundness "
        "guarantee is claimed. The intensional subclass score is 1 - cos + (|c_i| - "
        "|c_j| (confirmed in the LaTeX source: a signed difference, unbounded below; the "
        "paper says the super concept has the larger module). Negatives follow Bordes et al. "
        "(head or tail corrupted, one of each per positive); the ranking loss pairs a "
        "positive with its own corruptions. Margins and alpha are model arguments; the "
        "paper's values are searched on the grid {0.1, 0.3, 0.4, 0.5, 1, 2} (margins) and "
        "{0.1, 0.5, 1, 2} (alpha), dimension 100, SGD lr 0.001, 1000-2000 epochs; the best "
        "configurations are in the Appendix of the authors' repository "
        "(keyu-wang-2002/EIKE) and are the recipes of the benchmarks on YAGO39K and "
        "M-YAGO39K (triple classification, link prediction, published numbers of Tables 2-4). "
        "The authors' code trains differently (SGD, mixed mini-batches, per-step "
        "normalisation, a hard-negative schedule), uses the L1 norm in the relational score "
        "in most best configurations and a Bernoulli corruption: none of these is in mOWL. "
        "The language model is an optional input: pass pre-computed vectors "
        "(dict class IRI -> vector, or the path of a file written by "
        "EIKE.save_intensional_vectors) as intensional_vectors; mOWL imports and downloads no "
        "language model (EIKE.encode_class_texts accepts any encoder callable, "
        "EIKE.pca_vectors reduces them to the embedding dimension). The paper's code used "
        "all-MiniLM-L6-v2 (384 dimensions, frozen) and a trained linear map to 100 "
        "dimensions. Its third dataset, DB99K-242, has no released splits or negatives and is "
        "not in mOWL."),
)

BOXSEL = MethodCard(
    key="boxsel",
    assumption="closed world over stated axioms",
    evaluation=("common-filtered-ranking",),
    expressivity="EL⊥ with ABox",
    tasks=("subsumption", "satisfiability", "disjointness", "instance-checking",
           "role-assertion"),
    name="BoxSEL (Statistical EL)",
    family=Family.GEOMETRIC,
    region="box",
    summary=(
        "Embeds Statistical EL TBoxes, whose conditional statements (D | C)[l, u] bound the "
        "proportion of C-instances that are D, as boxes in the style of BoxEL; conditional "
        "probabilities are read off as volume ratios."),
    logic="Statistical EL (SEL): EL normal form plus conditionals (D | C)[l, u]",
    concepts="axis-parallel box",
    roles="affine map (diagonal scaling and translation), as in BoxEL",
    individuals="point (BoxEL), for the deterministic part",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.boxsel.model.BoxSEL",
    module="mowl.nn.el.boxsel.module.BoxSELModule",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion"),
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "If the loss of a conditional (D | C)[l, u] is zero (exact volumes), the "
            "geometric interpretation satisfies l·Vol(C) <= Vol(D ∩ C) <= u·Vol(C); with "
            "the soundness of the BoxEL losses, a zero TBox loss gives a model of the SEL "
            "TBox. mOWL checks the conditional part (Lemma 2); the BoxEL part is the "
            "guarantee of the boxel card.",
            "zhu2024, Lemma 2 and Theorem 1",
            condition="zero loss, exact (not softplus) volumes",
            verified_in_mowl="tests/models/test_boxsel_model.py::"
                             "test_volume_ratio_and_conditional_loss_follow_equations_3_to_5"),
        Guarantee(
            K.SOUNDNESS,
            "If the loss is zero and T ⊨ (D | C)[l, u], the probability p computed from the "
            "embedding lies in [l, u].",
            "zhu2024, Theorem 3",
            condition="zero loss"),
        Guarantee(
            K.EXPRESSIVITY,
            "For concepts C, D, with non-empty, non-full regions, p with Iw ⊨ (D | C)[p] is "
            "computed in O((c + d)·n) time (c, d concept sizes, n dimension). mOWL computes "
            "it as an exact volume ratio, BoxSEL.probability.",
            "zhu2024, Theorem 2"),
    ),
    intersection_closed=True,
    helly="2 (axis-parallel boxes; eckhoff1988)",
    related=("boxel",),
    references=(
        Reference("zhu2024",
                  "Zhu, Y., Potyka, N., Xiong, B., Tran, T.-K., Nayyeri, M., Kharlamov, E., "
                  "Staab, S. (2024). Approximating Probabilistic Inference in Statistical EL "
                  "with Knowledge Graph Embeddings. arXiv:2407.11821.",
                  "https://arxiv.org/abs/2407.11821"),),
    notes=(
        "ADAPTATION. Implemented: the BoxEL geometry and losses (reused from BoxEL, as in "
        "Appendix D), the conditional losses of Eqs. 4, 5 and 7 with softplus volumes "
        "(optional cooling of the sharpness, Appendix E.1), the location regulariser of "
        "Eq. 9, probabilities as volume ratios, and the interval estimate over several "
        "embeddings (BoxSEL.probability_interval, Section 8.3). Input: OWL cannot express "
        "(D | C)[l, u], so the conditionals are a Python argument (Conditional tuples) "
        "next to the OWL ontology with the deterministic axioms; there is no file format. "
        "Premises and conclusions may be names, conjunctions and existentials, nested "
        "(the paper's prototype supports (B|A), (B|A1⊓A2), (B|∃r.A), (∃r.B|A)); the "
        "region of ∃r.C is the preimage of Box(C) under the role map, as in the "
        "left-existential BoxEL loss. ⊤ is not allowed in conditionals (safe TBoxes, "
        "Definition 2). Not implemented: the SEL normalisation SNF0 (introducing a name "
        "for a complex concept in a conditional; define the name in the ontology and use "
        "it), the YAGO3-derived Person/Country/Hybrid datasets and the PMP evaluation "
        "protocol. The volume regulariser is implemented exactly as in the LaTeX source "
        "(L_vol(w) = sum_i [beta^n - Vol(C_i) - epsilon]_+; the paper says it helps "
        "in its ablation, Table 12) but is off by default (volume_weight=0): it is zero "
        "only for boxes that fill the cube [0, beta]^n up to epsilon, which cannot hold "
        "together with conditionals below 1 for nested or partly overlapping concepts "
        "(Vol(D ∩ C) <= u Vol(C) < Vol(C)); the source does not say how it is weighted "
        "or what epsilon is, so it is an option, not a default. The conditional losses are averaged per pattern group (the paper sums "
        "over axioms). The authors' code was not located and not checked; hyperparameters "
        "(dimension, beta, learning rate) are the user's choice."),
)

DFALC = MethodCard(
    key="dfalc",
    assumption="open world",
    evaluation=("common-filtered-ranking",),
    expressivity="ALC with ABox",
    tasks=("subsumption", "instance-checking", "role-assertion"),
    name="Differentiable fuzzy ALC (DF-ALC)",
    family=Family.FUZZY,
    region="fuzzy set over a finite domain",
    summary=(
        "A fuzzy grounding of ALC over a finite domain (the individuals): every concept "
        "name is a vector of truth values over the domain, every role name a matrix of "
        "truth values, complex concepts use the Gödel semantics. The TBox is enforced by "
        "a hierarchical loss on fuzzy inclusions (C^I(a) <= D^I(a) for all a), which "
        "revises an initial (e.g. perceptual) grounding into a model of the ontology."),
    logic="ALC (TBox GCIs, EquivalentClasses, DisjointClasses; class and object "
          "property assertions)",
    concepts="fuzzy membership vector over the domain (sigmoid of logits)",
    roles="fuzzy membership matrix over pairs of domain elements",
    individuals="domain element (index = individual)",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.dfalc.model.DFALCModel",
    module="mowl.nn.alc.dfalc.module.DFALCModule",
    axioms=("subclass_of", "equivalent_classes", "disjoint_classes", "class_assertion",
            "object_property_assertion"),
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "Fuzzy entailment in the translated DF-ALC ontology is consistent with "
            "entailment in ALC (three-valued crisp transformation with threshold alpha).",
            "wu2022, Proposition 1"),
        Guarantee(
            K.SOUNDNESS,
            "When the hierarchical loss converges to 0, the learned interpretation is a "
            "model of the (normalised) ALC ontology: every inclusion holds pointwise, "
            "C^I(a) <= D^I(a) (Eq. 6). In mOWL the loss is applied to un-normalised "
            "inclusions, for which the statement holds directly.",
            "wu2022, Proposition 2 and Eq. 8",
            condition="zero loss",
            verified_in_mowl="tests/models/test_dfalc_model.py::"
                             "test_perceptual_grounding_is_kept_where_consistent; "
                             "tests/models/test_dfalc_model.py::"
                             "test_hierarchical_loss_is_zero_iff_pointwise_inclusion"),
        Guarantee(
            K.LIMITATION,
            "C ⊓ ¬C ≅ ⊥ and C ⊔ ¬C ≅ ⊤ do not hold in the Gödel/fuzzy ALC semantics "
            "(only (C ⊔ ¬C)^I >= 0.5 and (C ⊓ ¬C)^I <= 0.5).",
            "wu2022, Table 2 and the paragraph after it"),
    ),
    references=(
        Reference("wu2022",
                  "Wu, X., Zhu, X., Zhao, Y., Dai, X. (2022). Differentiable Fuzzy ALC: A "
                  "Neural-Symbolic Representation Language for Symbol Grounding. "
                  "arXiv:2211.12006.",
                  "https://arxiv.org/abs/2211.12006"),),
    related=("falcon",),
    notes=(
        "ADAPTATION. Implemented: the DF-ALC representation (truth-value vectors and "
        "matrices over Delta = the individuals, plus optional anonymous elements), Gödel "
        "operators (Eqs. 1-5), the hierarchical loss (Eq. 8), the rule-based loss of "
        "normal forms 1-3 (Eq. 9) and 4-7 (loss_type='rule') and initialisation from a perceptual grounding (DFALCModel.set_grounding). "
        "The rule-based losses of normal forms 4-7 (LaTeX source Eqs. 11-14, i.e. "
        "Loss_NF4..NF7) are implemented for inclusions of the shapes L ⊑ ∃r.L, L ⊑ ∀r.L, "
        "∃r.L ⊑ L, ∀r.L ⊑ L (L a concept name or its negation) when loss_type='rule', "
        "with the product t-norm, G(v, w) = ReLU(v - w) excluded from the gradient and "
        "one threshold alpha = alpha' (default 0.8). The sum index of the second term of "
        "Loss_NF4 is not written in the source; it is read as a sum over a. Other "
        "quantified inclusions use Eq. 8. Not implemented: the normalisation (the "
        "hierarchical loss is applied to arbitrary ALC inclusions, which it handles "
        "directly), the masked-ABox-revision and conjunctive-query experiments, and the "
        "neural classifiers that produce a perceptual grounding (image models are out of "
        "scope; pass the grounding in). ABox assertions are fuzzy assertions with truth "
        "value 1 (loss: distance to 1). The paper's code is not linked and was not "
        "checked. Truth values live in a table over the domain, so the representation is "
        "transductive (O(|Delta|^2) per role) and only suits small domains. Axiom scores "
        "for the benchmarks are the hierarchical loss of the candidate inclusion."),
)

CARDS = (EIKE, BOXSEL, DFALC)
