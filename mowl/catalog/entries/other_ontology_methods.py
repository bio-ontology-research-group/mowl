"""Further ontology embedding methods that mOWL catalogues but does not implement."""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind as K, MethodCard, Origin,
                               Reference, Status)

EIKE = MethodCard(
    key="eike",
    name="EIKE",
    family=Family.GEOMETRIC,
    region="axis-aligned ellipsoid (plus language-model vectors)",
    summary=(
        "Embeds an ontology in two spaces: in the extensional space concepts are "
        "axis-aligned ellipsoids and instances are points; in the intensional space "
        "concepts are vectors initialised from a pre-trained language model applied to "
        "their names and descriptions. Trained on SubClassOf, InstanceOf and instance "
        "relation triples."),
    logic="Ontology as triples: SubClassOf, InstanceOf and instance relations, plus text",
    concepts="axis-aligned ellipsoid (extensional) and vector (intensional)",
    roles="relation vector between instances",
    individuals="point",
    not_implemented_reason=(
        "Uses only the triple view of an ontology (subclass and instance-of links), not "
        "DL axioms, and needs a pre-trained language model; within mOWL's scope its "
        "geometric part is covered by taxonomy/graph methods and its textual part by the "
        "language-model family."),
    intersection_closed=False,
    helly="d+1 (ellipsoids are convex sets in R^d)",
    references=(
        Reference("wang2024eike",
                  "Wang, K., Qi, G., Chen, J., Huang, Y., Wu, T. (2024). Embedding "
                  "Ontologies via Incorporating Extensional and Intensional Knowledge. "
                  "arXiv:2402.01677.",
                  "https://arxiv.org/abs/2402.01677"),),
)

BOXSEL = MethodCard(
    key="boxsel",
    name="BoxSEL (Statistical EL)",
    family=Family.GEOMETRIC,
    region="box",
    summary=(
        "Embeds Statistical EL TBoxes, whose conditional statements (D | C)[l, u] bound the "
        "proportion of C-instances that are D, as boxes in the style of BoxEL; conditional "
        "probabilities are read off as volume ratios."),
    logic="Statistical EL (SEL)",
    concepts="axis-parallel box",
    roles="as in BoxEL, on which the method builds",
    not_implemented_reason=(
        "Statistical EL conditionals (D | C)[l, u] cannot be expressed in OWL and are not "
        "part of mOWL's datasets; the method would need its own input format."),
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "If the TBox loss is zero, the induced geometric interpretation satisfies the "
            "SEL TBox.",
            "zhu2024, Theorem 1",
            condition="zero loss"),
        Guarantee(
            K.SOUNDNESS,
            "If the loss is zero and T ⊨ (D | C)[l, u], the probability p computed from the "
            "embedding lies in [l, u].",
            "zhu2024, Theorem 3",
            condition="zero loss"),
        Guarantee(
            K.EXPRESSIVITY,
            "For concepts C, D with non-empty, non-full regions, p with Iw ⊨ (D | C)[p] is "
            "computed in O((c + d)·n) time (c, d concept sizes, n dimension).",
            "zhu2024, Theorem 2"),
    ),
    intersection_closed=True,
    helly="2 (axis-parallel boxes; eckhoff1988)",
    references=(
        Reference("zhu2024",
                  "Zhu, Y., Potyka, N., Xiong, B., Tran, T.-K., Nayyeri, M., Kharlamov, E., "
                  "Staab, S. (2024). Approximating Probabilistic Inference in Statistical EL "
                  "with Knowledge Graph Embeddings. arXiv:2407.11821.",
                  "https://arxiv.org/abs/2407.11821"),),
)

DFALC = MethodCard(
    key="dfalc",
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
        "normal forms 1-3 (Eq. 9, loss_type='rule'; inclusions with quantifiers keep Eq. "
        "8) and initialisation from a perceptual grounding (DFALCModel.set_grounding). "
        "Not implemented: the normalisation (the hierarchical loss is applied to "
        "arbitrary ALC inclusions, which it handles directly), the rule-based losses of "
        "normal forms 4-7 (Eqs. 10-13, whose printed formulas are partly illegible in the "
        "PDF text), the masked-ABox-revision and conjunctive-query experiments, and the "
        "neural classifiers that produce a perceptual grounding (image models are out of "
        "scope; pass the grounding in). ABox assertions are fuzzy assertions with truth "
        "value 1 (loss: distance to 1). The paper's code is not linked and was not "
        "checked. Truth values live in a table over the domain, so the representation is "
        "transductive (O(|Delta|^2) per role) and only suits small domains. Axiom scores "
        "for the benchmarks are the hierarchical loss of the candidate inclusion."),
)

CARDS = (EIKE, BOXSEL, DFALC)
