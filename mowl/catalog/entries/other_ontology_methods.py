"""Further ontology embedding methods that mOWL catalogues but does not implement."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Reference

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
    name="Differentiable fuzzy ALC (DF-ALC)",
    family=Family.FUZZY,
    region="fuzzy set over grounded individuals",
    summary=(
        "A neural-symbolic representation language: concept and role names are grounded "
        "as fuzzy membership values of individuals (e.g. produced by image classifiers), "
        "and a normalised ALC ontology is enforced through a hierarchical loss on fuzzy "
        "concept inclusions, revising the grounding."),
    logic="ALC (normalised)",
    concepts="fuzzy membership vector over individuals",
    roles="fuzzy membership matrix over pairs of individuals",
    individuals="domain element (grounded, e.g. by an image)",
    not_implemented_reason=(
        "Designed for symbol grounding of perceptual data (e.g. images), with membership "
        "degrees coming from external neural classifiers; it does not learn vector "
        "embeddings of an ontology's concepts. FALCON (card 'falcon') is the closest "
        "method in mOWL."),
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "Fuzzy entailment in the translated DF-ALC ontology is consistent with "
            "entailment in ALC.",
            "wu2022, Proposition 1"),
        Guarantee(
            K.SOUNDNESS,
            "When the hierarchical loss converges to 0, the learned interpretation is a "
            "model of the given ALC ontology.",
            "wu2022, Proposition 2",
            condition="zero loss"),
    ),
    references=(
        Reference("wu2022",
                  "Wu, X., Zhu, X., Zhao, Y., Dai, X. (2022). Differentiable Fuzzy ALC: A "
                  "Neural-Symbolic Representation Language for Symbol Grounding. "
                  "arXiv:2211.12006.",
                  "https://arxiv.org/abs/2211.12006"),),
)

CARDS = (EIKE, BOXSEL, DFALC)
