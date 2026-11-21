"""Partial-order and lattice embeddings of hierarchies (catalogued only).

These methods embed a partial order (a taxonomy, a DAG of hypernymy links) or a lattice of
events, not a description logic ontology. They are classified in the ``order`` family;
the ``region`` field gives the shape used to encode the order.
"""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Reference

_ORDER_ONLY = (
    "Embeds a partial order or DAG of subsumption links only; there is no interpretation "
    "of conjunction with other concepts, existential restrictions or roles, so it is not "
    "defined for DL ontologies beyond their taxonomy. mOWL can embed taxonomies with "
    "graph methods (card 'taxonomy_walks').")

ORDER_EMBEDDINGS = MethodCard(
    key="order_embeddings",
    name="Order embeddings",
    family=Family.ORDER,
    region="reversed product order on the positive orthant (cone x + R^n_+)",
    summary=(
        "Maps each element to a vector so that x ⪯ y iff y ≤ x coordinatewise, i.e. "
        "each element denotes the orthant cone above its vector; trained with a max-margin "
        "loss on order violations."),
    logic="Partial order (hypernymy, entailment, caption-image order)",
    concepts="vector (orthant above it)",
    roles="not represented",
    not_implemented_reason=_ORDER_ONLY,
    references=(
        Reference("vendrov2016",
                  "Vendrov, I., Kiros, R., Fidler, S., Urtasun, R. (2016). Order-Embeddings "
                  "of Images and Language. In Proceedings of ICLR 2016.",
                  "https://arxiv.org/abs/1511.06361"),),
)

HYPERBOLIC_CONES = MethodCard(
    key="hyperbolic_cones",
    name="Hyperbolic entailment cones",
    family=Family.ORDER,
    region="cone (geodesically convex, in the Poincaré ball)",
    summary=(
        "Each concept is a point of the Poincaré ball with an entailment cone attached; "
        "v is a subconcept of u when v lies in u's cone. Cone apertures are chosen so that "
        "the order is transitive."),
    logic="Partial order / DAG (e.g. WordNet hypernymy)",
    concepts="point and entailment cone in hyperbolic space",
    roles="not represented",
    not_implemented_reason=_ORDER_ONLY,
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "With the aperture function of Eqs. 25-26, the cone relation is transitive.",
            "ganea2018, Theorems 3 and 4"),
    ),
    references=(
        Reference("ganea2018",
                  "Ganea, O.-E., Bécigneul, G., Hofmann, T. (2018). Hyperbolic Entailment "
                  "Cones for Learning Hierarchical Embeddings. In Proceedings of ICML 2018.",
                  "https://arxiv.org/abs/1804.01882"),),
)

SHADOW_CONES = MethodCard(
    key="shadow_cones",
    name="Shadow cones",
    family=Family.ORDER,
    region="cone (umbral or penumbral shadow in hyperbolic space)",
    summary=(
        "Concepts are objects in hyperbolic space and the order is inclusion of the "
        "shadows they cast from a light source; entailment cones are a special case."),
    logic="Partial order / DAG",
    concepts="hyperbolic point (and ball) with its shadow",
    roles="not represented",
    not_implemented_reason=_ORDER_ONLY,
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Shadow cone relations are transitive partial orders, and entailment cones are "
            "equivalent to penumbral cones with a ball-shaped light source at the origin.",
            "yu2023shadow, Theorems 3.1 and 3.2"),
        Guarantee(
            K.LIMITATION,
            "Penumbral cones are geodesically convex, umbral cones are not; the shadow is "
            "undefined near the light source (a hole of radius r).",
            "yu2023shadow, Theorem 3.3 and Remark 1"),
    ),
    references=(
        Reference("yu2023shadow",
                  "Yu, T., Liu, T. J. B., Tseng, A., De Sa, C. (2024). Shadow Cones: A "
                  "Generalized Framework for Partial Order Embeddings. In Proceedings of "
                  "ICLR 2024.",
                  "https://arxiv.org/abs/2305.15215"),),
)

DISK_EMBEDDINGS = MethodCard(
    key="disk_embeddings",
    name="Hyperbolic disk embeddings",
    family=Family.ORDER,
    region="disk (ball) in a quasi-metric space, e.g. hyperbolic",
    summary=(
        "Each node of a DAG is a formal disk (centre and signed radius); the partial order "
        "is inclusion of lower cones, realised as disk inclusion. Order embeddings and "
        "hyperbolic entailment cones are special cases."),
    logic="Partial order / DAG",
    concepts="disk (centre, signed radius)",
    roles="not represented",
    not_implemented_reason=_ORDER_ONLY,
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Order embeddings and hyperbolic entailment cones are order isomorphic to disk "
            "embeddings (in Euclidean space and on the (n−1)-sphere, respectively).",
            "suzuki2019, Theorems 1 and 3"),
    ),
    references=(
        Reference("suzuki2019",
                  "Suzuki, R., Takahama, R., Onoda, S. (2019). Hyperbolic Disk Embeddings "
                  "for Directed Acyclic Graphs. In Proceedings of ICML 2019.",
                  "https://arxiv.org/abs/1902.04335"),),
)

BOX_LATTICE = MethodCard(
    key="box_lattice",
    name="Box lattice embeddings",
    family=Family.ORDER,
    region="box (probabilistic, with volume as probability)",
    summary=(
        "Concepts are boxes; box volume gives a probability, so joint "
        "and conditional probabilities come from intersection volumes and the boxes form a "
        "lattice (meet = intersection, join = smallest enclosing box)."),
    logic="Partial order / lattice with probabilities (e.g. WordNet, Flickr captions)",
    concepts="box",
    roles="not represented",
    not_implemented_reason=_ORDER_ONLY,
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Boxes ordered by inclusion form a lattice (meet: intersection or ⊥; join: "
            "smallest enclosing box) that is strictly more general than the order-embedding "
            "lattice in any dimension.",
            "vilnis2018, Section 4 and Appendix B"),
    ),
    intersection_closed=True,
    helly="2 (axis-parallel boxes; eckhoff1988)",
    references=(
        Reference("vilnis2018",
                  "Vilnis, L., Li, X., Murty, S., McCallum, A. (2018). Probabilistic "
                  "Embedding of Knowledge Graphs with Box Lattice Measures. In Proceedings of "
                  "ACL 2018.",
                  "https://arxiv.org/abs/1805.06627"),),
)

GUMBEL_BOX = MethodCard(
    key="gumbel_box",
    name="Gumbel box embeddings",
    family=Family.ORDER,
    region="box with Gumbel-distributed corners",
    summary=(
        "Box corners are min- and max-Gumbel random variables, which keeps the space closed "
        "under intersection while giving gradient signal for disjoint boxes (local "
        "identifiability)."),
    logic="Partial order / lattice with probabilities",
    concepts="random box (Gumbel corners)",
    roles="not represented",
    not_implemented_reason=_ORDER_ONLY,
    guarantees=(
        Guarantee(
            K.CLOSURE,
            "The space of Gumbel boxes is closed under intersection (max of max-Gumbels is "
            "max-Gumbel).",
            "dasgupta2020, abstract and Lemma 1"),
    ),
    intersection_closed=True,
    references=(
        Reference("dasgupta2020",
                  "Dasgupta, S. S., Boratko, M., Zhang, D., Vilnis, L., Li, X. L., "
                  "McCallum, A. (2020). Improving Local Identifiability in Probabilistic Box "
                  "Embeddings. In Proceedings of NeurIPS 2020.",
                  "https://arxiv.org/abs/2010.04831"),),
)

CARDS = (ORDER_EMBEDDINGS, HYPERBOLIC_CONES, SHADOW_CONES, DISK_EMBEDDINGS, BOX_LATTICE,
         GUMBEL_BOX)
