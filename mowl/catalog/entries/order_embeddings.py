"""Partial-order and lattice embeddings of hierarchies (catalogued only).

These methods embed a partial order (a taxonomy, a DAG of hypernymy links) or a lattice of
events, not a description logic ontology. They are classified in the ``order`` family;
the ``region`` field gives the shape used to encode the order.
"""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind as K, MethodCard, Origin,
                               Reference, Status)

_ORDER_ONLY = (
    "Embeds a partial order or DAG of subsumption links only; there is no interpretation "
    "of conjunction with other concepts, existential restrictions or roles, so it is not "
    "defined for DL ontologies beyond their taxonomy. mOWL can embed taxonomies with "
    "graph methods (card 'taxonomy_walks').")

_TAXONOMY = (
    "Embeds the told subclass order between named classes (C ⊑ D, the gci0 normal form) "
    "only; conjunction, existential restrictions, roles, individuals and disjointness have no "
    "interpretation and the axioms that use them are ignored.")

_TAXONOMY_NOTE = (
    "Adaptation to ontologies: the subclass axioms C ⊑ D between named classes are the order "
    "pairs; the training data is the 'gci0' normal form restricted to named classes. "
    "Conjunction, existential restrictions, bottom, assertions and role axioms are not used "
    "(dropped from training, no score). Tasks: subsumption prediction and inference between "
    "named classes. Training is the generic full-batch EL loop with negatives from corrupting "
    "the stated pairs; the paper's mini-batches and early stopping are not reproduced. "
    "Reference code was not consulted (not fetched); the implementation follows the paper text.")

ORDER_EMBEDDINGS = MethodCard(
    key="order_embeddings",
    assumption="closed world over stated axioms",
    training=("pairwise-margin-ranking", "either-side-corruption", "transitive-closure-positives",),
    evaluation=("hierarchy-threshold-f1", "common-filtered-ranking",),
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
    expressivity="taxonomy",
    tasks=("subsumption",),
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.order_embeddings.model.OrderEmbeddings",
    module="mowl.nn.el.order_embeddings.module.OrderEmbeddingsModule",
    axioms=("gci0",),
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "The violation E(x, y) = ||max(0, y - x)||_2 is zero iff x ⪯ y in the reversed "
            "product order; a zero loss on stated pairs therefore implies the pairs of "
            "their transitive closure (transitivity of the order).",
            "vendrov2016, Section 2 (Eq. 2)",
            condition="zero loss",
            verified_in_mowl="tests/models/test_order_embeddings_model.py::"
                             "test_violation_is_zero_iff_order_holds_and_order_is_transitive"),
    ),
    limitations=(
        "Only the subclass order between named classes; no conjunction, existentials, "
        "bottom, assertions or roles.",
        "The paper's order embeddings are a distributive vector lattice; they cannot model "
        "disjointness as box lattices can (Vilnis et al. 2018, Appendix B).",
    ),
    notes=_TAXONOMY_NOTE,
    references=(
        Reference("vendrov2016",
                  "Vendrov, I., Kiros, R., Fidler, S., Urtasun, R. (2016). Order-Embeddings "
                  "of Images and Language. In Proceedings of ICLR 2016.",
                  "https://arxiv.org/abs/1511.06361"),),
)

HYPERBOLIC_CONES = MethodCard(
    key="hyperbolic_cones",
    assumption="closed world over stated axioms",
    training=("pairwise-margin-ranking", "transitive-closure-positives", "known-positive-filtering",),
    evaluation=("hierarchy-threshold-f1", "common-filtered-ranking",),
    name="Hyperbolic entailment cones",
    family=Family.ORDER,
    region="cone (geodesically convex, in the Poincaré ball)",
    summary=(
        "Each concept is a point of the Poincaré ball with an entailment cone attached; "
        "v is a subconcept of u when v lies in u's cone. Cone apertures are chosen so that "
        "the order is transitive."),
    logic="Partial order / DAG (e.g. WordNet hypernymy); in mOWL the subclass order of an ontology",
    concepts="point and entailment cone in hyperbolic space",
    roles="not represented",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.hyperbolic_cones.model.HyperbolicCones",
    module="mowl.nn.order.module.HyperbolicConesModule",
    expressivity="taxonomy",
    tasks=("subsumption",),
    axioms=("gci0",),
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "With the aperture function of Eqs. 25-26, the cone relation is transitive.",
            "ganea2018, Theorems 3 and 4",
            verified_in_mowl="tests/models/test_hyperbolic_cones_model.py::"
                             "test_cone_relation_is_transitive"),
        Guarantee(
            K.LIMITATION,
            "No cone can be defined at the origin, so points are kept outside an "
            "epsilon-ball around it (epsilon fixed by K).",
            "ganea2018, Lemma 2 and Theorem 3"),
    ),
    limitations=(_TAXONOMY,),
    references=(
        Reference("ganea2018",
                  "Ganea, O.-E., Bécigneul, G., Hofmann, T. (2018). Hyperbolic Entailment "
                  "Cones for Learning Hierarchical Embeddings. In Proceedings of ICML 2018.",
                  "https://arxiv.org/abs/1804.01882"),),
    code_url="https://github.com/dalab/hyperbolic_cones",
    notes=(
        "Geometry and energy are the paper's (Eqs. 26, 28, 33) and the loss is the max-margin "
        "loss of Eq. 32 (means instead of sums). Adaptations: the paper initialises from 100 "
        "epochs of Poincaré embeddings (not implemented; random initialisation instead); "
        "training uses the generic full-batch loop of mOWL with Riemannian SGD (exponential "
        "map) instead of batches of 10; sampled negatives are re-drawn when they are told "
        "axioms but not when they are entailed; the Euclidean cones are not implemented. "
        "The ontology is used through its subclass axioms between named classes only. The "
        "score of an axiom is the energy E(D, C), which is 0 for every contained pair, so "
        "rankings have ties among contained candidates."),
)

SHADOW_CONES = MethodCard(
    key="shadow_cones",
    assumption="closed world over stated axioms",
    training=("logsumexp-contrastive", "softmax-energy-contrastive", "transitive-closure-positives", "known-positive-filtering",),
    evaluation=("hierarchy-threshold-f1", "common-filtered-ranking",),
    name="Shadow cones",
    family=Family.ORDER,
    region="cone (umbral or penumbral shadow in hyperbolic space)",
    summary=(
        "Concepts are objects in hyperbolic space and the order is inclusion of the "
        "shadows they cast from a light source; entailment cones are a special case."),
    logic="Partial order / DAG; in mOWL the subclass order of an ontology",
    concepts="hyperbolic point (and ball) with its shadow",
    roles="not represented",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.shadow_cones.model.ShadowCones",
    module="mowl.nn.order.module.ShadowConesModule",
    expressivity="taxonomy",
    tasks=("subsumption",),
    axioms=("gci0",),
    code_url="https://github.com/ydtydr/ShadowCones",
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Shadow cone relations are transitive partial orders, and entailment cones are "
            "equivalent to penumbral cones with a ball-shaped light source at the origin.",
            "yu2023shadow, Theorems 3.1 and 3.2",
            verified_in_mowl="tests/models/test_shadow_cones_model.py::"
                             "test_penumbral_ball_is_the_entailment_cone"),
        Guarantee(
            K.LIMITATION,
            "Penumbral cones are geodesically convex, umbral cones are not; the shadow is "
            "undefined near the light source (a hole of radius r).",
            "yu2023shadow, Theorem 3.3 and Remark 1"),
    ),
    limitations=(_TAXONOMY,),
    references=(
        Reference("yu2023shadow",
                  "Yu, T., Liu, T. J. B., Tseng, A., De Sa, C. (2024). Shadow Cones: A "
                  "Generalized Framework for Partial Order Embeddings. In Proceedings of "
                  "ICLR 2024.",
                  "https://arxiv.org/abs/2305.15215"),),
    notes=(
        "neg=True of the module is the hinge max(0, gamma1 - E) (negative='energy' returns the raw "
        "energy, the form Eq. 1 is written on; the loss of Eq. 1 is the same for both). "
        "All four formulations of the paper (umbral and penumbral, Poincaré half-space and "
        "ball), the two-case distance energy of Section 4 and the loss of Eq. 1. "
        "Adaptations: curvature is fixed at -1; Riemannian SGD only (the paper uses "
        "RiemannianAdam for the ball) and the half-space update is a retraction; no burn-in "
        "phase; generic full-batch training. The default loss is Eq. 1 of the paper; the authors' code trains with another loss by default (a softmax cross-entropy including the positive, ``loss_cross``), which is the regime ``softmax-energy-contrastive`` (attach ``SoftmaxEnergyContrastiveLoss(gamma2=margin)`` to ``training_regime``). For umbral-ball the aperture and altitude "
        "were re-derived with the hyperbolic law of sines because the printed Appendix F.2 "
        "appears to contain typographical errors (least certain formulation; no published "
        "numbers reproduced). The radius is not trained. Rankings use the signed distance "
        "to the cone, so contained candidates are not tied."),
)

DISK_EMBEDDINGS = MethodCard(
    key="disk_embeddings",
    assumption="closed world over stated axioms",
    training=("pairwise-margin-ranking", "transitive-closure-positives", "known-positive-filtering",),
    evaluation=("hierarchy-threshold-f1", "common-filtered-ranking",),
    name="Hyperbolic disk embeddings",
    family=Family.ORDER,
    region="disk (ball) in a quasi-metric space, e.g. hyperbolic",
    summary=(
        "Each node of a DAG is a formal disk (centre and signed radius); the partial order "
        "is inclusion of lower cones, realised as disk inclusion. Order embeddings and "
        "hyperbolic entailment cones are special cases."),
    logic="Partial order / DAG; in mOWL the subclass order of an ontology",
    concepts="disk (centre, signed radius)",
    roles="not represented",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.disk_embeddings.model.DiskEmbeddings",
    module="mowl.nn.order.module.DiskEmbeddingsModule",
    expressivity="taxonomy",
    tasks=("subsumption",),
    axioms=("gci0",),
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Order embeddings and hyperbolic entailment cones are order isomorphic to disk "
            "embeddings (in Euclidean space and on the (n−1)-sphere, respectively).",
            "suzuki2019, Theorems 1 and 3"),
        Guarantee(
            K.EXPRESSIVITY,
            "Disk inclusion d(x, y) <= r_x - r_y is transitive (triangle inequality) and "
            "invariant under adding a constant to all radii (translational symmetry, Eq. 4).",
            "suzuki2019, Section 2.3",
            verified_in_mowl="tests/models/test_disk_embeddings_model.py::"
                             "test_inclusion_is_transitive_and_shift_invariant"),
    ),
    limitations=(_TAXONOMY,),
    references=(
        Reference("suzuki2019",
                  "Suzuki, R., Takahama, R., Onoda, S. (2019). Hyperbolic Disk Embeddings "
                  "for Directed Acyclic Graphs. In Proceedings of ICML 2019.",
                  "https://arxiv.org/abs/1902.04335"),),
    notes=(
        "Hyperbolic (Lorentz model), spherical and Euclidean disk embeddings with the "
        "protrusion of Eq. 9, the margin loss of Eq. 10 with E = l (Eq. 11) and Riemannian "
        "SGD (Eqs. 13-14). Adaptations: generic full-batch training instead of the paper's "
        "sampling; the spherical model is not initialised from Poincaré embeddings; the "
        "polyhedral quasi-metrics used for the equivalence with order embeddings are not "
        "implemented. Rankings use the signed protrusion l, not its positive part. The manifold "
        "distances are exactly 0 for coincident points (they used to have a small floor), so "
        "a disk inside itself costs exactly 0."),
)

BOX_LATTICE = MethodCard(
    key="box_lattice",
    assumption="closed world over stated axioms",
    training=("probabilistic-bce", "transitive-closure-positives",),
    evaluation=("hierarchy-threshold-f1", "common-filtered-ranking",),
    name="Box lattice embeddings",
    family=Family.ORDER,
    region="box (probabilistic, with volume as probability)",
    summary=(
        "Concepts are boxes; box volume gives a probability, so joint "
        "and conditional probabilities come from intersection volumes and the boxes form "
        "a lattice (meet = intersection, join = smallest enclosing box)."),
    logic="Partial order / lattice with probabilities (e.g. WordNet, Flickr captions)",
    concepts="box",
    roles="not represented",
    expressivity="taxonomy",
    tasks=("subsumption",),
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.box_lattice.model.BoxLattice",
    module="mowl.nn.el.box_lattice.module.BoxLatticeModule",
    axioms=("gci0",),
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Boxes ordered by inclusion form a lattice (meet: intersection or ⊥; join: "
            "smallest enclosing box) that is strictly more general than the order-embedding "
            "lattice in any dimension (the box lattice is non-distributive).",
            "vilnis2018, Section 4 and Appendix B",
            verified_in_mowl="tests/models/test_box_lattice_model.py::"
                             "test_lattice_operations_and_non_distributivity"),
    ),
    limitations=(
        "Only the subclass order between named classes; no conjunction, existentials, "
        "bottom, assertions or roles. The box meet is not the class conjunction.",
        "The complement of a box is not a box (vilnis2018, Section 4.2).",
    ),
    notes=_TAXONOMY_NOTE + (
        " The positive loss of an included pair is exactly 0 (zero_at_inclusion=True; False is "
        "the earlier -log(1 + eps) < 0). Probabilities: stated pairs get the target P(D|C) = 1, corrupted pairs P(D|C) = 0 "
        "(the paper uses calibrated targets for Flickr and unary marginals for WordNet; the "
        "latter are optional via unary_weight). Ours: for disjoint boxes the positive loss "
        "is the linear form of the paper's lower bound of the joint; the paper's printed "
        "log of the bound has an inconsistent sign and is undefined for non-positive "
        "bounds. min_delta defaults to 0.01 (paper: 1e-6 for WordNet; with 1e-6 boxes "
        "collapse in a dimension on small taxonomies)."),
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
    assumption="closed world over stated axioms",
    training=("probabilistic-bce", "transitive-closure-positives",),
    evaluation=("hierarchy-threshold-f1", "common-filtered-ranking",),
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
    expressivity="taxonomy",
    tasks=("subsumption",),
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.gumbel_box.model.GumbelBox",
    module="mowl.nn.el.gumbel_box.module.GumbelBoxModule",
    axioms=("gci0",),
    guarantees=(
        Guarantee(
            K.CLOSURE,
            "The space of Gumbel boxes is closed under intersection (max of max-Gumbels is "
            "max-Gumbel).",
            "dasgupta2020, abstract and Lemma 1",
            verified_in_mowl="tests/models/test_gumbel_box_model.py::"
                             "test_lemma_1_max_of_max_gumbels"),
    ),
    limitations=(
        "Only the subclass order between named classes; no conjunction, existentials, "
        "bottom, assertions or roles.",
        "Volumes and conditionals use the softplus approximation of the Bessel expression "
        "and the first-order approximation of the conditional (dasgupta2020, Sections 4.1 "
        "and 4.2); P(D|C) < 1 even for identical boxes at finite beta.",
    ),
    notes=_TAXONOMY_NOTE + (
        " The logarithms of side lengths and of 1 - P use the guard of the binary code boxes "
        "(log_guard='stable'; 'epsilon' is the earlier 1e-30 / 1e-9 guard, which floored the "
        "log volume of a collapsed side with zero gradient). Locations are kept in the unit cube by projection. The signs of Algorithm 1's "
        "log-sum-exp terms are swapped relative to Lemma 1; the implementation follows "
        "Lemma 1. The paper's code and hyperparameter search (Bayesian, Hyperband) were not "
        "reproduced."),
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
