"""GroundedEL (element-grounded multi-box model) and the least-model solver it builds on."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

_SOURCE = "mOWL (original): docstring of mowl.models.grounded.model.GroundedEL"

_EL_TASKS = ("subsumption", "satisfiability", "disjointness", "instance-checking",
             "role-assertion", "query-answering")

GROUNDED = MethodCard(
    key="grounded",
    expressivity="ELH(∘)⊥ with ABox",
    tasks=_EL_TASKS,
    name="GroundedEL",
    family=Family.GEOMETRIC,
    region="union of boxes, grounded in domain elements",
    summary=(
        "Concepts are unions of K axis-parallel boxes and the domain is a finite set of "
        "elements, one per class of the normalised ontology and one per individual, each "
        "a point with a 'dead' score. Axioms are trained both on the elements (Łukasiewicz "
        "hinges on memberships) and on sampled points of the regions, so that the learned "
        "regions and the element interpretation agree. Training uses no reasoner."),
    logic="Normalised EL++ with role inclusions, role chains and ABox assertions; inverse "
          "roles in translation mode",
    concepts="union of K axis-parallel boxes",
    roles="union of product boxes (head x tail), or translation boxes (role_mode="
          "'translation')",
    individuals="domain element (a point)",
    status=Status.LEARNABLE,
    origin=Origin.ORIGINAL,
    implementation="mowl.models.grounded.model.GroundedEL",
    module="mowl.nn.el.grounded.module.GroundedELModule",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion", "role_inclusion", "role_chain"),
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "Exact-limit realisation: initialised from the labels of a least model "
            "(init='elements'), the model has no element violation and no region "
            "violation at zero training epochs, i.e. it is a model of the ontology.",
            _SOURCE + ", guarantee R2",
            condition="by construction (init='elements' with least-model labels)",
            verified_in_mowl="tests/models/test_grounded_model.py::"
                             "test_t3_elements_init_zero_epochs_exact"),
        Guarantee(
            K.COMPLETENESS,
            "For every n, an ELHO(∘) ontology is satisfiable iff it has a multibox model "
            "with domain R^n; the number of boxes per concept is not bounded.",
            "federico2026a, Theorem 3",
            condition="enough boxes per concept (not bounded by a fixed K)"),
        Guarantee(
            K.LIMITATION,
            "Training from a random initialisation is non-convex and is not guaranteed to "
            "reach a zero-violation model for every seed or at a too small K.",
            _SOURCE + ", non-guarantees",
            condition="random initialisation"),
    ),
    limitations=(
        "Corrupted-fact negatives (neg_weight > 0) are not filtered against the ABox: a "
        "corruption may coincide with a stated or entailed fact.",
        "The training procedure (temperature annealing, element and region sampling, "
        "idle-component re-seeding, Lagrangian dual weights) is implemented inside the "
        "model and is not yet available as training regimes for other methods.",
    ),
    intersection_closed="finite unions of boxes are closed under intersection, but the "
                        "number of boxes multiplies (K·K), so unions of at most K boxes "
                        "are not",
    helly="No Helly-2 bound: unions of K ≥ 2 boxes are not convex (federico2026a, "
          "Theorem 3).",
    related=("multiboxel", "canonical_model", "least_model", "query2box", "betae"),
    references=(R.FEDERICO2026_MULTIBOX, R.AHARONI2024, R.KAZAKOV2014),
    notes=(
        "Logical query answering (mowl.nn.el.grounded.queries): the query types 1p, 2p, "
        "3p, 2i, 3i, 2u, ip, pi and up of the query embedding literature are compiled to "
        "unions of boxes (anchor, projection as Minkowski sum, intersection, union) and "
        "scored over entities or concepts, or scored by a beam (max-min) search over "
        "witness elements; available in translation mode. The box algebra is exact while "
        "the number of boxes stays below max_boxes."),
)

LEAST_MODEL = MethodCard(
    key="least_model",
    expressivity="ELH(∘)⊥ with ABox",
    tasks=("subsumption", "satisfiability", "disjointness", "instance-checking",
           "role-assertion", "classification"),
    name="Least model (sparse saturation)",
    family=Family.CONSTRUCTIVE,
    region="finite least interpretation over the indexed syntactic domain",
    summary=(
        "Computes the least model of a normalised EL++ ontology over GroundedEL's domain "
        "(one element per class and per individual) as the least fixed point of the "
        "completion rules, with sparse matrices; also a literal LP formulation for small "
        "domains, an independent finite-model checker, and a realisation of the model by "
        "intervals of the real line."),
    logic="Normalised EL++ tables (NF1-NF4, ⊥ forms, role inclusions and chains, ABox "
          "assertions)",
    concepts="set of domain elements; intervals in the one-dimensional realisation",
    roles="relation between domain elements",
    individuals="domain element",
    status=Status.CONSTRUCTIVE,
    origin=Origin.ORIGINAL,
    implementation="mowl.geometric.least_model.LeastModel",
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "The iteration computes the least Boolean fixed point of the completion "
            "rules; the result is a finite model of the supplied tables, as verified by "
            "the independent checker check_model.",
            "mOWL (original): module docstring of mowl.geometric.least_model",
            condition="by construction",
            verified_in_mowl="tests/geometric/test_least_model.py::"
                             "test_rules_lp_checker_and_pruning"),
        Guarantee(
            K.LIMITATION,
            "A valid finite interpretation does not imply that every class is "
            "satisfiable; the LP method is intended for small finite domains only.",
            "mOWL (original): module docstring of mowl.geometric.least_model",
            condition="always"),
    ),
    related=("canonical_model", "grounded", "interval_union"),
    references=(R.KAZAKOV2014, R.BAADER2005),
    notes="Used to initialise GroundedEL exactly (init='elements') and to check it.",
)

CARDS = (GROUNDED, LEAST_MODEL)
