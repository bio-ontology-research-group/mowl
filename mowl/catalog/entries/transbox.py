"""TransBox (Yang, Chen & Sattler, 2025)."""

from mowl.catalog.card import Family, Guarantee, GuaranteeKind as K, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="transbox",
    expressivity="ELH(∘)⊥ with ABox",
    tasks=("subsumption", "satisfiability", "disjointness", "instance-checking",
           "role-assertion"),
    name="TransBox",
    family=Family.GEOMETRIC,
    region="box",
    summary=(
        "Concepts and roles are both axis-parallel boxes; a role box is a set of "
        "translations, so r(a, b) holds when x_a − x_b lies in the role box. Every EL++ "
        "constructor maps boxes to boxes, so complex concepts can be scored directly."),
    logic="EL++ (including role inclusions and role chains)",
    concepts="axis-parallel box (centre, offset)",
    roles="axis-parallel box of translations",
    individuals="point",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation="mowl.models.transbox.model.TransBox",
    module="mowl.nn.el.transbox.module.TransBoxModule",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion", "role_inclusion", "role_chain"),
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "If the TransBox loss of every axiom of O is zero, the induced geometric "
            "interpretation is a model of O; this is not affected by the semantic "
            "enhancement or by the extended intersections.",
            "yang2025, Theorem 5.1 (proof: Appendix A.6)",
            condition="zero loss"),
        Guarantee(
            K.CLOSURE,
            "EL++-closed: the intersection of two boxes is a box, ∃r.B is the box with "
            "centre c(r)+c(B) and offset o(r)+o(B), and the composition of two role boxes "
            "is the role box with summed centres and offsets.",
            "yang2025, Definition 3.1, Propositions 4.1 and 4.2"),
        Guarantee(
            K.EXPRESSIVITY,
            "Role inclusion r ⊑ s is represented as Box(r) ⊆ Box(s) while r and s keep "
            "different embeddings; roles can be one-to-many and many-to-many.",
            "yang2025, Section 4.1"),
        Guarantee(
            K.LIMITATION,
            "Under its box semantics TransBox is a box interpretation in the sense of "
            "Leemhuis & Kutz, so the Helly property applies: some satisfiable ontologies "
            "have no box model in any dimension, hence no zero-loss TransBox embedding.",
            "leemhuis2025, Example 1 and Proposition 5"),
    ),
    limitations=(
        "Two random boxes with centres in [−1, 1]^n and offsets in (0, 1]^n intersect with "
        "probability (2/3)^n (yang2025, Theorem 4.5); TransBox therefore uses 'extended' "
        "intersections over (R ∪ {∅})^n during training. Federico et al. (2026, multibox "
        "paper, Section 3.1) describe this as an approximated intersection that loosens "
        "the link to classical semantics.",
    ),
    intersection_closed=True,
    helly="2 (axis-parallel boxes, any dimension; eckhoff1988, as stated in leemhuis2025, "
          "Proposition 4)",
    references=(R.YANG2025, R.LEEMHUIS2025, R.FEDERICO2026_MULTIBOX),
    code_url="https://github.com/HuiYang1997/TransBox",
    notes=(
        "mOWL implements the semantic enhancement (∃r_all, ``use_enhancement=True``) and "
        "the masked extended intersection. GCIs C ⊑ ∃R.D are negated on both sides "
        "(two negatives per positive), as in the paper's Section 4.1."),
)
