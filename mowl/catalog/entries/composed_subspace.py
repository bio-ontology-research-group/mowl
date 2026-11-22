"""ComposedSubspace: a model of subspace logic assembled from parts."""

from mowl.catalog.card import Family, MethodCard, Origin, Status
from mowl.catalog.entries.e2r import _GARG

CARD = MethodCard(
    key="composed_subspace",
    expressivity="E2R fragment with ABox",
    tasks=("subsumption", "disjointness", "role-subsumption", "instance-checking",
           "role-assertion"),
    name="ComposedSubspace (a model of subspace logic from parts)",
    family=Family.ALGEBRAIC,
    region="linear subspace (quantum logic)",
    summary=(
        "Not a method of its own: a model whose classes and relations are linear subspaces, "
        "assembled from a class model, a relation model, an individual model, a measure, an "
        "objective and regularisers (mowl.representation), any of which can be swapped by "
        "name. Embed2Reason (E2R) is available as the named combination and gives the same "
        "losses as its direct implementation."),
    logic="The fragment read by the subspace geometry: named-concept inclusion, "
          "conjunction, disjunction and negation definitions of named concepts, range "
          "restrictions, role inclusion, concept and role assertions",
    concepts="the chosen class model (axis-parallel subspace by default)",
    roles="the chosen relation model",
    individuals="the chosen individual model (unit vector by default)",
    status=Status.LEARNABLE,
    origin=Origin.ORIGINAL,
    implementation="mowl.models.composed.subspace.ComposedSubspace",
    module="mowl.representation.library.logic_subspace.SubspaceGeometry",
    axioms=("class_assertion", "object_property_assertion", "subclass_of",
            "equivalent_classes", "disjoint_classes", "sub_object_property_of",
            "object_property_range"),
    limitations=(
        "Guarantees depend on the parts chosen; those stated for E2R hold for the E2R "
        "combination only.",
        "A measure that cannot be combined with subspace regions is refused.",),
    references=(_GARG,),
    related=("e2r",),
    notes="See the documentation page on the parts of a method.",
)
