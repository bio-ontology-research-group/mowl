"""ComposedALC: a model for ALC axioms assembled from parts."""

from mowl.catalog.card import Family, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="composed-alc",
    expressivity="ALC with ABox",
    tasks=("subsumption", "satisfiability", "disjointness",
           "instance-checking", "role-assertion",),
    name="ComposedALC (a model for ALC axioms from parts)",
    family=Family.GEOMETRIC,
    region="chosen (fuzzy set, half-axis cone, ...)",
    summary=(
        "Not a method of its own: a model for axioms with class expressions of any depth, "
        "assembled from a class model, a relation model, a measure and an objective "
        "(mowl.representation) that reads such axioms. DF-ALC, FALCON and Al-cone are "
        "available by name and give the same losses as their own modules; any other "
        "combination is a few lines."),
    logic="ALC (subclass, equivalence and disjointness axioms between class expressions "
          "with intersection, union, complement, existential and universal restriction; "
          "class and object property assertions)",
    concepts="the chosen class model",
    roles="the chosen relation model",
    individuals="the elements of the domain, or regions of the same kind as classes",
    status=Status.LEARNABLE,
    origin=Origin.ORIGINAL,
    implementation="mowl.models.composed.alc.ComposedALC",
    module="mowl.representation.library.logic_alc.ExpressionGeometry",
    axioms=("subclass_of", "equivalent_classes", "disjoint_classes", "class_assertion",
            "object_property_assertion"),
    limitations=(
        "Guarantees depend on the parts chosen: a measure with a strict margin that is "
        "exact for the region family makes a zero loss imply the axiom, a graded measure "
        "does not.",),
    references=(R.FALCON2022,),
    notes="See the documentation page on the parts of a method.",
)
