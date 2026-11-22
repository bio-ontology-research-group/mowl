"""ComposedEL: a model assembled from parts."""

from mowl.catalog.card import Family, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="composed",
    expressivity="EL⊥ with ABox",
    tasks=("subsumption", "satisfiability", "disjointness",
           "instance-checking", "role-assertion",),
    name="ComposedEL (a model from parts)",
    family=Family.GEOMETRIC,
    region="chosen (ball, box, ...)",
    summary=(
        "Not a method of its own: a model assembled from a class model, a relation "
        "model, a measure and an objective (mowl.representation). The published methods "
        "that are combinations of such parts are available by name and give the same "
        "losses as their own modules; any other combination is one line."),
    logic="EL++ in normal form",
    concepts="the chosen class model",
    roles="the chosen relation model",
    individuals="regions of the same kind as classes",
    status=Status.LEARNABLE,
    origin=Origin.ORIGINAL,
    implementation="mowl.models.composed.model.ComposedEL",
    module="mowl.representation.geometry.ComposedGeometry",
    axioms=("gci0", "gci1", "gci2", "gci3", "gci0_bot", "gci1_bot", "gci3_bot",
            "class_assertion", "object_property_assertion"),
    limitations=(
        "Guarantees depend on the parts chosen: a strict margin with an exact measure "
        "makes a zero loss imply the axiom; a slack margin does not.",),
    references=(R.KULMANOV2019, R.PENG2020),
    notes="See the documentation page on the parts of a method.",
)
