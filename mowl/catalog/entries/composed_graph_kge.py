"""GraphKGEModel: a knowledge graph embedding of a projection, assembled from parts."""

from mowl.catalog.card import Family, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="composed_graph_kge",
    expressivity="EL with ABox",
    tasks=(),
    head_tasks=("subsumption",),
    name="GraphKGEModel (knowledge graph embedding of a projection, from parts)",
    family=Family.GRAPH,
    region="chosen (points, boxes, parallelograms, matrices)",
    summary=(
        "Not a method of its own: a model assembled from the parts of a registered method "
        "(graph_kge, gda_projection, boxe, expressive, reshuffle): a projection of the "
        "ontology into triples, an entity model, a relation model, the triple score, a loss, "
        "a negative sampler and regularisers, any of which can be swapped by name. BoxE, "
        "ExpressivE and RESHUFFLE are available as named combinations."),
    logic="Only the axiom patterns the projection turns into triples",
    concepts="point (or encoded point)",
    roles="chosen relation model",
    individuals="point",
    status=Status.LEARNABLE,
    origin=Origin.ORIGINAL,
    implementation="mowl.models.graph_kge.triple_model.GraphKGEModel",
    limitations=(
        "Guarantees of the knowledge graph methods refer to the projected triples, not to "
        "the description logic semantics; axioms the projector does not read are ignored.",),
    references=(R.ALI2021,),
    related=("graph_kge", "boxe", "expressive", "reshuffle", "gda_projection"),
    notes="See the documentation page on knowledge graph models on a projection.",
)
