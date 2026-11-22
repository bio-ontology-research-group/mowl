"""ComposedSequenceModel: a model that learns from sequences, assembled from parts."""

from mowl.catalog.card import Family, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

CARD = MethodCard(
    key="composed_sequences",
    expressivity="OWL 2 DL",
    tasks=(),
    head_tasks=("subsumption",),
    name="ComposedSequenceModel (walks and sentences from parts)",
    family=Family.GRAPH,
    region="chosen (walks or sentences + an encoder)",
    summary=(
        "Not a method of its own: a model assembled from a view of the ontology (a graph "
        "projection or an axiom or annotation corpus, optionally with entailed axioms), a "
        "source of sequences (random walks, sentences, both), an encoder (Word2Vec, "
        "fastText), the reading of an entity's vector, a measure of similarity and a scorer "
        "of candidate axioms (mowl.representation.sequence). DL2Vec, OWL2Vec*, taxonomy "
        "walks, Onto2Vec and OPA2Vec are available by name."),
    logic="Depends on the view (only the axiom patterns of the projection or the corpus)",
    concepts="vector",
    roles="vector (relation tokens in the sequences)",
    individuals="vector",
    status=Status.LEARNABLE,
    origin=Origin.ORIGINAL,
    implementation="mowl.models.composed.sequence.ComposedSequenceModel",
    limitations=(
        "The embedding reflects co-occurrence of tokens in the sequences, not the semantics "
        "of the axioms; the measure is a similarity (symmetric).",),
    references=(R.MIKOLOV2013,),
    notes="See the documentation page on methods that learn from sequences.",
)
