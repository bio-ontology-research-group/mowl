"""Exact-model check of trained region embeddings."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._benchmark_refs import LEEMHUIS2025

REGIME = RegimeCard(
    key="exact-model-check",
    name="Exact-model and Helly check of trained regions",
    kind=RegimeKind.EVALUATION,
    summary=(
        "Reads the class regions off a trained region embedding and decides exactly, "
        "independently of the training loss, which propositional normal-form axioms "
        "(C ⊑ D, C ⊓ D ⊑ E, C ⊑ ⊥, C ⊓ D ⊑ ⊥) they satisfy, whether every satisfiable "
        "concept name has a non-empty region and a region with non-empty interior, and "
        "how many Helly certificates (pairwise satisfiable, jointly unsatisfiable "
        "triples) the regions violate. exact_model requires non-empty interiors; "
        "exact_model_nonempty only non-empty regions."),
    components=("model check", "degeneracy", "metrics"),
    applies_to=("models whose module has a reader in "
                "mowl.evaluation.exact_model.REGION_READERS: single boxes (ELBE, Box2EL, "
                "TransBox, ELK-Em, RegD-ELBE, BoxEL), balls (ELEm, EmEL++, EmEL(var), "
                "RegD-ELEm), prisms (PrismEL), products of octagons (OctagonEL), unions of "
                "boxes (MultiBoxEL, MultiELBE, volume MultiBoxEL); propositional EL_bot "
                "ontologies"),
    references=(LEEMHUIS2025,),
    standalone=False,
    implementation="mowl.evaluation.exact_model.ExactModelRegime",
    limitations=(
        "Role axioms are not checked.",
        "Decided exactly (closed regions, absolute tolerance tol): boxes, prisms and "
        "octagons in closed form (octagon tightening of Charpenay and Schockaert 2024); "
        "ball in ball and lens (two balls) in ball in closed form; emptiness of an "
        "intersection of any number of balls by exact support enumeration of the dual "
        "of min_x max_i (|x - c_i|^2 - r_i^2); unions of boxes in unions of boxes by "
        "exact box subtraction. Not decided (reported as unchecked, exact_model = 0): "
        "an intersection of three or more balls inside a ball, and a non-box piece "
        "covered only by several pieces of a union; neither occurs in binary normal "
        "forms of the supported models.",
        "The interior test asks for an inscribed ball of radius larger than tol / 2; "
        "for octagons this uses a small linear programme (scipy).",
        "Not region models: CanonicalMixture (membership matrix over the canonical "
        "domain), E2R (algebraic), ALCone (ALC, a different model family).",
    ),
    notes=(
        "Defined in mOWL for the helly-tk benchmark; it composes the exact checks of "
        "mowl.evaluation.regions with mowl.geometric (CanonicalModel, helly_check). The "
        "Helly certificates follow Leemhuis and Kutz (NeSy 2025). For single boxes the "
        "verdicts are those of mowl.geometric.checker.ModelChecker."),
)
