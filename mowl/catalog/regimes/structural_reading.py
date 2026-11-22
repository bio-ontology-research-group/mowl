"""Regime cards: the regimes of the structural and the point readings of MultiBoxEL and
MultiELBE (structural tables, extra penalties, sampled point loss)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.entries._refs import FEDERICO2026_MULTIBOX

REGIMES = [
    RegimeCard(
        key="structural-tables",
        name="Structural tables compiled when training starts",
        kind=RegimeKind.TRAINING,
        summary=(
            "At the start of training the definitions A = X and Y, the satisfiable classes "
            "and pairs and the witness targets are compiled from the training axioms and "
            "installed in the geometry: defined classes become intersections of their "
            "definers, the existence and witness penalties get their targets."),
        components=("positives",),
        applies_to="Geometries with set_structural_tables (structural reading: slots of "
                   "boxes, prisms)",
        references=(FEDERICO2026_MULTIBOX,),
        standalone=False,
        introduced_with=("multiboxel",),
        used_by=("multiboxel",),
        implementation="mowl.training.structural.StructuralTables",
        notes="The structural reading of MultiBoxEL (objective='structural').",
    ),
    RegimeCard(
        key="extra-penalties",
        name="Penalties of the geometry added at every step",
        kind=RegimeKind.TRAINING,
        summary=(
            "Adds the penalties of the geometry that its regularisation loss does not count "
            "(the existence, size and witness penalties of the structural reading) to the "
            "loss of every step."),
        components=("loss combination",),
        applies_to="Geometries with extra_penalties()",
        references=(FEDERICO2026_MULTIBOX,),
        standalone=False,
        introduced_with=("multiboxel",),
        used_by=("multiboxel",),
        implementation="mowl.training.structural.ExtraPenalties",
        notes="The structural reading of MultiBoxEL.",
    ),
    RegimeCard(
        key="sampled-point-loss",
        name="Point loss on a sample of canonical elements",
        kind=RegimeKind.TRAINING,
        summary=(
            "Every step draws pairs and role triples of canonical elements (seed + epoch) "
            "and adds the binary cross-entropy of their soft memberships and the log-volume "
            "penalty of the named classes."),
        components=("loss combination",),
        applies_to="Geometries with point terms and models with point supervision",
        references=(FEDERICO2026_MULTIBOX,),
        standalone=False,
        introduced_with=("multiboxel",),
        used_by=("multiboxel", "multielbe"),
        implementation="mowl.training.structural.SampledPointLoss",
        notes="The point, mixed and hybrid readings of MultiBoxEL and MultiELBE.",
    ),
]
