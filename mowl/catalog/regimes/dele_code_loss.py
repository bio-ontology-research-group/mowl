"""Regime card: how the DELE reference code combines per-axiom losses."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2024DELE

REGIME = RegimeCard(
    key="dele-code-loss",
    name="Loss combination of the DELE reference code",
    kind=RegimeKind.TRAINING,
    summary=("Per normal form: mean of squared positive plus mean of squared negative "
             "losses (ELBE, Box2EL; ELEm only for GCI2 with negatives, plain means "
             "otherwise), as the training loops of the DELE reference code do."),
    components=("loss combination",),
    applies_to="ELEmModule, ELBEModule, BoxSquaredELModule",
    references=(MASHKOVA2024DELE,),
    standalone=False,
    introduced_with=("elem", "elbe", "box2el"),
    used_by=("elem", "elbe", "box2el"),
    implementation="mowl.training.dele.DELECodeLoss",
    notes=("From models/*.py of bio-ontology-research-group/DELE (commit 5974c7d). The "
           "paper (Eq. 23) only says 'the sum of individual GCI losses'. ELEm: "
           "mean(l+^2) + mean(l-^2) for GCI2 when it has negatives, mean(l+) + mean(l-) "
           "otherwise. ELBE: every normal form mean(l+^2) + mean(l-^2). Box2EL: the loop "
           "squares the module losses; the reference's GCI2 and GCI3 negative losses "
           "already return the batch mean of (delta - d)^2, so they enter squared again "
           "(mean(l-)^2). Without this regime mOWL's ELBE squares the means and pulls the "
           "negative loss to 1, which with the DELE negative losses (zero iff the "
           "negative is satisfied) trains negatives to be violated: ELBE on the yeast "
           "PPI benchmark then ranks close to random (macro AUC 0.59, run "
           "dele-yp-elbe-base-s0, before this regime existed)."),
)
