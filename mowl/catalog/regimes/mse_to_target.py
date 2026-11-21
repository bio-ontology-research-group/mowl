"""Regime card: Squared error to target scores."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import PENG2020

REGIME = RegimeCard(
    key='mse-to-target',
    name='Squared error to target scores',
    kind=RegimeKind.TRAINING,
    summary=('Positive scores are regressed to 0 and negative scores to a target t with a squared'
             ' error, per axiom or on the mean.'),
    components=('loss combination',),
    applies_to='EmbeddingELModel',
    references=(PENG2020,),
    standalone=False,
    introduced_with=('elbe',),
    used_by=(),
    implementation='mowl.training.losses.MSETargetLoss',
    notes=('Not described in the ELBE paper; the reference implementation (EL2Box_embedding, '
           'model/ELBoxlModel.py) applies MSELoss per axiom with target 0 to positives and target'
           " 2 to negatives. mOWL's ELBE squares the mean score with targets 0 and 1 "
           "(MSETargetLoss(neg_target=1, reduce='mean_first') reproduces it exactly, tested)."),
)
