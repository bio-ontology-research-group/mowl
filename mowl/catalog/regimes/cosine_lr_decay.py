"""Regime card: Constant then cosine-decayed learning rate."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='cosine-lr-decay',
    name='Constant then cosine-decayed learning rate',
    kind=RegimeKind.TRAINING,
    summary='Constant learning rate for a fraction of training, then cosine decay to a floor.',
    components=('optimiser schedule',),
    applies_to='EmbeddingELModel',
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=('grounded',),
    implementation='mowl.training.schedules.ConstantThenCosine',
    notes=("ELK-Em, Section 5.2 ('held constant for the first third of training and then "
           "cosine-decayed to 10^-4')."),
)
