"""Regime card: negatives drawn once and reused."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2026

REGIME = RegimeCard(
    key='draw-negatives-once',
    assumption="closed world over stated axioms",
    name='Negatives drawn once and reused',
    kind=RegimeKind.TRAINING,
    summary=('Wraps a negative sampler and keeps its first draw per normal form for all '
             'later epochs, so the model always sees the same corruptions.'),
    components=('negative sampling',),
    applies_to='EmbeddingELModel',
    references=(MASHKOVA2026,),
    standalone=False,
    introduced_with=(),
    used_by=('elem', 'emelpp', 'box2el'),
    implementation='mowl.training.kbc.DrawOnce',
    notes=('The data loader of the faithfulness reference code (kbc, dataloader.py) calls '
           'np.random.seed(0) before every batch; with one batch per epoch the corruptions '
           'are identical in every epoch and for every run seed.'),
)
