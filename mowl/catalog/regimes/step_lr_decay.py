"""Regime card: Step decay of the learning rate."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import JACKERMEIER2024, SINGH2026

REGIME = RegimeCard(
    key='step-lr-decay',
    name='Step decay of the learning rate',
    kind=RegimeKind.TRAINING,
    summary='The learning rate is multiplied by a factor (0.1) after given epochs.',
    components=('optimiser schedule',),
    applies_to='EmbeddingELModel',
    references=(JACKERMEIER2024, SINGH2026),
    standalone=False,
    introduced_with=(),
    used_by=('box2el', 'elkem'),
    implementation='mowl.training.schedules.StepDecay',
    notes=("Box2EL, Appendix H ('decay the learning rate with a factor of 0.1 after 2000 training"
           " epochs' for some dataset/task pairs; reference code MultiStepLR); ELK-Em, Section "
           "5.1 ('learning rate 0.01 decayed by 0.1 after 5,000 epochs'). One mOWL epoch is one "
           'optimisation step, as in both reference implementations.'),
)
