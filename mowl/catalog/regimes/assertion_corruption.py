"""Regime card: Corruption of ABox assertions."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='assertion-corruption',
    name='Corruption of ABox assertions',
    kind=RegimeKind.TRAINING,
    summary=(
        'Negatives of class assertions C(a) (the class replaced by a class the model does'
        'not fix) and of role assertions r(a, b) (the object replaced by an individual);'
        'corruptions that are observed assertions are redrawn. Other axiom types get no'
        'negatives.'),
    components=('negative sampling',),
    applies_to='EL models with a negative loss for assertions (ELK-Em)',
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=(),
    implementation='mowl.training.elkem.AssertionCorruption',
    notes=(
        'ELK-Em, Section 5.2 (abox_k_neg corruptions per assertion, 2 in the paper).'
        'Samples from the global torch generator.'),
)
