"""Regime card: Correlation term on corrupted subsumptions."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='pearson-contrastive-term',
    name='Correlation term on corrupted subsumptions',
    kind=RegimeKind.TRAINING,
    summary=(
        'Once per step, the batch of gci0 and gci2 is corrupted k times per axiom and one'
        'minus the Pearson correlation of the margin-free scores exp(-loss) with the'
        'labels is added, weighted. Used only when the training data has no assertions.'),
    components=('negative sampling', 'loss combination'),
    applies_to='Modules with score(rows, form) = exp(-margin-free loss) (ELK-Em)',
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=(),
    implementation='mowl.training.elkem.PearsonContrastiveTerm',
    notes=(
        'ELK-Em, Section 4.2, Eq. 5. Computed in the additional-loss hook from the'
        'batches of the step, so it is independent of the per-form losses; the per-form'
        'variant is pearson-correlation-loss.'),
)
