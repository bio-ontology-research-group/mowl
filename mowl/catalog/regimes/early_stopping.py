"""Regime card: Early stopping on validation loss."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2024NESY, MASHKOVA2024DELE

REGIME = RegimeCard(
    key='early-stopping',
    name='Early stopping on validation loss',
    kind=RegimeKind.TRAINING,
    summary='Training ends when the validation loss has not improved for a number of epochs.',
    components=('model selection', 'schedule'),
    applies_to='Any model with validation data',
    references=(MASHKOVA2024NESY, MASHKOVA2024DELE),
    standalone=False,
    introduced_with=(),
    used_by=(),
    not_implemented_reason=('No TrainingRegime hook can end the training loop (a hook such as '
                            'should_stop(epoch) -> bool, consulted after every epoch, is needed).'
                            ' ReduceOnPlateau records the epoch at which training would have '
                            'stopped.'),
    notes=("NeSy 2024, Section 4.4 and DELE, Section 6.3 ('early stopping if validation loss does"
           " not improve for 20 epochs')."),
)
