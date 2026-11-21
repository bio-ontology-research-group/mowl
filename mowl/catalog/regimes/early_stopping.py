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
    implementation='mowl.training.schedules.EarlyStopping',
    notes=("NeSy 2024, Section 4.4 and DELE, Section 6.3 ('early stopping if validation loss does"
           " not improve for 20 epochs'). Implemented by EarlyStopping(patience=20) (and by "
           "ReduceOnPlateau(stop=True) together with a learning-rate reduction): the training loop asks the regime's should_stop(epoch) after "
           "every epoch."),
)
