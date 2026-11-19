"""Regime card: Reduce learning rate on plateau."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2024NESY, MASHKOVA2024DELE

REGIME = RegimeCard(
    key='reduce-lr-on-plateau',
    name='Reduce learning rate on plateau',
    kind=RegimeKind.TRAINING,
    summary=('The learning rate is reduced when the validation loss has not improved for a number'
             ' of epochs (patience 10).'),
    components=('optimiser schedule',),
    applies_to='EmbeddingELModel with validation data',
    references=(MASHKOVA2024NESY, MASHKOVA2024DELE),
    standalone=False,
    introduced_with=(),
    used_by=(),
    implementation='mowl.training.schedules.ReduceOnPlateau',
    limitations=(
        ('The early stopping that accompanies it in both papers (20 epochs without improvement) '
         'cannot be expressed with the TrainingRegime hooks; the regime only records the epoch at'
         " which it would stop (see 'early-stopping')."),
    ),
    notes=("NeSy 2024, Section 4.4 and DELE, Section 6.3 ('Adam optimizer and ReduceLROnPlateau "
           "scheduler with patience parameter 10'); the reference code monitors the validation "
           'loss of the predicted normal form (gci2).'),
)
