"""Regime card: Reduce learning rate on plateau."""

from mowl.catalog.card import Guarantee, GuaranteeKind, RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2024NESY, MASHKOVA2024DELE

REGIME = RegimeCard(
    key='reduce-lr-on-plateau',
    name='Reduce learning rate on plateau',
    kind=RegimeKind.TRAINING,
    summary=('The learning rate is reduced when the validation loss has not improved for a number'
             ' of epochs (patience 10); training can end after a longer patience (20).'),
    components=('optimiser schedule', 'early stopping'),
    applies_to='EmbeddingELModel with validation data',
    references=(MASHKOVA2024NESY, MASHKOVA2024DELE),
    standalone=False,
    introduced_with=(),
    used_by=('elem', 'elbe', 'box2el'),
    implementation='mowl.training.schedules.ReduceOnPlateau',
    properties=(
        Guarantee(
            GuaranteeKind.REGIME_PROPERTY,
            'With stop=True training ends after stop_patience monitored epochs without '
            'validation-loss improvement, as in the reference code.',
            'DELE, Section 6.3 ("early stopping if validation loss does not improve for 20 '
            'epochs")',
            condition='stop=True and a monitored validation loss'),
    ),
    notes=(
        "NeSy 2024, Section 4.4 and DELE, Section 6.3 ('Adam optimizer and ReduceLROnPlateau "
        "scheduler with patience parameter 10'); the reference code monitors the validation "
        'loss of the predicted normal form (gci2 for the GO & STRING data, gci0 for '
        'subsumption), once per paper epoch. With a per-normal-form-minibatch regime one '
        'mOWL epoch is one step; pass every=None to monitor once per paper epoch. With '
        'every=1 (the default) the regime only records the early-stop epoch (stop_epoch) '
        'unless stop=True is given.',
    ),
)
