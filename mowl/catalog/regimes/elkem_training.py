"""Regime card: ELK-Em training (paper design)."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='elkem-training',
    name='ELK-Em training (paper design)',
    kind=RegimeKind.TRAINING,
    summary=(
        'The default regime of ELK-Em: one random mini-batch per normal form and step,'
        'assertion negatives and weights, the correlation term when there are no'
        'assertions, an optional lexical regulariser, a step decay of the learning rate'
        'and checkpoint selection by validation median rank. A composition of the regimes'
        'fixed-concept-routing, assertion-corruption, per-normal-form-minibatch, loss-'
        'sum, pearson-contrastive-term, lexical-regularisation, step-lr-decay and'
        'validation-checkpoint.'),
    components=('batching', 'negative sampling', 'loss combination', 'optimiser schedule', 'model selection'),
    applies_to='mowl.models.ELKEm and ComposedEL(method="elkem")',
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=(),
    implementation='mowl.training.elkem.ELKEmTraining',
    notes=(
        'Follows the paper text (unsquared losses, batches without replacement,'
        'corruptions that avoid the subclass); elkem-code-training is the design of the'
        'authors code. Reproduces mowl.baselines.models.elkem step by step from the same'
        'seed.'),
)
