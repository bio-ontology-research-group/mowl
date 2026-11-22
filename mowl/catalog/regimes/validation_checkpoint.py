"""Regime card: Checkpoint file selected by a validation value."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='validation-checkpoint',
    name='Checkpoint file selected by a validation value',
    kind=RegimeKind.TRAINING,
    summary=(
        'Every few epochs a validation value (median rank of validation subsumptions, or'
        'the validation loss of the evaluation form) is computed and the parameters are'
        'saved to the model file when it improves; the final state is saved when nothing'
        'improved or there is no validation data. The module keeps its final parameters.'),
    components=('model selection',),
    applies_to='EmbeddingELModel and models built on it',
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=('mcmultibox',),
    implementation='mowl.training.elkem.ValidationCheckpoint',
    notes=(
        'ELK-Em, Section 5.1 (median rank every 100 epochs); Volume MultiBoxEL (Aharoni'
        '2024, Chapter 7: best validation loss). Replaces the built-in validation of the'
        'generic loop while attached. Unlike validation-rank-selection it saves to the'
        'file and does not restore the best state into the module.'),
)
