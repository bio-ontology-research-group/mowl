"""Regime card: Checkpoint selection by a validation rank metric."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import JACKERMEIER2024, SINGH2026

REGIME = RegimeCard(
    key='validation-rank-selection',
    name='Checkpoint selection by a validation rank metric',
    kind=RegimeKind.TRAINING,
    summary=('Every k epochs the validation axioms are ranked and the parameters with the best '
             'median rank are kept.'),
    components=('model selection',),
    applies_to='EmbeddingELModel',
    references=(JACKERMEIER2024, SINGH2026),
    standalone=False,
    introduced_with=('box2el',),
    used_by=('elkem',),
    implementation='mowl.training.selection.ValidationRankSelection',
    notes=("Box2EL, Section 4.2 ('evaluate the models on a fraction of the validation set every "
           "100 epochs and choose the embeddings that achieve the best performance'; reference "
           'train.py: median rank of the first 1,000 validation nf1 axioms); ELK-Em, Section 5.1 '
           "('Checkpoints are selected by median rank on the validation split'). Call finalize() "
           'after train().'),
)
