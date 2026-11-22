"""Regime card: Positive against its own group of negatives (softmax)."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import YANG2025_REGD

YU2024 = Reference(
    "yu2024shadowcones",
    "Yu, T., Liu, T. J. B., Tseng, A., Sa, C. D. (2024). Shadow Cones: A Generalized "
    "Framework for Partial Order Embeddings. ICLR 2024.",
    "https://openreview.net/forum?id=O2jsRQ2UB9")

REGIME = RegimeCard(
    key='grouped-logsumexp-contrastive',
    name='Positive against its own group of negatives (softmax)',
    kind=RegimeKind.TRAINING,
    summary=(
        'For every positive the clamped positive score plus the log-sum-exp of the'
        'negative losses of its own K corruptions, averaged over the positives: the'
        'softmax cross-entropy of a positive against K negatives.'),
    components=('loss combination',),
    applies_to='Any EL model whose negatives come in K blocks of one corruption per positive',
    references=(YU2024, YANG2025_REGD,),
    standalone=False,
    introduced_with=('regd', 'shadow_cones'),
    used_by=(),
    implementation='mowl.training.losses.GroupedLogSumExpContrastiveLoss',
    notes=(
        'Eq. 1 of Shadow Cones (clamp gamma_2) and Eq. 10 of RegD (clamp gamma_1, gci0'
        'and gci2 only). logsumexp-contrastive pools all negatives of the step instead.'),
)
