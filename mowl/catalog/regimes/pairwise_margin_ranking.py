"""Regime card: Pairwise margin ranking (triplet) loss."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import HE2024HIT, YANG2025ONT

REGIME = RegimeCard(
    key='pairwise-margin-ranking',
    name='Pairwise margin ranking (triplet) loss',
    kind=RegimeKind.TRAINING,
    summary='max(0, γ + s(positive) − s(negative)) for every positive and each of its negatives.',
    components=('loss combination',),
    applies_to='Any model with per-axiom scores',
    references=(HE2024HIT, YANG2025ONT),
    standalone=False,
    introduced_with=('hit',),
    used_by=('ont',),
    implementation='mowl.training.losses.PairwiseMarginRankingLoss',
    notes=('HiT, Eq. 3 (hyperbolic clustering loss, margin α); OnT, Section 3 (hierarchical '
           'contrastive loss). Both add a centripetal loss on hyperbolic norms, which is '
           'representation-specific and not part of this regime. Negatives are re-scored with the'
           ' positive-form score.'),
)
