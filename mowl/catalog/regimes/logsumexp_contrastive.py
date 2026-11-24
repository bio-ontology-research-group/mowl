"""Regime card: Log-sum-exp contrastive loss (RegD / shadow cones)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import YANG2025_REGD

REGIME = RegimeCard(
    key='logsumexp-contrastive',
    name='Log-sum-exp contrastive loss (RegD / shadow cones)',
    kind=RegimeKind.TRAINING,
    summary='Σ max(E(positive), γ1) + log Σ exp(max(γ2 − d(negative), 0)).',
    components=('loss combination',),
    applies_to='Any model with per-axiom scores',
    references=(YANG2025_REGD,),
    standalone=False,
    introduced_with=('regd',),
    used_by=(),
    implementation='mowl.training.losses.LogSumExpContrastiveLoss',
    notes=('RegD, Section 3.3, Eq. 9, taken from shadow cones (Yu et al. 2024, not read). RegD '
           'scores negatives with the boundary dissimilarity only; here both sides use the '
           "module's score. The positive sum is averaged over the batch (not specified in RegD)."),
)
