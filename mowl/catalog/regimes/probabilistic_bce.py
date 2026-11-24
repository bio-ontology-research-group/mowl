"""Regime card: Binary cross-entropy on probabilities exp(−score)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import ZHANG2022, DASGUPTA2020

REGIME = RegimeCard(
    key='probabilistic-bce',
    name='Binary cross-entropy on probabilities exp(−score)',
    kind=RegimeKind.TRAINING,
    summary=('Scores are read as negative log-probabilities; positives maximise log p, negatives '
             'maximise log(1 − p), optionally with self-adversarial weights.'),
    components=('loss combination',),
    applies_to='Any model with per-axiom scores',
    references=(ZHANG2022, DASGUPTA2020),
    standalone=False,
    introduced_with=('bcbox',),
    used_by=('gumbel_box',),
    implementation='mowl.training.losses.ProbabilisticBCELoss',
    notes=('Binary-code boxes, Section 4.4, Eq. 10 (BCE, 32 negatives, self-adversarial weights);'
           ' Gumbel boxes, Section 5 (KL / cross-entropy of conditional probabilities against '
           "targets; mOWL's version uses binary targets)."),
)
