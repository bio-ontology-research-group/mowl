"""Regime card: Ranking by the margin-free loss (ELK-Em)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='margin-free-loss-scoring',
    name='Ranking by the margin-free loss (ELK-Em)',
    kind=RegimeKind.EVALUATION,
    summary=('Candidate axioms are scored with the per-axiom loss with the margin set to 0; '
             'ELK-Em reports exp(−loss), which ranks identically.'),
    components=('scores',),
    applies_to="EmbeddingELModel whose module has a 'margin' or 'gamma' attribute",
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=(),
    implementation='mowl.evaluation.scoring_rules.margin_free_loss_scores',
    notes=("ELK-Em, Section 4.2, Eq. 4 and Section 5.1. Differs from 'loss-scoring' whenever the "
           'trained margin is non-zero.'),
)
