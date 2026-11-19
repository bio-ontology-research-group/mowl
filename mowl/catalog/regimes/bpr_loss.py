"""Regime card: Bayesian personalised ranking loss."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import FALCON2022

REGIME = RegimeCard(
    key='bpr-loss',
    name='Bayesian personalised ranking loss',
    kind=RegimeKind.TRAINING,
    summary='−log σ(s(negative) − s(positive)) for every positive and each of its negatives.',
    components=('loss combination',),
    applies_to='Any model with per-axiom scores',
    references=(FALCON2022,),
    standalone=False,
    introduced_with=('falcon',),
    used_by=(),
    implementation='mowl.training.losses.BPRLoss',
    notes=("FALCON, Section 5.1: 'we apply uniform negative sampling for each ABox axiom and we "
           "use BPR loss (Rendle et al. 2012) for ABox optimization' (4, 8 or 16 negatives). "
           "mowl's FALCONModel trains with its own loop."),
)
