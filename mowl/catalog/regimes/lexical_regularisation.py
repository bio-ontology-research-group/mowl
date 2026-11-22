"""Regime card: Lexical regulariser of individuals."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='lexical-regularisation',
    name='Lexical regulariser of individuals',
    kind=RegimeKind.TRAINING,
    summary=(
        'On a random batch of individuals with external representations the pairwise'
        'cosine similarity is matched by the geometric similarity exp(-lambda * squared'
        'distance / n) of their points.'),
    components=('regularisation',),
    applies_to='Modules with lexical_loss(indices, features, decay) (ELK-Em)',
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=(),
    implementation='mowl.training.elkem.LexicalRegularisation',
    notes=(
        'ELK-Em, Section 4.3, Eq. 6; protein function prediction (Table 4) uses weight'
        '10.0 and decay 1.0.'),
)
