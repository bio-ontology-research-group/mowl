"""Regime card: Ranking by centre distance."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import JACKERMEIER2024, YANG2025, MONDAL2021, AHARONI2024

REGIME = RegimeCard(
    key='centre-distance-scoring',
    name='Ranking by centre distance',
    kind=RegimeKind.EVALUATION,
    summary=('Candidate axioms are scored by the distance between the centres of the regions of '
             'both sides (with model-specific formulas for existential restrictions), not by the '
             'training loss.'),
    components=('scores',),
    applies_to='Ball and box models with a registered scorer',
    references=(JACKERMEIER2024, YANG2025, MONDAL2021, AHARONI2024),
    standalone=False,
    introduced_with=('box2el',),
    used_by=('transbox', 'emelpp', 'emelvar', 'mcmultibox'),
    implementation='mowl.evaluation.el_scoring.centre_distance_scores',
    limitations=(
        ('Centre distance is symmetric for C ⊑ D and ignores region size, so an embedding whose '
         'regions are nested can rank badly and vice versa; numbers are not comparable with '
         'loss-scored ones (ELK-Em, Section 5.1, compares its exp(−loss) ranks with '
         'centre-distance baselines).'),
    ),
    notes=('Box2EL, Appendix F; TransBox, Appendix F (with masks for empty components); EmEL++, '
           "Section 4.4 ('distance from D'); Aharoni (2024), Section 7.1.2."),
)
