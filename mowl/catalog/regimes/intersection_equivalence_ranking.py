"""Regime card: Ranking of intersection equivalences (ELBE)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import PENG2020

REGIME = RegimeCard(
    key='intersection-equivalence-ranking',
    assumption="closed world over stated axioms",
    task="subsumption-prediction",
    target="inductive prediction",
    name='Ranking of intersection equivalences (ELBE)',
    kind=RegimeKind.EVALUATION,
    summary=('For triples E ⊑ C, E ⊑ D, C ⊓ D ⊑ E held out from GO, rank E by the distance of its'
             ' centre to the centre of the intersection of C and D.'),
    components=('test set', 'candidate set', 'scores', 'metrics'),
    applies_to='Ball and box models',
    references=(PENG2020,),
    standalone=False,
    introduced_with=('elbe',),
    used_by=(),
    not_implemented_reason=('Needs the synthetic GO test set of ELBE (2,131 NF2 axioms with their'
                            ' NF1 companions, 1,000 held out), which is not distributed with '
                            'mOWL; the scoring is the NF2 centre distance of '
                            "'centre-distance-scoring'."),
    notes='ELBE, Sections 5.1 and 5.3, Table 5 (H@1/3/10, MR).',
)
