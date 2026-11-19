"""Regime card: Protein-centric F1 at a validation threshold (CAFA)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='protein-centric-threshold-f1',
    name='Protein-centric F1 at a validation threshold (CAFA)',
    kind=RegimeKind.EVALUATION,
    summary=('Function predictions for held-out proteins are propagated through GO, a score '
             'threshold is chosen on validation proteins, and the protein-centric F1 at that '
             'threshold is reported on test proteins.'),
    components=('test set', 'candidate set', 'metrics'),
    applies_to='Models with individuals and class-assertion scores',
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=(),
    not_implemented_reason=('Needs the CAFA5-derived benchmark and GO propagation of predictions,'
                            ' which are not in mOWL.'),
    notes=('ELK-Em, Section 5.2, Table 4; differs from CAFA Fmax, which maximises over thresholds'
           ' on the test set.'),
)
