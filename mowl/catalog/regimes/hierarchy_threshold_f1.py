"""Regime card: Threshold classification of subsumptions with 1:10 negatives."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import HE2024HIT, DASGUPTA2020

REGIME = RegimeCard(
    key='hierarchy-threshold-f1',
    task="subsumption-prediction",
    name='Threshold classification of subsumptions with 1:10 negatives',
    kind=RegimeKind.EVALUATION,
    summary=('Every test subsumption is paired with 10 negatives (random, or hard siblings); a '
             'score threshold tuned on validation decides, and precision/recall/F1 are reported.'),
    components=('test set', 'metrics'),
    applies_to='Any model with subsumption scores',
    references=(HE2024HIT, DASGUPTA2020),
    standalone=False,
    introduced_with=('hit',),
    used_by=('gumbel_box',),
    not_implemented_reason=("'regd-sampled-f1' implements the random-negative variant on the "
                            "child side; HiT's parent-side and sibling (hard) test negatives and "
                            'its transfer setting are not implemented.'),
    notes=('HiT, Section 4 (Multi-hop Inference and Mixed-hop Prediction, Table 2; transfer, '
           'Table 3); Gumbel boxes, Section 5.3 (F1 with 1:10 negatives on WordNet). Compare '
           "'regd-sampled-f1' (RegD)."),
)
