"""Regime card: Transitive reduction of the told hierarchy as positives."""

from mowl.catalog.card import RegimeCard, RegimeKind, Guarantee, GuaranteeKind
from mowl.catalog.regimes._regime_refs import YANG2025_REGD, HE2024HIT, DASGUPTA2020

REGIME = RegimeCard(
    key='transitive-reduction-positives',
    name='Transitive reduction of the told hierarchy as positives',
    kind=RegimeKind.TRAINING,
    summary=('Atomic subsumptions implied by a chain of other asserted atomic subsumptions are '
             'removed from the training data, so the model is trained on the basic edges only.'),
    components=('positives',),
    applies_to='Models trained on atomic subsumptions (gci0)',
    references=(YANG2025_REGD, HE2024HIT, DASGUPTA2020),
    standalone=False,
    introduced_with=(),
    used_by=('regd', 'hit', 'gumbel_box'),
    implementation='mowl.training.positives.TransitiveReductionPositives',
    properties=(
        Guarantee(
            kind=GuaranteeKind.EXPRESSIVITY,
            statement=('The transitive closure of the kept axioms equals that of the asserted '
                       'gci0 axioms.'),
            source='by construction (TransitiveReductionPositives)',
            condition='by construction',
            verified_in_mowl='tests/training/test_regime_positives.py',
        ),
    ),
    notes=("RegD, Section 5.1 ('trained exclusively on basic edges'); HiT, Section 4 (Multi-hop "
           'Inference: train on one-hop subsumptions); Gumbel boxes, Section 5.3 (train on the '
           'transitive reduction of WordNet). Only gci0 is changed; on cycles all cycle edges are'
           ' kept.'),
)
