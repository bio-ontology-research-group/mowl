"""Regime card: Link classification AUC with sampled test negatives."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import ZHANG2022

REGIME = RegimeCard(
    key='sampled-negative-auc',
    assumption="closed world over stated axioms",
    task="link-prediction",
    target="inductive prediction",
    name='Link classification AUC with sampled test negatives',
    kind=RegimeKind.EVALUATION,
    summary=('Test edges are scored against sampled negatives (uniform pairs, or corruptions of '
             'the test edges) and the ROC AUC is reported.'),
    components=('test set', 'metrics'),
    applies_to='Graph/hierarchy embeddings',
    references=(ZHANG2022,),
    standalone=False,
    introduced_with=('bcbox',),
    used_by=(),
    not_implemented_reason=('Not needed by any mOWL benchmark yet; the two test-negative '
                            'strategies change the AUC substantially (Table 3 of the paper '
                            'reports both).'),
    notes="Binary-code boxes, Section 5, Table 3 ('unif.' and 'corr.' test negatives).",
)
