"""Regime card: Filtered link prediction on role or membership assertions."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import LOURENCO2026, CHARPENAY2024, GARG2019

REGIME = RegimeCard(
    key='kg-filtered-link-prediction',
    task="link-prediction",
    target="inductive prediction",
    name='Filtered link prediction on role or membership assertions',
    kind=RegimeKind.EVALUATION,
    summary=('Each test assertion r(a, b) (or C(a)) is ranked against all corruptions of one '
             'entity; corruptions that are true in the training (and validation) data are removed'
             ' (filtered setting); MRR and Hits@k.'),
    components=('test set', 'candidate set', 'filtering', 'metrics'),
    applies_to='Models with individuals and role assertions',
    references=(LOURENCO2026, CHARPENAY2024, GARG2019),
    standalone=False,
    introduced_with=(),
    used_by=('boxlite', 'octagon', 'e2r'),
    implementation="mowl.evaluation.assertion_ranking.RoleAssertionRankingRegime",
    notes=('BoxLitE, Section 6 (inferred role assertions, filtered MRR/H@k); octagon embeddings, '
           'Section 6 (FB15k-237, WN18RR); E2R, Section 5 (filtered, membership queries averaged '
           'per query on LUBM1U).'),
)
