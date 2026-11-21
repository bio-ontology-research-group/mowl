"""Regime card: Raw and filtered link prediction on role assertions through model scores."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.benchmarks.entries._refs import WANG2024EIKE, LV2018TRANSC, HUANG2023

REGIME = RegimeCard(
    key='assertion-link-prediction',
    assumption="closed world over stated axioms",
    task="link-prediction",
    target="inductive prediction",
    name='Raw and filtered link prediction on role assertions between individuals',
    kind=RegimeKind.EVALUATION,
    summary=('Every test assertion r(a, b) is ranked as the object of r(a, ?) (and, unless only '
             'tails are asked for, as the subject of r(?, b)) among all individuals; raw ranks '
             'and ranks after removing the candidates that are known training, validation or '
             'test assertions; MRR, MR and Hits@k.'),
    components=('test set', 'candidate set', 'filtering', 'metrics'),
    applies_to=('Any model with individuals that scores object_property_assertion axioms '
                'through the model scoring interface (mowl.evaluation.model_scoring)'),
    references=(WANG2024EIKE, LV2018TRANSC, HUANG2023),
    standalone=False,
    introduced_with=(),
    used_by=('eike', 'concept2box'),
    implementation='mowl.evaluation.filtered_ranking.AssertionLinkPredictionRegime',
    limitations=('Tied candidates share the average rank; the reference code of EIKE orders '
                 'them by position. Unlike RoleAssertionRankingRegime, which needs a DL-Lite '
                 'knowledge base and role_scores, this regime scores axiom rows, so any EL '
                 'model can be evaluated.',),
    notes=('EIKE, Section 5.4 and the reference link_prediction.py (both directions, raw and '
           'filtered MRR, filtered Hits@1/3/10); Concept2Box, Section 5.2 (tails only, '
           'filtering not stated). Compare kg-filtered-link-prediction (assertions between '
           'DL-Lite individuals) and the kg-* benchmarks (triples encoded between classes).'),
)
