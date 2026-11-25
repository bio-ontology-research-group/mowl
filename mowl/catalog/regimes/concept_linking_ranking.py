"""Regime card: Concept linking, ranking the concepts of an entity."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.benchmarks.entries._refs import HUANG2023, HAO2019JOIE

REGIME = RegimeCard(
    key='concept-linking-ranking',
    assumption="closed world over stated axioms",
    task="type-prediction",
    target="inductive prediction",
    name='Concept linking: rank the concepts of an entity',
    kind=RegimeKind.EVALUATION,
    summary=('For every held-out class assertion C(a), all concept candidates are ranked for '
             'the entity a with the model\'s cost of C(a); raw and filtered MRR and Hits@k '
             '(filtered: the other concepts known for a are removed).'),
    components=('test set', 'candidate set', 'filtering', 'metrics'),
    applies_to=('Any model with individuals that scores class_assertion axioms '
                '(mowl.evaluation.filtered_ranking.FilteredRankingRegime with '
                'gci_name="class_assertion", target="tail")'),
    references=(HUANG2023, HAO2019JOIE),
    standalone=False,
    introduced_with=(),
    used_by=('concept2box',),
    implementation='mowl.evaluation.filtered_ranking.FilteredRankingRegime',
    limitations=('Concept2Box does not say whether it filters; both are reported.',),
    notes=('Concept2Box, Section 5.3 (MRR, Hits@1 called accuracy, Hits@3); JOIE, Section 5 '
           '(instance typing).'),
)
