"""Regime card: Tail prediction in the views of a two-view knowledge graph."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.benchmarks.entries._refs import HUANG2023

REGIME = RegimeCard(
    key='two-view-kg-completion',
    assumption="closed world over stated axioms",
    task="link-prediction",
    target="inductive prediction",
    name='Tail prediction in the ontology view or the instance view',
    kind=RegimeKind.EVALUATION,
    summary=('Every held-out triple (h, r, t) of one view is a query (h, r, ?t): the ontology '
             'view as C ⊑ ∃r.D over the concepts of the view, the instance view as r(a, b) over '
             'all entities; raw and filtered MRR and Hits@k.'),
    components=('test set', 'candidate set', 'filtering', 'metrics'),
    applies_to=('Models that score gci2 (ontology view; FilteredRankingRegime) or '
                'object_property_assertion (instance view; AssertionLinkPredictionRegime) '
                'axioms'),
    references=(HUANG2023,),
    standalone=False,
    introduced_with=(),
    used_by=('concept2box',),
    implementation='mowl.evaluation.filtered_ranking.FilteredRankingRegime',
    limitations=('The paper does not say whether it filters known triples: raw and filtered '
                 'metrics are both reported.',),
    notes='Concept2Box, Section 5.2 (queries (h, r, ?t), MRR, Hits@1, Hits@10).',
)
