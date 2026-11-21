"""Regime card: Complex query answering on incomplete KGs."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import REN2020, ZHANG2021CONE, HE2024ACONE

REGIME = RegimeCard(
    key='query-answering',
    task="query-answering",
    target="inductive prediction",
    name='Complex query answering on incomplete KGs',
    kind=RegimeKind.EVALUATION,
    summary=('For logical queries of fixed structures (1p, 2i, ip, ...), the non-trivial answers '
             '(those not reachable in the training graph) are ranked among all entities, '
             'filtering the other answers; MRR and Hits@k per query type.'),
    components=('test set', 'candidate set', 'filtering', 'metrics'),
    applies_to='Models with score_query(structure, queries): GroundedEL (translation mode)',
    references=(REN2020, ZHANG2021CONE, HE2024ACONE),
    standalone=False,
    introduced_with=('query2box',),
    used_by=('cone_qe', 'acone', 'grounded'),
    implementation='mowl.evaluation.query_answering.LogicalQueryAnswering',
    notes=("Query2Box, Section 4 and Appendix ('non-trivial answers v ∈ [[q]]test \\ [[q]]val'); "
           'ConE, Section 5; AConE, Section 6 (NELL-QA, WN18RR-QA). In mOWL the complete graph '
           'is given by the caller: held-out assertions (inductive prediction, as in these '
           'papers) or the assertions entailed by the ontology (deductive inference, the '
           'TBox as a constraint on the answers). Queries are sampled by '
           'mowl.evaluation.sample_queries; the released query datasets are not shipped.'),
)
