"""Regime card: Inductive link prediction with 50 sampled negatives per side."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import TERU2020, PAVLOVIC2025

REGIME = RegimeCard(
    key='inductive-sampled-ranking',
    assumption="closed world over stated axioms",
    task="link-prediction",
    target="inductive prediction",
    name='Inductive link prediction with 50 sampled negatives per side',
    kind=RegimeKind.EVALUATION,
    summary=('The model is trained on one graph and tested on another graph with different '
             'entities and the same relations. Every test triple h ⊑ ∃r.t is ranked against 50 '
             'corrupted tails and, separately, 50 corrupted heads, drawn uniformly from the '
             'entities of the test graph that give no known true triple (facts of the test '
             'graph and test triples); the rank is the average rank among the 51 (realistic '
             'rank). Hits@k, MRR and MR over both directions; the literature reports Hits@10.'),
    components=('test set', 'candidate set', 'negatives', 'filtering', 'metrics'),
    applies_to=('Any model with inductive_costs(graph, rows): it must score triples over a '
                'graph whose entities it has not seen in training, from the facts of that '
                'graph (RESHUFFLE). A model with one vector per training entity is refused '
                'with UnsupportedInductiveModel.'),
    references=(TERU2020, PAVLOVIC2025),
    standalone=False,
    introduced_with=(),
    used_by=('reshuffle',),
    implementation='mowl.evaluation.inductive_ranking.InductiveSampledRankingRegime',
    limitations=("Candidates are 50 sampled entities, not all entities: the metrics are "
                 "optimistic estimates of the ones over all entities (PyKEEN's "
                 "SampledRankBasedEvaluator cites arXiv:2106.06935 for this) and are not "
                 "comparable with the filtered full ranking of "
                 "kg-filtered-link-prediction.",
                 "The negatives are one fixed random draw per seed; the reference code draws "
                 "them from Python's global random state."),
    notes=('RESHUFFLE, Section 6 and Appendix "Evaluation Protocol"; the protocol of GraIL '
           '(Teru et al. 2020). Code: Utils.analyze_checkpoints of '
           'https://github.com/AleksVap/RESHUFFLE (PyKEEN SampledRankBasedEvaluator, 50 '
           'negatives, additional filter triples: the facts of the test graph). The model '
           'receives the test graph through InductiveKnowledgeGraphDataset.inference_graph.'),
)
