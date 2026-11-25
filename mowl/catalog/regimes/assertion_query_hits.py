"""Regime card: Hits@k of concept and role assertion queries over the individuals."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.benchmarks.entries._refs import LACERDA2024FAITHEL

REGIME = RegimeCard(
    key='assertion-query-hits',
    assumption="closed world over stated axioms",
    task="link-prediction",
    target="inductive prediction",
    name='Hits@k of C(?) and r(a, ?) assertion queries (one hit per query)',
    kind=RegimeKind.EVALUATION,
    summary=('Every class of a held-out class assertion is a query C(?) and every pair (a, r) of '
             'a held-out role assertion a query r(a, ?); all individuals are ranked by the '
             "model's cost of the assertion and a query is a hit at k if any of its held-out "
             'answers is among the k best. Hits@k and the reciprocal rank of the best answer, '
             'per query kind; known answers are not filtered unless asked.'),
    components=('test set', 'candidate set', 'filtering', 'metrics'),
    applies_to=('Any model with scores for class_assertion and object_property_assertion '
                '(FaithEL, ELBE and the EL models with individuals, BoxLitE-style KB '
                'embeddings)'),
    references=(LACERDA2024FAITHEL,),
    standalone=False,
    introduced_with=('faithel',),
    used_by=(),
    implementation='mowl.evaluation.assertion_queries.AssertionQueryHitsRegime',
    limitations=(
        'The code counts a query as a hit if ANY of its held-out answers is in the top k, '
        'which is more lenient than the per-answer Hits@k of the knowledge graph protocol '
        "('kg-filtered-link-prediction'); the two are not comparable.",
        'Known answers are not filtered (as in the code); with filtered=True they are.'),
    notes=('utils.py of victorlacerdab/FaithEL_method (get_hits_at_k_concept_assertions, '
           'get_hits_at_k_role_assertions; k = 1, 3, 10, 100, all). The paper itself could '
           'not be read; the regime follows the code.'),
)
