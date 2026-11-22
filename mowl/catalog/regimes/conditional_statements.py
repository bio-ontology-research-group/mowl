"""Regime card: Loss of probabilistic conditionals."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind

ZHU2024 = Reference(
    "zhu2024boxsel",
    "Zhu, Y., et al. (2024). Approximating Probabilistic Inference in Statistical EL with "
    "Knowledge Graph Embeddings (BoxSEL). ACL 2024.",
    "https://aclanthology.org/2024.acl-long.20/")

REGIME = RegimeCard(
    key='conditional-statements',
    name='Loss of probabilistic conditionals',
    kind=RegimeKind.TRAINING,
    summary=(
        'For every group of conditionals (D | C)[l, u] with the same structure of'
        'conclusion and premise, the mean of the hinged violations l*Vol(C) - Vol(D and'
        'C) and Vol(D and C) - u*Vol(C) is added once per step with a weight.'),
    components=('loss combination', 'positives'),
    applies_to='Geometries that read the form "conditional" (BoxSEL)',
    references=(ZHU2024,),
    standalone=False,
    introduced_with=('boxsel',),
    used_by=(),
    implementation='mowl.training.conditional.ConditionalStatements',
    notes=(
        'BoxSEL (Zhu et al. 2024). The statements are given to the regime or are the'
        'conditionals of the model; the cooling schedule of the paper is temperature-'
        'annealing.'),
)
