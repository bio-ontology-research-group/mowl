"""Regime card: Transitive closure of the told hierarchy as positives."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import VENDROV2016, ZHAPA2023

REGIME = RegimeCard(
    key='transitive-closure-positives',
    name='Transitive closure of the told hierarchy as positives',
    kind=RegimeKind.TRAINING,
    summary=('Every atomic subsumption reachable through a chain of asserted atomic subsumptions '
             'is added as a training positive.'),
    components=('positives',),
    applies_to='Models trained on atomic subsumptions (gci0)',
    references=(VENDROV2016, ZHAPA2023),
    standalone=False,
    introduced_with=(),
    used_by=('order_embeddings', 'cate'),
    implementation='mowl.training.positives.TransitiveClosurePositives',
    limitations=(
        'The closure can be quadratic in the number of classes (max_new caps it).',
    ),
    notes=('Order embeddings, Section 4.1 (train and test on the transitive closure of WordNet); '
           "CatE, Section 4.5 ('Tr' saturation of the lattice). Uses only the asserted gci0 "
           "axioms; for the deductive closure of all normal forms see 'closure-positives'."),
)
