"""Regime card: Cyclical learning rate."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import ZHAPA2023

REGIME = RegimeCard(
    key='cyclic-lr',
    name='Cyclical learning rate',
    kind=RegimeKind.TRAINING,
    summary=('The learning rate oscillates between a lower and an upper bound (triangular '
             'cyclical learning rate).'),
    components=('optimiser schedule',),
    applies_to='EmbeddingELModel',
    references=(ZHAPA2023,),
    standalone=False,
    introduced_with=('cate',),
    used_by=(),
    implementation='mowl.training.schedules.CyclicLearningRate',
    notes=("CatE, Section 4.1 ('Adam optimizer with a Cyclic Learning Rate scheduler', Smith "
           '2017, not read). The paper gives neither bounds nor cycle length; the mOWL regime '
           'steps once per epoch (= one optimisation step).'),
)
