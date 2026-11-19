"""Regime card: Sum of per-normal-form losses."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import KULMANOV2019, MONDAL2021, XIONG2022, MASHKOVA2024DELE

REGIME = RegimeCard(
    key='loss-sum',
    name='Sum of per-normal-form losses',
    kind=RegimeKind.TRAINING,
    summary=('Mean positive loss plus (weighted) mean negative loss for every normal form, summed'
             ' over normal forms.'),
    components=('loss combination',),
    applies_to='EmbeddingELModel',
    references=(KULMANOV2019, MONDAL2021, XIONG2022, MASHKOVA2024DELE),
    standalone=False,
    introduced_with=('elem',),
    used_by=('emelpp', 'boxel', 'box2el', 'transbox'),
    implementation='mowl.training.losses.SumLoss',
    notes=('The neutral default of mOWL (EmbeddingELModel.compute_loss). ELEmbeddings, Section '
           '3.4 and Algorithm 1; EmEL++, Section 3.3; DELE, Eq. 23; BoxEL, Section 5.2 weights '
           'the non-subsumption loss by φ = 0.05 (neg_weight). Box2EL squares every per-axiom '
           'loss inside the module (reference code: loss.square().mean(), (δ − μ)² for '
           "negatives), which mOWL's BoxSquaredELModule also does, so the sum gives its "
           'objective.'),
)
