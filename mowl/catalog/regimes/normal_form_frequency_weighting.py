"""Regime card: Weighting of normal forms by frequency."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2024NESY

REGIME = RegimeCard(
    key='normal-form-frequency-weighting',
    name='Weighting of normal forms by frequency',
    kind=RegimeKind.TRAINING,
    summary=('The loss of every normal form is multiplied by its share of the training '
             'mini-batches, ceil(n_t/b) / Σ_u ceil(n_u/b).'),
    components=('loss combination',),
    applies_to='EmbeddingELModel',
    references=(MASHKOVA2024NESY,),
    standalone=False,
    introduced_with=(),
    used_by=(),
    implementation='mowl.training.losses.NormalFormFrequencyWeighting',
    notes=("NeSy 2024 paper, Section 4.4, Eq. 4 ('weigh individual GCI losses based on frequency "
           "of the axiom types sampled during one epoch'); the paper gives no formula, the one "
           'used is that of the reference implementation (geometric_embeddings, '
           'models/elembeddings.py, loss_weight=True). Whether DELE uses it is not stated in the '
           'DELE paper (Eq. 23 is unweighted).'),
)
