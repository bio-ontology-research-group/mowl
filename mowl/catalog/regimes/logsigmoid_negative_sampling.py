"""Regime card: Log-sigmoid negative-sampling loss (optionally self-adversarial)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import (
    REN2020, ZHANG2021CONE, HE2024ACONE, CHARPENAY2024, ZHANG2022)

REGIME = RegimeCard(
    key='logsigmoid-negative-sampling',
    name='Log-sigmoid negative-sampling loss (optionally self-adversarial)',
    kind=RegimeKind.TRAINING,
    summary=('−log σ(γ − d(positive)) − Σ α_k log σ(d(negative_k) − γ), with uniform weights α or'
             ' self-adversarial weights softmax(−T·d) over the negatives of each positive.'),
    components=('loss combination',),
    applies_to='Any model with per-axiom distance scores',
    references=(REN2020, ZHANG2021CONE, HE2024ACONE, CHARPENAY2024, ZHANG2022),
    standalone=False,
    introduced_with=(),
    used_by=('query2box', 'cone_qe', 'acone', 'octagon', 'bcbox'),
    implementation='mowl.training.losses.LogSigmoidLoss',
    notes=("Query2Box, Eq. 4 (uniform weights 1/k); ConE, Section 4 'Training Objective'; AConE, "
           'Eq. 4; octagon embeddings, Section 5 (self-adversarial negative sampling of Sun et '
           'al. 2019); binary-code boxes use self-adversarial weights with a BCE objective (see '
           "'probabilistic-bce'). The self-adversarial weights are detached from the gradient as "
           'in Sun et al. (2019, not read).'),
)
