"""Regime card: Pearson correlation contrastive loss (ELK-Em)."""

from mowl.catalog.card import RegimeCard, RegimeKind, Guarantee, GuaranteeKind
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='pearson-correlation-loss',
    name='Pearson correlation contrastive loss (ELK-Em)',
    kind=RegimeKind.TRAINING,
    summary=('One minus the Pearson correlation between the axiom scores exp(−loss) of positives '
             'and corruptions and their binary labels; no margin hyperparameter.'),
    components=('loss combination',),
    applies_to='Any model with per-axiom loss scores',
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=(),
    implementation='mowl.training.losses.PearsonCorrelationLoss',
    properties=(
        Guarantee(
            kind=GuaranteeKind.EXPRESSIVITY,
            statement=('The correlation term is invariant to positive affine rescaling of the '
                       'transformed scores, so it needs no margin.'),
            source=("definition of the Pearson correlation; ELK-Em, Section 4.2 ('without an "
                    "additional margin hyperparameter')"),
            condition='always',
        ),
    ),
    notes=('ELK-Em, Section 4.2, Eq. 5; used only when training on the TBox alone (Section 5.1). '
           "Built into mowl.models.elkem.model.ELKEm. ELK-Em's scores use margin 0; give the "
           'module margin 0 to match.'),
)
