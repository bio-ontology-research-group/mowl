"""Regime card: Ranking by the training loss."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2024DELE, KULMANOV2019

REGIME = RegimeCard(
    key='loss-scoring',
    name='Ranking by the training loss',
    kind=RegimeKind.EVALUATION,
    summary=("Candidate axioms are scored with the model's own loss for the normal form (lower is"
             ' more plausible).'),
    components=('scores',),
    applies_to='Any EmbeddingELModel',
    references=(MASHKOVA2024DELE, KULMANOV2019),
    standalone=False,
    introduced_with=('elem',),
    used_by=('falcon', 'cate'),
    implementation='mowl.evaluation.el_scoring.loss_scores',
    limitations=(
        ('With a positive margin γ the hinge assigns the same score 0 to every candidate inside '
         "the margin, producing ties; ELEmbeddings 'ranks thousands of axioms at the same rank' "
         '(NeSy 2024, Section 5). Results then depend on tie handling.'),
    ),
    notes=('ELEmbeddings, Eq. 21 (the NF3 loss without the normalisation terms); DELE, Section '
           "6.2 ('we use the corresponding loss expressions to score axioms'); mOWL's evaluators."
           ' A component of evaluation regimes (resolve_scorer).'),
)
