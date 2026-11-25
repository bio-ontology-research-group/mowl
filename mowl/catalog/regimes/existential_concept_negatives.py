"""Regime card: Negative concept regularisation on existential concepts (BoxLitE)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import LOURENCO2026

REGIME = RegimeCard(
    key='existential-concept-negatives',
    assumption="closed world over stated axioms",
    name='Negative concept regularisation on existential concepts (BoxLitE)',
    kind=RegimeKind.TRAINING,
    summary=('Instead of sampled corruptions, convex regularisation terms push individuals out of'
             ' the boxes of existential concepts they are not asserted to belong to, which keeps '
             'the training problem a second-order cone program.'),
    components=('negative sampling',),
    applies_to='Convex (SOCP) training of box embeddings for DL-Lite_H',
    references=(LOURENCO2026,),
    standalone=False,
    introduced_with=('boxlite',),
    used_by=(),
    not_implemented_reason=('Part of a convex program solved with an external conic solver '
                            "(CVXPY/MOSEK); it does not fit mOWL's gradient-based training loop."),
    notes=("BoxLitE, Section 5.2 ('Negative Concept Regularization') and the comparison with "
           'sampled negatives in Section 7.'),
)
