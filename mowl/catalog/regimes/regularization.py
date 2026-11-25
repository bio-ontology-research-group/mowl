"""Regime card: regularisation as a component (penalties and projections)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import KULMANOV2019

REGIME = RegimeCard(
    key='regularization',
    name='Penalties and projections on embedding tables',
    kind=RegimeKind.TRAINING,
    summary=(
        'Constraints on where embeddings may lie, applied to any model: penalties are '
        'added to the loss of every step (norm near a target, norm below a bound, power '
        'of the coordinates), projections are applied after every step (onto a sphere, '
        'into a ball, into an interval). They are used in addition to or instead of the '
        "regularisation term of the model's own module."),
    components=('regularisation',),
    applies_to='EmbeddingELModel',
    references=(KULMANOV2019,),
    standalone=False,
    introduced_with=(),
    used_by=('elem',),
    implementation='mowl.training.regularization.Regularization',
    limitations=(
        'Terms a module adds inside the loss of every axiom are not switched off.',
        'Constraints a representation needs to be well defined (points inside the '
        'Poincaré ball, boxes inside the unit cube) stay in the model.'),
    notes=(
        "NormPenalty with its defaults is the unit-norm penalty of ELEmbeddings on the "
        "class centres; SphereProjection is the same constraint enforced exactly, as the "
        "entity normalisation of TransE does."),
)
