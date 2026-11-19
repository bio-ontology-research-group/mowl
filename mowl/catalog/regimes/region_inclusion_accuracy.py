"""Regime card: Geometric inclusion accuracy."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MONDAL2021, XIONG2022

REGIME = RegimeCard(
    key='region-inclusion-accuracy',
    name='Geometric inclusion accuracy',
    kind=RegimeKind.EVALUATION,
    summary=('Fraction of the atomic subsumptions of a split (train, validation, test) whose '
             'subclass region lies inside the superclass region.'),
    components=('test set', 'metrics'),
    applies_to=('Ball models (ELEmbeddings, EmEL++) and single-box models (ELBE, Box2EL, '
                'TransBox)'),
    references=(MONDAL2021, XIONG2022),
    standalone=False,
    introduced_with=('emelpp',),
    used_by=('boxel',),
    implementation='mowl.evaluation.inclusion_accuracy.InclusionAccuracyRegime',
    notes=('EmEL++, Section 4.5, Table 4 (training, testing and inferences sets); BoxEL, Section '
           "5.2, Table 3 ('accuracy based measure'). Unlike 'exact-model-check' it reports per "
           'split, including held-out axioms.'),
)
