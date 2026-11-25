"""Regime card: Sibling hard negatives."""

from mowl.catalog.card import RegimeCard, RegimeKind, Guarantee, GuaranteeKind
from mowl.catalog.regimes._regime_refs import HE2024HIT

REGIME = RegimeCard(
    key='sibling-hard-negatives',
    assumption="closed world over stated axioms",
    name='Sibling hard negatives',
    kind=RegimeKind.TRAINING,
    summary=('For C ⊑ D, negatives C ⊑ S with S a sibling of C in the told hierarchy that is not '
             'a told superclass of C; random non-superclasses when C has no sibling.'),
    components=('negative sampling',),
    applies_to='Models trained on atomic subsumptions (gci0)',
    references=(HE2024HIT,),
    standalone=False,
    introduced_with=('hit',),
    used_by=(),
    implementation='mowl.training.negatives.SiblingHardNegatives',
    properties=(
        Guarantee(
            kind=GuaranteeKind.SOUNDNESS,
            statement=('No negative is a subsumption of the transitive closure of the asserted '
                       'gci0 axioms.'),
            source=('by construction (SiblingHardNegatives); HiT, Section 2.3 defines negatives '
                    'this way'),
            condition='by construction',
            verified_in_mowl='tests/training/test_regime_negatives.py',
        ),
    ),
    limitations=(
        ('Only the told hierarchy is used: entailments that need other axiom types are not '
         'excluded.'),
    ),
    notes=("HiT, Section 2.3 ('a hard negative is identified when e1 and e2 are also siblings "
           "that share the same parent') and Section 4 (10 negatives per positive, random "
           'negatives supplement missing siblings). HiT uses the hard negatives both for training'
           ' and for its hard test sets.'),
)
