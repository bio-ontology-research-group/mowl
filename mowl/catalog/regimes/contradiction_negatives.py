"""Regime card: Contradiction negatives from asserted disjointness."""

from mowl.catalog.card import RegimeCard, RegimeKind, Guarantee, GuaranteeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2026, KULMANOV2019

REGIME = RegimeCard(
    key='contradiction-negatives',
    name='Contradiction negatives from asserted disjointness',
    kind=RegimeKind.TRAINING,
    summary=("A fraction p⊥ of the negatives of C ⊑ D (and C(a)) uses a class D' that is disjoint"
             ' with a told superclass of C through an asserted disjointness axiom, so the '
             'negative is provably false; the rest are uniform corruptions.'),
    components=('negative sampling',),
    applies_to='EmbeddingELModel on ontologies with disjointness axioms',
    references=(MASHKOVA2026, KULMANOV2019),
    standalone=False,
    introduced_with=(),
    used_by=(),
    implementation='mowl.training.negatives.ContradictionNegatives',
    properties=(
        Guarantee(
            kind=GuaranteeKind.SOUNDNESS,
            statement=('Every negative drawn from the contradictory pool contradicts the ontology'
                       ' unless its subclass (or individual) is unsatisfiable.'),
            source='by construction (ContradictionNegatives)',
            condition='satisfiable subclass',
            verified_in_mowl='tests/training/test_regime_negatives.py',
        ),
    ),
    limitations=(
        ('Needs disjointness axioms; natural candidates such as GALEN have none (Mashkova and '
         'Hoehndorf 2026, Section 7).'),
        ('Training only on contradiction negatives lowers the faithfulness violation rate but '
         'does not make the models faithful (FVR above 0.43 in most settings) and MRR often moves'
         ' the other way (Mashkova and Hoehndorf 2026, Section 6.3).'),
    ),
    notes=("Mashkova and Hoehndorf (2026), Section 6.3 'type-aware negative sampling': a fraction"
           " p⊥ of negatives from S⊥ 'via cheap reasoning shortcuts (asserted disjointness, "
           "domain/range violations)'. mOWL implements the disjointness shortcut for gci0 and "
           'class assertions only; domain/range violations and the other normal forms of the '
           'paper (GCI2, GCI1-bottom, GCI3) are not implemented. Kulmanov et al. (2019, Section '
           "5) proposed to 'infer further negatives by exploring disjointness' as future work."),
)
