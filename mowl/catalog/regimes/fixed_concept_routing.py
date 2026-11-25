"""Regime card: Routing of axioms of classes fixed to the empty box."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='fixed-concept-routing',
    name='Routing of axioms of classes fixed to the empty box',
    kind=RegimeKind.TRAINING,
    summary=(
        'Axioms whose class the class model fixes to the empty box are not trained as'
        'they are: a subsumption with a fixed-empty subclass holds trivially and is'
        'dropped, an intersection or existential axiom with a fixed-empty superclass is'
        'trained as its bottom normal form.'),
    components=('positives',),
    applies_to='EL models whose class model fixes classes (a bot_mask): ELK-Em',
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=(),
    implementation='mowl.training.elkem.FixedConceptRouting',
    notes=(
        'ELK-Em, Table 2 and the training code. The classes are read from the class model'
        'of the module, so the regime also applies to a model built with'
        'ComposedEL(method="elkem").'),
)
