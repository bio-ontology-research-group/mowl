"""Regime card: Entailment classification under injected inconsistency (FALCON)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import FALCON2022

REGIME = RegimeCard(
    key='paraconsistent-entailment-classification',
    task="axiom-ranking",
    target="logical status",
    name='Entailment classification under injected inconsistency (FALCON)',
    kind=RegimeKind.EVALUATION,
    summary=('Reasoner-entailed axioms versus axioms that are neither asserted nor entailed are '
             "classified by the model's membership degrees, after adding N inconsistent axioms; "
             'MAE, AUC, AUPR, Fmax.'),
    components=('test set', 'metrics'),
    applies_to='ALC models with graded membership (FALCON)',
    references=(FALCON2022,),
    standalone=False,
    introduced_with=('falcon',),
    used_by=(),
    not_implemented_reason=('Needs HermiT entailments and the inconsistency-injection protocol of'
                            ' the FALCON paper, which are not in mOWL; the closest implemented '
                            "regime is 'faithfulness-hierarchy' (entailed / undetermined / "
                            'contradictory).'),
    notes='FALCON, Section 5.1-5.2, Table 1 (Pizza with 0-1000 inconsistent axioms).',
)
