"""Regime card: Interval estimates of probabilistic conditionals (Statistical EL)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.benchmarks.entries._refs import ZHU2024

REGIME = RegimeCard(
    key='sel-pmp-intervals',
    assumption="open world",
    task="subsumption",
    target="deductive inference",
    name='Interval estimates of probabilistic conditionals by probabilistic modus ponens',
    kind=RegimeKind.EVALUATION,
    summary=('A model is trained on the learning set of a Statistical EL ontology (the '
             'conditionals (D|C)[l, u] minus a held-out 30% of the most general ones). For '
             'every held-out query (E|C) whose premises (D|C) and (E|C ⊓ D) are in the '
             'learning set, the probabilistic modus ponens gives the entailed interval; the '
             "model's volume ratios (min and max over N embeddings) give the estimate. "
             'Soundness accuracy, soundness error and approximation gap.'),
    components=('split', 'queries', 'entailed intervals', 'estimate', 'metrics'),
    applies_to='Models of probabilistic conditionals with volume ratios (BoxSEL).',
    references=(ZHU2024,),
    standalone=False,
    introduced_with=('boxsel',),
    used_by=(),
    implementation='mowl.evaluation.sel_probability.SELProbabilityRegime',
    limitations=(
        'The ontologies of the paper are not released and cannot be rebuilt exactly, so the '
        'published numbers are not reproducible; the regime runs on ontologies built with '
        'mowl.datasets.builtin.sel_conditionals.',
        'The paper\'s estimate uses N = 60 embeddings; one run gives one embedding, the '
        'ensemble needs SELProbabilityRegime.evaluate_ensemble.',
        'The appendix pseudocode of the paper uses premises (A|Q1) and (Q2|A) and bounds '
        'q1 q2 and min(1, q1 q2 + 1 - q2), which differ from the main text (Proposition 1: '
        '(D|C) and (E|C ⊓ D); l1 l2 and min(1, u1 u2 + 1 - l1)); the regime follows the '
        'main text.'),
    notes=('arXiv:2407.11821 (e-print dated June 2026, UAI template), Section "Experiments" '
           '(Evaluation Protocol, Table "Approximation quality") and the appendices on '
           'dataset construction and the PMP evaluation.'),
)
