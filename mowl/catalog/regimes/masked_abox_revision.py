"""Regime card: Masked ABox revision (success rate of the revised grounding)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.benchmarks.entries._refs import WU2022

REGIME = RegimeCard(
    key='masked-abox-revision',
    assumption="closed world over entailed axioms",
    task="subsumption",
    target="deductive inference",
    name='Masked ABox revision: success rate of the revised grounding',
    kind=RegimeKind.EVALUATION,
    summary=('The ABox of an ontology completed by a reasoner is the ideal grounding (1 for '
             'entailed assertions, 0.5 unknown elsewhere); a fraction of it is replaced by '
             'random values in the unknown region, imitating a neural network. The model '
             'starts from this masked grounding, is trained on the TBox alone, and the '
             "revised grounding is made crisp (above 0.5 true, otherwise false, as in the "
             "authors' evaluator); the success rate is the fraction of TBox axioms that the "
             'crisp interpretation over the individuals satisfies.'),
    components=('ideal grounding', 'masking', 'starting point', 'crisp transformation',
                'metric'),
    applies_to=('Models whose learned object is a grounding over a finite domain with a '
                'set_grounding method (DF-ALC).'),
    references=(WU2022,),
    standalone=False,
    introduced_with=('dfalc',),
    used_by=(),
    implementation='mowl.evaluation.abox_revision.MaskedABoxRevisionRegime',
    limitations=(
        'The paper compares with the masked grounding (M), Logical Tensor Networks and '
        'DF-ALC with the rule-based loss; only the last two DF-ALC variants are mOWL '
        'methods. The success rate counts the axioms mOWL trains on (subclass, equivalence, '
        'disjointness of ALC expressions), not the paper\'s axiom counts.',
        'The paper\'s conjunctive query answering and PASCAL-Part experiments are not part '
        'of the regime.'),
    notes=('arXiv:2211.12006, Section "Performance Evaluation" (settings, Table 3); the '
           'released code (training/Evaluation.py: MaskABox) fills the ground truth with 0.5, '
           'masks int(rate * |C| * |I|) concept cells and int(rate * |R| * |I|) role cells '
           'with values uniform in (1 - alpha, alpha), and seeds NumPy with 42.'),
)
