"""Regime card: Triple classification with per-relation thresholds on released negatives."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.benchmarks.entries._refs import WANG2024EIKE, LV2018TRANSC

REGIME = RegimeCard(
    key='triple-classification-thresholds',
    assumption="closed world over stated axioms",
    task="axiom-ranking",
    target="inductive prediction",
    name='Triple classification with one threshold per relation',
    kind=RegimeKind.EVALUATION,
    summary=('Held-out axioms of one form (InstanceOf C(a), SubClassOf C ⊑ D, or relational '
             'r(a, b)) and as many negatives (corruptions released with the dataset, or sampled) '
             'are scored; a cost threshold per relation is chosen to maximise accuracy on the '
             'validation axioms; an axiom is true iff its cost is below its relation\'s '
             'threshold; accuracy, precision, recall and F1 on the test axioms.'),
    components=('test set', 'negatives', 'thresholds', 'metrics'),
    applies_to=('Any model that scores the form (class_assertion, gci0 or '
                'object_property_assertion) through the model scoring interface'),
    references=(WANG2024EIKE, LV2018TRANSC),
    standalone=False,
    introduced_with=(),
    used_by=('eike',),
    implementation='mowl.evaluation.triple_classification.TripleClassificationRegime',
    limitations=('The released YAGO39K negatives are corruptions that are not in the knowledge '
                 'base; the world assumption is closed (a true but unstated axiom is a '
                 'negative). The threshold grid of 100 steps between the lowest and highest '
                 'validation cost is that of the reference code (grid=None searches every '
                 'midpoint).',),
    notes=('EIKE, Section 5.3 (Tables 2-4); TransC, Section 5. The task is a classification of '
           'candidate axioms; the closest task key is axiom-ranking.'),
)
