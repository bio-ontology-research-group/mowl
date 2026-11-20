"""Regime card: the subsumption-ranking evaluation of ELK-Em."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._benchmark_refs import JACKERMEIER2024
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='elkem-subsumption-ranking',
    name='Unfiltered subsumption ranking with averaged ties (ELK-Em)',
    kind=RegimeKind.EVALUATION,
    summary=(
        'For every test subsumption C ⊑ D, the superclass D is ranked among all classes '
        'except owl:Thing and C by the margin-free loss ℓ0 (equivalently the score '
        'exp(−ℓ0)). Tied candidates share their average rank; ranks are unfiltered; '
        'Hits@k counts fractional ranks ≤ k; AUC is mean((n − r)/(n − 1)).'),
    components=('test set', 'candidate set', 'scoring', 'metrics'),
    applies_to='EmbeddingELModel (modules with zero_margin_loss; others are scored by '
               'their loss)',
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=(),
    implementation='mowl.evaluation.normal_form_ranking.ELKEmSubsumptionRankingRegime',
    limitations=(
        ('owl:Thing is removed from the candidates, although the authors\' own '
         "specification (§8 D8) keeps it; their evaluation code removes it because the "
         'full box of owl:Thing saturates the score and would dominate every tie block.'),
        ('With average-rank ties a model that scores many candidates equally cannot get '
         'Hits@1, so the numbers are not comparable with tie-by-position rankings such as '
         "the 'inferred-subsumption-ranking' regime of Box2EL on the same data."),
    ),
    notes=(
        'ELK-Em, Section 5.1, Table 3; paper/ranking/eval.py (evaluate_ranking_metrics) '
        'of city-artificial-intelligence/elk-em. The test pairs are the atomic subsumptions '
        'inferred by an EL++ reasoner but not asserted, on the 90/10 test/validation split '
        'of the Box2EL reference release (Jackermeier et al. 2024, data.zip, '
        '<ONT>/inferences/). Benchmarks: elkem-{galen,go,anatomy}-subsumption. The paper '
        'does not describe tie handling; the code averages ties (§8 D7) and masks '
        'owl:Thing, and both are part of this regime.'),
)
