"""Regime card: pooled mean of all per-axiom losses."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2026

REGIME = RegimeCard(
    key='pooled-axiom-mean',
    name='Pooled mean of all per-axiom losses',
    kind=RegimeKind.TRAINING,
    summary=('The losses of all positive and negative axioms of all normal forms are summed '
             'and divided by their total number, so that large normal forms weigh more than '
             'in a sum of per-form means.'),
    components=('loss combination',),
    applies_to='EmbeddingELModel',
    references=(MASHKOVA2026,),
    standalone=True,
    introduced_with=(),
    used_by=('elem', 'emelpp', 'box2el'),
    implementation='mowl.training.kbc.PooledAxiomMeanLoss',
    limitations=('Assumes a fixed number of rows per epoch (full batch).',),
    notes=("weight_mode='uniform' of the faithfulness reference code (kbc, box2el.py and "
           'elembeddings.py train()), the default of its run_el.py.'),
)
