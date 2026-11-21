"""Regime card: the training design of the faithfulness reference code (kbc)."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2026

REGIME = RegimeCard(
    key='kbc-code-training',
    assumption="closed world over stated axioms",
    name='Faithfulness reference code training (kbc run_el.py)',
    kind=RegimeKind.TRAINING,
    summary=('How run_el.py of the faithfulness paper trains ELEmbeddings, EmEL++ and '
             'Box2EL: one fixed corruption per C ⊑ ∃r.D axiom (superclass replaced, drawn '
             'once), the sum of all positive and negative per-axiom losses divided by their '
             'number, Adam with ReduceLROnPlateau (patience 10) on the validation loss of '
             'the normal form under test, early stopping after 20 epochs without '
             'improvement, best validation loss kept.'),
    components=('negative sampling', 'loss combination', 'optimiser schedule',
                'model selection'),
    applies_to='EmbeddingELModel with a negative loss for gci2 (ELEm, EmEL++, Box2EL)',
    references=(MASHKOVA2026,),
    standalone=False,
    introduced_with=(),
    used_by=('elem', 'emelpp', 'box2el'),
    implementation='mowl.training.kbc.KBCCodeTraining',
    limitations=(
        ("The reference code's own copies of the three models (box2el.py, "
         "elembeddings.py with loss_type='leaky_relu' and reg_mode='relaxed', emelpp.py) "
         "are not reimplemented; mOWL's modules are used, so per-axiom losses can differ "
         "(for ELEmbeddings: ReLU instead of leaky ReLU)."),
        ("In the released data every val.owl is train.owl under another ontology IRI (the "
         "files differ only in that IRI), so the 'validation loss' that drives the "
         "schedule, the early stopping and the checkpoint is the training loss of the "
         "normal form under test; mOWL keeps this as released."),
        ('Full batch only: the pooled denominator and the reuse of the negatives assume one '
         'batch per epoch (true for every split of the benchmark at batch size 32,768).'),
    ),
    notes=('bio-ontology-research-group/kbc at commit 905ec72: run_el.py defaults '
           '(neg_types gci2, weight_mode uniform, val_mode new, random_neg_fraction 1), '
           'box2el.py train() (ReduceLROnPlateau(patience=10), epochs_no_improve=20, '
           'checkpoint on the best validation loss) and dataloader.py (np.random.seed(0) at '
           'every batch, hence the same negatives in every epoch). The fallback '
           'hyperparameters of run_el.py are in the recipes '
           '(mowl.benchmarks.recipe_sets.mashkova2026).'),
)
