"""Regime card: the training design of the ELK-Em code."""

from mowl.catalog.card import RegimeCard, RegimeKind, Guarantee, GuaranteeKind
from mowl.catalog.regimes._regime_refs import SINGH2026

REGIME = RegimeCard(
    key='elkem-code-training',
    assumption="closed world over stated axioms",
    name='ELK-Em code training',
    kind=RegimeKind.TRAINING,
    summary=(
        "How the ranking experiments of the ELK-Em paper were trained: one optimisation "
        "step per epoch on a mini-batch of every normal form, the sum over normal forms of "
        "the mean *squared* per-axiom losses, a Pearson correlation contrastive term "
        "(weight 3.0) on the margin-free scores of k_neg corruptions of C ⊑ D (D replaced) "
        "and of C ⊑ ∃r.D (either side), a step learning-rate decay, and checkpoint "
        "selection by the median rank of the validation subsumptions."),
    components=('batching', 'negative sampling', 'loss combination', 'optimiser schedule',
                'model selection'),
    applies_to='mowl.models.elkem.model.ELKEm (TBox training)',
    references=(SINGH2026,),
    standalone=False,
    introduced_with=('elkem',),
    used_by=(),
    implementation='mowl.training.reference.ELKEmCodeTraining',
    properties=(
        Guarantee(
            kind=GuaranteeKind.REGIME_PROPERTY,
            statement=('Corruptions are not filtered against entailed axioms, as the paper '
                       "states ('without filtering corruptions that happen to be "
                       "entailed'); ⊤, ⊥ and unsatisfiable concepts are never drawn as "
                       'replacements.'),
            source='ELK-Em, Section 4.2; ELKEm._class_pool',
            condition='always',
        ),
    ),
    limitations=(
        ('The ABox losses and the lexical regulariser of the protein-function experiments '
         '(Table 4) are not part of this regime; they are the regimes assertion-corruption and lexical-regularisation of the default regime elkem-training.'),
        ("The code multiplies the total loss by 1,000; Adam is invariant to a constant "
         "loss factor up to its epsilon, so mOWL leaves the factor out."),
        ('A replacement equal to the corrupted column is redrawn, but C ⊑ C can still be '
         'drawn for C ⊑ D (the code also redraws D = C).'),
        ("The code draws the C ⊑ ∃r.D positives of the Pearson term and of the positive "
         "loss as two independent batches; mOWL uses one batch for both."),
        ("The geometry of the code (interval-arithmetic existential boxes, an overlap term "
         "in the C ⊑ ∃r.D loss) belongs to the model: ELKEm(variant='reference')."),
    ),
    notes=(
        'ELK-Em, Section 5.1 and paper/ranking/train.py of '
        'city-artificial-intelligence/elk-em: Adam, learning rate 0.01 decayed by 0.1 '
        'after 5,000 epochs, margin 0.1, batch 1,024 per normal form, 10,000 epochs, '
        'k_neg 3 (GALEN, Anatomy) or 5 (GO), contrastive factor 3.0 and box-size '
        'regulariser 1.0 from the script defaults, validation median rank every 100 '
        'epochs. Parameters positive_power (2: code, 1: paper), margin_free_contrast '
        '(True: code) and corruption_pool (\'code\': all classes but owl:Thing and '
        'owl:Nothing; \'satisfiable\': also without unsatisfiable classes) switch between '
        'the code and the paper text. Axioms whose concept is fixed to the empty box are '
        'dropped or rerouted to their _bot normal form and nf1_bot is not trained. The '
        'published Table 3 numbers were obtained with this regime (recipes: '
        'mowl.benchmarks.recipe_sets.singh2026).'),
)
