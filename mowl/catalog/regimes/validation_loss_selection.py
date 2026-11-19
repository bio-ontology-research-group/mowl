"""Regime card: Checkpoint selection by validation loss."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2024NESY, MASHKOVA2024DELE

REGIME = RegimeCard(
    key='validation-loss-selection',
    name='Checkpoint selection by validation loss',
    kind=RegimeKind.TRAINING,
    summary=('The parameters with the lowest loss on the validation axioms of the predicted '
             'normal form are kept.'),
    components=('model selection',),
    applies_to='Any model with validation data',
    references=(MASHKOVA2024NESY, MASHKOVA2024DELE),
    standalone=False,
    introduced_with=(),
    used_by=(),
    implementation='mowl.training.selection.ValidationSelection',
    limitations=(
        ('The validation loss of positive axioms alone does not penalise trivial models in which '
         'every axiom holds (NeSy 2024, Section 5 and Appendix I, on collapsed models); a rank '
         'metric does.'),
    ),
    notes=("Built into mOWL's EmbeddingELModel.train (best state_dict saved to model_filepath, "
           'eval_gci_name only). NeSy 2024, Section 4.4 and DELE, Section 6.3 (reference code: '
           'validation loss of gci2). The regime keeps the best state in memory; call finalize() '
           'after train().'),
)
