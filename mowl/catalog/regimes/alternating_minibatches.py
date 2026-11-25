"""Regime cards: the training of FaithEL as parts."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind

LACERDA2024FAITHEL = Reference(
    "lacerda2024faithel",
    "Lacerda, V., Ozaki, A., Guimarães, R. (2024). FaithEL: Strongly TBox Faithful "
    "Knowledge Base Embeddings for EL. In Rules and Reasoning (RuleML+RR 2024), 191-199.")

_FOR = "Models trained by the loop of mowl.training.loop or by the generic EL loop"

REGIMES = (
    RegimeCard(
        key="alternating-minibatches",
        name="Forms in turn, in minibatches",
        kind=RegimeKind.TRAINING,
        summary=("Per epoch the rows of each form in turn, in a fresh random order and in "
                 "minibatches; one optimisation step per minibatch."),
        components=("batching",),
        applies_to=_FOR,
        references=(LACERDA2024FAITHEL,),
        standalone=False,
        introduced_with=("faithel",),
        used_by=(),
        implementation="mowl.training.alternating.AlternatingMinibatches",
        notes=("The loop of the authors' code of FaithEL: the concept assertions, then the "
               "role assertions. Every step takes the loss of one form only; per-normal-form-"
               "minibatch takes a minibatch of every form in every step."),
    ),
    RegimeCard(
        key="full-batch",
        name="All rows of the chosen forms in one step",
        kind=RegimeKind.TRAINING,
        summary=("One step per epoch on all rows of the chosen forms; the loss is the sum of "
                 "the mean losses of the forms."),
        components=("batching",),
        applies_to=_FOR,
        references=(LACERDA2024FAITHEL,),
        standalone=False,
        introduced_with=(),
        used_by=("faithel",),
        implementation="mowl.training.alternating.FullBatch",
        notes="The generic loop of the EL models restricted to the ABox forms of FaithEL.",
    ),
    RegimeCard(
        key="form-selection",
        name="Train on some forms only",
        kind=RegimeKind.TRAINING,
        summary="Only the chosen forms are trained; the others are left out of the loop.",
        components=("positives",),
        applies_to=_FOR,
        references=(LACERDA2024FAITHEL,),
        standalone=False,
        introduced_with=("faithel",),
        used_by=(),
        implementation="mowl.training.alternating.FormSelection",
        notes=("FaithEL learns the ABox only: its TBox forms are read from a constructed "
               "model and have no parameters."),
    ),
)
