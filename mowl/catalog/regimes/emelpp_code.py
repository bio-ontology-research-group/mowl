"""Regime cards: the training design of the EmEL++ reference code (kracr/EmELpp), found by
comparing ``experiments/*/models/*_EmEL.py`` and ``*_EL.py`` with the paper (Mondal,
Bhatia, Mutharaju, AAAI-MAKE 2021) and with mOWL."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import MONDAL2021

REGIMES = (
    RegimeCard(
        key="validation-axioms-as-positives",
        name="Validation axioms in the training data",
        kind=RegimeKind.TRAINING,
        summary=(
            "The validation axioms of a normal form are added to the training data. "
            "Reproduces reference code that trains on an ontology from which only the "
            "test axioms were removed."),
        components=("positives",),
        applies_to="EmbeddingELModel",
        references=(MONDAL2021,),
        standalone=False,
        introduced_with=("emelpp",),
        implementation="mowl.training.positives.ValidationAxiomsAsPositives",
        limitations=(
            "Model selection on validation axioms that are also training axioms selects "
            "for fit, not for generalisation.",),
        notes=(
            "kracr/EmELpp trains on experiments/*/data/*_norm_mod.owl. Checked against the "
            "released files: every validation pair (GALEN 5,573, GO 17,093, ANATOMY "
            "24,355) is an asserted SubClassOf axiom of that ontology, no test pair is. "
            "The paper (Section 4.3) says that validation and test pairs were removed."),
    ),
    RegimeCard(
        key="emelpp-code-objective",
        name="Objective of the EmEL++ Keras code",
        kind=RegimeKind.TRAINING,
        summary=(
            "Per step the losses of row i of every normal form (positives, negatives "
            "without hinge, role axioms, the radius of owl:Thing pulled to 5, a "
            "negative-radius penalty) are summed and the mean of the squared row sums "
            "is minimised (Keras 'mse' against zero). C ⊑ D is trained with a learned "
            "SubClassOf translation; EmEL++ radii are max(0, r)."),
        components=("loss", "regularisation"),
        applies_to="ELEm, EmEL++",
        references=(MONDAL2021,),
        standalone=False,
        introduced_with=("emelpp",),
        used_by=("elem",),
        implementation="mowl.training.loss_forms.EmELppCodeObjective",
        limitations=(
            "Squaring the row sum lets a negative (unbounded) negative loss cancel "
            "positive losses of other normal forms in the same row; the objective is "
            "not the sum of the losses of the paper.",),
        notes=(
            "model.compile(optimizer, loss='mse') on the per-row output 'loss1 + ... + "
            "loss6 - loss7' of ELModel.call in *_EmEL.py (and 'loss1 + ... + "
            "loss_nf3_neg' in *_EL.py); nf1 axioms are stored as (C, SubClassOf, D) and "
            "scored as ‖c + r_sub - d‖; top_loss |r_Thing - 5|; radius_loss min(0, r)."),
    ),
    RegimeCard(
        key="emelpp-code-training",
        name="Training design of the EmEL++ code",
        kind=RegimeKind.TRAINING,
        summary=(
            "Training data with the validation pairs, 256 random axioms per normal form "
            "and step, negatives drawn once (different classes from the subsumption "
            "pairs), joint unit-norm initialisation, the Keras objective, 1,000 epochs "
            "of ceil(N/256) steps, and selection by the mean rank of the first 100 "
            "validation pairs after every epoch."),
        components=("positives", "batching", "negative sampling", "initialisation", "loss",
                    "model selection"),
        applies_to="ELEm, EmEL++",
        references=(MONDAL2021,),
        standalone=False,
        introduced_with=("emelpp",),
        used_by=("elem",),
        implementation="mowl.training.reference.EmELppCodeTraining",
        notes=(
            "A composition of validation-axioms-as-positives, per-normal-form-minibatch, "
            "fixed-negatives, joint-normalised-initialisation, emelpp-code-objective and "
            "validation-rank-selection. From experiments/{GALEN,GO,ANATOMY}/models/"
            "*_EmEL.py and *_EL.py of kracr/EmELpp (commit 267f2c7): batch_size 256, "
            "learning rate 3e-4, 1,000 epochs, steps_per_epoch = ceil(max table size / "
            "256). The paper gives neither the batch size, the learning rate nor the "
            "number of epochs."),
    ),
)
