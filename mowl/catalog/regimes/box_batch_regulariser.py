"""Regime card: Box regulariser of the batches (Box2EL code)."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind

JACKERMEIER2024B = Reference(
    "jackermeier2024box2elcode",
    "Jackermeier, M., Chen, J., Horrocks, I. (2024). Dual Box Embeddings for the Description "
    "Logic EL++ (code of the BoxEL baseline, model/BoxEL.py). WWW 2024.",
    "https://doi.org/10.1145/3589334.3645648")

REGIME = RegimeCard(
    key='box-batch-regulariser',
    name='Box regulariser of the batches (Box2EL code)',
    kind=RegimeKind.TRAINING,
    summary=(
        'Every box of every batch of a step (the classes of an axiom, the intersection,'
        'the translated box, corrupted axioms included) adds the mean excess of the upper'
        'corners over 1 plus the excess of the norm of the lower corners over 1; this'
        'replaces the regularisation term of the step.'),
    components=('regularisation',),
    applies_to='Geometries with box classes and an affine relation model (BoxEL, BoxSEL)',
    references=(JACKERMEIER2024B,),
    standalone=False,
    introduced_with=('boxel',),
    used_by=(),
    implementation='mowl.training.box_regularization.BoxBatchRegulariser',
    notes=(
        'The BoxEL baseline of the Box2EL code (l2_side_regularizer); part of box2el-'
        'baseline-code training. It depends on the batches, so it is a regime and not a'
        'penalty on the tables.'),
)
